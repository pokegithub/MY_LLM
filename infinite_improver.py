"""
infinite_improver.py — Self-improvement loop via uncertainty-guided
                        synthetic data generation

Architecture:
  1. IDENTIFY: Use the confidence head to find high-uncertainty prompts
  2. GENERATE: Sample N diverse candidate completions per prompt
  3. SCORE: Rank candidates via self-consistency, execution (for code),
            and cross-entropy under the model itself
  4. PAIR: Create (chosen, rejected) preference pairs from ranked candidates
  5. REFINE: Fine-tune via DPO on synthetic pairs
  6. ITERATE: Repeat with the improved model

Safety mechanisms:
  - Diversity penalty prevents mode collapse
  - Original training data mixed into each iteration
    - Automatic quality gate: iteration is rolled back if uncertainty regresses
  - Maximum iteration count as hard stop

This implements the "self-play" improvement paradigm adapted for SLMs,
where the model's own uncertainty signal drives curriculum selection.
"""

import os
import sys
import json
import time
import math
import random
import hashlib
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

sys.path.insert(0, ".")
from config import (
    ImproverConfig, DPOConfig,
    model_cfg, improver_cfg, dpo_cfg,
)
from core.sequence_ops import gather_token_log_probs
from tokenizer import BPETokenizer
from model import LLM
from security.validator import (
    ValidationError,
    execute_python_snippet,
    get_allowed_data_roots,
    iter_jsonl,
)
from core.checkpoint_io import (
    atomic_torch_save,
    extract_state_dict,
    safe_torch_load,
)


def _set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Prompt sources for self-improvement
# ---------------------------------------------------------------------------

SEED_PROMPTS = [
    # Reasoning
    "Explain step by step how to solve: If a train travels 120 miles in 2 hours, what is its average speed?",
    "What is the logical fallacy in: 'Everyone I know likes pizza, therefore all humans like pizza'?",
    "Solve: A farmer has 17 sheep. All but 9 die. How many are left?",
    # Code
    "Write a Python function to find the longest palindromic substring.",
    "Implement a binary search tree with insert, delete, and search operations in Python.",
    "Write a function that merges two sorted linked lists into one sorted linked list.",
    # Knowledge
    "Explain the difference between TCP and UDP protocols.",
    "What causes the seasons on Earth? Explain the physics.",
    "Describe how a transformer neural network processes a sequence of tokens.",
    # Math
    "Prove that the square root of 2 is irrational.",
    "What is the derivative of x^x and why?",
    "Explain the Monty Hall problem and its solution.",
    # Creative
    "Write a short story about a robot learning to paint.",
    "Explain quantum entanglement using only metaphors from cooking.",
]


# ---------------------------------------------------------------------------
# Uncertainty-based prompt selection
# ---------------------------------------------------------------------------

def identify_uncertain_prompts(
    model: LLM,
    tok: BPETokenizer,
    prompts: List[str],
    device: str,
    threshold: float = 0.5,
    max_prompts: int = 100,
) -> List[Tuple[str, float]]:
    """
    Score prompts by model uncertainty via the confidence head.
    Returns prompts sorted by uncertainty (highest first).
    """
    was_training = model.training
    model.eval()
    scored = []

    for prompt in prompts:
        ids = tok.encode(prompt, add_bos=True)
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(ids_tensor)

        if out.unc_scores is not None:
            # Average uncertainty across sequence positions
            # unc_scores[..., 0] = correctness confidence
            # Lower confidence = higher uncertainty
            avg_conf = out.unc_scores[..., 0].mean().item()
            uncertainty = 1.0 - avg_conf
        else:
            # Fallback: use entropy of output distribution
            logits = out.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * probs.log().clamp(min=-100)).sum().item()
            max_ent = math.log(model.cfg.vocab_size)
            uncertainty = entropy / max_ent

        scored.append((prompt, uncertainty))

    # Sort by uncertainty (highest first)
    scored.sort(key=lambda x: x[1], reverse=True)
    if was_training:
        model.train()
    return scored[:max_prompts]


# ---------------------------------------------------------------------------
# Diverse candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    model: LLM,
    tok: BPETokenizer,
    prompt: str,
    device: str,
    n_candidates: int = 4,
    max_len: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    diversity_penalty: float = 0.1,
) -> List[Tuple[str, Optional[torch.Tensor]]]:
    """
    Generate diverse candidate completions for a prompt.
    Uses varied temperatures for diversity.
    Returns list of (completion_text, uncertainty_scores).
    """
    was_training = model.training
    model.eval()
    candidates = []

    # Vary temperature for diversity
    temperatures = [
        max(0.1, temperature - 0.3),
        temperature,
        min(1.5, temperature + 0.3),
        min(1.8, temperature + 0.5),
    ][:n_candidates]
    while len(temperatures) < n_candidates:
        temperatures.append(temperature + random.uniform(-0.2, 0.4))
    temperatures = [max(0.1, min(2.0, t)) for t in temperatures]

    prompt_text = f"{tok.USR}\n{prompt}{tok.END}\n{tok.AST}\n"
    prompt_ids = tok.encode(prompt_text, add_bos=True)
    prompt_tensor = torch.tensor(
        [prompt_ids], dtype=torch.long, device=device
    )

    for temp in temperatures:
        try:
            generated, unc = model.generate(
                prompt_tensor,
                max_new_tokens=max_len,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=tok.eos_token_id,
            )
            # Decode only the new tokens
            new_ids = generated[0, len(prompt_ids):].tolist()
            text = tok.decode(new_ids, skip_special=True).strip()
            if len(text) > 10:
                candidates.append((text, unc))
        except Exception as e:
            print(f"    Gen error: {str(e)[:50]}", flush=True)

    # Deduplicate by content similarity
    if diversity_penalty > 0:
        unique = []
        seen_hashes = set()
        for text, unc in candidates:
            # Simple hash-based dedup
            h = hashlib.md5(text[:200].encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append((text, unc))
        candidates = unique

    if was_training:
        model.train()
    return candidates


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def score_candidate(
    model: LLM,
    tok: BPETokenizer,
    prompt: str,
    completion: str,
    device: str,
    unc_scores: Optional[torch.Tensor] = None,
) -> float:
    """
    Score a candidate completion. Higher = better.

    Combines:
      1. Negative perplexity (model's own assessment)
      2. Confidence head score (if available)
      3. Length-normalized log probability
      4. Structural quality heuristics
    """
    # Encode full sequence
    full_text = f"{tok.USR}\n{prompt}{tok.END}\n{tok.AST}\n{completion}{tok.END}"
    ids = tok.encode(full_text, add_bos=True)
    if len(ids) < 5:
        return -100.0

    ids_tensor = torch.tensor(
        [ids[:model.cfg.max_seq_len]], dtype=torch.long, device=device
    )

    with torch.no_grad():
        out = model(ids_tensor)

    logits = out.logits[:, :-1, :]
    targets = ids_tensor[:, 1:]
    token_lp = gather_token_log_probs(logits, targets)

    # Length-normalized log probability
    seq_lp = token_lp.sum().item() / max(token_lp.shape[1], 1)

    # Confidence score
    conf_score = 0.0
    if out.unc_scores is not None:
        conf_score = out.unc_scores[..., 0].mean().item()

    # Structural quality
    struct_score = 0.0
    if len(completion) > 50:
        struct_score += 0.1
    if any(c in completion for c in ".!?"):
        struct_score += 0.1
    if completion.count("\n") > 0:
        struct_score += 0.05
    # Penalize very short or very repetitive outputs
    words = completion.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        struct_score += 0.1 * unique_ratio

    # Combined score
    return seq_lp + 0.5 * conf_score + struct_score


def rank_and_pair(
    scored_candidates: List[Tuple[str, float]],
) -> List[Tuple[str, str]]:
    """
    Given scored candidates [(text, score)], create preference pairs.
    Pairs the best with the worst, second-best with second-worst, etc.
    """
    if len(scored_candidates) < 2:
        return []

    sorted_cands = sorted(
        scored_candidates, key=lambda x: x[1], reverse=True
    )
    pairs = []

    n = len(sorted_cands)
    for i in range(n // 2):
        chosen = sorted_cands[i][0]
        rejected = sorted_cands[n - 1 - i][0]
        # Only pair if there's meaningful score difference
        if sorted_cands[i][1] - sorted_cands[n - 1 - i][1] > 0.01:
            pairs.append((chosen, rejected))

    return pairs


# ---------------------------------------------------------------------------
# Code execution verification
# ---------------------------------------------------------------------------

def verify_code_output(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Attempt to execute Python code and verify it runs without errors.
    Returns (success, output_or_error).
    """
    return execute_python_snippet(code, timeout=timeout)


# ---------------------------------------------------------------------------
# Main improvement loop
# ---------------------------------------------------------------------------

def improve(cfg: ImproverConfig = improver_cfg):
    _set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tok = BPETokenizer()
    tok.load(cfg.tokenizer_path)
    model_cfg.vocab_size = tok.vocab_size_

    # Load model
    model = LLM(model_cfg).to(device)
    if os.path.exists(cfg.model_checkpoint):
        try:
            st, path = safe_torch_load(cfg.model_checkpoint, map_location=device)
            model.load_state_dict(extract_state_dict(st))
            print(f"  Loaded model: {path}")
        except FileNotFoundError:
            print("  No checkpoint found; starting from current initialized model.")

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.synth_data_dir, exist_ok=True)

    # Collect seed prompts + any user-provided prompts
    prompts = list(SEED_PROMPTS)
    allowed_roots = get_allowed_data_roots()
    max_json_bytes = int(
        os.environ.get("MYLLM_MAX_DATA_JSON_BYTES", "536870912")
    )
    max_jsonl_line_bytes = int(
        os.environ.get("MYLLM_MAX_DATA_JSONL_LINE_BYTES", "4000000")
    )
    prompt_file = os.path.join(cfg.synth_data_dir, "prompts.jsonl")
    if os.path.exists(prompt_file):
        try:
            for item in iter_jsonl(
                prompt_file,
                max_bytes=max_json_bytes,
                max_line_bytes=max_jsonl_line_bytes,
                allowed_roots=allowed_roots,
            ):
                if isinstance(item, dict):
                    prompts.append(item.get("prompt", item.get("text", "")))
        except (ValidationError, OSError, ValueError) as exc:
            print(f"  Prompt ingestion warning: {str(exc)[:120]}")
        prompts = [p for p in prompts if len(p) > 10]

    print(f"\n{'=' * 60}")
    print(f"Infinite Improver: {cfg.n_iterations} iterations")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Candidates per prompt: {cfg.n_candidates}")
    print(f"  Uncertainty threshold: {cfg.unc_threshold}")
    print(f"{'=' * 60}")
    current_model_checkpoint = cfg.model_checkpoint
    iter_reports = []

    for iteration in range(cfg.n_iterations):
        print(f"\n--- Iteration {iteration + 1}/{cfg.n_iterations} ---")
        iter_t0 = time.time()

        # Step 1: Identify uncertain prompts
        print("  Step 1: Identifying uncertain prompts...")
        uncertain = identify_uncertain_prompts(
            model, tok, prompts, device,
            threshold=cfg.unc_threshold,
            max_prompts=cfg.samples_per_iter,
        )
        high_unc = [
            (p, u) for p, u in uncertain if u > cfg.unc_threshold
        ]
        if not high_unc:
            high_unc = uncertain[:min(50, len(uncertain))]
        print(
            f"    Found {len(high_unc)} high-uncertainty prompts "
            f"(avg unc: {sum(u for _, u in high_unc) / max(len(high_unc), 1):.3f})"
        )

        # Step 2: Generate candidates
        print("  Step 2: Generating candidates...")
        all_pairs = []
        for pi, (prompt, unc) in enumerate(high_unc):
            if pi % 10 == 0:
                print(
                    f"    Prompt {pi + 1}/{len(high_unc)} "
                    f"(unc={unc:.3f})",
                    flush=True,
                )

            candidates = generate_candidates(
                model, tok, prompt, device,
                n_candidates=cfg.n_candidates,
                max_len=cfg.max_gen_len,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                diversity_penalty=cfg.diversity_penalty,
            )

            if len(candidates) < 2:
                continue

            # Step 3: Score candidates
            scored = []
            for text, unc_tensor in candidates:
                score = score_candidate(
                    model, tok, prompt, text, device, unc_tensor
                )

                # Bonus for code that executes
                if "def " in text or "import " in text:
                    # Extract code blocks
                    code = text
                    if "```python" in text:
                        parts = text.split("```python")
                        if len(parts) > 1:
                            code = parts[1].split("```")[0]
                    success, _ = verify_code_output(code)
                    if success:
                        score += 0.5

                scored.append((text, score))

            # Step 4: Create preference pairs
            pairs = rank_and_pair(scored)
            for chosen, rejected in pairs:
                all_pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "iteration": iteration,
                })

        print(f"    Generated {len(all_pairs)} synthetic preference pairs")

        # Mix anchor preference data to reduce drift across iterations.
        anchor_added = 0
        if os.path.exists(dpo_cfg.data_path):
            try:
                for item in iter_jsonl(
                    dpo_cfg.data_path,
                    max_bytes=max_json_bytes,
                    max_line_bytes=max_jsonl_line_bytes,
                    allowed_roots=allowed_roots,
                ):
                    if anchor_added >= max(50, len(all_pairs) // 2):
                        break
                    if isinstance(item, dict) and all(
                        k in item for k in ("prompt", "chosen", "rejected")
                    ):
                        all_pairs.append(
                            {
                                "prompt": item["prompt"],
                                "chosen": item["chosen"],
                                "rejected": item["rejected"],
                                "iteration": iteration,
                                "anchor": True,
                            }
                        )
                        anchor_added += 1
            except (ValidationError, OSError, ValueError) as e:
                print(f"    Anchor mix skipped: {str(e)[:80]}")

        if anchor_added:
            print(f"    Added {anchor_added} anchor pairs from {dpo_cfg.data_path}")

        if len(all_pairs) < 10:
            print("    Too few pairs, skipping iteration")
            iter_reports.append({
                "iteration": iteration,
                "status": "skipped",
                "high_uncertainty_prompts": len(high_unc),
                "synthetic_pairs": len(all_pairs),
                "anchor_pairs": anchor_added,
                "reason": "too_few_pairs",
                "elapsed_sec": round(time.time() - iter_t0, 2),
            })
            continue

        # Save synthetic data
        synth_path = os.path.join(
            cfg.synth_data_dir,
            f"synth_iter_{iteration:03d}.jsonl",
        )
        with open(synth_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair) + "\n")
        print(f"    Saved: {synth_path}")

        # Step 5: DPO training on synthetic + anchor pairs
        print("  Step 5: DPO refinement...")
        prev_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        iter_dpo_cfg = DPOConfig(
            checkpoint_path=current_model_checkpoint,
            ref_checkpoint=current_model_checkpoint,
            output_dir=os.path.join(
                cfg.output_dir, f"iter_{iteration:03d}"
            ),
            tokenizer_path=cfg.tokenizer_path,
            data_path=synth_path,
            total_steps=cfg.dpo_steps_per_iter,
            batch_size=8,
            grad_accum=4,
            max_lr=2e-6,
            beta=0.1,
        )

        # Import and run DPO
        from alignment import dpo_train
        dpo_train(iter_dpo_cfg)

        # Load improved model
        iter_dir = iter_dpo_cfg.output_dir
        if os.path.exists(iter_dir):
            try:
                st, path = safe_torch_load(iter_dir, map_location=device)
                model.load_state_dict(extract_state_dict(st))
                print(f"    Loaded improved model: {path}")
            except FileNotFoundError:
                print("    No improved checkpoint found in iteration output.")

        # Step 6: Quality gate
        # Quick self-eval: check if uncertainty decreased
        print("  Step 6: Quality gate check...")
        post_uncertain = identify_uncertain_prompts(
            model, tok,
            [p for p, _ in high_unc[:20]],
            device,
        )
        avg_post_unc = sum(u for _, u in post_uncertain) / max(
            len(post_uncertain), 1
        )
        avg_pre_unc = sum(u for _, u in high_unc[:20]) / max(
            min(20, len(high_unc)), 1
        )

        print(
            f"    Uncertainty: {avg_pre_unc:.3f} -> {avg_post_unc:.3f} "
            f"({'improved' if avg_post_unc < avg_pre_unc else 'REGRESSED'})"
        )

        if avg_post_unc > avg_pre_unc + 0.1:
            print(
                "    QUALITY GATE FAILED — reverting to previous model"
            )
            model.load_state_dict(prev_state)
            print("    Reverted model weights for this iteration")
            reverted = True
        else:
            # Update checkpoint reference for next iteration
            current_model_checkpoint = iter_dir
            reverted = False

        iter_reports.append({
            "iteration": iteration,
            "status": "completed",
            "high_uncertainty_prompts": len(high_unc),
            "synthetic_pairs": len(all_pairs),
            "anchor_pairs": anchor_added,
            "avg_uncertainty_before": round(avg_pre_unc, 6),
            "avg_uncertainty_after": round(avg_post_unc, 6),
            "quality_gate_reverted": reverted,
            "checkpoint_used": current_model_checkpoint,
            "elapsed_sec": round(time.time() - iter_t0, 2),
        })

        elapsed = time.time() - iter_t0
        print(
            f"  Iteration {iteration + 1} complete: "
            f"{elapsed / 60:.1f}min, {len(all_pairs)} pairs"
        )

    # Final save
    final_path = os.path.join(cfg.output_dir, "improved_final.pt")
    atomic_torch_save(
        {"model_state_dict": model.state_dict()},
        final_path,
    )
    report_path = os.path.join(cfg.output_dir, "improver_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "iterations_requested": cfg.n_iterations,
                "iterations": iter_reports,
                "final_checkpoint": final_path,
            },
            f,
            indent=2,
        )
    print(f"\nImprovement complete! Final model: {final_path}")
    print(f"Improvement report: {report_path}")


if __name__ == "__main__":
    improve()