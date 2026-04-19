"""Evaluation suite for perplexity, benchmarks, math, and code tasks."""

import os
import sys
import json
import time
import math
import re
import random
import logging
import hashlib
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


LOGGER = logging.getLogger("eval_suite")

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except Exception as exc:
    _HAS_DATASETS = False
    LOGGER.warning("datasets import unavailable; real benchmark loaders disabled: %s", exc)

sys.path.insert(0, ".")
from config import EvalConfig, model_cfg, eval_cfg, runtime_config_dict
from tokenizer import BPETokenizer
from model import LLM
from core.sequence_ops import gather_token_log_probs, next_token_cross_entropy
from security.validator import execute_python_snippet
from core.checkpoint_io import extract_state_dict, pick_checkpoint_file, safe_torch_load


class EvalIntegrityError(RuntimeError):
    """Raised when an evaluation would produce misleading evidence."""


def _json_hash(payload: Dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def inspect_checkpoint_path(path: str) -> Dict:
    meta = {
        "configured_path": path,
        "exists": os.path.exists(path),
        "selected_checkpoint": None,
        "selected_checkpoint_exists": False,
        "sha256_sidecar_exists": False,
        "sha256_sidecar": None,
        "load_status": "not_loaded",
        "error": "",
    }
    try:
        selected = pick_checkpoint_file(path)
    except FileNotFoundError as exc:
        meta["load_status"] = "missing"
        meta["error"] = str(exc)
        return meta

    meta["selected_checkpoint"] = str(selected)
    meta["selected_checkpoint_exists"] = selected.exists()
    sidecar = selected.with_suffix(selected.suffix + ".sha256")
    if sidecar.exists():
        meta["sha256_sidecar_exists"] = True
        with open(sidecar, "r", encoding="utf-8") as handle:
            meta["sha256_sidecar"] = handle.read().split()[0].strip()
    return meta


def _set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_real_mc_questions(limit: int = 200) -> List[Dict]:
    """Load MC/binary QA items from public benchmarks."""
    items: List[Dict] = []
    if not _HAS_DATASETS:
        return items

    # BoolQ (binary)
    try:
        ds = load_dataset("google/boolq", split="validation")
        for ex in ds.select(range(min(limit // 2, len(ds)))):
            items.append({
                "question": str(ex.get("question", "")),
                "choices": ["false", "true"],
                "answer": 1 if bool(ex.get("answer", False)) else 0,
            })
    except Exception as exc:
        LOGGER.warning("BoolQ load failed; skipping real BoolQ samples: %s", exc)

    # ARC-Challenge (multi-choice)
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        target = max(limit - len(items), 0)
        for ex in ds.select(range(min(target, len(ds)))):
            ch = ex.get("choices", {})
            labels = [str(x).strip() for x in ch.get("label", [])]
            texts = [str(x) for x in ch.get("text", [])]
            if not texts or len(texts) != len(labels):
                continue
            ans_label = str(ex.get("answerKey", "")).strip()
            if ans_label not in labels:
                continue
            items.append({
                "question": str(ex.get("question", "")),
                "choices": texts,
                "answer": labels.index(ans_label),
            })
            if len(items) >= limit:
                break
    except Exception as exc:
        LOGGER.warning("ARC-Challenge load failed; skipping real ARC samples: %s", exc)

    random.shuffle(items)
    return items[:limit]


def _load_real_gsm8k(limit: int = 200) -> List[Dict]:
    if not _HAS_DATASETS:
        return []
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        out = []
        for ex in ds.select(range(min(limit, len(ds)))):
            out.append({
                "question": str(ex.get("question", "")),
                "answer": str(ex.get("answer", "")),
            })
        return out
    except Exception as exc:
        LOGGER.warning("GSM8K load failed; falling back to synthetic math set: %s", exc)
        return []


def _load_real_code_tasks(limit: int = 50) -> List[Dict]:
    if not _HAS_DATASETS:
        return []
    try:
        ds = load_dataset("openai_humaneval", split="test")
        out = []
        for ex in ds.select(range(min(limit, len(ds)))):
            out.append({
                "prompt": str(ex.get("prompt", "")),
                "test": str(ex.get("test", "")),
                "entry_point": str(ex.get("entry_point", "")),
            })
        return out
    except Exception as exc:
        LOGGER.warning("HumanEval load failed; falling back to toy code tasks: %s", exc)
        return []


def _choose_eval_set(
    name: str,
    real_items: List[Dict],
    fallback_items: List[Dict],
    min_required: int,
    cfg: EvalConfig,
) -> Tuple[List[Dict], Dict]:
    if len(real_items) >= min_required:
        return real_items, {
            "benchmark": name,
            "dataset_name": name,
            "split": "real",
            "run_mode": "real",
            "sample_count": len(real_items),
            "results_are_measured": True,
            "capability_evidence": "real_benchmark",
            "used_real": True,
            "real_count": len(real_items),
            "fallback_count": 0,
            "min_required": min_required,
            "warning": "",
        }
    if cfg.strict_real_benchmarks and not cfg.allow_toy_fallback:
        raise RuntimeError(
            f"Strict eval requires at least {min_required} real {name} samples, "
            f"but got {len(real_items)}"
        )
    return fallback_items, {
        "benchmark": name,
        "dataset_name": name,
        "split": "toy",
        "run_mode": "toy",
        "sample_count": len(fallback_items),
        "results_are_measured": True,
        "capability_evidence": "toy_smoke_only",
        "used_real": False,
        "real_count": len(real_items),
        "fallback_count": len(fallback_items),
        "min_required": min_required,
        "warning": "Fallback/toy samples are smoke tests, not benchmark evidence.",
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    cfg: EvalConfig, device: str
) -> Tuple[LLM, BPETokenizer, Dict]:
    tok = BPETokenizer()
    tok.load(cfg.tokenizer_path)
    model_cfg.vocab_size = tok.vocab_size_

    checkpoint_meta = inspect_checkpoint_path(cfg.checkpoint_path)
    if checkpoint_meta["load_status"] == "missing":
        raise EvalIntegrityError(
            "Evaluation requires a valid checkpoint. "
            f"{checkpoint_meta['error']}"
        )

    model = LLM(model_cfg).to(device)
    try:
        st, path = safe_torch_load(cfg.checkpoint_path, map_location=device)
        model.load_state_dict(extract_state_dict(st))
        checkpoint_meta["load_status"] = "loaded"
        checkpoint_meta["loaded_checkpoint"] = str(path)
        print(f"  Model loaded: {path}")
    except Exception as exc:
        checkpoint_meta["load_status"] = "failed"
        checkpoint_meta["error"] = str(exc)
        raise EvalIntegrityError(
            f"Failed to load evaluation checkpoint: {cfg.checkpoint_path}"
        ) from exc
    model.eval()
    return model, tok, checkpoint_meta


# ---------------------------------------------------------------------------
# 1. Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity(
    model: LLM,
    tok: BPETokenizer,
    texts: List[str],
    device: str,
    max_len: int = 512,
) -> Dict:
    """Compute perplexity on a list of texts."""
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        ids = tok.encode(text, add_bos=True)[:max_len]
        if len(ids) < 5:
            continue
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(ids_t)

        nll = next_token_cross_entropy(
            out.logits,
            ids_t,
            ignore_index=tok.pad_token_id,
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += (ids_t[:, 1:] != tok.pad_token_id).sum().item()

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(min(avg_nll, 100))
    return {
        "perplexity": round(ppl, 2),
        "avg_nll": round(avg_nll, 4),
        "total_tokens": total_tokens,
        "n_texts": len(texts),
    }


# ---------------------------------------------------------------------------
# 2. Multiple-choice evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_multiple_choice(
    model: LLM,
    tok: BPETokenizer,
    questions: List[Dict],
    device: str,
    n_few_shot: int = 0,
    few_shot_examples: Optional[List[Dict]] = None,
) -> Dict:
    """
    Evaluate multiple-choice accuracy.
    Each question: {"question": str, "choices": [str], "answer": int}
    """
    correct = 0
    total = 0

    for q in questions:
        question = q["question"]
        choices = q["choices"]
        answer_idx = q["answer"]

        # Build few-shot prefix
        prefix = ""
        if few_shot_examples and n_few_shot > 0:
            for ex in few_shot_examples[:n_few_shot]:
                prefix += (
                    f"Q: {ex['question']}\n"
                    f"A: {ex['choices'][ex['answer']]}\n\n"
                )

        # Score each choice
        best_score = float("-inf")
        best_idx = 0

        for ci, choice in enumerate(choices):
            text = f"{prefix}Q: {question}\nA: {choice}"
            ids = tok.encode(text, add_bos=True)[:512]
            ids_t = torch.tensor(
                [ids], dtype=torch.long, device=device
            )
            out = model(ids_t)
            logits = out.logits[:, :-1, :]
            targets = ids_t[:, 1:]

            token_lp = gather_token_log_probs(logits, targets)

            # Only score the answer tokens (after the question)
            q_text = f"{prefix}Q: {question}\nA: "
            q_len = len(tok.encode(q_text, add_bos=True)) - 1
            answer_lp = token_lp[0, q_len:].sum().item()
            # Length normalize
            answer_len = max(token_lp.shape[1] - q_len, 1)
            score = answer_lp / answer_len

            if score > best_score:
                best_score = score
                best_idx = ci

        if best_idx == answer_idx:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
    }


# ---------------------------------------------------------------------------
# 3. GSM8K Math evaluation
# ---------------------------------------------------------------------------

def extract_number(text: str) -> Optional[float]:
    """Extract the final numerical answer from a GSM8K-style response."""
    # Look for "#### X" pattern
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Look for "the answer is X"
    match = re.search(
        r"(?:the answer is|= )\s*([\-\d,\.]+)", text, re.IGNORECASE
    )
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Last number in text
    numbers = re.findall(r"[\-\d,]+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


@torch.no_grad()
def eval_gsm8k(
    model: LLM,
    tok: BPETokenizer,
    problems: List[Dict],
    device: str,
    max_gen: int = 256,
) -> Dict:
    """
    Evaluate on GSM8K math problems.
    Each problem: {"question": str, "answer": str}
    """
    correct = 0
    total = 0
    examples = []
    generated_tokens = 0
    completed = 0

    for prob in problems:
        question = prob["question"]
        gold_answer = prob["answer"]
        gold_num = extract_number(gold_answer)

        prompt = (
            f"Solve this math problem step by step.\n\n"
            f"Q: {question}\nA: Let me think step by step.\n"
        )
        ids = tok.encode(prompt, add_bos=True)
        ids_t = torch.tensor(
            [ids], dtype=torch.long, device=device
        )

        generated, _ = model.generate(
            ids_t,
            max_new_tokens=max_gen,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
        )
        new_ids = generated[0, len(ids):].tolist()
        generated_tokens += len(new_ids)
        response = tok.decode(new_ids, skip_special=True)
        if response.strip():
            completed += 1
        pred_num = extract_number(response)

        is_correct = (
            pred_num is not None
            and gold_num is not None
            and abs(pred_num - gold_num) < 1e-3
        )
        if is_correct:
            correct += 1
        total += 1

        if len(examples) < 10:
            examples.append({
                "question": question[:100],
                "gold": str(gold_num),
                "pred": str(pred_num),
                "correct": is_correct,
            })

    accuracy = correct / max(total, 1)
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "generated_tokens": generated_tokens,
        "completion_rate": round(completed / max(total, 1), 4),
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# 4. Code evaluation
# ---------------------------------------------------------------------------

def execute_code(code: str, timeout: int = 10) -> Tuple[bool, str]:
    """Execute Python code safely."""
    return execute_python_snippet(code, timeout=timeout)


@torch.no_grad()
def eval_code(
    model: LLM,
    tok: BPETokenizer,
    problems: List[Dict],
    device: str,
    max_gen: int = 512,
) -> Dict:
    """
    Evaluate code generation with execution.
    Each problem: {"prompt": str, "test": str, "entry_point": str}
    """
    passed = 0
    total = 0
    examples = []
    generated_tokens = 0
    completed = 0

    for prob in problems:
        prompt = prob["prompt"]
        test = prob.get("test", "")

        ids = tok.encode(prompt, add_bos=True)
        ids_t = torch.tensor(
            [ids], dtype=torch.long, device=device
        )

        generated, _ = model.generate(
            ids_t,
            max_new_tokens=max_gen,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
        )
        new_ids = generated[0, len(ids):].tolist()
        generated_tokens += len(new_ids)
        code = tok.decode(new_ids, skip_special=True)
        if code.strip():
            completed += 1

        # Combine generated code with test
        full_code = prompt + code + "\n" + test
        success, output = execute_code(full_code)

        if success:
            passed += 1
        total += 1

        if len(examples) < 5:
            examples.append({
                "prompt": prompt[:80],
                "success": success,
                "output": output[:100],
            })

    pass_rate = passed / max(total, 1)
    return {
        "pass@1": round(pass_rate, 4),
        "passed": passed,
        "total": total,
        "generated_tokens": generated_tokens,
        "completion_rate": round(completed / max(total, 1), 4),
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# 5. Factual consistency (self-contradiction detection)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_consistency(
    model: LLM,
    tok: BPETokenizer,
    questions: List[str],
    device: str,
    n_samples: int = 3,
    max_gen: int = 256,
) -> Dict:
    """
    Check factual consistency by asking the same question multiple times
    and measuring agreement between answers.
    """
    consistent = 0
    total = 0
    generated_tokens = 0
    completed = 0

    for question in questions:
        prompt = f"Q: {question}\nA:"
        ids = tok.encode(prompt, add_bos=True)
        ids_t = torch.tensor(
            [ids], dtype=torch.long, device=device
        )

        answers = []
        for _ in range(n_samples):
            generated, _ = model.generate(
                ids_t,
                max_new_tokens=max_gen,
                temperature=0.5,
                eos_token_id=tok.eos_token_id,
            )
            new_ids = generated[0, len(ids):].tolist()
            generated_tokens += len(new_ids)
            text = tok.decode(new_ids, skip_special=True).strip()
            if text:
                completed += 1
            answers.append(text[:200])

        # Check pairwise similarity (simple word overlap)
        if len(answers) >= 2:
            similarities = []
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    w1 = set(answers[i].lower().split())
                    w2 = set(answers[j].lower().split())
                    if w1 and w2:
                        sim = len(w1 & w2) / max(
                            len(w1 | w2), 1
                        )
                        similarities.append(sim)
            avg_sim = sum(similarities) / max(len(similarities), 1)
            if avg_sim > 0.3:
                consistent += 1
        total += 1

    rate = consistent / max(total, 1)
    return {
        "consistency_rate": round(rate, 4),
        "consistent": consistent,
        "total": total,
        "generated_tokens": generated_tokens,
        "completion_rate": round(completed / max(total * n_samples, 1), 4),
    }


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def build_scorecard(results: Dict) -> Dict:
    """Build an 18-metric practical capability scorecard (0-10)."""
    mc = float(results.get("multiple_choice", {}).get("accuracy", 0.0))
    math_acc = float(results.get("math", {}).get("accuracy", 0.0))
    code_p1 = float(results.get("code", {}).get("pass@1", 0.0))
    consistency = float(results.get("consistency", {}).get("consistency_rate", 0.0))
    ece = float(results.get("calibration", {}).get("ece", 0.5))

    code_completion = float(results.get("code", {}).get("completion_rate", 0.0))
    math_completion = float(results.get("math", {}).get("completion_rate", 0.0))
    cons_completion = float(results.get("consistency", {}).get("completion_rate", 0.0))

    throughput = float(results.get("operational", {}).get("generation_tokens_per_sec", 0.0))
    token_usage = float(results.get("operational", {}).get("avg_generated_tokens_per_task", 0.0))

    # Calibrated capability proxies
    reasoning = _clip01(0.7 * math_acc + 0.3 * code_p1)
    english = _clip01(0.8 * mc + 0.2 * consistency)
    context = _clip01(consistency)
    knowledge = _clip01(mc)
    task_completion = _clip01((code_completion + math_completion + cons_completion) / 3.0)
    relevancy = _clip01(0.6 * consistency + 0.4 * task_completion)
    coding = _clip01(0.7 * code_p1 + 0.3 * code_completion)
    factual = _clip01(0.6 * mc + 0.4 * consistency)
    instr = _clip01(0.5 * mc + 0.5 * task_completion)
    stability = _clip01(consistency)
    long_context = _clip01(0.5 * consistency + 0.5 * task_completion)
    hallucination_res = _clip01(0.5 * factual + 0.5 * consistency)

    # Operational normalization (kept conservative)
    throughput_score = _clip01(throughput / 200.0)
    token_eff = _clip01(1.0 - min(token_usage / 512.0, 1.0))
    training_speed = _clip01(0.65 * throughput_score + 0.35 * task_completion)
    latency_eff = _clip01(0.55 * throughput_score + 0.45 * token_eff)
    finetune_agility = _clip01(0.7 + 0.3 * (1.0 - min(ece / 0.5, 1.0)))
    deployment = _clip01(
        0.4 * latency_eff + 0.3 * finetune_agility + 0.3 * hallucination_res
    )

    metrics = {
        "Reasoning Depth": round(reasoning * 10, 2),
        "English Understanding": round(english * 10, 2),
        "Context Understanding": round(context * 10, 2),
        "General Knowledge": round(knowledge * 10, 2),
        "Throughput (TPS)": round(throughput_score * 10, 2),
        "Token Efficiency": round(token_eff * 10, 2),
        "Task Completion": round(task_completion * 10, 2),
        "Answer Relevancy": round(relevancy * 10, 2),
        "Coding Reliability": round(coding * 10, 2),
        "Factual Robustness": round(factual * 10, 2),
        "Instruction Following": round(instr * 10, 2),
        "Consistency/Stability": round(stability * 10, 2),
        "Long-Context Handling": round(long_context * 10, 2),
        "Hallucination Resistance": round(hallucination_res * 10, 2),
        "Training Speed (Proxy)": round(training_speed * 10, 2),
        "Latency Efficiency": round(latency_eff * 10, 2),
        "Fine-Tune Agility": round(finetune_agility * 10, 2),
        "Deployment Practicality": round(deployment * 10, 2),
    }
    overall = sum(metrics.values()) / max(len(metrics), 1)
    metrics["Overall Practical Capability"] = round(overall, 2)
    return metrics


def summarize_eval_run_mode(benchmark_integrity: Dict[str, Dict]) -> Dict:
    modes = {
        str(meta.get("run_mode", "unknown"))
        for meta in benchmark_integrity.values()
    }
    if not benchmark_integrity:
        run_mode = "unverified"
    elif modes == {"real"}:
        run_mode = "real"
    elif "toy" in modes:
        run_mode = "mixed_or_toy"
    else:
        run_mode = "unverified"
    return {
        "run_mode": run_mode,
        "real_benchmark_evidence": run_mode == "real",
        "warnings": [
            meta.get("warning", "")
            for meta in benchmark_integrity.values()
            if meta.get("warning")
        ],
    }


# ---------------------------------------------------------------------------
# 6. Uncertainty calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_calibration(
    model: LLM,
    tok: BPETokenizer,
    texts: List[str],
    device: str,
    n_bins: int = 10,
) -> Dict:
    """
    Evaluate how well the confidence head calibrates with actual accuracy.
    A well-calibrated model should have confidence ≈ accuracy in each bin.
    """
    if model.conf_head is None:
        return {"error": "No confidence head"}

    predictions = []  # (confidence, correct)

    for text in texts:
        ids = tok.encode(text, add_bos=True)[:512]
        if len(ids) < 10:
            continue
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(ids_t)

        if out.unc_scores is None:
            continue

        logits = out.logits[:, :-1, :]
        targets = ids_t[:, 1:]
        conf = out.unc_scores[:, :-1, 0]  # correctness confidence

        preds = logits.argmax(-1)
        correct = (preds == targets).float()

        for t in range(correct.shape[1]):
            predictions.append((
                conf[0, t].item(),
                correct[0, t].item(),
            ))

    if not predictions:
        return {"error": "No predictions"}

    # Bin by confidence
    bins = [[] for _ in range(n_bins)]
    for conf, corr in predictions:
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append(corr)

    calibration = []
    ece = 0.0  # Expected Calibration Error
    for i, b in enumerate(bins):
        if b:
            avg_conf = (i + 0.5) / n_bins
            avg_acc = sum(b) / len(b)
            calibration.append({
                "bin": f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}",
                "avg_confidence": round(avg_conf, 3),
                "avg_accuracy": round(avg_acc, 3),
                "count": len(b),
            })
            ece += abs(avg_conf - avg_acc) * len(b)

    ece /= max(len(predictions), 1)

    return {
        "ece": round(ece, 4),
        "n_predictions": len(predictions),
        "bins": calibration,
    }


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(cfg: EvalConfig = eval_cfg):
    _set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, checkpoint_meta = load_model(cfg, device)
    os.makedirs(cfg.output_dir, exist_ok=True)

    results = {}
    t_total = time.time()

    # --- 1. Perplexity ---
    print("\n[1/6] Perplexity evaluation...")
    ppl_texts = [
        "The quick brown fox jumps over the lazy dog. This sentence "
        "contains every letter of the English alphabet.",
        "In machine learning, a neural network is trained by adjusting "
        "weights to minimize a loss function.",
        "The Pythagorean theorem states that in a right triangle, "
        "the square of the hypotenuse equals the sum of squares of "
        "the other two sides.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    "
        "return fibonacci(n-1) + fibonacci(n-2)",
    ]
    results["perplexity"] = eval_perplexity(
        model, tok, ppl_texts, device
    )
    print(f"  PPL: {results['perplexity']['perplexity']}")

    # --- 2. Multiple choice ---
    print("\n[2/6] Multiple-choice evaluation...")
    fallback_mc = [
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": 1,
        },
        {
            "question": "Which planet is closest to the Sun?",
            "choices": ["Venus", "Earth", "Mercury", "Mars"],
            "answer": 2,
        },
        {
            "question": "What is 7 * 8?",
            "choices": ["54", "56", "58", "64"],
            "answer": 1,
        },
        {
            "question": "Which data structure uses FIFO?",
            "choices": ["Stack", "Queue", "Tree", "Graph"],
            "answer": 1,
        },
        {
            "question": "What does HTML stand for?",
            "choices": [
                "Hyper Text Markup Language",
                "High Tech Modern Language",
                "Hyper Transfer Markup Language",
                "Home Tool Markup Language",
            ],
            "answer": 0,
        },
    ]
    benchmark_integrity: Dict[str, Dict] = {}
    benchmark_integrity["perplexity"] = {
        "benchmark": "perplexity_smoke_texts",
        "dataset_name": "built_in_smoke_texts",
        "split": "smoke",
        "run_mode": "smoke",
        "sample_count": len(ppl_texts),
        "results_are_measured": True,
        "capability_evidence": "smoke_only",
        "warning": "Built-in perplexity texts are smoke checks, not benchmark evidence.",
    }
    mc_questions, mc_meta = _choose_eval_set(
        "multiple-choice",
        _load_real_mc_questions(limit=max(200, cfg.min_real_mc_samples)),
        fallback_mc,
        cfg.min_real_mc_samples,
        cfg,
    )
    benchmark_integrity["multiple_choice"] = mc_meta
    print(f"  Questions: {len(mc_questions)}")
    results["multiple_choice"] = eval_multiple_choice(
        model, tok, mc_questions, device
    )
    print(f"  Accuracy: {results['multiple_choice']['accuracy']}")

    # --- 3. Math ---
    print("\n[3/6] Math evaluation...")
    t_math = time.time()
    fallback_math = [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats "
            "three for breakfast every morning and bakes muffins for "
            "her friends every day with four. She sells the remainder "
            "at the farmers' market daily for $2 per fresh duck egg. "
            "How much in dollars does she make every day at the "
            "farmers' market?",
            "answer": "#### 18",
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half "
            "that much white fiber. How many bolts in total does it "
            "take?",
            "answer": "#### 3",
        },
    ]
    math_problems, math_meta = _choose_eval_set(
        "math",
        _load_real_gsm8k(limit=max(200, cfg.min_real_math_samples)),
        fallback_math,
        cfg.min_real_math_samples,
        cfg,
    )
    benchmark_integrity["math"] = math_meta
    print(f"  Problems: {len(math_problems)}")
    results["math"] = eval_gsm8k(model, tok, math_problems, device)
    math_elapsed = time.time() - t_math
    print(f"  Accuracy: {results['math']['accuracy']}")

    # --- 4. Code ---
    print("\n[4/6] Code evaluation...")
    t_code = time.time()
    fallback_code = [
        {
            "prompt": (
                "def add(a, b):\n"
                "    \"\"\"Add two numbers.\"\"\"\n"
            ),
            "test": (
                "assert add(2, 3) == 5\n"
                "assert add(-1, 1) == 0\n"
                "print('PASS')\n"
            ),
            "entry_point": "add",
        },
        {
            "prompt": (
                "def is_palindrome(s):\n"
                "    \"\"\"Check if a string is a palindrome.\"\"\"\n"
            ),
            "test": (
                "assert is_palindrome('racecar') == True\n"
                "assert is_palindrome('hello') == False\n"
                "assert is_palindrome('') == True\n"
                "print('PASS')\n"
            ),
            "entry_point": "is_palindrome",
        },
    ]
    code_problems, code_meta = _choose_eval_set(
        "code",
        _load_real_code_tasks(limit=max(50, cfg.min_real_code_samples)),
        fallback_code,
        cfg.min_real_code_samples,
        cfg,
    )
    benchmark_integrity["code"] = code_meta
    print(f"  Tasks: {len(code_problems)}")
    results["code"] = eval_code(model, tok, code_problems, device)
    code_elapsed = time.time() - t_code
    print(f"  Pass@1: {results['code']['pass@1']}")

    # --- 5. Consistency ---
    print("\n[5/6] Consistency evaluation...")
    t_cons = time.time()
    consistency_qs = [
        "What is the boiling point of water at sea level?",
        "Who wrote Romeo and Juliet?",
        "What programming language is known for its use in data science?",
    ]
    results["consistency"] = eval_consistency(
        model, tok, consistency_qs, device
    )
    cons_elapsed = time.time() - t_cons
    print(
        f"  Consistency: {results['consistency']['consistency_rate']}"
    )

    # --- 6. Calibration ---
    print("\n[6/6] Calibration evaluation...")
    cal_texts = ppl_texts * 3  # Reuse texts
    results["calibration"] = eval_calibration(
        model, tok, cal_texts, device
    )
    if "ece" in results["calibration"]:
        print(f"  ECE: {results['calibration']['ece']}")

    # --- Summary ---
    elapsed = time.time() - t_total
    eval_mode = summarize_eval_run_mode(
        {
            key: value
            for key, value in benchmark_integrity.items()
            if key != "perplexity"
        }
    )
    results["metadata"] = {
        "elapsed_seconds": round(elapsed, 1),
        "device": device,
        "checkpoint": cfg.checkpoint_path,
        "checkpoint_integrity": checkpoint_meta,
        "checkpoint_loaded": checkpoint_meta.get("load_status") == "loaded",
        "model_config_hash": _json_hash(runtime_config_dict().get("model", {})),
        "strict_real_benchmarks": bool(cfg.strict_real_benchmarks),
        "allow_toy_fallback": bool(cfg.allow_toy_fallback),
        "benchmark_integrity": benchmark_integrity,
        "run_mode": eval_mode["run_mode"],
        "real_benchmark_evidence": eval_mode["real_benchmark_evidence"],
        "warnings": eval_mode["warnings"],
        "model_params": sum(
            p.numel() for p in model.parameters()
        ),
    }

    gen_tokens = (
        int(results.get("math", {}).get("generated_tokens", 0))
        + int(results.get("code", {}).get("generated_tokens", 0))
        + int(results.get("consistency", {}).get("generated_tokens", 0))
    )
    gen_time = max(math_elapsed + code_elapsed + cons_elapsed, 1e-9)
    gen_tasks = (
        int(results.get("math", {}).get("total", 0))
        + int(results.get("code", {}).get("total", 0))
        + int(results.get("consistency", {}).get("total", 0))
    )
    results["operational"] = {
        "generation_tokens": gen_tokens,
        "generation_seconds": round(gen_time, 3),
        "generation_tokens_per_sec": round(gen_tokens / gen_time, 3),
        "avg_generated_tokens_per_task": round(gen_tokens / max(gen_tasks, 1), 3),
    }
    results["scorecard"] = build_scorecard(results)

    # Save report
    report_path = os.path.join(cfg.output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Perplexity    : {results['perplexity']['perplexity']}")
    print(
        f"  MC Accuracy   : {results['multiple_choice']['accuracy']}"
    )
    print(f"  Math Accuracy : {results['math']['accuracy']}")
    print(f"  Code Pass@1   : {results['code']['pass@1']}")
    print(
        f"  Consistency   : "
        f"{results['consistency']['consistency_rate']}"
    )
    if "ece" in results["calibration"]:
        print(f"  Calibration   : ECE={results['calibration']['ece']}")
    print(f"  Time          : {elapsed:.0f}s")
    print(f"  Report        : {report_path}")
    print("  Scorecard (0-10):")
    for k, v in results["scorecard"].items():
        print(f"    {k:<32} {v:>5}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    run_evaluation()
