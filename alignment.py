"""Direct Preference Optimization training loop for preference pairs."""

import os
import sys
import json
import time
import random
import copy
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

sys.path.insert(0, ".")
from config import DPOConfig, model_cfg, dpo_cfg
from tokenizer import BPETokenizer
from model import LLM
from core.checkpoint_io import (
    atomic_torch_save,
    extract_state_dict,
    safe_torch_load,
)
from core.lr_schedule import cosine_warmup_lr
from core.sequence_ops import masked_next_token_sequence_log_probs
from core.training_lifecycle import (
    set_optimizer_lr,
    should_run_interval,
    should_sync_step,
)
from engine.optimizer_factory import (
    AmpPolicySettings,
    OptimizerSettings,
    build_adamw,
    build_amp_policy,
)
from core.logging import configure_logging, get_logger
from observability.training_telemetry import (
    TrainingTelemetry,
    default_telemetry_path,
)
from security.validator import (
    ValidationError,
    get_allowed_data_roots,
    iter_jsonl,
    safe_load_json,
)


LOGGER = get_logger("dpo")


def _set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader_kwargs(cfg: DPOConfig, is_train: bool) -> dict:
    workers = int(getattr(cfg, "data_num_workers", 0))
    if workers < 0:
        workers = 0

    kwargs = {
        "num_workers": workers,
        "pin_memory": bool(
            getattr(cfg, "data_pin_memory", torch.cuda.is_available())
        ),
    }

    if workers > 0:
        kwargs["persistent_workers"] = bool(
            getattr(cfg, "data_persistent_workers", True)
        )
        kwargs["prefetch_factor"] = max(
            1, int(getattr(cfg, "data_prefetch_factor", 2))
        )

    if not is_train and kwargs.get("persistent_workers") and workers <= 1:
        kwargs["persistent_workers"] = False

    return kwargs


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_dpo_data(path: str) -> List[Dict]:
    """Load DPO preference pairs."""
    data = []
    allowed_roots = get_allowed_data_roots()
    max_json_bytes = int(
        os.environ.get("MYLLM_MAX_DATA_JSON_BYTES", "536870912")
    )
    max_jsonl_line_bytes = int(
        os.environ.get("MYLLM_MAX_DATA_JSONL_LINE_BYTES", "4000000")
    )

    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith((".json", ".jsonl"))
        ]
    else:
        files = [path]

    for fp in files:
        try:
            if fp.endswith(".jsonl"):
                records = iter_jsonl(
                    fp,
                    max_bytes=max_json_bytes,
                    max_line_bytes=max_jsonl_line_bytes,
                    allowed_roots=allowed_roots,
                )
                for item in records:
                    if isinstance(item, dict) and all(
                        k in item for k in ("prompt", "chosen", "rejected")
                    ):
                        data.append(item)
            else:
                content = safe_load_json(
                    fp,
                    max_bytes=max_json_bytes,
                    allowed_roots=allowed_roots,
                )
                if isinstance(content, list):
                    data.extend(
                        item
                        for item in content
                        if isinstance(item, dict)
                        and all(
                            k in item for k in ("prompt", "chosen", "rejected")
                        )
                    )
        except (ValidationError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"DPO data validation failed for {fp}: {str(exc)[:180]}"
            ) from exc
    return data


def encode_preference_pair(
    tok: BPETokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    max_len: int = 1024,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """
    Encode a preference pair into tokenized sequences with loss masks.

    Returns:
        chosen_ids:    [max_len] — token ids for prompt + chosen
        chosen_mask:   [max_len] — 1.0 where response tokens are scored
        rejected_ids:  [max_len] — token ids for prompt + rejected
        rejected_mask: [max_len] — 1.0 where response tokens are scored
    """
    # Encode prompt
    prompt_text = f"{tok.USR}\n{prompt}{tok.END}\n{tok.AST}\n"
    prompt_ids = tok.encode(prompt_text, add_bos=True)
    prompt_len = len(prompt_ids)

    # Encode chosen completion
    chosen_text = f"{chosen}{tok.END}"
    chosen_completion_ids = tok.encode(chosen_text, add_bos=False)
    chosen_full = prompt_ids + chosen_completion_ids

    # Encode rejected completion
    rejected_text = f"{rejected}{tok.END}"
    rejected_completion_ids = tok.encode(rejected_text, add_bos=False)
    rejected_full = prompt_ids + rejected_completion_ids

    # Truncate
    chosen_full = chosen_full[:max_len]
    rejected_full = rejected_full[:max_len]

    # Create masks (1.0 for response tokens, 0.0 for prompt tokens)
    chosen_mask = [0.0] * min(prompt_len, len(chosen_full))
    chosen_mask.extend(
        [1.0] * (len(chosen_full) - min(prompt_len, len(chosen_full)))
    )

    rejected_mask = [0.0] * min(prompt_len, len(rejected_full))
    rejected_mask.extend(
        [1.0] * (len(rejected_full) - min(prompt_len, len(rejected_full)))
    )

    # Pad to max_len
    def pad_to(seq, length, pad_val):
        return seq + [pad_val] * (length - len(seq))

    chosen_ids = pad_to(chosen_full, max_len, tok.pad_token_id)
    chosen_mask = pad_to(chosen_mask, max_len, 0.0)
    rejected_ids = pad_to(rejected_full, max_len, tok.pad_token_id)
    rejected_mask = pad_to(rejected_mask, max_len, 0.0)

    return (
        torch.tensor(chosen_ids, dtype=torch.long),
        torch.tensor(chosen_mask, dtype=torch.float),
        torch.tensor(rejected_ids, dtype=torch.long),
        torch.tensor(rejected_mask, dtype=torch.float),
    )


# ---------------------------------------------------------------------------
# DPO Dataset
# ---------------------------------------------------------------------------

class DPODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: List[Dict],
        tok: BPETokenizer,
        max_len: int = 1024,
    ):
        self.samples = []
        skipped = 0
        for item in data:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            try:
                c_ids, c_mask, r_ids, r_mask = (
                    encode_preference_pair(
                        tok, prompt, chosen, rejected, max_len
                    )
                )
                # Skip if either response is too short
                if c_mask.sum() < 5 or r_mask.sum() < 5:
                    skipped += 1
                    continue
                self.samples.append({
                    "chosen_ids": c_ids,
                    "chosen_mask": c_mask,
                    "rejected_ids": r_ids,
                    "rejected_mask": r_mask,
                })
            except Exception:
                skipped += 1

        LOGGER.info(
            "DPO dataset: %s pairs (skipped %s)",
            len(self.samples),
            skipped,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# DPO Loss
# ---------------------------------------------------------------------------

def compute_log_probs(
    model: LLM,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log probabilities for the given sequence,
    masked to only include response tokens.

    Returns: [batch_size] — summed log probs per sequence
    """
    out = model(input_ids)
    return masked_next_token_sequence_log_probs(
        out.logits,
        input_ids,
        mask,
    )


def dpo_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute DPO loss.

    Args:
        policy_chosen_lp: log P_policy(chosen | prompt)
        policy_rejected_lp: log P_policy(rejected | prompt)
        ref_chosen_lp: log P_ref(chosen | prompt)
        ref_rejected_lp: log P_ref(rejected | prompt)
        beta: temperature parameter
        label_smoothing: label smoothing factor
        loss_type: "sigmoid" or "hinge"

    Returns:
        (loss, chosen_reward, rejected_reward)
    """
    # Log-ratio of policy vs reference
    pi_logratios = policy_chosen_lp - policy_rejected_lp
    ref_logratios = ref_chosen_lp - ref_rejected_lp
    logits = beta * (pi_logratios - ref_logratios)

    if loss_type == "sigmoid":
        if label_smoothing > 0:
            loss = (
                -label_smoothing * F.logsigmoid(-logits)
                - (1 - label_smoothing) * F.logsigmoid(logits)
            )
        else:
            loss = -F.logsigmoid(logits)
    elif loss_type == "hinge":
        loss = torch.relu(1 - logits)
    else:
        raise ValueError(f"Unknown DPO loss type: {loss_type}")

    # Implicit rewards
    chosen_rewards = beta * (
        policy_chosen_lp - ref_chosen_lp
    ).detach()
    rejected_rewards = beta * (
        policy_rejected_lp - ref_rejected_lp
    ).detach()

    return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()


# ---------------------------------------------------------------------------
# DPO Trainer
# ---------------------------------------------------------------------------

def dpo_train(cfg: DPOConfig = dpo_cfg):
    _set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = build_amp_policy(
        AmpPolicySettings(
            dtype_name=cfg.dtype,
            device_type=("cuda" if torch.cuda.is_available() else "cpu"),
            use_grad_scaler=cfg.amp_grad_scaler,
            init_scale=cfg.amp_init_scale,
            growth_interval=cfg.amp_growth_interval,
        )
    )

    # Load tokenizer
    tok = BPETokenizer()
    tok.load(cfg.tokenizer_path)
    model_cfg.vocab_size = tok.vocab_size_

    # Load policy model (from SFT checkpoint)
    policy = LLM(model_cfg).to(device)
    if os.path.exists(cfg.checkpoint_path):
        try:
            st, path = safe_torch_load(cfg.checkpoint_path, map_location=device)
            policy.load_state_dict(extract_state_dict(st))
            LOGGER.info("Policy loaded: %s", path)
        except FileNotFoundError:
            LOGGER.info("No policy checkpoint found; using fresh policy initialization.")

    # Create reference model (frozen deep copy)
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    LOGGER.info("Reference model created (frozen)")

    # Load data
    if not os.path.exists(cfg.data_path):
        LOGGER.error("DPO data not found at %s", cfg.data_path)
        LOGGER.error("Expected JSONL format with prompt/chosen/rejected")
        return

    all_data = load_dpo_data(cfg.data_path)
    if len(all_data) < 2:
        LOGGER.error("DPO requires at least 2 preference pairs.")
        return
    random.shuffle(all_data)
    val_size = max(1, min(len(all_data) - 1, len(all_data) // 20))
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    if not train_data:
        LOGGER.error("No training data left after validation split.")
        return

    train_ds = DPODataset(train_data, tok, cfg.seq_len)
    val_ds = DPODataset(val_data, tok, cfg.seq_len)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        **_loader_kwargs(cfg, is_train=True),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **_loader_kwargs(cfg, is_train=False),
    )

    # Optimizer — lower LR than SFT for stability
    opt = build_adamw(
        policy,
        OptimizerSettings(
            lr=cfg.max_lr,
            weight_decay=cfg.weight_decay,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            use_fused=True,
        ),
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    configure_logging(
        level=os.environ.get("MYLLM_LOG_LEVEL", "INFO"),
        log_file=os.path.join(cfg.output_dir, "dpo.log"),
    )
    telemetry_path = default_telemetry_path(cfg.output_dir, phase="dpo")
    telemetry = TrainingTelemetry(
        telemetry_path,
        enabled=True,
        gradient_every=max(1, cfg.log_every),
        hardware_every=max(1, cfg.log_every // 2),
        loss_spike_threshold=float(
            os.environ.get("MYLLM_LOSS_SPIKE_THRESHOLD", "1.4")
        ),
        phase="dpo",
    )
    LOGGER.info("Telemetry stream: %s", os.path.abspath(telemetry_path))

    # Training loop
    policy.train()
    step = 0
    micro_step = 0
    best_val_loss = float("inf")
    t0 = time.time()

    LOGGER.info("DPO Training: %s steps", cfg.total_steps)
    LOGGER.info("Beta: %s Loss: %s", cfg.beta, cfg.loss_type)
    LOGGER.info("Data: %s train, %s val", len(train_ds), len(val_ds))

    non_blocking = bool(
        cfg.device_non_blocking
        and torch.cuda.is_available()
        and cfg.data_pin_memory
    )

    if len(train_ds) == 0:
        LOGGER.error("DPO training dataset is empty after filtering.")
        return
    if len(val_ds) == 0:
        LOGGER.error("DPO validation dataset is empty after filtering.")
        return

    while step < cfg.total_steps:
        for batch in train_loader:
            if step >= cfg.total_steps:
                break

            c_ids = batch["chosen_ids"].to(device, non_blocking=non_blocking)
            c_mask = batch["chosen_mask"].to(device, non_blocking=non_blocking)
            r_ids = batch["rejected_ids"].to(device, non_blocking=non_blocking)
            r_mask = batch["rejected_mask"].to(device, non_blocking=non_blocking)

            with amp.autocast():
                # Policy log probs
                policy_chosen_lp = compute_log_probs(
                    policy, c_ids, c_mask
                )
                policy_rejected_lp = compute_log_probs(
                    policy, r_ids, r_mask
                )

                # Reference log probs (no grad)
                with torch.no_grad():
                    ref_chosen_lp = compute_log_probs(
                        ref_model, c_ids, c_mask
                    )
                    ref_rejected_lp = compute_log_probs(
                        ref_model, r_ids, r_mask
                    )

                loss, chosen_rew, rejected_rew = dpo_loss(
                    policy_chosen_lp,
                    policy_rejected_lp,
                    ref_chosen_lp,
                    ref_rejected_lp,
                    beta=cfg.beta,
                    label_smoothing=cfg.label_smoothing,
                    loss_type=cfg.loss_type,
                )
                loss = loss / cfg.grad_accum

            if not torch.isfinite(loss.detach()):
                LOGGER.critical(
                    "Non-finite loss at step %s; skipping batch.",
                    step,
                )
                opt.zero_grad(set_to_none=True)
                continue

            amp.backward(loss)

            sync = should_sync_step(micro_step, cfg.grad_accum)
            micro_step += 1

            if not sync:
                continue

            lr = cosine_warmup_lr(
                step=step,
                total_steps=cfg.total_steps,
                warmup_steps=cfg.warmup_steps,
                max_lr=cfg.max_lr,
                min_lr=cfg.min_lr,
            )
            set_optimizer_lr(opt, lr)
            if not torch.isfinite(loss.detach()):
                LOGGER.critical(
                    "Non-finite loss at step %s; skipping optimizer step.",
                    step,
                )
                opt.zero_grad(set_to_none=True)
                continue
            grad_norm_value = amp.clip_grad_norm_(
                opt,
                policy.parameters(),
                cfg.grad_clip,
            )
            amp.step(opt)
            amp.update()

            telemetry_payload = telemetry.log_step(
                step=step,
                loss=loss.item() * cfg.grad_accum,
                lr=lr,
                grad_norm=grad_norm_value,
                model=policy,
                device=device,
                extra={
                    "reward_margin": round(
                        float((chosen_rew - rejected_rew).item()), 6
                    ),
                    "chosen_reward": round(float(chosen_rew.item()), 6),
                    "rejected_reward": round(float(rejected_rew.item()), 6),
                },
            )
            if telemetry_payload.get("loss_spike"):
                LOGGER.critical(
                    "Loss spike detected at step %s: loss=%s baseline=%s ratio=%s",
                    step,
                    telemetry_payload.get("loss"),
                    telemetry_payload.get("loss_baseline"),
                    telemetry_payload.get("loss_ratio"),
                )
            opt.zero_grad()

            # Logging
            if should_run_interval(step, cfg.log_every, include_zero=True):
                margin = (chosen_rew - rejected_rew).item()
                acc = (chosen_rew > rejected_rew).float().mean().item()
                LOGGER.info(
                    "DPO step %s/%s | loss:%.4f | margin:%.3f | acc:%.2f | lr:%.2e",
                    step,
                    cfg.total_steps,
                    loss.item() * cfg.grad_accum,
                    margin,
                    acc,
                    lr,
                )

            # Validation
            if should_run_interval(step, cfg.save_every):
                policy.eval()
                val_loss_sum = 0.0
                val_n = 0
                with torch.no_grad():
                    for vb in val_loader:
                        vc_ids = vb["chosen_ids"].to(device, non_blocking=non_blocking)
                        vc_mask = vb["chosen_mask"].to(device, non_blocking=non_blocking)
                        vr_ids = vb["rejected_ids"].to(device, non_blocking=non_blocking)
                        vr_mask = vb["rejected_mask"].to(device, non_blocking=non_blocking)

                        with amp.autocast():
                            pc = compute_log_probs(
                                policy, vc_ids, vc_mask
                            )
                            pr = compute_log_probs(
                                policy, vr_ids, vr_mask
                            )
                            rc = compute_log_probs(
                                ref_model, vc_ids, vc_mask
                            )
                            rr = compute_log_probs(
                                ref_model, vr_ids, vr_mask
                            )
                            vl, _, _ = dpo_loss(
                                pc, pr, rc, rr,
                                beta=cfg.beta,
                                loss_type=cfg.loss_type,
                            )
                        val_loss_sum += vl.item()
                        val_n += 1

                avg_val = val_loss_sum / max(val_n, 1)
                LOGGER.info("DPO VAL loss=%.4f", avg_val)

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    atomic_torch_save(
                        {
                            "step": step,
                            "model_state_dict": policy.state_dict(),
                            "val_loss": avg_val,
                        },
                        os.path.join(cfg.output_dir, "dpo_best.pt"),
                    )
                    LOGGER.info("DPO new best checkpoint saved.")

                # Regular checkpoint
                atomic_torch_save(
                    {
                        "step": step,
                        "model_state_dict": policy.state_dict(),
                    },
                    os.path.join(
                        cfg.output_dir, f"dpo_step_{step:06d}.pt"
                    ),
                )
                policy.train()

            step += 1

    # Final save
    atomic_torch_save(
        {"step": step, "model_state_dict": policy.state_dict()},
        os.path.join(cfg.output_dir, "dpo_final.pt"),
    )
    elapsed = time.time() - t0
    LOGGER.info(
        "DPO complete: %s steps in %.1fmin | best_val=%.4f",
        step,
        elapsed / 60,
        best_val_loss,
    )


if __name__ == "__main__":
    dpo_train()