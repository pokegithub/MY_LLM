"""Supervised fine-tuning loop for chat-style datasets."""

import os
import sys
import json
import time
import random
import torch
from typing import List, Dict, Tuple

sys.path.insert(0, ".")
from config import SFTConfig, model_cfg, sft_cfg
from tokenizer import BPETokenizer
from model import LLM
from core.checkpoint_io import (
    atomic_torch_save,
    extract_state_dict,
    safe_torch_load,
)
from core.lr_schedule import cosine_warmup_lr
from core.sequence_ops import next_token_cross_entropy
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


LOGGER = get_logger("sft")


def _set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader_kwargs(cfg: SFTConfig, is_train: bool) -> dict:
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

def load_sft_data(path: str) -> List[Dict]:
    """Load SFT data from JSONL or JSON files."""
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
                for item in iter_jsonl(
                    fp,
                    max_bytes=max_json_bytes,
                    max_line_bytes=max_jsonl_line_bytes,
                    allowed_roots=allowed_roots,
                ):
                    if isinstance(item, dict):
                        data.append(item)
            else:
                content = safe_load_json(
                    fp,
                    max_bytes=max_json_bytes,
                    allowed_roots=allowed_roots,
                )
                if isinstance(content, list):
                    data.extend(item for item in content if isinstance(item, dict))
                elif isinstance(content, dict):
                    data.append(content)
        except (ValidationError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"SFT data validation failed for {fp}: {str(exc)[:180]}"
            ) from exc
    return data


def tokenize_conversation(
    tok: BPETokenizer,
    messages: List[Dict],
    system: str = "",
    max_len: int = 1024,
) -> Tuple[List[int], List[int]]:
    """
    Tokenize a multi-turn conversation.
    Returns (input_ids, labels) where labels=-100 for non-assistant tokens.
    """
    input_ids = [tok.bos_token_id]
    labels = [-100]

    if system:
        sys_text = f"{tok.SYS}\n{system}{tok.END}\n"
        sys_ids = tok.encode(sys_text, add_bos=False)
        input_ids.extend(sys_ids)
        labels.extend([-100] * len(sys_ids))

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            text = f"{tok.USR}\n{content}{tok.END}\n"
            ids = tok.encode(text, add_bos=False)
            input_ids.extend(ids)
            labels.extend([-100] * len(ids))
        elif role == "assistant":
            prefix = f"{tok.AST}\n"
            prefix_ids = tok.encode(prefix, add_bos=False)
            input_ids.extend(prefix_ids)
            labels.extend([-100] * len(prefix_ids))

            content_text = f"{content}{tok.END}\n"
            content_ids = tok.encode(content_text, add_bos=False)
            input_ids.extend(content_ids)
            labels.extend(content_ids)  # supervise assistant output
        elif role == "system":
            text = f"{tok.SYS}\n{content}{tok.END}\n"
            ids = tok.encode(text, add_bos=False)
            input_ids.extend(ids)
            labels.extend([-100] * len(ids))

    # Truncate
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]

    return input_ids, labels


def pack_sequences(
    sequences: List[Tuple[List[int], List[int]]],
    max_len: int,
    pad_id: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Pack multiple conversations into fixed-length sequences.
    Uses greedy bin-packing for efficiency.
    """
    packed = []
    current_ids: List[int] = []
    current_labels: List[int] = []

    random.shuffle(sequences)

    for ids, labels in sequences:
        if len(current_ids) + len(ids) <= max_len:
            current_ids.extend(ids)
            current_labels.extend(labels)
        else:
            if current_ids:
                # Pad to max_len
                pad_len = max_len - len(current_ids)
                current_ids.extend([pad_id] * pad_len)
                current_labels.extend([-100] * pad_len)
                packed.append((
                    torch.tensor(current_ids, dtype=torch.long),
                    torch.tensor(current_labels, dtype=torch.long),
                ))
            current_ids = list(ids)
            current_labels = list(labels)

    if current_ids:
        pad_len = max_len - len(current_ids)
        current_ids.extend([pad_id] * pad_len)
        current_labels.extend([-100] * pad_len)
        packed.append((
            torch.tensor(current_ids, dtype=torch.long),
            torch.tensor(current_labels, dtype=torch.long),
        ))

    return packed


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

class SFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: List[Dict],
        tok: BPETokenizer,
        cfg: SFTConfig,
        is_val: bool = False,
    ):
        self.tok = tok
        self.cfg = cfg

        # Tokenize all conversations
        sequences = []
        for item in data:
            messages = item.get("messages", item.get("conversations", []))
            system = item.get("system", "")

            # Handle different formats
            if not messages and "instruction" in item:
                messages = [
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item.get("output", "")},
                ]

            if messages:
                ids, labels = tokenize_conversation(
                    tok, messages, system, cfg.seq_len
                )
                if len(ids) > 10:  # skip too-short sequences
                    sequences.append((ids, labels))

        # Pack or pad
        if cfg.packing and not is_val:
            self.samples = pack_sequences(
                sequences, cfg.seq_len, tok.pad_token_id
            )
        else:
            self.samples = []
            for ids, labels in sequences:
                pad_len = cfg.seq_len - len(ids)
                ids_padded = ids + [tok.pad_token_id] * pad_len
                labels_padded = labels + [-100] * pad_len
                self.samples.append((
                    torch.tensor(ids_padded[:cfg.seq_len], dtype=torch.long),
                    torch.tensor(
                        labels_padded[:cfg.seq_len], dtype=torch.long
                    ),
                ))

        LOGGER.info(
            "SFT %s: %s packed sequences from %s conversations",
            "val" if is_val else "train",
            len(self.samples),
            len(data),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return {"input_ids": ids, "labels": labels}


# ---------------------------------------------------------------------------
# SFT Trainer
# ---------------------------------------------------------------------------

def sft_train(cfg: SFTConfig = sft_cfg):
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

    # Load model from pretrained checkpoint
    model = LLM(model_cfg).to(device)
    ckpt_dir = cfg.checkpoint_path
    if os.path.exists(ckpt_dir):
        try:
            st, path = safe_torch_load(ckpt_dir, map_location=device)
            model.load_state_dict(extract_state_dict(st))
            LOGGER.info("Loaded pretrained: %s", path)
        except FileNotFoundError:
            LOGGER.info("No pretrained checkpoint found; training from initialization.")

    # Load data
    if not os.path.exists(cfg.data_path):
        LOGGER.error("SFT data not found at %s", cfg.data_path)
        LOGGER.error("Expected JSONL with 'messages' field")
        return

    all_data = load_sft_data(cfg.data_path)
    if len(all_data) < 2:
        LOGGER.error("SFT requires at least 2 samples.")
        return
    random.shuffle(all_data)
    val_size = max(1, min(len(all_data) - 1, len(all_data) // 20))
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    if not train_data:
        LOGGER.error("No training samples left after validation split.")
        return

    train_ds = SFTDataset(train_data, tok, cfg, is_val=False)
    val_ds = SFTDataset(val_data, tok, cfg, is_val=True)

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

    # Optimizer
    opt = build_adamw(
        model,
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
        log_file=os.path.join(cfg.output_dir, "sft.log"),
    )
    telemetry_path = default_telemetry_path(cfg.output_dir, phase="sft")
    telemetry = TrainingTelemetry(
        telemetry_path,
        enabled=True,
        gradient_every=max(1, cfg.log_every),
        hardware_every=max(1, cfg.log_every // 2),
        loss_spike_threshold=float(
            os.environ.get("MYLLM_LOSS_SPIKE_THRESHOLD", "1.4")
        ),
        phase="sft",
    )
    LOGGER.info("Telemetry stream: %s", os.path.abspath(telemetry_path))

    # Training loop
    model.train()
    step = 0
    micro_step = 0
    best_val_loss = float("inf")
    t0 = time.time()

    LOGGER.info("SFT Training: %s steps", cfg.total_steps)
    LOGGER.info("Batch: %s x %s accum", cfg.batch_size, cfg.grad_accum)
    LOGGER.info("Data: %s train, %s val", len(train_ds), len(val_ds))

    non_blocking = bool(
        cfg.device_non_blocking
        and torch.cuda.is_available()
        and cfg.data_pin_memory
    )

    if len(train_ds) == 0:
        LOGGER.error("SFT training dataset is empty after preprocessing.")
        return
    if len(val_ds) == 0:
        LOGGER.error("SFT validation dataset is empty after preprocessing.")
        return

    while step < cfg.total_steps:
        for batch in train_loader:
            if step >= cfg.total_steps:
                break

            ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            labels = batch["labels"].to(device, non_blocking=non_blocking)

            with amp.autocast():
                logits = model(ids).logits
                loss = next_token_cross_entropy(
                    logits,
                    labels,
                    ignore_index=-100,
                )
                loss = loss / cfg.grad_accum

            if not torch.isfinite(loss.detach()):
                LOGGER.critical(
                    "Non-finite loss at step %s; skipping batch.",
                    step,
                )
                opt.zero_grad()
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
                opt.zero_grad()
                continue
            grad_norm_value = amp.clip_grad_norm_(
                opt,
                model.parameters(),
                cfg.grad_clip,
            )
            amp.step(opt)
            amp.update()

            telemetry_payload = telemetry.log_step(
                step=step,
                loss=loss.item() * cfg.grad_accum,
                lr=lr,
                grad_norm=grad_norm_value,
                model=model,
                device=device,
                extra={
                    "global_batch_tokens": int(
                        cfg.batch_size * cfg.seq_len * cfg.grad_accum
                    ),
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
                LOGGER.info(
                    "SFT step %s/%s | loss:%.4f | lr:%.2e",
                    step,
                    cfg.total_steps,
                    loss.item() * cfg.grad_accum,
                    lr,
                )

            # Validation
            if should_run_interval(step, cfg.val_every):
                model.eval()
                val_loss = 0.0
                val_n = 0
                with torch.no_grad():
                    for vb in val_loader:
                        vids = vb["input_ids"].to(
                            device, non_blocking=non_blocking
                        )
                        vlabels = vb["labels"].to(
                            device, non_blocking=non_blocking
                        )
                        with amp.autocast():
                            vlogits = model(vids).logits
                            vl = next_token_cross_entropy(
                                vlogits,
                                vlabels,
                                ignore_index=-100,
                            )
                        val_loss += vl.item()
                        val_n += 1
                        if val_n >= cfg.val_steps:
                            break
                avg_val = val_loss / max(val_n, 1)
                LOGGER.info("SFT VAL loss=%.4f", avg_val)
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    sd = model.state_dict()
                    atomic_torch_save(
                        {"step": step, "model_state_dict": sd,
                         "val_loss": avg_val},
                        os.path.join(cfg.output_dir, "sft_best.pt"),
                    )
                    LOGGER.info("SFT new best checkpoint saved.")
                model.train()

            # Save checkpoint
            if should_run_interval(step, cfg.save_every):
                sd = model.state_dict()
                atomic_torch_save(
                    {"step": step, "model_state_dict": sd},
                    os.path.join(
                        cfg.output_dir, f"sft_step_{step:06d}.pt"
                    ),
                )

            step += 1

    # Final save
    sd = model.state_dict()
    atomic_torch_save(
        {"step": step, "model_state_dict": sd},
        os.path.join(cfg.output_dir, "sft_final.pt"),
    )
    elapsed = time.time() - t0
    LOGGER.info(
        "SFT complete: %s steps in %.1fmin | best_val=%.4f",
        step,
        elapsed / 60,
        best_val_loss,
    )


if __name__ == "__main__":
    sft_train()