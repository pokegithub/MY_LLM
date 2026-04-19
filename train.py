"""Training loop for pretraining with validation, telemetry, and checkpoints."""

import os
import sys
import time
import json
import random
import platform
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, ".")
from config import (
    ModelConfig,
    TrainConfig,
    apply_overrides,
    model_cfg,
    runtime_config_dict,
    show_config,
    train_cfg,
)
from tokenizer import BPETokenizer
from model import LLM, LossOutput
from data import get_dataloader, get_val_dataloader, DataPipelineError
from core.checkpoint_io import (
    atomic_torch_save,
    extract_state_dict,
    safe_torch_load,
)
from core.config_manager import ConfigManager
from core.lr_schedule import cosine_warmup_lr_from_tokens
from core.training_lifecycle import (
    set_optimizer_lr,
    should_run_interval,
    should_sync_step,
)
from core.hardware_optim import (
    apply_profile_attention_policy,
    resolve_profile_dtype_policy,
)
from engine.optimizer_factory import (
    AmpPolicySettings,
    OptimizerSettings,
    build_adamw,
    build_amp_policy,
)
from core.logging import configure_logging, get_logger
from hardware_profiles import load_profile
from observability.training_telemetry import (
    PerformanceMonitor,
    TrainingTelemetry,
    default_telemetry_path,
)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

_dist_active = False
LOGGER = get_logger("train")


def _backend() -> str:
    if torch.cuda.is_available() and platform.system() != "Windows":
        return "nccl"
    return "gloo"


def setup() -> tuple[int, int, int]:
    global _dist_active
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=_backend())
        _dist_active = True
        rank = dist.get_rank()
        local = int(os.environ.get("LOCAL_RANK", "0"))
        world = dist.get_world_size()
    else:
        rank, local, world = 0, 0, 1
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, local, world


def cleanup() -> None:
    if _dist_active and dist.is_initialized():
        dist.destroy_process_group()


def _broadcast_stop(flag: bool, device: str) -> bool:
    if not _dist_active:
        return flag
    t = torch.tensor(int(flag), device=device)
    dist.broadcast(t, src=0)
    return t.item() == 1


def _collect_train_config_paths() -> List[str]:
    paths: List[str] = []

    joined = os.environ.get("MYLLM_CONFIG_PATHS", "").strip()
    if joined:
        for part in joined.split(os.pathsep):
            part = part.strip()
            if part:
                paths.append(part)

    single = os.environ.get("MYLLM_CONFIG", "").strip()
    if single:
        paths.append(single)

    deduped = []
    seen = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _apply_runtime_config(master: bool = False) -> None:
    include_env = os.environ.get(
        "MYLLM_DISABLE_ENV_OVERRIDES", "0"
    ).strip().lower() not in {"1", "true", "yes", "y", "on"}

    manager = ConfigManager()
    snapshot = manager.load(
        config_paths=_collect_train_config_paths(),
        include_env=include_env,
    )

    if not snapshot.sources:
        return

    result = apply_overrides(snapshot.data, strict=False)

    if not master:
        return

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    snapshot_path = os.path.join(
        train_cfg.checkpoint_dir, "runtime_config_snapshot.json"
    )
    effective_path = os.path.join(
        train_cfg.checkpoint_dir, "runtime_config_effective.json"
    )

    ConfigManager.save_json(snapshot, snapshot_path)
    with open(effective_path, "w", encoding="utf-8") as f:
        json.dump(runtime_config_dict(), f, indent=2)

    LOGGER.info("%s", "=" * 60)
    LOGGER.info("TRAIN RUNTIME CONFIG")
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("sources      : %s", ", ".join(snapshot.sources))
    LOGGER.info("config_hash  : %s", snapshot.config_hash)
    LOGGER.info("applied_keys : %s", result["applied"])
    if result["unknown_sections"]:
        LOGGER.warning(
            "unknown sections: %s",
            ", ".join(result["unknown_sections"]),
        )
    if result["unknown_fields"]:
        LOGGER.warning(
            "unknown fields  : %s",
            ", ".join(result["unknown_fields"]),
        )
    LOGGER.info("snapshot     : %s", snapshot_path)
    LOGGER.info("effective    : %s", effective_path)
    LOGGER.info("%s", "=" * 60)


def _apply_attention_runtime_policy(master: bool = False) -> None:
    try:
        profile = load_profile()
    except Exception as exc:
        if master:
            LOGGER.warning("Hardware profile load skipped: %s", exc)
        return

    backend, use_flash = apply_profile_attention_policy(
        profile,
        current_backend=model_cfg.attention_backend,
        current_use_flash=model_cfg.use_flashattention,
    )
    dtype_name, use_grad_scaler = resolve_profile_dtype_policy(
        profile,
        current_dtype=train_cfg.dtype,
        current_amp_grad_scaler=train_cfg.amp_grad_scaler,
    )
    model_cfg.attention_backend = backend
    model_cfg.use_flashattention = use_flash
    train_cfg.dtype = dtype_name
    train_cfg.amp_grad_scaler = use_grad_scaler

    max_batch_size = profile.get("max_batch_size")
    if max_batch_size is not None:
        try:
            max_batch_size = int(max_batch_size)
        except (TypeError, ValueError):
            if master:
                LOGGER.warning(
                    "Ignoring invalid max_batch_size=%r in profile=%s",
                    max_batch_size,
                    profile.get("name", "unknown"),
                )
        else:
            if max_batch_size > 0 and train_cfg.batch_size > max_batch_size:
                prev_batch_size = train_cfg.batch_size
                train_cfg.batch_size = max_batch_size
                if master:
                    LOGGER.warning(
                        "Clamped batch_size from %s to profile max_batch_size=%s (profile=%s)",
                        prev_batch_size,
                        max_batch_size,
                        profile.get("name", "unknown"),
                    )

    if master:
        LOGGER.info(
            "Runtime policy: backend=%s flash=%s dtype=%s grad_scaler=%s profile=%s",
            model_cfg.attention_backend,
            model_cfg.use_flashattention,
            train_cfg.dtype,
            train_cfg.amp_grad_scaler,
            profile.get("name", "unknown"),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: Any,
    cfg: TrainConfig,
    device: str,
    amp_policy: Any,
) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    non_blocking = bool(
        getattr(cfg, "device_non_blocking", False)
        and torch.cuda.is_available()
        and getattr(cfg, "data_pin_memory", False)
    )

    for batch in val_loader:
        ids = batch["input_ids"].to(device, non_blocking=non_blocking)
        tgt = batch["targets"].to(device, non_blocking=non_blocking)

        with amp_policy.autocast():
            out = model(ids, targets=tgt)

        if isinstance(out, LossOutput):
            total_loss += out.ce_loss.item()
        else:
            total_loss += out[0].item()
        n_batches += 1

        if n_batches >= cfg.val_steps:
            break

    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_ckpt(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    step: int,
    loss: float,
    cfg: TrainConfig,
    master: bool,
    tag: str = "",
) -> None:
    if not master:
        return
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    sd = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )
    name = f"step_{step:07d}{tag}.pt"
    path = os.path.join(cfg.checkpoint_dir, name)
    atomic_torch_save(
        {
            "step": step,
            "model_state_dict": sd,
            "optim_state_dict": opt.state_dict(),
            "loss": loss,
        },
        path,
    )
    ckpts = sorted(
        f
        for f in os.listdir(cfg.checkpoint_dir)
        if f.endswith(".pt") and not f.endswith("_best.pt")
    )
    for old in ckpts[:-3]:
        os.remove(os.path.join(cfg.checkpoint_dir, old))
    LOGGER.info("Saved checkpoint: %s", path)


def load_ckpt(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: int,
) -> int:
    if not os.path.exists(cfg.checkpoint_dir):
        return 0
    ckpts = sorted(
        f
        for f in os.listdir(cfg.checkpoint_dir)
        if f.endswith(".pt") and not f.endswith("_best.pt")
    )
    if not ckpts:
        LOGGER.info("Fresh start.")
        return 0
    path = os.path.join(cfg.checkpoint_dir, ckpts[-1])
    loc = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    st, _ = safe_torch_load(path, map_location=loc, prefer_best=False)
    model.load_state_dict(extract_state_dict(st))
    optim_state = st.get("optim_state_dict")
    if isinstance(optim_state, dict):
        opt.load_state_dict(optim_state)
    step = int(st.get("step", 0))
    LOGGER.info("Resumed from %s step=%s", path, f"{step:,}")
    return step


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Validation-loss-based early stopping.

    Checks validation loss at val_every cadence. Stops when validation
    loss hasn't improved for `patience` consecutive checks.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        warmup: int = 200,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.best_val = float("inf")
        self.best_step = 0
        self.count = 0
        self.history: list = []

    def update(
        self, step: int, train_loss: float, val_loss: float
    ) -> bool:
        self.history.append({
            "step": step,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
        })
        if step < self.warmup:
            return False
        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            self.best_step = step
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience

    def status(self) -> str:
        if self.count == 0:
            return f"improving (best_val={self.best_val:.4f}@{self.best_step})"
        return (
            f"stale {self.count}/{self.patience} "
            f"(best_val={self.best_val:.4f}@{self.best_step})"
        )

    def save(self, path: str = "./loss_history.json") -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "history": self.history,
                    "best_val_loss": self.best_val,
                    "best_step": self.best_step,
                    "stopped_early": self.count >= self.patience,
                },
                f,
                indent=2,
            )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train() -> None:
    rank, local, world = setup()
    master = rank == 0
    use_ddp = world > 1 and _dist_active
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"

    configure_logging(level=os.environ.get("MYLLM_LOG_LEVEL", "INFO"))

    _apply_runtime_config(master=master)
    _apply_attention_runtime_policy(master=master)

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    configure_logging(
        level=os.environ.get("MYLLM_LOG_LEVEL", "INFO"),
        log_file=os.path.join(train_cfg.checkpoint_dir, "train.log"),
    )

    telemetry_path = default_telemetry_path(
        train_cfg.checkpoint_dir,
        phase="pretrain",
    )
    telemetry = TrainingTelemetry(
        telemetry_path,
        enabled=master,
        gradient_every=max(1, train_cfg.log_every),
        hardware_every=max(1, train_cfg.log_every // 2),
        loss_spike_threshold=float(
            os.environ.get("MYLLM_LOSS_SPIKE_THRESHOLD", "1.4")
        ),
        phase="pretrain",
    )
    if master:
        LOGGER.info("Telemetry stream: %s", os.path.abspath(telemetry_path))

    amp = build_amp_policy(
        AmpPolicySettings(
            dtype_name=train_cfg.dtype,
            device_type=("cuda" if torch.cuda.is_available() else "cpu"),
            use_grad_scaler=train_cfg.amp_grad_scaler,
            init_scale=train_cfg.amp_init_scale,
            growth_interval=train_cfg.amp_growth_interval,
        )
    )

    random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + rank)

    if master:
        show_config()

    # --- Tokenizer ---
    tok = BPETokenizer()
    tok_path = os.path.join(train_cfg.tokenizer_path, "encoder.json")
    if not os.path.exists(tok_path):
        if master:
            LOGGER.error("Tokenizer not found. Run: python train_tokenizer.py")
        cleanup()
        return
    tok.load(train_cfg.tokenizer_path)
    model_cfg.vocab_size = tok.vocab_size_

    # --- Model ---
    model = LLM(model_cfg).to(device)

    if train_cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            if master:
                LOGGER.info("torch.compile: enabled")
        except Exception as e:
            if master:
                LOGGER.warning(
                    "torch.compile: failed (%s), continuing without",
                    e,
                )

    # --- Optimizer ---
    opt = build_adamw(
        model,
        OptimizerSettings(
            lr=train_cfg.max_lr,
            weight_decay=train_cfg.weight_decay,
            beta1=train_cfg.beta1,
            beta2=train_cfg.beta2,
            eps=train_cfg.eps,
            use_fused=True,
        ),
    )

    start = load_ckpt(model, opt, train_cfg, local)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local] if torch.cuda.is_available() else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    if master:
        LOGGER.info("%s", "Single GPU" if not use_ddp else f"DDP x{world}")

    # --- Data ---
    loader = get_dataloader(
        tok,
        train_cfg,
        start_step=start,
        rank=rank,
        world=world,
    )
    val_loader = get_val_dataloader(tok, train_cfg)
    tps = (
        train_cfg.batch_size
        * train_cfg.seq_len
        * train_cfg.grad_accum
        * world
    )
    perf_monitor = PerformanceMonitor(tokens_per_step=tps)
    total = train_cfg.total_tokens // tps
    if train_cfg.total_tokens > 1_000_000_000:
        es = EarlyStopping(
            patience=sys.maxsize,
            min_delta=0.001,
            warmup=200,
        )
        if master:
            LOGGER.info(
                "Early stopping disabled for long-run token budget: %s",
                f"{train_cfg.total_tokens:,}",
            )
    else:
        es = EarlyStopping(patience=20, min_delta=0.001, warmup=200)

    if master:
        cs = train_cfg.curriculum_tokens // tps
        LOGGER.info(
            "Training %s step %s/%s",
            "RESUMED" if start > 0 else "started",
            f"{start:,}",
            f"{total:,}",
        )
        LOGGER.info(
            "tokens/step=%s total=%s curriculum_end=%s",
            f"{tps:,}",
            f"{total:,}",
            f"{cs:,}",
        )

    model.train()
    opt.zero_grad()
    step = start
    micro_step = 0
    accum_loss = 0.0
    accum_ce = 0.0
    accum_gate = 0.0
    accum_zl = 0.0
    accum_unc = 0.0
    accum_n = 0
    consecutive_oom_count = 0
    t0 = time.time()
    non_blocking = bool(
        train_cfg.device_non_blocking
        and torch.cuda.is_available()
        and train_cfg.data_pin_memory
    )
    loss_coeffs = {
        "ce": 1.0,
        "gate": float(train_cfg.gate_loss_coeff),
        "z": float(train_cfg.z_loss_coeff),
        "aux": float(train_cfg.aux_loss_coeff),
        "unc": float(train_cfg.unc_loss_coeff),
    }

    try:
        for batch in loader:
            if step >= total:
                if master:
                    LOGGER.info("Token budget reached.")
                break

            ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            tgt = batch["targets"].to(device, non_blocking=non_blocking)
            sync = should_sync_step(micro_step, train_cfg.grad_accum)

            try:
                with amp.autocast():
                    out = model(
                        ids,
                        targets=tgt,
                        loss_coeffs=loss_coeffs,
                    )

                    if isinstance(out, LossOutput):
                        loss = out.total_loss / train_cfg.grad_accum
                    else:
                        loss = out[0] / train_cfg.grad_accum

                if not torch.isfinite(loss.detach()):
                    if master:
                        LOGGER.critical(
                            "Non-finite loss at step %s; skipping batch.",
                            step,
                        )
                    opt.zero_grad()
                    accum_loss = 0.0
                    accum_ce = 0.0
                    accum_gate = 0.0
                    accum_zl = 0.0
                    accum_unc = 0.0
                    accum_n = 0
                    continue

                if use_ddp and not sync:
                    with model.no_sync():
                        amp.backward(loss)
                else:
                    amp.backward(loss)

                micro_step += 1

                accum_loss += loss.item() * train_cfg.grad_accum
                if isinstance(out, LossOutput):
                    accum_ce += out.ce_loss.item()
                    accum_gate += out.load_balance_loss.item()
                    accum_zl += out.z_loss.item()
                    accum_unc += out.unc_loss.item()
                accum_n += 1
                consecutive_oom_count = 0

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    consecutive_oom_count += 1
                    if master:
                        LOGGER.warning(
                            "OOM step %s - skip (%s/3)",
                            step,
                            consecutive_oom_count,
                        )
                    opt.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if consecutive_oom_count > 3:
                        raise RuntimeError("Fatal OOM loop detected") from e
                    continue
                raise

            if not sync:
                continue

            if not torch.isfinite(loss.detach()):
                if master:
                    LOGGER.critical(
                        "Non-finite loss at step %s; skipping optimizer step.",
                        step,
                    )
                opt.zero_grad()
                accum_loss = 0.0
                accum_ce = 0.0
                accum_gate = 0.0
                accum_zl = 0.0
                accum_unc = 0.0
                accum_n = 0
                continue

            grad_norm_value = amp.clip_grad_norm_(
                opt,
                model.parameters(),
                train_cfg.grad_clip,
            )
            lr = cosine_warmup_lr_from_tokens(
                step=step,
                tokens_per_step=tps,
                total_tokens=train_cfg.total_tokens,
                warmup_tokens=train_cfg.warmup_tokens,
                max_lr=train_cfg.max_lr,
                min_lr=train_cfg.min_lr,
            )
            set_optimizer_lr(opt, lr)
            amp.step(opt)
            amp.update()

            avg_loss = accum_loss / max(accum_n, 1)
            should_stop = False
            perf_payload = perf_monitor.sample(step=step, device=device)

            telemetry_payload = telemetry.log_step(
                step=step,
                loss=avg_loss,
                lr=lr,
                grad_norm=grad_norm_value,
                model=model.module if use_ddp else model,
                device=device,
                extra={
                    "tokens_seen_billion": round((step * tps) / 1e9, 6),
                    **perf_payload,
                },
            )
            if master and telemetry_payload.get("loss_spike"):
                LOGGER.critical(
                    "Loss spike detected at step %s: loss=%s baseline=%s ratio=%s",
                    step,
                    telemetry_payload.get("loss"),
                    telemetry_payload.get("loss_baseline"),
                    telemetry_payload.get("loss_ratio"),
                )
            opt.zero_grad()

            # --- Logging ---
            if master and should_run_interval(step, train_cfg.log_every):
                elapsed = time.time() - t0
                spd = (
                    tps * train_cfg.log_every / max(elapsed, 1e-9)
                )
                ce = train_cfg.curriculum_tokens // tps
                phase = "curr" if step < ce else "full"

                log_parts = [
                    f"step {step:>6}/{total}",
                    f"loss:{avg_loss:.4f}",
                ]
                if accum_n > 0:
                    log_parts.extend([
                        f"ce:{accum_ce / accum_n:.4f}",
                        f"gate:{accum_gate / accum_n:.4f}",
                        f"zl:{accum_zl / accum_n:.5f}",
                        f"unc:{accum_unc / accum_n:.4f}",
                    ])
                log_parts.extend([
                    f"lr:{lr:.2e}",
                    f"grad:{grad_norm_value:.3f}",
                    f"{spd / 1e3:.1f}Kt/s",
                    f"{step * tps / 1e9:.2f}B",
                    f"[{phase}]",
                ])
                LOGGER.info("%s", " | ".join(log_parts))

                t0 = time.time()
                accum_loss = 0.0
                accum_ce = 0.0
                accum_gate = 0.0
                accum_zl = 0.0
                accum_unc = 0.0
                accum_n = 0

            # --- Validation & early stopping ---
            if (
                master
                and should_run_interval(step, train_cfg.val_every)
            ):
                val_loss = validate(
                    model.module if use_ddp else model,
                    val_loader,
                    train_cfg,
                    device,
                    amp,
                )
                LOGGER.info("VAL loss=%.4f | %s", val_loss, es.status())
                hit = es.update(step, avg_loss, val_loss)
                if val_loss <= es.best_val:
                    save_ckpt(
                        model, opt, step, val_loss,
                        train_cfg, master, tag="_best"
                    )
                if hit:
                    LOGGER.warning("EARLY STOP step %s", step)
                    save_ckpt(
                        model, opt, step, es.best_val,
                        train_cfg, master,
                    )
                    es.save()
                    should_stop = True
                model.train()

            should_stop = _broadcast_stop(should_stop, device)
            if should_stop:
                cleanup()
                return

            if should_run_interval(step, train_cfg.checkpoint_every):
                save_ckpt(
                    model, opt, step, avg_loss,
                    train_cfg, master,
                )
            step += 1

    except DataPipelineError as e:
        if master:
            LOGGER.critical("FATAL DATA ERROR: %s", e)
        save_ckpt(
            model, opt, step, float("inf"),
            train_cfg, master, tag="_emergency"
        )
        cleanup()
        raise

    if master:
        LOGGER.info("Training complete!")
        save_ckpt(model, opt, step, 0.0, train_cfg, master)
        es.save()
    cleanup()


if __name__ == "__main__":
    train()