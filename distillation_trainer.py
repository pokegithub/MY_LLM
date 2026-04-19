"""Teacher-student distillation training loop."""

import os
import time
import random
from types import SimpleNamespace
import torch
import torch.nn.functional as F

from config import DistillConfig, distill_cfg, model_cfg, train_cfg
from tokenizer import BPETokenizer
from model import LLM
from data import get_dataloader, get_val_dataloader
from hardware_profiles import load_profile
from core.hardware_optim import apply_profile_attention_policy
from core.checkpoint_io import (
    atomic_torch_save,
    extract_state_dict,
    safe_torch_load,
)
from core.lr_schedule import cosine_warmup_lr
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


LOGGER = get_logger("distill")


def _set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_data_cfg(cfg: DistillConfig, tok: BPETokenizer):
    """Adapt distillation config to data.py expected TrainConfig-like shape."""
    return SimpleNamespace(
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        grad_accum=cfg.grad_accum,
        val_steps=25,
        max_open_sources=train_cfg.max_open_sources,
        curriculum_tokens=train_cfg.curriculum_tokens,
        dedup_ngram_size=train_cfg.dedup_ngram_size,
        dedup_threshold=train_cfg.dedup_threshold,
        min_doc_length=train_cfg.min_doc_length,
        max_doc_length=train_cfg.max_doc_length,
        quality_threshold=train_cfg.quality_threshold,
        excluded_benchmark_sources=train_cfg.excluded_benchmark_sources,
        val_data_dir=cfg.val_data_dir,
        data_num_workers=cfg.data_num_workers,
        data_prefetch_queue=cfg.data_prefetch_queue,
        data_prefetch_factor=cfg.data_prefetch_factor,
        data_persistent_workers=cfg.data_persistent_workers,
        data_pin_memory=cfg.data_pin_memory,
        device_non_blocking=cfg.device_non_blocking,
    )


def _apply_attention_runtime_policy() -> None:
    """Resolve attention backend from hardware profile for distillation runs."""
    try:
        profile = load_profile()
    except Exception as exc:
        LOGGER.warning("Hardware profile load skipped: %s", exc)
        return

    requested_backend = str(
        profile.get("attention_backend", model_cfg.attention_backend)
    ).strip().lower()
    requested_flash = bool(
        profile.get("use_flashattention", model_cfg.use_flashattention)
    )

    backend, use_flash = apply_profile_attention_policy(
        profile,
        current_backend=model_cfg.attention_backend,
        current_use_flash=model_cfg.use_flashattention,
    )
    model_cfg.attention_backend = backend
    model_cfg.use_flashattention = use_flash

    if requested_backend not in {"sdpa", "flash2", "auto"}:
        LOGGER.warning(
            "Profile requested unsupported backend '%s'; resolved to %s.",
            requested_backend,
            backend,
        )

    if requested_flash and requested_backend in {"auto", "flash2"} and backend != "flash2":
        LOGGER.warning(
            "Flash attention requested by profile '%s' but unavailable at runtime; falling back to %s.",
            profile.get("name", "unknown"),
            backend,
        )

    LOGGER.info(
        "Distill attention policy: backend=%s flash=%s profile=%s",
        model_cfg.attention_backend,
        model_cfg.use_flashattention,
        profile.get("name", "unknown"),
    )


def _validate_distill_recipe(cfg: DistillConfig) -> None:
    """Enforce stable distillation recipe for reproducible student behavior."""
    tol = 1e-8
    errors = []

    if abs(float(cfg.temperature) - 2.0) > tol:
        errors.append(f"temperature must be 2.0 (got {cfg.temperature})")
    if abs(float(cfg.alpha_kd) - 0.8) > tol:
        errors.append(f"alpha_kd must be 0.8 (got {cfg.alpha_kd})")
    if abs(float(cfg.alpha_ce) - 0.2) > tol:
        errors.append(f"alpha_ce must be 0.2 (got {cfg.alpha_ce})")
    if abs(float(cfg.gate_loss_coeff) - 0.01) > tol:
        errors.append(f"gate_loss_coeff must be 0.01 (got {cfg.gate_loss_coeff})")
    if int(cfg.data_num_workers) != 0:
        errors.append(f"data_num_workers must be 0 on T4-stability path (got {cfg.data_num_workers})")
    if int(cfg.data_prefetch_queue) != 8:
        errors.append(f"data_prefetch_queue must be 8 (got {cfg.data_prefetch_queue})")

    if errors:
        raise ValueError("Distillation recipe validation failed: " + "; ".join(errors))


def distill_train(cfg: DistillConfig = distill_cfg):
    _set_seed(42)
    _validate_distill_recipe(cfg)
    _apply_attention_runtime_policy()
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

    tok = BPETokenizer()
    tok.load(cfg.tokenizer_path)
    model_cfg.vocab_size = tok.vocab_size_

    teacher = LLM(model_cfg).to(device)
    student = LLM(model_cfg).to(device)

    if not os.path.exists(cfg.teacher_checkpoint):
        raise FileNotFoundError(
            f"Teacher checkpoint path not found: {cfg.teacher_checkpoint}"
        )
    t_st, t_path = safe_torch_load(cfg.teacher_checkpoint, map_location=device)
    teacher.load_state_dict(extract_state_dict(t_st))
    LOGGER.info("Teacher loaded: %s", t_path)

    if os.path.exists(cfg.student_checkpoint):
        try:
            s_st, s_path = safe_torch_load(cfg.student_checkpoint, map_location=device)
            student.load_state_dict(extract_state_dict(s_st))
            LOGGER.info("Student loaded: %s", s_path)
        except FileNotFoundError:
            LOGGER.info("Student checkpoint dir is empty; training from fresh student init.")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    opt = build_adamw(
        student,
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
        log_file=os.path.join(cfg.output_dir, "distill.log"),
    )
    telemetry_path = default_telemetry_path(cfg.output_dir, phase="distill")
    telemetry = TrainingTelemetry(
        telemetry_path,
        enabled=True,
        gradient_every=max(1, cfg.log_every),
        hardware_every=max(1, cfg.log_every // 2),
        loss_spike_threshold=float(
            os.environ.get("MYLLM_LOSS_SPIKE_THRESHOLD", "1.4")
        ),
        phase="distill",
    )
    LOGGER.info("Telemetry stream: %s", os.path.abspath(telemetry_path))

    data_cfg = _make_data_cfg(cfg, tok)
    train_loader = get_dataloader(tok, data_cfg, start_step=0)
    val_loader = get_val_dataloader(tok, data_cfg)

    step = 0
    micro_step = 0
    best_val = float("inf")
    t0 = time.time()
    student.train()
    non_blocking = bool(
        cfg.device_non_blocking
        and torch.cuda.is_available()
        and cfg.data_pin_memory
    )

    LOGGER.info(
        "Distillation: steps=%s batch=%s seq=%s temp=%s alpha_kd=%s alpha_ce=%s gate=%s z=%s",
        cfg.total_steps,
        cfg.batch_size,
        cfg.seq_len,
        cfg.temperature,
        cfg.alpha_kd,
        cfg.alpha_ce,
        cfg.gate_loss_coeff,
        cfg.z_loss_coeff,
    )

    while step < cfg.total_steps:
        for batch in train_loader:
            if step >= cfg.total_steps:
                break

            ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            targets = batch["targets"].to(device, non_blocking=non_blocking)

            with torch.no_grad(), amp.autocast():
                t_out = teacher(ids)
                t_logits = t_out.logits

            with amp.autocast():
                s_out = student(ids)
                s_logits = s_out.logits

                temp = max(cfg.temperature, 1e-3)
                kd = F.kl_div(
                    F.log_softmax(s_logits / temp, dim=-1),
                    F.softmax(t_logits / temp, dim=-1),
                    reduction="batchmean",
                ) * (temp * temp)

                ce = F.cross_entropy(
                    s_logits.reshape(-1, s_logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=tok.pad_token_id,
                )

                z_loss_coeff = float(cfg.z_loss_coeff)
                moe_gate = cfg.gate_loss_coeff * s_out.load_balance_loss
                moe_z = z_loss_coeff * s_out.z_loss
                aux_terms = [
                    layer.ffn.router_prob_cv
                    for layer in student.layers
                    if layer.is_moe
                ]
                aux_term = (
                    cfg.aux_loss_coeff * torch.stack(aux_terms).mean()
                    if aux_terms
                    else ce.new_zeros(())
                )
                moe_reg = moe_gate + moe_z + aux_term

                loss = cfg.alpha_kd * kd + cfg.alpha_ce * ce + moe_reg
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

            grad_norm_value = None
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
                student.parameters(),
                cfg.grad_clip,
            )
            amp.step(opt)
            amp.update()

            telemetry_payload = telemetry.log_step(
                step=step,
                loss=loss.item() * cfg.grad_accum,
                lr=lr,
                grad_norm=grad_norm_value,
                model=student,
                device=device,
                extra={
                    "kd_loss": round(float(kd.item()), 6),
                    "ce_loss": round(float(ce.item()), 6),
                    "gate_loss": round(float(s_out.load_balance_loss.item()), 6),
                    "z_loss": round(float(s_out.z_loss.item()), 6),
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
            opt.zero_grad(set_to_none=True)

            if should_run_interval(step, cfg.log_every, include_zero=True):
                elapsed = max(time.time() - t0, 1e-9)
                LOGGER.info(
                    "distill step %s/%s | loss:%.4f | kd:%.4f | ce:%.4f | gate:%.4f | z:%.5f | lr:%.2e | %.1fKt/s",
                    step,
                    cfg.total_steps,
                    loss.item() * cfg.grad_accum,
                    kd.item(),
                    ce.item(),
                    s_out.load_balance_loss.item(),
                    s_out.z_loss.item(),
                    opt.param_groups[0]["lr"],
                    (cfg.batch_size * cfg.seq_len) / elapsed / 1e3,
                )
                t0 = time.time()

            if should_run_interval(step, cfg.save_every):
                student.eval()
                vloss = 0.0
                vn = 0
                with torch.no_grad():
                    for vb in val_loader:
                        vids = vb["input_ids"].to(device, non_blocking=non_blocking)
                        vt = vb["targets"].to(device, non_blocking=non_blocking)
                        with amp.autocast():
                            vout = student(vids)
                            vl = F.cross_entropy(
                                vout.logits.reshape(-1, vout.logits.size(-1)),
                                vt.reshape(-1),
                                ignore_index=tok.pad_token_id,
                            )
                        vloss += vl.item()
                        vn += 1
                        if vn >= 10:
                            break
                vmean = vloss / max(vn, 1)
                LOGGER.info("distill val loss: %.4f", vmean)

                save_obj = {
                    "step": step,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": vmean,
                }
                out = os.path.join(cfg.output_dir, f"distill_step_{step:07d}.pt")
                atomic_torch_save(save_obj, out)
                if vmean < best_val:
                    best_val = vmean
                    best_out = os.path.join(cfg.output_dir, "distill_best.pt")
                    atomic_torch_save(save_obj, best_out)
                    LOGGER.info("Distill new best: %s", best_out)
                student.train()

            step += 1
            if step >= cfg.total_steps:
                break

    final_path = os.path.join(cfg.output_dir, "distill_final.pt")
    atomic_torch_save({"model_state_dict": student.state_dict()}, final_path)
    LOGGER.info("Distillation complete: %s", final_path)


if __name__ == "__main__":
    distill_train()
