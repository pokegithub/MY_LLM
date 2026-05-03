"""Training loop for pretraining with validation, telemetry, and checkpoints."""

import os
import sys
import time
import json
import hashlib
import random
import platform
import subprocess
from typing import Any, Dict, List, Optional

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
from core.checkpoint_io import (
    atomic_torch_save,
    extract_state_dict,
    safe_torch_load,
)
from core.config_manager import ConfigManager
from core.dependency_checks import check_requirement_file
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
    get_amp_dtype,
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
TOKEN_CACHE_DIR = os.path.join(".", "data_cache", "tokens")
TOKEN_ARTIFACT_MANIFEST_PATH = os.path.join(
    ".", "data_cache", "token_artifacts_manifest.json"
)
TRAINING_REQUIREMENTS_PATH = os.path.join(".", "requirement.txt")


class TrainingPreflightError(RuntimeError):
    """Raised when training readiness checks fail before training starts."""


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
# Training readiness preflight
# ---------------------------------------------------------------------------

def set_training_seed(
    seed: int = 42,
    rank: int = 0,
    *,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """Set process-local training seeds and report the applied policy."""
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": effective_seed,
        "deterministic_algorithms": bool(deterministic),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _add_preflight_check(
    checks: List[Dict[str, Any]],
    name: str,
    status: str,
    detail: str,
    **extra: Any,
) -> None:
    if status not in {"pass", "warn", "fail", "unverified"}:
        raise ValueError(f"invalid preflight status: {status}")
    item = {"name": name, "status": status, "detail": detail}
    item.update(extra)
    checks.append(item)


def _dense_parameter_lower_bound(cfg: ModelConfig, vocab_size: int) -> Optional[int]:
    if cfg.enabled_experimental_features():
        return None
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    attn = (cfg.dim * cfg.dim) + (2 * cfg.dim * kv_dim) + (cfg.dim * cfg.dim)
    ffn = 3 * cfg.dim * cfg.ffn_dim
    norms = 2 * cfg.dim
    per_layer = attn + ffn + norms
    return (vocab_size * cfg.dim) + (cfg.n_layers * per_layer) + cfg.dim


def _load_token_artifact_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("token artifact manifest root must be an object")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("token artifact manifest must contain an artifacts list")
    return payload


def run_training_preflight(
    cfg: TrainConfig = train_cfg,
    mcfg: ModelConfig = model_cfg,
    *,
    token_cache_dir: str = TOKEN_CACHE_DIR,
    token_artifact_manifest_path: str = TOKEN_ARTIFACT_MANIFEST_PATH,
    dependency_requirements_path: Optional[str] = TRAINING_REQUIREMENTS_PATH,
) -> Dict[str, Any]:
    """Validate training readiness without constructing dataloaders or training."""
    checks: List[Dict[str, Any]] = []
    tokenizer_vocab: Optional[int] = None

    if dependency_requirements_path is not None:
        dep_report = check_requirement_file(dependency_requirements_path)
        failed = [
            item for item in dep_report["items"]
            if item.get("status") == "fail"
        ]
        unverified = [
            item for item in dep_report["items"]
            if item.get("status") == "unverified"
        ]
        if dep_report["ok"]:
            _add_preflight_check(
                checks,
                "dependency_environment",
                "pass",
                "installed packages satisfy declared requirement file",
                requirement_path=dependency_requirements_path,
                package_count=len(dep_report["items"]),
            )
        else:
            status = "fail" if failed else "unverified"
            _add_preflight_check(
                checks,
                "dependency_environment",
                status,
                "declared dependency environment is not verified for training",
                requirement_path=dependency_requirements_path,
                failed_packages=[item["package"] for item in failed[:8]],
                unverified_packages=[item["package"] for item in unverified[:8]],
                summary=dep_report["summary"],
            )

    try:
        if cfg.seq_len > mcfg.max_seq_len:
            _add_preflight_check(
                checks,
                "config_profile_resolution",
                "fail",
                "train.seq_len exceeds model.max_seq_len",
                seq_len=cfg.seq_len,
                max_seq_len=mcfg.max_seq_len,
                profile_name=mcfg.profile_name,
            )
        else:
            tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accum
            _add_preflight_check(
                checks,
                "config_profile_resolution",
                "pass",
                "model profile and train sequence geometry are internally consistent",
                profile_name=mcfg.profile_name,
                seq_len=cfg.seq_len,
                max_seq_len=mcfg.max_seq_len,
                tokens_per_step=tokens_per_step,
            )
    except (TypeError, ValueError) as exc:
        _add_preflight_check(
            checks,
            "config_profile_resolution",
            "fail",
            f"config/profile check failed: {exc}",
        )

    encoder_path = os.path.join(cfg.tokenizer_path, "encoder.json")
    merges_path = os.path.join(cfg.tokenizer_path, "merges.json")
    if not os.path.isfile(encoder_path) or not os.path.isfile(merges_path):
        _add_preflight_check(
            checks,
            "tokenizer_loadability",
            "fail",
            "tokenizer artifacts are missing",
            encoder_path=encoder_path,
            merges_path=merges_path,
        )
    else:
        try:
            tok = BPETokenizer()
            tok.load(cfg.tokenizer_path)
            tokenizer_vocab = tok.vocab_size_
            _add_preflight_check(
                checks,
                "tokenizer_loadability",
                "pass",
                "tokenizer artifacts loaded",
                tokenizer_path=cfg.tokenizer_path,
                vocab_size=tok.vocab_size_,
                special_tokens_ok=all(t in tok.encoder for t in tok.SPECIAL),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            _add_preflight_check(
                checks,
                "tokenizer_loadability",
                "fail",
                f"tokenizer load failed: {exc}",
                tokenizer_path=cfg.tokenizer_path,
            )

    token_files: List[str] = []
    if os.path.isdir(token_cache_dir):
        token_files = [
            os.path.join(token_cache_dir, name)
            for name in os.listdir(token_cache_dir)
            if name.endswith(".bin")
        ]
    if not token_files:
        _add_preflight_check(
            checks,
            "token_cache_presence",
            "fail",
            "no local .bin token cache found; this preflight does not verify streaming dataset access",
            token_cache_dir=token_cache_dir,
        )
    else:
        total_bytes = sum(os.path.getsize(path) for path in token_files)
        _add_preflight_check(
            checks,
            "token_cache_presence",
            "pass",
            "local token cache files found",
            token_cache_dir=token_cache_dir,
            file_count=len(token_files),
            size_bytes=total_bytes,
        )

    if not os.path.isfile(token_artifact_manifest_path):
        status = "unverified" if token_files else "fail"
        _add_preflight_check(
            checks,
            "token_artifact_manifest",
            status,
            "token artifact manifest is missing; token cache provenance is not inspectable",
            manifest_path=token_artifact_manifest_path,
        )
    else:
        try:
            manifest = _load_token_artifact_manifest(token_artifact_manifest_path)
            artifacts = manifest["artifacts"]
            existing = [
                item for item in artifacts
                if isinstance(item, dict) and item.get("exists") is True
            ]
            if not existing:
                _add_preflight_check(
                    checks,
                    "token_artifact_manifest",
                    "fail",
                    "manifest has no existing token artifacts",
                    manifest_path=token_artifact_manifest_path,
                    artifact_count=len(artifacts),
                )
            else:
                _add_preflight_check(
                    checks,
                    "token_artifact_manifest",
                    "pass",
                    "token artifact manifest is loadable and reports existing artifacts",
                    manifest_path=token_artifact_manifest_path,
                    artifact_count=len(artifacts),
                    existing_artifact_count=len(existing),
                    schema=manifest.get("schema", "unknown"),
                )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            _add_preflight_check(
                checks,
                "token_artifact_manifest",
                "fail",
                f"token artifact manifest is invalid: {exc}",
                manifest_path=token_artifact_manifest_path,
            )

    ckpt_dir = cfg.checkpoint_dir
    if os.path.exists(ckpt_dir) and not os.path.isdir(ckpt_dir):
        _add_preflight_check(
            checks,
            "checkpoint_directory",
            "fail",
            "checkpoint_dir exists but is not a directory",
            checkpoint_dir=ckpt_dir,
        )
    elif os.path.isdir(ckpt_dir):
        ckpts = [name for name in os.listdir(ckpt_dir) if name.endswith(".pt")]
        _add_preflight_check(
            checks,
            "checkpoint_directory",
            "pass",
            "checkpoint directory exists",
            checkpoint_dir=ckpt_dir,
            checkpoint_count=len(ckpts),
        )
    else:
        parent = os.path.dirname(os.path.abspath(ckpt_dir)) or "."
        if os.path.isdir(parent):
            _add_preflight_check(
                checks,
                "checkpoint_directory",
                "warn",
                "checkpoint directory does not exist yet; training will create it",
                checkpoint_dir=ckpt_dir,
            )
        else:
            _add_preflight_check(
                checks,
                "checkpoint_directory",
                "fail",
                "checkpoint directory parent does not exist",
                checkpoint_dir=ckpt_dir,
                parent=parent,
            )

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        amp_dtype = get_amp_dtype(cfg.dtype)
        detail = "dtype is supported by AMP policy"
        status = "pass"
        if device_type == "cpu" and amp_dtype is torch.float16:
            status = "warn"
            detail = "float16 CPU training is not an efficient or validated path"
        elif device_type == "cuda" and amp_dtype is torch.bfloat16:
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
            if callable(bf16_supported) and not bf16_supported():
                status = "warn"
                detail = "CUDA bfloat16 is unsupported; AMP policy will fall back to float16"
        _add_preflight_check(
            checks,
            "dtype_device_compatibility",
            status,
            detail,
            dtype=cfg.dtype,
            device_type=device_type,
        )
    except ValueError as exc:
        _add_preflight_check(
            checks,
            "dtype_device_compatibility",
            "fail",
            f"unsupported dtype: {exc}",
            dtype=cfg.dtype,
            device_type=device_type,
        )

    vocab_for_estimate = tokenizer_vocab or mcfg.vocab_size
    dense_params = _dense_parameter_lower_bound(mcfg, vocab_for_estimate)
    if dense_params is None:
        _add_preflight_check(
            checks,
            "memory_lower_bound",
            "unverified",
            "parameter lower-bound estimate is not implemented for experimental model features",
            profile_name=mcfg.profile_name,
        )
    else:
        bytes_per_param = 2 if str(cfg.dtype).lower() in {"bfloat16", "float16"} else 4
        lower_bound_gb = dense_params * bytes_per_param / (1024 ** 3)
        grad_gb = dense_params * 4 / (1024 ** 3)
        adam_state_gb = dense_params * 8 / (1024 ** 3)
        activation_bytes = (
            int(cfg.batch_size)
            * int(cfg.seq_len)
            * int(mcfg.n_layers)
            * int(mcfg.dim)
            * bytes_per_param
            * 6
        )
        activation_gb = activation_bytes / (1024 ** 3)
        training_estimate_gb = lower_bound_gb + grad_gb + adam_state_gb + activation_gb
        status = "unverified"
        detail = (
            "training memory estimate computed from explicit assumptions; "
            "compile overhead, allocator fragmentation, dataloader memory, and "
            "hardware fit are not measured"
        )
        extra = {}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024 ** 3)
            extra["cuda_total_memory_gb"] = round(total_gb, 3)
            if training_estimate_gb > total_gb:
                status = "fail"
                detail = "estimated training memory exceeds visible CUDA memory"
            elif training_estimate_gb > total_gb * 0.6:
                status = "warn"
                detail = "estimated training memory consumes most visible CUDA memory"
            else:
                status = "pass"
        _add_preflight_check(
            checks,
            "memory_lower_bound",
            status,
            detail,
            estimated_params_lower_bound=dense_params,
            parameter_memory_lower_bound_gb=round(lower_bound_gb, 3),
            gradient_memory_estimate_gb=round(grad_gb, 3),
            adam_state_memory_estimate_gb=round(adam_state_gb, 3),
            activation_memory_estimate_gb=round(activation_gb, 3),
            estimated_training_memory_gb=round(training_estimate_gb, 3),
            estimate_scope="rough_training_memory_estimate_not_measured",
            **extra,
        )

    failures = [item for item in checks if item["status"] == "fail"]
    warnings = [item for item in checks if item["status"] == "warn"]
    unverified = [item for item in checks if item["status"] == "unverified"]
    ready = not failures and not unverified
    return {
        "ok": not failures,
        "ready_for_training": ready,
        "checks": checks,
        "summary": {
            "failures": len(failures),
            "warnings": len(warnings),
            "unverified": len(unverified),
            "ready_for_training": ready,
        },
    }


def assert_training_preflight_ready(report: Dict[str, Any]) -> None:
    if report.get("ready_for_training") is True:
        return
    failed = [
        item for item in report.get("checks", [])
        if item.get("status") in {"fail", "unverified"}
    ]
    detail = "; ".join(
        f"{item.get('name')}: {item.get('status')} - {item.get('detail')}"
        for item in failed
    )
    raise TrainingPreflightError(
        "training preflight did not establish readiness: " + detail
    )


def _first_token_artifact(
    token_cache_dir: str,
    dtype: Any,
    min_tokens: int,
) -> str:
    if not os.path.isdir(token_cache_dir):
        raise TrainingPreflightError(
            f"token cache directory is missing: {token_cache_dir}"
        )
    try:
        import numpy as np
        itemsize = np.dtype(dtype).itemsize
    except (TypeError, ValueError) as exc:
        raise TrainingPreflightError(f"invalid token dtype: {dtype}") from exc

    for name in sorted(os.listdir(token_cache_dir)):
        if not name.endswith(".bin"):
            continue
        path = os.path.join(token_cache_dir, name)
        if os.path.getsize(path) >= min_tokens * itemsize:
            return path
    raise TrainingPreflightError(
        f"no token artifact in {token_cache_dir} has at least {min_tokens} tokens"
    )


def run_tiny_real_data_validation(
    *,
    token_cache_dir: str = TOKEN_CACHE_DIR,
    tokenizer_path: str = "./tokenizer_data",
    checkpoint_dir: str = "./run_artifacts/tiny_real_data_validation",
    seq_len: int = 16,
    seed: int = 123,
    deterministic: bool = False,
    data_loader_smoke: bool = False,
) -> Dict[str, Any]:
    """Run one tiny update from real local token artifacts.

    This validates artifact readability, one optimizer step, checkpoint
    save/load, and missing-checkpoint eval gating. It is not a training run and
    does not measure model quality.
    """
    import numpy as np
    from config import EDGE_0_5B_GQA, build_model_config
    from eval_suite import inspect_checkpoint_path

    if seq_len < 4:
        raise ValueError("seq_len must be at least 4 for validation")

    started_at = time.time()
    seed_report = set_training_seed(seed, deterministic=deterministic)
    tok = BPETokenizer()
    tok.load(tokenizer_path)
    dtype = np.uint16 if tok.vocab_size_ <= 65535 else np.uint32
    artifact_path = _first_token_artifact(
        token_cache_dir,
        dtype,
        min_tokens=seq_len + 1,
    )
    raw = np.fromfile(artifact_path, dtype=dtype, count=seq_len + 1)
    if raw.size < seq_len + 1:
        raise TrainingPreflightError(
            f"token artifact has insufficient tokens: {artifact_path}"
        )
    max_token = int(raw.max(initial=0))
    if max_token >= tok.vocab_size_:
        raise TrainingPreflightError(
            f"token id {max_token} exceeds tokenizer vocab size {tok.vocab_size_}"
        )

    tiny_cfg = build_model_config(
        EDGE_0_5B_GQA,
        vocab_size=tok.vocab_size_,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=max(32, seq_len),
        sliding_window=8,
    )
    tiny_config_payload = {
        "profile_name": tiny_cfg.profile_name,
        "vocab_size": tiny_cfg.vocab_size,
        "dim": tiny_cfg.dim,
        "n_layers": tiny_cfg.n_layers,
        "n_heads": tiny_cfg.n_heads,
        "n_kv_heads": tiny_cfg.n_kv_heads,
        "ffn_dim": tiny_cfg.ffn_dim,
        "max_seq_len": tiny_cfg.max_seq_len,
        "sliding_window": tiny_cfg.sliding_window,
    }
    tiny_config_hash = hashlib.sha256(
        json.dumps(tiny_config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    tiny_train_cfg = TrainConfig(
        tokenizer_path=tokenizer_path,
        checkpoint_dir=checkpoint_dir,
        batch_size=1,
        seq_len=seq_len,
        grad_accum=1,
        total_tokens=seq_len + 1,
        warmup_tokens=4,
        curriculum_tokens=seq_len + 1,
        compile_model=False,
        data_num_workers=0,
        data_persistent_workers=False,
        data_pin_memory=False,
        device_non_blocking=False,
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = LLM(tiny_cfg).train()
    opt = build_adamw(
        model,
        OptimizerSettings(lr=1e-3, weight_decay=0.0, use_fused=False),
    )
    amp = build_amp_policy(
        AmpPolicySettings(
            dtype_name="bfloat16",
            device_type="cpu",
            use_grad_scaler=False,
        )
    )
    ids = torch.tensor(raw[:-1], dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(raw[1:], dtype=torch.long).unsqueeze(0)

    with amp.autocast():
        out = model(ids, targets=targets)
    if not isinstance(out, LossOutput) or not torch.isfinite(out.total_loss.detach()):
        raise TrainingPreflightError("tiny real-data validation produced invalid loss")

    amp.backward(out.total_loss)
    grad_norm = amp.clip_grad_norm_(opt, model.parameters(), 1.0)
    if not torch.isfinite(torch.tensor(float(grad_norm))):
        raise TrainingPreflightError(
            "tiny real-data validation produced invalid grad norm"
        )
    amp.step(opt)
    amp.update()
    opt.zero_grad()

    save_ckpt(
        model,
        opt,
        step=1,
        loss=float(out.total_loss.detach().cpu().item()),
        cfg=tiny_train_cfg,
        master=True,
        tag="_tiny_real_data_validation",
    )
    checkpoint_path = os.path.join(
        checkpoint_dir,
        "step_0000001_tiny_real_data_validation.pt",
    )
    restored = LLM(tiny_cfg)
    restored_opt = build_adamw(
        restored,
        OptimizerSettings(lr=1e-3, weight_decay=0.0, use_fused=False),
    )
    resumed_step = load_ckpt(restored, restored_opt, tiny_train_cfg, device=0)
    if resumed_step != 1:
        raise TrainingPreflightError(
            f"checkpoint resume step mismatch: expected 1, got {resumed_step}"
        )

    valid_meta = inspect_checkpoint_path(checkpoint_dir)
    if valid_meta.get("selected_checkpoint_exists") is not True:
        raise TrainingPreflightError(
            "eval checkpoint inspection did not find the tiny checkpoint"
        )

    missing_meta = inspect_checkpoint_path(
        os.path.join(checkpoint_dir, "missing_eval_checkpoint")
    )
    if missing_meta.get("load_status") != "missing":
        raise TrainingPreflightError(
            "eval missing-checkpoint gate did not report missing"
        )

    loader_report = {
        "enabled": bool(data_loader_smoke),
        "status": "not_run",
    }
    if data_loader_smoke:
        import contextlib
        import io
        from data import get_dataloader

        loader_cfg = TrainConfig(
            tokenizer_path=tokenizer_path,
            checkpoint_dir=checkpoint_dir,
            batch_size=1,
            seq_len=seq_len,
            grad_accum=1,
            total_tokens=seq_len + 1,
            warmup_tokens=4,
            curriculum_tokens=seq_len + 1,
            compile_model=False,
            data_num_workers=0,
            data_persistent_workers=False,
            data_pin_memory=False,
            device_non_blocking=False,
        )
        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            batch = next(iter(get_dataloader(tok, loader_cfg)))
        input_shape = list(batch["input_ids"].shape)
        target_shape = list(batch["targets"].shape)
        if input_shape != [1, seq_len] or target_shape != [1, seq_len]:
            raise TrainingPreflightError(
                f"data loader smoke produced wrong shapes: {input_shape}, {target_shape}"
            )
        loader_report = {
            "enabled": True,
            "status": "pass",
            "input_shape": input_shape,
            "target_shape": target_shape,
            "captured_stdout": captured_stdout.getvalue(),
        }

    return {
        "schema": "tiny_real_data_validation_v1",
        "ok": True,
        "artifact_path": artifact_path,
        "tokens_read": int(raw.size),
        "token_dtype": np.dtype(dtype).name,
        "tokenizer_vocab_size": tok.vocab_size_,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": os.path.isfile(checkpoint_path),
        "checkpoint_inspection": valid_meta,
        "resumed_step": resumed_step,
        "loss": float(out.total_loss.detach().cpu().item()),
        "grad_norm": float(grad_norm),
        "eval_missing_checkpoint_gate": missing_meta.get("load_status"),
        "data_loader_smoke": loader_report,
        "model_config_hash": tiny_config_hash,
        "model_config": tiny_config_payload,
        "seed_report": seed_report,
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "runtime_seconds": round(time.time() - started_at, 4),
        "quality_claim": "none",
        "capability_evidence": "none",
    }


def _cuda_device_reports() -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    if not torch.cuda.is_available():
        return devices
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
            "capability": list(props.major_minor)
            if hasattr(props, "major_minor")
            else [props.major, props.minor],
            "multi_processor_count": props.multi_processor_count,
        })
    return devices


def _parse_nvidia_smi_csv(stdout: str) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    for index, line in enumerate(stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue
        memory_mib: Optional[int]
        try:
            memory_mib = int(float(parts[1]))
        except ValueError:
            memory_mib = None
        devices.append({
            "index": index,
            "name": parts[0],
            "total_memory_mib": memory_mib,
            "total_memory_gb": (
                round(memory_mib / 1024, 3) if memory_mib is not None else None
            ),
            "driver_version": parts[2],
            "compute_capability": parts[3],
        })
    return devices


def _nvidia_smi_report() -> Dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return {
            "available": False,
            "gpus": [],
            "error": "nvidia-smi executable not found",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "available": False,
            "gpus": [],
            "error": f"nvidia-smi timed out after {exc.timeout} seconds",
        }

    if proc.returncode != 0:
        return {
            "available": False,
            "gpus": [],
            "error": (proc.stderr or proc.stdout or "").strip()[:500],
        }
    devices = _parse_nvidia_smi_csv(proc.stdout)
    return {
        "available": bool(devices),
        "gpus": devices,
        "error": "",
    }


def _probe_cuda_tensor_ops() -> Dict[str, Any]:
    report = {
        "cuda_tensor_op_success": False,
        "status": "not_run_cuda_unavailable",
        "dtype_paths": {
            "float32": "unverified_cuda_unavailable",
            "float16": "unverified_cuda_unavailable",
            "bfloat16": "unverified_cuda_unavailable",
        },
        "errors": {},
    }
    if not torch.cuda.is_available():
        return report

    report["status"] = "running"
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    for name, dtype in dtype_map.items():
        if name == "bfloat16":
            checker = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(checker) and not checker():
                report["dtype_paths"][name] = "unavailable_not_supported"
                continue
        try:
            x = torch.ones((2, 2), device="cuda", dtype=dtype)
            y = (x + 1).sum()
            if not torch.isfinite(y.detach().float()):
                raise RuntimeError("non-finite CUDA tensor result")
            torch.cuda.synchronize()
            report["dtype_paths"][name] = "pass"
        except torch.cuda.OutOfMemoryError as exc:
            report["dtype_paths"][name] = "oom"
            report["errors"][name] = str(exc)
        except RuntimeError as exc:
            report["dtype_paths"][name] = "fail"
            report["errors"][name] = str(exc)

    report["cuda_tensor_op_success"] = report["dtype_paths"]["float32"] == "pass"
    report["status"] = "pass" if report["cuda_tensor_op_success"] else "fail"
    return report


def _classify_gpu_evidence(
    *,
    cuda_available: bool,
    system_gpu_detected: bool,
    tensor_op_success: bool,
    validation: Optional[Dict[str, Any]],
) -> str:
    if not cuda_available and system_gpu_detected:
        return "gpu_detected_but_unusable"
    if not cuda_available:
        return "gpu_unavailable"
    if not tensor_op_success:
        return "gpu_detected_but_unusable"
    if not validation:
        return "gpu_partial_unverified"
    if validation.get("status") == "pass":
        return "gpu_tiny_step_passed"
    if validation.get("error_type") == "cuda_oom":
        return "gpu_tiny_step_oom"
    return "gpu_partial_unverified"


def _repo_local_python_report(
    *,
    repo_root: Optional[str] = None,
    executable: Optional[str] = None,
) -> Dict[str, Any]:
    root = os.path.abspath(repo_root or os.getcwd())
    exe = os.path.abspath(executable or sys.executable)
    candidates = [
        os.path.abspath(os.path.join(root, ".venv", "Scripts", "python.exe")),
        os.path.abspath(os.path.join(root, ".venv", "bin", "python")),
    ]
    normalized_exe = os.path.normcase(exe)
    normalized_candidates = [os.path.normcase(path) for path in candidates]
    return {
        "executable": exe,
        "expected_repo_local_candidates": candidates,
        "is_repo_local_venv": normalized_exe in normalized_candidates,
        "source_of_truth": os.path.abspath(
            os.path.join(root, ".venv", "Scripts", "python.exe")
        ),
    }


def _cuda_dtype_supported(dtype_name: str) -> bool:
    if dtype_name == "float32":
        return torch.cuda.is_available()
    if dtype_name == "float16":
        return torch.cuda.is_available()
    if dtype_name == "bfloat16":
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        return bool(torch.cuda.is_available() and callable(checker) and checker())
    return False


def _preferred_cuda_validation_dtype() -> str:
    return "bfloat16" if _cuda_dtype_supported("bfloat16") else "float16"


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(exc).lower()
    )


def _classify_gpu_fit_attempt(attempt: Dict[str, Any]) -> str:
    if attempt.get("error_type") == "cuda_oom":
        return "does_not_fit"
    if attempt.get("status") == "pass":
        if (
            attempt.get("forward_passed")
            and attempt.get("backward_optimizer_passed")
            and attempt.get("checkpoint_write_passed")
            and attempt.get("resume_reload_passed")
        ):
            return "fits_mechanics_only"
        return "partially_unverified"
    if str(attempt.get("status", "")).startswith("skipped"):
        return "partially_unverified"
    return "partially_unverified"


def _gpu_fit_matrix_candidates() -> List[Dict[str, Any]]:
    preferred = _preferred_cuda_validation_dtype()
    return [
        {
            "name": "micro_fp16_64d_2l_s16",
            "dtype_name": "float16",
            "dim": 64,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_dim": 128,
            "seq_len": 16,
            "batch_size": 1,
        },
        {
            "name": "micro_bf16_64d_2l_s16",
            "dtype_name": "bfloat16",
            "dim": 64,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_dim": 128,
            "seq_len": 16,
            "batch_size": 1,
        },
        {
            "name": f"short_{preferred}_96d_3l_s32",
            "dtype_name": preferred,
            "dim": 96,
            "n_layers": 3,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_dim": 192,
            "seq_len": 32,
            "batch_size": 1,
        },
        {
            "name": f"bounded_{preferred}_128d_4l_s64",
            "dtype_name": preferred,
            "dim": 128,
            "n_layers": 4,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_dim": 256,
            "seq_len": 64,
            "batch_size": 1,
        },
        {
            "name": f"upper_tiny_{preferred}_192d_6l_s64",
            "dtype_name": preferred,
            "dim": 192,
            "n_layers": 6,
            "n_heads": 6,
            "n_kv_heads": 2,
            "ffn_dim": 512,
            "seq_len": 64,
            "batch_size": 1,
        },
    ]


def _validation_model_config(
    *,
    vocab_size: int,
    seq_len: int,
    overrides: Dict[str, Any],
) -> ModelConfig:
    from config import EDGE_0_5B_GQA, build_model_config

    cfg_values = {
        "vocab_size": int(vocab_size),
        "dim": int(overrides["dim"]),
        "n_layers": int(overrides["n_layers"]),
        "n_heads": int(overrides["n_heads"]),
        "n_kv_heads": int(overrides["n_kv_heads"]),
        "ffn_dim": int(overrides["ffn_dim"]),
        "max_seq_len": max(int(overrides.get("max_seq_len", seq_len)), int(seq_len), 32),
        "sliding_window": min(int(overrides.get("sliding_window", 16)), int(seq_len)),
        "use_flashattention": False,
        "attention_backend": "sdpa",
    }
    return build_model_config(EDGE_0_5B_GQA, **cfg_values)


def _model_config_payload(cfg: ModelConfig) -> Dict[str, Any]:
    return {
        "profile_name": cfg.profile_name,
        "vocab_size": cfg.vocab_size,
        "dim": cfg.dim,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "ffn_dim": cfg.ffn_dim,
        "max_seq_len": cfg.max_seq_len,
        "sliding_window": cfg.sliding_window,
    }


def _run_gpu_fit_attempt(
    *,
    candidate: Dict[str, Any],
    vocab_size: int,
    checkpoint_root: str,
    seed: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": candidate["name"],
        "device": "cuda",
        "dtype_requested": candidate["dtype_name"],
        "batch_size": int(candidate["batch_size"]),
        "seq_len": int(candidate["seq_len"]),
        "config_override": {
            key: candidate[key]
            for key in ("dim", "n_layers", "n_heads", "n_kv_heads", "ffn_dim")
        },
        "forward_passed": False,
        "backward_optimizer_passed": False,
        "checkpoint_write_passed": False,
        "resume_reload_passed": False,
        "quality_claim": "none",
    }
    dtype_name = str(candidate["dtype_name"])
    if not torch.cuda.is_available():
        result.update({
            "status": "skipped_cuda_unavailable",
            "fit_classification": "partially_unverified",
            "error": "CUDA is unavailable to PyTorch.",
        })
        return result
    if not _cuda_dtype_supported(dtype_name):
        result.update({
            "status": f"skipped_{dtype_name}_unavailable",
            "fit_classification": "partially_unverified",
            "error": f"{dtype_name} is not supported by this CUDA path.",
        })
        return result

    started_at = time.time()
    set_training_seed(seed, deterministic=False)
    device = torch.device("cuda")
    attempt_dir = os.path.join(checkpoint_root, candidate["name"])
    os.makedirs(attempt_dir, exist_ok=True)
    for name in os.listdir(attempt_dir):
        if name.endswith(".pt") or name.endswith(".pt.sha256"):
            os.remove(os.path.join(attempt_dir, name))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = torch.cuda.memory_allocated(device)
    memory_reserved_before = torch.cuda.memory_reserved(device)
    try:
        cfg = _validation_model_config(
            vocab_size=vocab_size,
            seq_len=int(candidate["seq_len"]),
            overrides=candidate,
        )
        result["model_config"] = _model_config_payload(cfg)
        validation_cfg = TrainConfig(
            checkpoint_dir=attempt_dir,
            batch_size=int(candidate["batch_size"]),
            seq_len=int(candidate["seq_len"]),
            grad_accum=1,
            dtype=dtype_name,
            total_tokens=int(candidate["batch_size"]) * int(candidate["seq_len"]),
            warmup_tokens=4,
            curriculum_tokens=int(candidate["batch_size"]) * int(candidate["seq_len"]),
            compile_model=False,
            data_num_workers=0,
            data_persistent_workers=False,
            data_pin_memory=False,
            device_non_blocking=False,
        )
        model = LLM(cfg).to(device).train()
        opt = build_adamw(
            model,
            OptimizerSettings(lr=1e-3, weight_decay=0.0, use_fused=False),
        )
        amp = build_amp_policy(
            AmpPolicySettings(
                dtype_name=dtype_name,
                device_type="cuda",
                use_grad_scaler=(dtype_name == "float16"),
            )
        )
        ids = torch.randint(
            0,
            cfg.vocab_size,
            (int(candidate["batch_size"]), int(candidate["seq_len"])),
            device=device,
        )
        targets = torch.randint(
            0,
            cfg.vocab_size,
            (int(candidate["batch_size"]), int(candidate["seq_len"])),
            device=device,
        )
        with amp.autocast():
            out = model(ids, targets=targets)
        if not isinstance(out, LossOutput) or not torch.isfinite(out.total_loss.detach()):
            raise TrainingPreflightError("GPU fit attempt produced invalid loss")
        result["forward_passed"] = True
        amp.backward(out.total_loss)
        grad_norm = amp.clip_grad_norm_(opt, model.parameters(), 1.0)
        if not torch.isfinite(torch.tensor(float(grad_norm))):
            raise TrainingPreflightError("GPU fit attempt produced invalid grad norm")
        amp.step(opt)
        amp.update()
        opt.zero_grad()
        torch.cuda.synchronize(device)
        result["backward_optimizer_passed"] = True
        result["loss"] = float(out.total_loss.detach().cpu().item())
        result["grad_norm"] = float(grad_norm)
        result["amp_dtype_used"] = str(amp.dtype).replace("torch.", "")
        result["grad_scaler_used"] = amp.uses_grad_scaler

        save_ckpt(
            model,
            opt,
            step=1,
            loss=result["loss"],
            cfg=validation_cfg,
            master=True,
            tag="_gpu_fit",
        )
        checkpoint_path = os.path.join(attempt_dir, "step_0000001_gpu_fit.pt")
        result["checkpoint_path"] = checkpoint_path
        result["checkpoint_write_passed"] = os.path.isfile(checkpoint_path)
        restored = LLM(cfg).to(device).train()
        restored_opt = build_adamw(
            restored,
            OptimizerSettings(lr=1e-3, weight_decay=0.0, use_fused=False),
        )
        resumed_step = load_ckpt(restored, restored_opt, validation_cfg, device=0)
        result["resumed_step"] = int(resumed_step)
        result["resume_reload_passed"] = resumed_step == 1
        torch.cuda.synchronize(device)
        result["status"] = "pass"
    except (RuntimeError, ValueError, TrainingPreflightError) as exc:
        result["status"] = "fail"
        result["error_type"] = "cuda_oom" if _is_cuda_oom(exc) else "runtime_error"
        result["error"] = str(exc)[:1000]
    finally:
        if torch.cuda.is_available():
            result["memory_allocated_before_bytes"] = memory_before
            result["memory_reserved_before_bytes"] = memory_reserved_before
            result["memory_allocated_after_bytes"] = torch.cuda.memory_allocated(device)
            result["memory_reserved_after_bytes"] = torch.cuda.memory_reserved(device)
            result["memory_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["memory_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            result["peak_allocated_gb"] = round(
                result["memory_peak_allocated_bytes"] / (1024 ** 3),
                4,
            )
            result["peak_reserved_gb"] = round(
                result["memory_peak_reserved_bytes"] / (1024 ** 3),
                4,
            )
            torch.cuda.empty_cache()
        result["runtime_seconds"] = round(time.time() - started_at, 4)

    result["fit_classification"] = _classify_gpu_fit_attempt(result)
    return result


def _default_training_fit_report() -> Dict[str, Any]:
    preflight = run_training_preflight()
    memory_checks = [
        item for item in preflight["checks"] if item.get("name") == "memory_lower_bound"
    ]
    memory_check = memory_checks[0] if memory_checks else None
    return {
        "fit_classification": "does_not_fit"
        if memory_check and memory_check.get("status") == "fail"
        else "partially_unverified",
        "preflight_ready_for_training": bool(preflight["ready_for_training"]),
        "memory_lower_bound_status": (
            memory_check.get("status") if memory_check else "unknown"
        ),
        "memory_lower_bound_detail": (
            memory_check.get("detail") if memory_check else "unknown"
        ),
        "evidence": "train-preflight",
    }


def run_gpu_constrained_fit_validation(
    *,
    checkpoint_root: str = "./run_artifacts/gpu_constrained_fit",
    tokenizer_path: str = "./tokenizer_data",
    token_cache_dir: str = TOKEN_CACHE_DIR,
    matrix_candidates: Optional[List[Dict[str, Any]]] = None,
    run_short_validation: bool = True,
    seed: int = 777,
) -> Dict[str, Any]:
    """Measure bounded GPU fit on constrained hardware without claiming training readiness."""
    started_at = time.time()
    interpreter = _repo_local_python_report()
    nvidia_smi = _nvidia_smi_report()
    cuda_available = bool(torch.cuda.is_available())
    devices = _cuda_device_reports()
    detected_vram_gb = None
    if nvidia_smi.get("gpus"):
        detected_vram_gb = nvidia_smi["gpus"][0].get("total_memory_gb")
    elif devices:
        detected_vram_gb = devices[0].get("total_memory_gb")

    report: Dict[str, Any] = {
        "schema": "gpu_constrained_fit_validation_v1",
        "ok": True,
        "interpreter": interpreter,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": devices,
        "nvidia_smi": nvidia_smi,
        "gpu_name": devices[0]["name"] if devices else None,
        "detected_total_vram_gb": detected_vram_gb,
        "vram_class": "4gb_constrained"
        if detected_vram_gb and detected_vram_gb <= 4.5
        else "unknown_or_larger",
        "preferred_dtype": _preferred_cuda_validation_dtype()
        if cuda_available
        else "unverified_cuda_unavailable",
        "hardware_classification": "validation_only_for_current_repo_target",
        "matrix": [],
        "smallest_fitting_config": None,
        "strongest_fitting_config": None,
        "failed_configs": [],
        "short_validation": None,
        "default_training_fit": None,
        "pretraining_readiness_changed": "no",
        "quality_claim": "none",
        "limitations": [
            "This command tests bounded validation configs only; it is not pretraining.",
            "A fitting reduced config does not prove the intended training config fits.",
            "4GB VRAM should be treated as validation-only for the current repo target.",
        ],
    }

    if not interpreter["is_repo_local_venv"]:
        report["ok"] = False
        report["status"] = "wrong_interpreter"
        report["limitations"].append(
            "Validation must be run with the repo-local .venv Python interpreter."
        )
        return report
    if not cuda_available:
        report["ok"] = False
        report["status"] = "cuda_unavailable"
        report["limitations"].append("PyTorch cannot use CUDA in this interpreter.")
        return report

    tok = BPETokenizer()
    tok.load(tokenizer_path)
    os.makedirs(checkpoint_root, exist_ok=True)
    candidates = matrix_candidates or _gpu_fit_matrix_candidates()
    for index, candidate in enumerate(candidates):
        attempt = _run_gpu_fit_attempt(
            candidate=candidate,
            vocab_size=tok.vocab_size_,
            checkpoint_root=os.path.join(checkpoint_root, "matrix"),
            seed=seed + index,
        )
        report["matrix"].append(attempt)

    fitting = [
        attempt
        for attempt in report["matrix"]
        if attempt.get("fit_classification") == "fits_mechanics_only"
    ]
    if fitting:
        report["smallest_fitting_config"] = fitting[0]["name"]
        report["strongest_fitting_config"] = fitting[-1]["name"]
    report["failed_configs"] = [
        attempt["name"]
        for attempt in report["matrix"]
        if attempt.get("fit_classification") == "does_not_fit"
    ]

    if fitting and run_short_validation:
        strongest = fitting[-1]
        overrides = {
            **strongest["config_override"],
            "max_seq_len": max(int(strongest["seq_len"]), 32),
            "sliding_window": min(16, int(strongest["seq_len"])),
        }
        try:
            torch.cuda.empty_cache()
            short = run_short_real_data_validation(
                token_cache_dir=token_cache_dir,
                tokenizer_path=tokenizer_path,
                checkpoint_dir=os.path.join(checkpoint_root, "short_real_token_validation"),
                seq_len=int(strongest["seq_len"]),
                pre_resume_steps=2,
                post_resume_steps=1,
                seed=seed + 100,
                deterministic=False,
                model_overrides=overrides,
                force_device_type="cuda",
            )
            short["fit_classification"] = "fits_short_validation"
            short["chosen_from_matrix"] = strongest["name"]
            report["short_validation"] = short
        except (RuntimeError, ValueError, OSError, TrainingPreflightError) as exc:
            report["short_validation"] = {
                "ok": False,
                "chosen_from_matrix": strongest["name"],
                "fit_classification": "does_not_fit"
                if _is_cuda_oom(exc)
                else "partially_unverified",
                "error_type": "cuda_oom" if _is_cuda_oom(exc) else "runtime_error",
                "error": str(exc)[:1000],
                "quality_claim": "none",
            }

    report["default_training_fit"] = _default_training_fit_report()
    report["runtime_seconds"] = round(time.time() - started_at, 4)
    if not fitting:
        report["ok"] = False
        report["status"] = "no_bounded_gpu_config_fit"
    else:
        report["status"] = "bounded_gpu_validation_evidence_collected"
    return report


def _run_device_step_validation(
    *,
    device_type: str,
    dtype_name: str,
    seq_len: int = 16,
    seed: int = 444,
) -> Dict[str, Any]:
    from config import EDGE_0_5B_GQA, build_model_config

    started_at = time.time()
    set_training_seed(seed, deterministic=(device_type == "cpu"))
    device = torch.device(device_type)
    cfg = build_model_config(
        EDGE_0_5B_GQA,
        vocab_size=512,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=max(32, seq_len),
        sliding_window=8,
    )
    model = LLM(cfg).to(device).train()
    opt = build_adamw(
        model,
        OptimizerSettings(lr=1e-3, weight_decay=0.0, use_fused=False),
    )
    amp = build_amp_policy(
        AmpPolicySettings(
            dtype_name=dtype_name,
            device_type=device_type,
            use_grad_scaler=(device_type == "cuda" and dtype_name == "float16"),
        )
    )

    memory_before: Optional[int] = None
    memory_after: Optional[int] = None
    memory_peak: Optional[int] = None
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)

    ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
    targets = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
    with amp.autocast():
        out = model(ids, targets=targets)
    if not isinstance(out, LossOutput) or not torch.isfinite(out.total_loss.detach()):
        raise TrainingPreflightError("hardware validation produced invalid loss")
    amp.backward(out.total_loss)
    grad_norm = amp.clip_grad_norm_(opt, model.parameters(), 1.0)
    if not torch.isfinite(torch.tensor(float(grad_norm))):
        raise TrainingPreflightError("hardware validation produced invalid grad norm")
    amp.step(opt)
    amp.update()
    opt.zero_grad()

    if device_type == "cuda":
        memory_after = torch.cuda.memory_allocated(device)
        memory_peak = torch.cuda.max_memory_allocated(device)

    return {
        "status": "pass",
        "device_type": device_type,
        "dtype_requested": dtype_name,
        "amp_dtype_used": str(amp.dtype).replace("torch.", ""),
        "grad_scaler_used": amp.uses_grad_scaler,
        "device_placement": "pass",
        "amp_step": "pass",
        "compile_attempted": False,
        "compile_status": "not_attempted_in_evidence_pass",
        "loss": float(out.total_loss.detach().cpu().item()),
        "grad_norm": float(grad_norm),
        "memory_allocated_before_bytes": memory_before,
        "memory_allocated_after_bytes": memory_after,
        "memory_peak_allocated_bytes": memory_peak,
        "runtime_seconds": round(time.time() - started_at, 4),
        "quality_claim": "none",
    }


def run_hardware_readiness_validation() -> Dict[str, Any]:
    """Collect current-machine hardware evidence without inferring readiness."""
    nvidia_smi = _nvidia_smi_report()
    system_gpu_detected = bool(nvidia_smi.get("gpus"))
    cuda_available = bool(torch.cuda.is_available())
    cuda_version = getattr(torch.version, "cuda", None)
    cudnn_available = bool(torch.backends.cudnn.is_available())
    bf16_supported = False
    if cuda_available:
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        bf16_supported = bool(checker()) if callable(checker) else False
    tensor_ops = _probe_cuda_tensor_ops()
    detected_vram_gb = None
    if nvidia_smi.get("gpus"):
        detected_vram_gb = nvidia_smi["gpus"][0].get("total_memory_gb")

    report: Dict[str, Any] = {
        "schema": "hardware_readiness_v1",
        "ok": True,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": cuda_version,
        "torch_build_has_cuda_runtime": cuda_version is not None,
        "cudnn_available": cudnn_available,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": _cuda_device_reports(),
        "nvidia_smi": nvidia_smi,
        "system_gpu_detected": system_gpu_detected,
        "cuda_bf16_supported": bf16_supported,
        "cuda_fp16_supported": cuda_available,
        "gpu_tensor_ops": tensor_ops,
        "gpu_result_classification": "unknown",
        "hardware_ready_for_training": False,
        "training_readiness_claim": "none",
        "quality_claim": "none",
        "evidence_level": "system_gpu_only_pytorch_cuda_unavailable"
        if system_gpu_detected and not cuda_available
        else ("cpu_only" if not cuda_available else "cuda_single_step"),
        "validation": None,
        "rtx_2050_4gb_realism": {
            "target_gpu_class": "NVIDIA GeForce RTX 2050 laptop GPU",
            "detected_total_vram_gb": detected_vram_gb,
            "vram_class": "4gb_constrained" if detected_vram_gb and detected_vram_gb <= 4.5 else "unknown_or_larger",
            "default_training_path_ready": False,
            "tiny_gpu_mechanics_only": False,
            "fit_claim": "none",
        },
        "limitations": [],
    }

    if not cuda_available:
        if system_gpu_detected:
            report["limitations"].append(
                "NVIDIA hardware is visible to nvidia-smi, but current PyTorch is not CUDA-enabled."
            )
        report["limitations"].append(
            "CUDA is not available in this environment; GPU readiness is unverified."
        )
        report["validation"] = _run_device_step_validation(
            device_type="cpu",
            dtype_name="bfloat16",
        )
        report["gpu_result_classification"] = _classify_gpu_evidence(
            cuda_available=False,
            system_gpu_detected=system_gpu_detected,
            tensor_op_success=False,
            validation=report["validation"],
        )
        return report

    dtype_name = "bfloat16" if bf16_supported else "float16"
    try:
        report["validation"] = _run_device_step_validation(
            device_type="cuda",
            dtype_name=dtype_name,
        )
    except (RuntimeError, ValueError, TrainingPreflightError) as exc:
        error_type = (
            "cuda_oom"
            if isinstance(exc, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(exc).lower()
            else "runtime_error"
        )
        report["ok"] = False
        report["validation"] = {
            "status": "fail",
            "device_type": "cuda",
            "dtype_requested": dtype_name,
            "error_type": error_type,
            "error": str(exc),
        }
        report["gpu_result_classification"] = _classify_gpu_evidence(
            cuda_available=True,
            system_gpu_detected=system_gpu_detected,
            tensor_op_success=bool(tensor_ops.get("cuda_tensor_op_success")),
            validation=report["validation"],
        )
        report["limitations"].append(
            "CUDA exists but the narrow validation step failed."
        )
        return report

    report["hardware_ready_for_training"] = False
    report["rtx_2050_4gb_realism"]["tiny_gpu_mechanics_only"] = True
    report["rtx_2050_4gb_realism"]["fit_claim"] = (
        "tiny validation fit only; production/default training fit is unverified"
    )
    report["gpu_result_classification"] = _classify_gpu_evidence(
        cuda_available=True,
        system_gpu_detected=system_gpu_detected,
        tensor_op_success=bool(tensor_ops.get("cuda_tensor_op_success")),
        validation=report["validation"],
    )
    report["limitations"].append(
        "A single CUDA validation step is not proof that full pretraining fits or is stable."
    )
    if detected_vram_gb and detected_vram_gb <= 4.5:
        report["limitations"].append(
            "4GB VRAM is constrained; passing tiny mechanics does not imply default training readiness."
        )
    return report


def run_short_real_data_validation(
    *,
    token_cache_dir: str = TOKEN_CACHE_DIR,
    tokenizer_path: str = "./tokenizer_data",
    checkpoint_dir: str = "./run_artifacts/short_real_data_validation",
    seq_len: int = 16,
    pre_resume_steps: int = 3,
    post_resume_steps: int = 2,
    seed: int = 321,
    deterministic: bool = True,
    model_overrides: Optional[Dict[str, Any]] = None,
    force_device_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a bounded multi-step real-token validation, not pretraining."""
    import numpy as np
    from eval_suite import inspect_checkpoint_path

    if seq_len < 4:
        raise ValueError("seq_len must be at least 4")
    if pre_resume_steps < 1 or post_resume_steps < 1:
        raise ValueError("pre_resume_steps and post_resume_steps must be >= 1")

    started_at = time.time()
    seed_report = set_training_seed(seed, deterministic=deterministic)
    tok = BPETokenizer()
    tok.load(tokenizer_path)
    dtype = np.uint16 if tok.vocab_size_ <= 65535 else np.uint32
    total_steps = int(pre_resume_steps) + int(post_resume_steps)
    tokens_needed = total_steps * (seq_len + 1)
    artifact_path = _first_token_artifact(
        token_cache_dir,
        dtype,
        min_tokens=tokens_needed,
    )
    raw = np.fromfile(artifact_path, dtype=dtype, count=tokens_needed)
    if raw.size < tokens_needed:
        raise TrainingPreflightError(
            f"token artifact has insufficient tokens for short validation: {artifact_path}"
        )
    max_token = int(raw.max(initial=0))
    if max_token >= tok.vocab_size_:
        raise TrainingPreflightError(
            f"token id {max_token} exceeds tokenizer vocab size {tok.vocab_size_}"
        )

    if force_device_type is not None and force_device_type not in {"cpu", "cuda"}:
        raise ValueError("force_device_type must be 'cpu', 'cuda', or None")
    if force_device_type == "cuda" and not torch.cuda.is_available():
        raise TrainingPreflightError("CUDA was required for validation but is unavailable")
    device_type = force_device_type or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    dtype_name = "bfloat16"
    if device_type == "cuda":
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        dtype_name = "bfloat16" if callable(checker) and checker() else "float16"

    overrides = {
        "dim": 96,
        "n_layers": 3,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_dim": 192,
        "max_seq_len": max(64, seq_len),
        "sliding_window": 8,
    }
    if model_overrides:
        overrides.update(model_overrides)
    tiny_cfg = _validation_model_config(
        vocab_size=tok.vocab_size_,
        seq_len=seq_len,
        overrides=overrides,
    )
    model_config = _model_config_payload(tiny_cfg)
    model_config_hash = hashlib.sha256(
        json.dumps(model_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    validation_cfg = TrainConfig(
        tokenizer_path=tokenizer_path,
        checkpoint_dir=checkpoint_dir,
        batch_size=1,
        seq_len=seq_len,
        grad_accum=1,
        total_tokens=tokens_needed,
        warmup_tokens=4,
        curriculum_tokens=tokens_needed,
        compile_model=False,
        data_num_workers=0,
        data_persistent_workers=False,
        data_pin_memory=False,
        device_non_blocking=False,
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    for name in os.listdir(checkpoint_dir):
        if "_short_validation" in name and (
            name.endswith(".pt") or name.endswith(".pt.sha256")
        ):
            os.remove(os.path.join(checkpoint_dir, name))

    def _build_model_and_opt() -> tuple[LLM, torch.optim.Optimizer, Any]:
        local_model = LLM(tiny_cfg).to(device).train()
        local_opt = build_adamw(
            local_model,
            OptimizerSettings(lr=1e-3, weight_decay=0.0, use_fused=False),
        )
        local_amp = build_amp_policy(
            AmpPolicySettings(
                dtype_name=dtype_name,
                device_type=device_type,
                use_grad_scaler=(device_type == "cuda" and dtype_name == "float16"),
            )
        )
        return local_model, local_opt, local_amp

    def _step(
        local_model: LLM,
        local_opt: torch.optim.Optimizer,
        local_amp: Any,
        step_index: int,
    ) -> Dict[str, Any]:
        start = step_index * (seq_len + 1)
        chunk = raw[start: start + seq_len + 1]
        ids = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        targets = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
        with local_amp.autocast():
            out = local_model(ids, targets=targets)
        if not isinstance(out, LossOutput) or not torch.isfinite(out.total_loss.detach()):
            raise TrainingPreflightError(
                f"short validation produced invalid loss at step {step_index + 1}"
            )
        local_amp.backward(out.total_loss)
        grad_norm = local_amp.clip_grad_norm_(local_opt, local_model.parameters(), 1.0)
        if not torch.isfinite(torch.tensor(float(grad_norm))):
            raise TrainingPreflightError(
                f"short validation produced invalid grad norm at step {step_index + 1}"
            )
        local_amp.step(local_opt)
        local_amp.update()
        local_opt.zero_grad()
        return {
            "step": step_index + 1,
            "loss": float(out.total_loss.detach().cpu().item()),
            "grad_norm": float(grad_norm),
        }

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    memory_before = (
        torch.cuda.memory_allocated(device) if device_type == "cuda" else None
    )

    model, opt, amp = _build_model_and_opt()
    step_reports: List[Dict[str, Any]] = []
    for step_index in range(pre_resume_steps):
        step_reports.append(_step(model, opt, amp, step_index))

    save_ckpt(
        model,
        opt,
        step=pre_resume_steps,
        loss=step_reports[-1]["loss"],
        cfg=validation_cfg,
        master=True,
        tag="_short_validation",
    )
    checkpoint_before_resume = os.path.join(
        checkpoint_dir,
        f"step_{pre_resume_steps:07d}_short_validation.pt",
    )

    restored, restored_opt, restored_amp = _build_model_and_opt()
    resumed_step = load_ckpt(restored, restored_opt, validation_cfg, device=0)
    if resumed_step != pre_resume_steps:
        raise TrainingPreflightError(
            f"short validation resume mismatch: expected {pre_resume_steps}, got {resumed_step}"
        )

    for step_index in range(pre_resume_steps, total_steps):
        step_reports.append(_step(restored, restored_opt, restored_amp, step_index))

    save_ckpt(
        restored,
        restored_opt,
        step=total_steps,
        loss=step_reports[-1]["loss"],
        cfg=validation_cfg,
        master=True,
        tag="_short_validation_final",
    )
    final_checkpoint = os.path.join(
        checkpoint_dir,
        f"step_{total_steps:07d}_short_validation_final.pt",
    )

    produced_meta = inspect_checkpoint_path(checkpoint_dir)
    if produced_meta.get("selected_checkpoint_exists") is not True:
        raise TrainingPreflightError(
            "eval checkpoint inspection did not find the short-validation checkpoint"
        )
    missing_meta = inspect_checkpoint_path(
        os.path.join(checkpoint_dir, "missing_eval_checkpoint")
    )
    if missing_meta.get("load_status") != "missing":
        raise TrainingPreflightError(
            "eval missing-checkpoint gate did not report missing"
        )

    memory_after = (
        torch.cuda.memory_allocated(device) if device_type == "cuda" else None
    )
    memory_peak = (
        torch.cuda.max_memory_allocated(device) if device_type == "cuda" else None
    )

    return {
        "schema": "short_real_data_validation_v1",
        "ok": True,
        "run_classification": "bounded_validation_not_pretraining",
        "quality_claim": "none",
        "capability_evidence": "none",
        "artifact_path": artifact_path,
        "tokens_read": int(raw.size),
        "token_dtype": np.dtype(dtype).name,
        "tokenizer_vocab_size": tok.vocab_size_,
        "pre_resume_steps": int(pre_resume_steps),
        "post_resume_steps": int(post_resume_steps),
        "total_optimizer_steps": int(total_steps),
        "seq_len": int(seq_len),
        "device": device_type,
        "dtype_requested": dtype_name,
        "amp_dtype_used": str(amp.dtype).replace("torch.", ""),
        "grad_scaler_used": amp.uses_grad_scaler,
        "compile_attempted": False,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_before_resume": checkpoint_before_resume,
        "final_checkpoint": final_checkpoint,
        "checkpoint_exists": os.path.isfile(final_checkpoint),
        "resumed_step": resumed_step,
        "resume_success": resumed_step == pre_resume_steps,
        "checkpoint_inspection": produced_meta,
        "eval_gate_on_produced_checkpoint": (
            "checkpoint_found"
            if produced_meta.get("selected_checkpoint_exists")
            else "checkpoint_missing"
        ),
        "eval_missing_checkpoint_gate": missing_meta.get("load_status"),
        "step_reports": step_reports,
        "model_config_hash": model_config_hash,
        "model_config": model_config,
        "seed_report": seed_report,
        "cuda_available": bool(torch.cuda.is_available()),
        "memory_allocated_before_bytes": memory_before,
        "memory_allocated_after_bytes": memory_after,
        "memory_peak_allocated_bytes": memory_peak,
        "runtime_seconds": round(time.time() - started_at, 4),
        "limitations": [
            "This is a bounded validation run, not pretraining.",
            "The checkpoint is not evidence of useful model quality.",
            "The tiny validation model is intentionally smaller than production profiles.",
        ],
    }


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
    from data import get_dataloader, get_val_dataloader, DataPipelineError

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

    set_training_seed(42, rank)

    if master:
        show_config()

    # --- Tokenizer ---
    tok = BPETokenizer()
    tok_path = os.path.join(train_cfg.tokenizer_path, "encoder.json")
    if not os.path.exists(tok_path):
        if master:
            LOGGER.error("Tokenizer not found. Run: python train_tokenizer.py")
        cleanup()
        raise TrainingPreflightError(
            f"Tokenizer not found at {tok_path}. Run: python train_tokenizer.py"
        )
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
                            "Non-finite loss at step %s; aborting training.",
                            step,
                        )
                    opt.zero_grad()
                    raise FloatingPointError(
                        f"Non-finite loss at step {step}; training aborted."
                    )

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
                        "Non-finite loss at step %s before optimizer step; aborting training.",
                        step,
                    )
                opt.zero_grad()
                raise FloatingPointError(
                    f"Non-finite loss before optimizer step at step {step}; training aborted."
                )

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
