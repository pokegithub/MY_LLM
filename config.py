"""Model and training configuration."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from core.logging import get_logger

LOGGER = get_logger("config")


@dataclass
class ModelConfig:
    vocab_size              : int   = 50000
    pad_token_id            : int   = 0
    bos_token_id            : int   = 1
    eos_token_id            : int   = 2
    dim                     : int   = 1024
    n_layers                : int   = 16
    n_heads                 : int   = 16
    n_kv_heads              : int   = 4
    max_seq_len             : int   = 2048
    ffn_dim                 : int   = 2816
    use_moe                 : bool  = True
    n_experts               : int   = 8
    n_experts_active        : int   = 2
    moe_freq                : int   = 2
    n_memory_tokens         : int   = 64
    n_loops                 : int   = 2
    yarn_scale              : float = 4.0
    sliding_window          : int   = 512
    adaptive_loop_threshold : float = 0.4
    max_refinement_loops    : int   = 1
    use_confidence_head     : bool  = True
    confidence_dim          : int   = 3
    use_residual_gates      : bool  = True
    norm_eps                : float = 1e-6
    qk_norm_eps             : float = 1e-6
    rope_theta              : float = 500000.0
    attention_backend       : str   = "auto"
    use_flashattention      : bool  = False
    use_mla                 : bool  = False
    mla_compression_ratio   : float = 0.5
    head_dim                : int   = field(init=False)
    latent_head_dim         : int   = field(init=False)

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, (
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by "
            f"n_kv_heads ({self.n_kv_heads})"
        )
        backend = str(self.attention_backend).strip().lower()
        if backend not in {"sdpa", "flash2", "auto"}:
            raise ValueError(
                "attention_backend must be one of: sdpa, flash2, auto"
            )

        ratio = float(self.mla_compression_ratio)
        if not (abs(ratio - 0.25) < 1e-8 or abs(ratio - 0.5) < 1e-8):
            raise ValueError(
                "mla_compression_ratio must be 0.25 or 0.5"
            )
        if float(self.qk_norm_eps) <= 0:
            raise ValueError("qk_norm_eps must be > 0")

        self.attention_backend = backend
        self.mla_compression_ratio = ratio
        self.qk_norm_eps = float(self.qk_norm_eps)
        self.head_dim = self.dim // self.n_heads
        self.latent_head_dim = max(
            1, int(self.head_dim * self.mla_compression_ratio)
        )


@dataclass
class TrainConfig:
    # --- Paths ---
    tokenizer_path    : str   = "./tokenizer_data"
    checkpoint_dir    : str   = "./checkpoints"
    val_data_dir      : str   = "./data_cache/val_tokens"

    # --- Core training ---
    vocab_size        : int   = 50000
    total_tokens      : int   = 10_000_000_000
    batch_size        : int   = 64
    seq_len           : int   = 512
    grad_accum        : int   = 4
    max_lr            : float = 3e-4
    min_lr            : float = 3e-5
    warmup_tokens     : int   = 200_000_000
    weight_decay      : float = 0.1
    grad_clip         : float = 1.0
    beta1             : float = 0.9
    beta2             : float = 0.95
    eps               : float = 1e-8
    dtype             : str   = "bfloat16"
    amp_grad_scaler   : bool  = True
    amp_init_scale    : float = 65536.0
    amp_growth_interval: int  = 2000
    compile_model     : bool  = True

    # --- Runtime / performance ---
    data_num_workers     : int   = 0
    data_prefetch_queue  : int   = 8
    data_prefetch_factor : int   = 2
    data_persistent_workers: bool = True
    data_pin_memory      : bool  = True
    device_non_blocking  : bool  = True

    # --- Logging & checkpointing ---
    checkpoint_every  : int   = 25
    log_every         : int   = 10
    val_every         : int   = 100
    val_steps         : int   = 50

    # --- Data ---
    max_open_sources  : int   = 8
    curriculum_tokens : int   = 500_000_000

    # --- Loss coefficients ---
    aux_loss_coeff    : float = 0.01
    gate_loss_coeff   : float = 0.01
    z_loss_coeff      : float = 0.001
    unc_loss_coeff    : float = 0.1

    # --- Data quality ---
    dedup_ngram_size  : int   = 13
    dedup_threshold   : float = 0.8
    min_doc_length    : int   = 50
    max_doc_length    : int   = 100000
    quality_threshold : float = 0.3

    # --- Data integrity ---
    # Exclude common benchmark datasets from pretrain/tokenizer corpora
    # to reduce train/eval contamination risk.
    excluded_benchmark_sources: Tuple[str, ...] = (
        "truthful_qa",
        "google/boolq",
        "allenai/ai2_arc",
        "winogrande",
        "openai/gsm8k",
    )


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning."""
    checkpoint_path   : str   = "./checkpoints"
    output_dir        : str   = "./sft_checkpoints"
    tokenizer_path    : str   = "./tokenizer_data"
    data_path         : str   = "./sft_data"
    batch_size        : int   = 16
    seq_len           : int   = 1024
    grad_accum        : int   = 4
    max_lr            : float = 2e-5
    min_lr            : float = 2e-6
    warmup_steps      : int   = 100
    total_steps       : int   = 5000
    weight_decay      : float = 0.05
    grad_clip         : float = 1.0
    dtype             : str   = "bfloat16"
    amp_grad_scaler   : bool  = True
    amp_init_scale    : float = 65536.0
    amp_growth_interval: int  = 2000
    log_every         : int   = 10
    save_every        : int   = 500
    val_every         : int   = 100
    val_steps         : int   = 25
    packing           : bool  = True
    max_packing_len   : int   = 1024
    data_num_workers  : int   = 0
    data_prefetch_factor: int = 2
    data_persistent_workers: bool = True
    data_pin_memory   : bool  = True
    device_non_blocking: bool = True


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""
    checkpoint_path   : str   = "./sft_checkpoints"
    ref_checkpoint    : str   = "./sft_checkpoints"
    output_dir        : str   = "./dpo_checkpoints"
    tokenizer_path    : str   = "./tokenizer_data"
    data_path         : str   = "./dpo_data"
    batch_size        : int   = 8
    seq_len           : int   = 1024
    grad_accum        : int   = 8
    max_lr            : float = 5e-6
    min_lr            : float = 5e-7
    warmup_steps      : int   = 50
    total_steps       : int   = 2000
    weight_decay      : float = 0.05
    grad_clip         : float = 1.0
    dtype             : str   = "bfloat16"
    amp_grad_scaler   : bool  = True
    amp_init_scale    : float = 65536.0
    amp_growth_interval: int  = 2000
    beta              : float = 0.1
    label_smoothing   : float = 0.0
    log_every         : int   = 5
    save_every        : int   = 200
    loss_type         : str   = "sigmoid"  # "sigmoid" or "hinge"
    data_num_workers  : int   = 0
    data_prefetch_factor: int = 2
    data_persistent_workers: bool = True
    data_pin_memory   : bool  = True
    device_non_blocking: bool = True


@dataclass
class ImproverConfig:
    """Configuration for self-improvement loop."""
    model_checkpoint  : str   = "./dpo_checkpoints"
    tokenizer_path    : str   = "./tokenizer_data"
    output_dir        : str   = "./improved_checkpoints"
    synth_data_dir    : str   = "./synth_data"
    n_iterations      : int   = 3
    samples_per_iter  : int   = 5000
    unc_threshold     : float = 0.5
    temperature       : float = 0.8
    top_k             : int   = 50
    top_p             : float = 0.9
    max_gen_len       : int   = 512
    n_candidates      : int   = 4
    sft_steps_per_iter: int   = 1000
    dpo_steps_per_iter: int   = 500
    diversity_penalty : float = 0.1


@dataclass
class EvalConfig:
    """Configuration for evaluation suite."""
    checkpoint_path   : str   = "./dpo_checkpoints"
    tokenizer_path    : str   = "./tokenizer_data"
    output_dir        : str   = "./eval_results"
    batch_size        : int   = 16
    max_gen_len       : int   = 512
    temperature       : float = 0.0
    n_few_shot        : int   = 5
    code_timeout      : int   = 10
    strict_real_benchmarks: bool = True
    allow_toy_fallback: bool = False
    min_real_mc_samples: int = 100
    min_real_math_samples: int = 100
    min_real_code_samples: int = 20


@dataclass
class DistillConfig:
    """Configuration for teacher-student distillation."""
    teacher_checkpoint : str   = "./dpo_checkpoints"
    student_checkpoint : str   = "./checkpoints"
    tokenizer_path     : str   = "./tokenizer_data"
    output_dir         : str   = "./distill_checkpoints"
    data_dir           : str   = "./data_cache/tokens"
    val_data_dir       : str   = "./data_cache/val_tokens"
    batch_size         : int   = 16
    seq_len            : int   = 512
    grad_accum         : int   = 4
    total_steps        : int   = 2000
    warmup_steps       : int   = 100
    max_lr             : float = 1e-4
    min_lr             : float = 1e-5
    weight_decay       : float = 0.05
    grad_clip          : float = 1.0
    dtype              : str   = "bfloat16"
    amp_grad_scaler    : bool  = True
    amp_init_scale     : float = 65536.0
    amp_growth_interval: int   = 2000
    temperature        : float = 2.0
    alpha_kd           : float = 0.8
    alpha_ce           : float = 0.2
    log_every          : int   = 10
    save_every         : int   = 200
    data_num_workers   : int   = 0
    data_prefetch_queue: int   = 8
    data_prefetch_factor: int  = 2
    data_persistent_workers: bool = True
    data_pin_memory    : bool  = True
    device_non_blocking: bool  = True
    aux_loss_coeff     : float = 0.01
    gate_loss_coeff    : float = 0.01
    z_loss_coeff       : float = 0.001


@dataclass
class QuantConfig:
    """Configuration for post-training quantization/export."""
    checkpoint_path     : str   = "./dpo_checkpoints"
    output_dir          : str   = "./quantized"
    tokenizer_path      : str   = "./tokenizer_data"
    quant_mode          : str   = "dynamic-int8"
    export_torchscript  : bool  = True
    export_onnx         : bool  = False
    max_seq_len         : int   = 512
    calibration_samples : int   = 128


@dataclass
class HardwareProfileConfig:
    """Configuration for hardware profile selection."""
    profile_dir   : str = "./hardware_configs"
    active_profile: str = "auto"


model_cfg = ModelConfig()
train_cfg = TrainConfig()
sft_cfg = SFTConfig()
dpo_cfg = DPOConfig()
improver_cfg = ImproverConfig()
eval_cfg = EvalConfig()
distill_cfg = DistillConfig()
quant_cfg = QuantConfig()
hardware_profile_cfg = HardwareProfileConfig()


def _config_targets() -> Dict[str, Any]:
    return {
        "model": model_cfg,
        "train": train_cfg,
        "sft": sft_cfg,
        "dpo": dpo_cfg,
        "improver": improver_cfg,
        "eval": eval_cfg,
        "distill": distill_cfg,
        "quant": quant_cfg,
        "hardware_profile": hardware_profile_cfg,
        # Backward-compatible aliases.
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "sft_cfg": sft_cfg,
        "dpo_cfg": dpo_cfg,
        "improver_cfg": improver_cfg,
        "eval_cfg": eval_cfg,
        "distill_cfg": distill_cfg,
        "quant_cfg": quant_cfg,
        "hardware_profile_cfg": hardware_profile_cfg,
    }


def _restore_runtime_state(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    targets = _config_targets()
    for section, values in snapshot.items():
        target = targets.get(section)
        if target is None or not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            if hasattr(target, key):
                setattr(target, key, value)
    for section_name, target in _config_targets().items():
        if not section_name.endswith("_cfg") and hasattr(target, "__post_init__"):
            target.__post_init__()


def validate_runtime_state() -> None:
    """Validate current runtime configuration using strict section schemas."""
    from core.config_schema import validate_runtime_sections

    validate_runtime_sections(runtime_config_dict())


def apply_overrides(
    overrides: Mapping[str, Any],
    strict: bool = False,
) -> Dict[str, Any]:
    """Apply layered runtime overrides to global config objects.

    Expected shape:
      {
        "train": {"batch_size": 32},
        "model": {"max_seq_len": 4096}
      }
    """
    if not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping")

    snapshot = runtime_config_dict()
    targets = _config_targets()
    unknown_sections = []
    unknown_fields = []
    applied = 0

    try:
        for raw_section, values in overrides.items():
            section = str(raw_section).strip().lower().replace("-", "_")
            target = targets.get(section)
            if target is None:
                unknown_sections.append(section)
                continue

            if not isinstance(values, Mapping):
                unknown_fields.append(f"{section} (non-mapping override)")
                continue

            for key, value in values.items():
                attr = str(key).strip()
                if hasattr(target, attr):
                    setattr(target, attr, value)
                    applied += 1
                else:
                    unknown_fields.append(f"{section}.{attr}")

        # Re-run post-init hooks where present (e.g., recompute derived fields).
        for section_name, target in _config_targets().items():
            if not section_name.endswith("_cfg") and hasattr(target, "__post_init__"):
                target.__post_init__()

        if strict and (unknown_sections or unknown_fields):
            raise KeyError(
                "Unknown config overrides: "
                + ", ".join(unknown_sections + unknown_fields)
            )

        validate_runtime_state()
    except Exception:
        _restore_runtime_state(snapshot)
        raise

    return {
        "applied": applied,
        "unknown_sections": sorted(set(unknown_sections)),
        "unknown_fields": sorted(set(unknown_fields)),
    }


def runtime_config_dict() -> Dict[str, Dict[str, Any]]:
    """Return current runtime config state as plain dictionaries."""
    return {
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
        "sft": asdict(sft_cfg),
        "dpo": asdict(dpo_cfg),
        "improver": asdict(improver_cfg),
        "eval": asdict(eval_cfg),
        "distill": asdict(distill_cfg),
        "quant": asdict(quant_cfg),
        "hardware_profile": asdict(hardware_profile_cfg),
    }


validate_runtime_state()


def show_config():
    """Print configs. Called explicitly, never at import time."""
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("Configuration")
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("Architecture      : %sd x %sL x %sH", model_cfg.dim, model_cfg.n_layers, model_cfg.n_heads)
    LOGGER.info("Vocab size        : %s", f"{model_cfg.vocab_size:,}")
    LOGGER.info("Memory tokens     : %s", model_cfg.n_memory_tokens)
    LOGGER.info("Recursive loops   : %s", model_cfg.n_loops)
    LOGGER.info("Max refinement    : %s", model_cfg.max_refinement_loops)
    LOGGER.info(
        "MoE               : %sE / %sA every %s layers",
        model_cfg.n_experts,
        model_cfg.n_experts_active,
        model_cfg.moe_freq,
    )
    LOGGER.info("Residual gates    : %s", model_cfg.use_residual_gates)
    LOGGER.info(
        "YaRN scale        : %sx (%s)",
        model_cfg.yarn_scale,
        int(model_cfg.max_seq_len * model_cfg.yarn_scale),
    )
    LOGGER.info("Sliding window    : %s", model_cfg.sliding_window)
    LOGGER.info(
        "Attn backend      : %s (flash=%s)",
        model_cfg.attention_backend,
        model_cfg.use_flashattention,
    )
    LOGGER.info(
        "MLA               : %s (ratio=%s, latent_hd=%s)",
        model_cfg.use_mla,
        model_cfg.mla_compression_ratio,
        model_cfg.latent_head_dim,
    )
    LOGGER.info("QK-Norm eps       : %s", model_cfg.qk_norm_eps)
    LOGGER.info(
        "Confidence head   : %s (dim=%s)",
        model_cfg.use_confidence_head,
        model_cfg.confidence_dim,
    )
    tps = train_cfg.batch_size * train_cfg.seq_len * train_cfg.grad_accum
    LOGGER.info("Tokens/step       : %s", f"{tps:,}")
    LOGGER.info("Total tokens      : %.1fB", train_cfg.total_tokens / 1e9)
    LOGGER.info("Curriculum tokens : %.0fM", train_cfg.curriculum_tokens / 1e6)
    LOGGER.info("Max LR            : %s", train_cfg.max_lr)
    LOGGER.info("torch.compile     : %s", train_cfg.compile_model)
    LOGGER.info(
        "Data loader       : workers=%s prefetch_q=%s prefetch_factor=%s pin_memory=%s",
        train_cfg.data_num_workers,
        train_cfg.data_prefetch_queue,
        train_cfg.data_prefetch_factor,
        train_cfg.data_pin_memory,
    )
    LOGGER.info(
        "Loss coeffs       : aux=%s gate=%s z=%s unc=%s",
        train_cfg.aux_loss_coeff,
        train_cfg.gate_loss_coeff,
        train_cfg.z_loss_coeff,
        train_cfg.unc_loss_coeff,
    )
    LOGGER.info("%s", "=" * 60)


if __name__ == "__main__":
    show_config()