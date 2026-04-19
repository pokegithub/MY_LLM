"""Pydantic schemas for runtime configuration validation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSection(_StrictModel):
    vocab_size: int = Field(ge=128)
    pad_token_id: int = Field(ge=0)
    bos_token_id: int = Field(ge=0)
    eos_token_id: int = Field(ge=0)
    dim: int = Field(ge=64)
    n_layers: int = Field(ge=1)
    n_heads: int = Field(ge=1)
    n_kv_heads: int = Field(ge=1)
    max_seq_len: int = Field(ge=64)
    ffn_dim: int = Field(ge=128)
    use_moe: bool
    n_experts: int = Field(ge=1)
    n_experts_active: int = Field(ge=1)
    moe_freq: int = Field(ge=1)
    n_memory_tokens: int = Field(ge=0)
    n_loops: int = Field(ge=1)
    yarn_scale: float = Field(gt=0)
    sliding_window: int = Field(ge=1)
    adaptive_loop_threshold: float = Field(ge=0.0, le=1.0)
    max_refinement_loops: int = Field(ge=0)
    use_confidence_head: bool
    confidence_dim: int = Field(ge=1)
    use_residual_gates: bool
    norm_eps: float = Field(gt=0)
    qk_norm_eps: float = Field(
        gt=0,
        description="RMS epsilon used for QK normalization before attention logits.",
    )
    rope_theta: float = Field(gt=0)
    attention_backend: str = Field(
        description="Attention backend policy: sdpa, flash2, or auto."
    )
    use_flashattention: bool = Field(
        description="Whether flash attention is allowed when backend policy permits."
    )
    use_mla: bool = Field(
        description="Enable latent KV compression (MLA-style) path in attention."
    )
    mla_compression_ratio: float = Field(
        gt=0,
        le=1,
        description="Latent compression ratio d'/d; constrained to 0.25 or 0.5.",
    )
    head_dim: int = Field(
        ge=1,
        description="Per-head attention dimension (derived from dim / n_heads).",
    )
    latent_head_dim: int = Field(
        ge=1,
        description="Per-head latent KV dimension derived from head_dim * ratio.",
    )

    @model_validator(mode="after")
    def validate_head_relations(self):
        if self.dim % self.n_heads != 0:
            raise ValueError("model.dim must be divisible by model.n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("model.n_heads must be divisible by model.n_kv_heads")
        if self.n_experts_active > self.n_experts:
            raise ValueError("model.n_experts_active cannot exceed model.n_experts")
        if self.attention_backend not in {"sdpa", "flash2", "auto"}:
            raise ValueError(
                "model.attention_backend must be one of: sdpa, flash2, auto"
            )

        ratio = float(self.mla_compression_ratio)
        if not (abs(ratio - 0.25) < 1e-8 or abs(ratio - 0.5) < 1e-8):
            raise ValueError(
                "model.mla_compression_ratio must be 0.25 or 0.5"
            )

        expected_latent = max(1, int(self.head_dim * ratio))
        if self.latent_head_dim != expected_latent:
            raise ValueError(
                "model.latent_head_dim must equal "
                "int(model.head_dim * model.mla_compression_ratio)"
            )
        return self


class TrainSection(_StrictModel):
    tokenizer_path: str
    checkpoint_dir: str
    val_data_dir: str
    vocab_size: int = Field(ge=128)
    total_tokens: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    seq_len: int = Field(ge=8)
    grad_accum: int = Field(ge=1)
    max_lr: float = Field(gt=0)
    min_lr: float = Field(gt=0)
    warmup_tokens: int = Field(ge=0)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(gt=0)
    beta1: float = Field(gt=0, lt=1)
    beta2: float = Field(gt=0, lt=1)
    eps: float = Field(gt=0)
    dtype: str
    amp_grad_scaler: bool
    amp_init_scale: float = Field(gt=0)
    amp_growth_interval: int = Field(ge=1)
    compile_model: bool
    data_num_workers: int = Field(ge=0)
    data_prefetch_queue: int = Field(ge=1)
    data_prefetch_factor: int = Field(ge=1)
    data_persistent_workers: bool
    data_pin_memory: bool
    device_non_blocking: bool
    checkpoint_every: int = Field(ge=1)
    log_every: int = Field(ge=1)
    val_every: int = Field(ge=1)
    val_steps: int = Field(ge=1)
    max_open_sources: int = Field(ge=1)
    curriculum_tokens: int = Field(ge=0)
    aux_loss_coeff: float = Field(ge=0)
    gate_loss_coeff: float = Field(
        ge=0,
        description="Coefficient for MoE gate/load-balancing loss.",
    )
    z_loss_coeff: float = Field(ge=0)
    unc_loss_coeff: float = Field(ge=0)
    dedup_ngram_size: int = Field(ge=2)
    dedup_threshold: float = Field(ge=0.0, le=1.0)
    min_doc_length: int = Field(ge=1)
    max_doc_length: int = Field(ge=1)
    quality_threshold: float = Field(ge=0.0, le=1.0)
    excluded_benchmark_sources: Tuple[str, ...]

    @model_validator(mode="after")
    def validate_train_schedule(self):
        if self.min_lr > self.max_lr:
            raise ValueError("train.min_lr cannot be greater than train.max_lr")
        if self.warmup_tokens > self.total_tokens:
            raise ValueError("train.warmup_tokens cannot exceed train.total_tokens")
        if self.max_doc_length < self.min_doc_length:
            raise ValueError("train.max_doc_length must be >= train.min_doc_length")
        return self


class SFTSection(_StrictModel):
    checkpoint_path: str
    output_dir: str
    tokenizer_path: str
    data_path: str
    batch_size: int = Field(ge=1)
    seq_len: int = Field(ge=8)
    grad_accum: int = Field(ge=1)
    max_lr: float = Field(gt=0)
    min_lr: float = Field(gt=0)
    warmup_steps: int = Field(ge=0)
    total_steps: int = Field(ge=1)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(gt=0)
    dtype: str
    amp_grad_scaler: bool
    amp_init_scale: float = Field(gt=0)
    amp_growth_interval: int = Field(ge=1)
    log_every: int = Field(ge=1)
    save_every: int = Field(ge=1)
    val_every: int = Field(ge=1)
    val_steps: int = Field(ge=1)
    packing: bool
    max_packing_len: int = Field(ge=8)
    data_num_workers: int = Field(ge=0)
    data_prefetch_factor: int = Field(ge=1)
    data_persistent_workers: bool
    data_pin_memory: bool
    device_non_blocking: bool


class DPOSection(_StrictModel):
    checkpoint_path: str
    ref_checkpoint: str
    output_dir: str
    tokenizer_path: str
    data_path: str
    batch_size: int = Field(ge=1)
    seq_len: int = Field(ge=8)
    grad_accum: int = Field(ge=1)
    max_lr: float = Field(gt=0)
    min_lr: float = Field(gt=0)
    warmup_steps: int = Field(ge=0)
    total_steps: int = Field(ge=1)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(gt=0)
    dtype: str
    amp_grad_scaler: bool
    amp_init_scale: float = Field(gt=0)
    amp_growth_interval: int = Field(ge=1)
    beta: float = Field(gt=0)
    label_smoothing: float = Field(ge=0.0, le=1.0)
    log_every: int = Field(ge=1)
    save_every: int = Field(ge=1)
    loss_type: str
    data_num_workers: int = Field(ge=0)
    data_prefetch_factor: int = Field(ge=1)
    data_persistent_workers: bool
    data_pin_memory: bool
    device_non_blocking: bool

    @model_validator(mode="after")
    def validate_loss_type(self):
        if self.loss_type not in {"sigmoid", "hinge"}:
            raise ValueError("dpo.loss_type must be one of: sigmoid, hinge")
        if self.min_lr > self.max_lr:
            raise ValueError("dpo.min_lr cannot be greater than dpo.max_lr")
        return self


class ImproverSection(_StrictModel):
    model_checkpoint: str
    tokenizer_path: str
    output_dir: str
    synth_data_dir: str
    n_iterations: int = Field(ge=1)
    samples_per_iter: int = Field(ge=1)
    unc_threshold: float = Field(ge=0.0, le=1.0)
    temperature: float = Field(gt=0)
    top_k: int = Field(ge=0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_gen_len: int = Field(ge=1)
    n_candidates: int = Field(ge=2)
    sft_steps_per_iter: int = Field(ge=1)
    dpo_steps_per_iter: int = Field(ge=1)
    diversity_penalty: float = Field(ge=0)


class EvalSection(_StrictModel):
    checkpoint_path: str
    tokenizer_path: str
    output_dir: str
    batch_size: int = Field(ge=1)
    max_gen_len: int = Field(ge=1)
    temperature: float = Field(ge=0)
    n_few_shot: int = Field(ge=0)
    code_timeout: int = Field(ge=1)
    strict_real_benchmarks: bool
    allow_toy_fallback: bool
    min_real_mc_samples: int = Field(ge=1)
    min_real_math_samples: int = Field(ge=1)
    min_real_code_samples: int = Field(ge=1)


class DistillSection(_StrictModel):
    teacher_checkpoint: str
    student_checkpoint: str
    tokenizer_path: str
    output_dir: str
    data_dir: str
    val_data_dir: str
    batch_size: int = Field(ge=1)
    seq_len: int = Field(ge=8)
    grad_accum: int = Field(ge=1)
    total_steps: int = Field(ge=1)
    warmup_steps: int = Field(ge=0)
    max_lr: float = Field(gt=0)
    min_lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(gt=0)
    dtype: str
    amp_grad_scaler: bool
    amp_init_scale: float = Field(gt=0)
    amp_growth_interval: int = Field(ge=1)
    temperature: float = Field(gt=0)
    alpha_kd: float = Field(ge=0)
    alpha_ce: float = Field(ge=0)
    log_every: int = Field(ge=1)
    save_every: int = Field(ge=1)
    data_num_workers: int = Field(ge=0)
    data_prefetch_queue: int = Field(ge=1)
    data_prefetch_factor: int = Field(ge=1)
    data_persistent_workers: bool
    data_pin_memory: bool
    device_non_blocking: bool
    aux_loss_coeff: float = Field(ge=0)
    gate_loss_coeff: float = Field(ge=0)
    z_loss_coeff: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_alpha_mix(self):
        if self.alpha_kd + self.alpha_ce <= 0:
            raise ValueError("distill.alpha_kd + distill.alpha_ce must be > 0")
        if self.min_lr > self.max_lr:
            raise ValueError("distill.min_lr cannot be greater than distill.max_lr")
        return self


class QuantSection(_StrictModel):
    checkpoint_path: str
    output_dir: str
    tokenizer_path: str
    quant_mode: str
    export_torchscript: bool
    export_onnx: bool
    max_seq_len: int = Field(ge=1)
    calibration_samples: int = Field(ge=1)


class HardwareProfileSection(_StrictModel):
    profile_dir: str
    active_profile: str


SECTION_MODELS = {
    "model": ModelSection,
    "train": TrainSection,
    "sft": SFTSection,
    "dpo": DPOSection,
    "improver": ImproverSection,
    "eval": EvalSection,
    "distill": DistillSection,
    "quant": QuantSection,
    "hardware_profile": HardwareProfileSection,
}


def validate_runtime_sections(runtime_state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Validate full runtime state and return normalized data per section."""
    normalized: Dict[str, Dict[str, Any]] = {}
    errors = []

    for section, model in SECTION_MODELS.items():
        if section not in runtime_state:
            errors.append(f"Missing section: {section}")
            continue
        try:
            normalized[section] = model.model_validate(
                runtime_state[section]
            ).model_dump()
        except ValidationError as exc:
            errors.append(f"Section '{section}' invalid: {exc}")

    if errors:
        raise ValueError(" | ".join(errors))

    return normalized
