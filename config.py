"""Model and training configuration."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from core.logging import get_logger

LOGGER = get_logger("config")

EDGE_0_5B_GQA = "edge_0_5b_gqa"
LOCAL_1_5B_GQA = "local_1_5b_gqa"
WORKSTATION_3B_GQA = "workstation_3b_gqa"
LEGACY_EXPERIMENTAL_439M = "legacy_experimental_439m"
DEPLOY_2GB_EDGE = "2gb_edge"
DEPLOY_8GB_LAPTOP = "8gb_laptop"
DEPLOY_16GB_WORKSTATION = "16gb_workstation"
MODEL_PROFILE_NAMES = (
    EDGE_0_5B_GQA,
    LOCAL_1_5B_GQA,
    WORKSTATION_3B_GQA,
    LEGACY_EXPERIMENTAL_439M,
)
DEPLOYMENT_TIER_NAMES = (
    DEPLOY_2GB_EDGE,
    DEPLOY_8GB_LAPTOP,
    DEPLOY_16GB_WORKSTATION,
)


@dataclass
class ModelConfig:
    profile_name            : str   = EDGE_0_5B_GQA
    vocab_size              : int   = 50000
    pad_token_id            : int   = 0
    bos_token_id            : int   = 1
    eos_token_id            : int   = 2
    dim                     : int   = 1536
    n_layers                : int   = 18
    n_heads                 : int   = 16
    n_kv_heads              : int   = 4
    max_seq_len             : int   = 2048
    ffn_dim                 : int   = 4096
    use_moe                 : bool  = False
    n_experts               : int   = 8
    n_experts_active        : int   = 2
    moe_freq                : int   = 2
    n_memory_tokens         : int   = 0
    n_loops                 : int   = 1
    yarn_scale              : float = 1.0
    sliding_window          : int   = 512
    adaptive_loop_threshold : float = 0.4
    max_refinement_loops    : int   = 0
    use_confidence_head     : bool  = False
    confidence_dim          : int   = 3
    use_residual_gates      : bool  = False
    use_mod_routing         : bool  = False
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
        if self.profile_name not in MODEL_PROFILE_NAMES:
            raise ValueError(
                "profile_name must be one of: "
                + ", ".join(MODEL_PROFILE_NAMES)
            )
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
        experimental = self.enabled_experimental_features()
        if experimental and not self.is_legacy_experimental:
            raise ValueError(
                "Experimental mechanisms require profile_name="
                f"{LEGACY_EXPERIMENTAL_439M}: {', '.join(experimental)}"
            )

    @property
    def is_legacy_experimental(self) -> bool:
        return self.profile_name == LEGACY_EXPERIMENTAL_439M

    def enabled_experimental_features(self) -> Tuple[str, ...]:
        enabled = []
        if self.use_moe:
            enabled.append("use_moe")
        if self.n_memory_tokens > 0:
            enabled.append("n_memory_tokens")
        if self.n_loops > 1:
            enabled.append("n_loops")
        if self.max_refinement_loops > 0:
            enabled.append("max_refinement_loops")
        if self.use_confidence_head:
            enabled.append("use_confidence_head")
        if self.use_residual_gates:
            enabled.append("use_residual_gates")
        if self.use_mod_routing:
            enabled.append("use_mod_routing")
        if self.use_mla:
            enabled.append("use_mla")
        return tuple(enabled)


MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    EDGE_0_5B_GQA: {
        "profile_name": EDGE_0_5B_GQA,
        "dim": 1536,
        "n_layers": 18,
        "n_heads": 16,
        "n_kv_heads": 4,
        "ffn_dim": 4096,
        "max_seq_len": 2048,
        "sliding_window": 512,
    },
    LOCAL_1_5B_GQA: {
        "profile_name": LOCAL_1_5B_GQA,
        "dim": 2048,
        "n_layers": 28,
        "n_heads": 16,
        "n_kv_heads": 4,
        "ffn_dim": 6144,
        "max_seq_len": 4096,
        "sliding_window": 1024,
    },
    WORKSTATION_3B_GQA: {
        "profile_name": WORKSTATION_3B_GQA,
        "dim": 2560,
        "n_layers": 36,
        "n_heads": 20,
        "n_kv_heads": 4,
        "ffn_dim": 8192,
        "max_seq_len": 4096,
        "sliding_window": 2048,
    },
    LEGACY_EXPERIMENTAL_439M: {
        "profile_name": LEGACY_EXPERIMENTAL_439M,
        "dim": 1024,
        "n_layers": 16,
        "n_heads": 16,
        "n_kv_heads": 4,
        "ffn_dim": 2816,
        "max_seq_len": 2048,
        "use_moe": True,
        "n_experts": 8,
        "n_experts_active": 2,
        "moe_freq": 2,
        "n_memory_tokens": 64,
        "n_loops": 2,
        "yarn_scale": 4.0,
        "sliding_window": 512,
        "max_refinement_loops": 1,
        "use_confidence_head": True,
        "use_residual_gates": True,
        "use_mod_routing": True,
    },
}


@dataclass(frozen=True)
class DeploymentTier:
    name: str
    target_vram_gb: int
    model_profile: str
    intended_model_size_band: str
    quantization_expectation: str
    active_context_tokens: int
    offload_expectation: str
    rag_expectation: str
    caveats: Tuple[str, ...]
    external_unverified_targets: Tuple[str, ...] = (
        "GGUF/llama.cpp",
        "GPTQ",
        "AWQ",
        "vLLM",
    )


REFERENCE_RUNTIME_NOTE = (
    "PyTorch CPU/GPU model execution is the current in-repo reference path. "
    "The serving boundary is fail-closed until a real model backend is wired."
)
IMPLEMENTED_QUANTIZATION_NOTE = (
    "The only implemented in-repo quantization mode is CPU dynamic-int8 for "
    "torch Linear modules, with optional TorchScript export when tracing succeeds."
)


DEPLOYMENT_TIERS: Dict[str, DeploymentTier] = {
    DEPLOY_2GB_EDGE: DeploymentTier(
        name=DEPLOY_2GB_EDGE,
        target_vram_gb=2,
        model_profile=EDGE_0_5B_GQA,
        intended_model_size_band=(
            "edge/sub-0.5B to roughly 0.5B dense; larger models require "
            "external quantized runtimes/offload not verified by this repo"
        ),
        quantization_expectation=(
            "2GB VRAM is edge/offload territory. Full PyTorch fp16/bf16 "
            "GPU inference is not claimed; 4-bit/GGUF-style paths are external "
            "and unverified here."
        ),
        active_context_tokens=512,
        offload_expectation=(
            "CPU fallback or external offload runtime expected; repo only "
            "verifies PyTorch reference behavior."
        ),
        rag_expectation=(
            "RAG is expected for document workflows; brute-force long context "
            "is not realistic at this tier."
        ),
        caveats=(
            "2GB VRAM does not make a 3B/5B model comfortable on GPU.",
            "KV cache grows with active context, especially in global layers.",
            "Deployment tier metadata is not validated runtime support.",
        ),
    ),
    DEPLOY_8GB_LAPTOP: DeploymentTier(
        name=DEPLOY_8GB_LAPTOP,
        target_vram_gb=8,
        model_profile=EDGE_0_5B_GQA,
        intended_model_size_band=(
            "roughly 0.5B PyTorch reference path, or larger only with "
            "quantization/offload paths not verified here"
        ),
        quantization_expectation=(
            "Dynamic-int8 CPU export is implemented; 4-bit GPU/local-runtime "
            "formats are not implemented in this repo."
        ),
        active_context_tokens=1024,
        offload_expectation=(
            "CPU fallback may be usable for slow local inference; GPU use still "
            "depends on checkpoint size, dtype, context, and runtime overhead."
        ),
        rag_expectation=(
            "RAG is recommended for long documents instead of increasing context "
            "blindly."
        ),
        caveats=(
            "8GB does not validate laptop deployment for every checkpoint.",
            "Lower-bound estimates exclude allocator and framework overhead.",
            "External runtimes remain future integration targets.",
        ),
    ),
    DEPLOY_16GB_WORKSTATION: DeploymentTier(
        name=DEPLOY_16GB_WORKSTATION,
        target_vram_gb=16,
        model_profile=LOCAL_1_5B_GQA,
        intended_model_size_band=(
            "local 1.5B-class reference target; 3B-class use is conditional on "
            "quantization/offload and has not been repo-verified"
        ),
        quantization_expectation=(
            "PyTorch fp16/bf16 is the reference path; dynamic-int8 CPU export "
            "exists, but GPTQ/AWQ/GGUF/vLLM are not implemented here."
        ),
        active_context_tokens=2048,
        offload_expectation=(
            "GPU inference may be feasible for smaller profiles; larger profiles "
            "need measured memory checks, not profile names."
        ),
        rag_expectation=(
            "RAG remains the practical path for large private document sets."
        ),
        caveats=(
            "16GB is not proof that workstation_3b_gqa fits with useful context.",
            "Global-layer KV cache still grows with context.",
            "Deployment readiness is separate from training success and benchmark quality.",
        ),
    ),
}


def available_model_profiles() -> Tuple[str, ...]:
    return MODEL_PROFILE_NAMES


def available_deployment_tiers() -> Tuple[str, ...]:
    return DEPLOYMENT_TIER_NAMES


def get_model_profile(profile_name: str) -> Dict[str, Any]:
    try:
        return dict(MODEL_PROFILES[profile_name])
    except KeyError as exc:
        raise ValueError(
            "Unknown model profile. Expected one of: "
            + ", ".join(MODEL_PROFILE_NAMES)
        ) from exc


def get_deployment_tier(name: str) -> Dict[str, Any]:
    try:
        return asdict(DEPLOYMENT_TIERS[name])
    except KeyError as exc:
        raise ValueError(
            "Unknown deployment tier. Expected one of: "
            + ", ".join(DEPLOYMENT_TIER_NAMES)
        ) from exc


def build_model_config(profile_name: str = EDGE_0_5B_GQA, **overrides: Any) -> ModelConfig:
    values = get_model_profile(profile_name)
    values.update(overrides)
    values["profile_name"] = profile_name
    return ModelConfig(**values)


def _apply_model_profile(target: ModelConfig, profile_name: str) -> None:
    for key, value in get_model_profile(profile_name).items():
        setattr(target, key, value)


def estimate_dense_parameter_count_lower_bound(
    cfg: ModelConfig,
    vocab_size: Optional[int] = None,
) -> Optional[int]:
    """Estimate dense production parameter count; return None for legacy paths."""
    if cfg.enabled_experimental_features():
        return None
    effective_vocab = int(vocab_size if vocab_size is not None else cfg.vocab_size)
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    attention_params = (
        (cfg.dim * cfg.dim)
        + (2 * cfg.dim * kv_dim)
        + (cfg.dim * cfg.dim)
    )
    ffn_params = 3 * cfg.dim * cfg.ffn_dim
    norm_params = 2 * cfg.dim
    return (
        (effective_vocab * cfg.dim)
        + (cfg.n_layers * (attention_params + ffn_params + norm_params))
        + cfg.dim
    )


def estimate_weight_memory_lower_bound_gb(
    cfg: ModelConfig,
    *,
    bytes_per_param: int = 2,
    vocab_size: Optional[int] = None,
) -> Optional[float]:
    params = estimate_dense_parameter_count_lower_bound(
        cfg,
        vocab_size=vocab_size,
    )
    if params is None:
        return None
    if int(bytes_per_param) <= 0:
        raise ValueError("bytes_per_param must be > 0")
    return (params * int(bytes_per_param)) / (1024 ** 3)


def estimate_kv_cache_lower_bound_gb(
    cfg: ModelConfig,
    *,
    active_context_tokens: int,
    batch_size: int = 1,
    bytes_per_value: int = 2,
) -> float:
    if int(active_context_tokens) <= 0:
        raise ValueError("active_context_tokens must be > 0")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(bytes_per_value) <= 0:
        raise ValueError("bytes_per_value must be > 0")

    local_layers = cfg.n_layers // 2 if cfg.sliding_window > 0 else 0
    global_layers = cfg.n_layers - local_layers
    local_tokens = min(int(active_context_tokens), int(cfg.sliding_window))
    token_slots = (local_layers * local_tokens) + (
        global_layers * int(active_context_tokens)
    )
    values = (
        int(batch_size)
        * token_slots
        * 2
        * cfg.n_kv_heads
        * cfg.head_dim
    )
    return (values * int(bytes_per_value)) / (1024 ** 3)


def build_deployment_report(
    tier_name: str,
    *,
    vocab_size: Optional[int] = None,
    weight_bytes_per_param: int = 2,
    kv_bytes_per_value: int = 2,
    batch_size: int = 1,
) -> Dict[str, Any]:
    tier = get_deployment_tier(tier_name)
    cfg = build_model_config(tier["model_profile"])
    params = estimate_dense_parameter_count_lower_bound(cfg, vocab_size=vocab_size)
    weight_gb = estimate_weight_memory_lower_bound_gb(
        cfg,
        bytes_per_param=weight_bytes_per_param,
        vocab_size=vocab_size,
    )
    kv_gb = estimate_kv_cache_lower_bound_gb(
        cfg,
        active_context_tokens=int(tier["active_context_tokens"]),
        batch_size=batch_size,
        bytes_per_value=kv_bytes_per_value,
    )
    return {
        "tier": tier,
        "model_profile": tier["model_profile"],
        "profile_is_validated_runtime_support": False,
        "reference_runtime": REFERENCE_RUNTIME_NOTE,
        "implemented_quantization": IMPLEMENTED_QUANTIZATION_NOTE,
        "repo_verified_external_runtimes": [],
        "external_unverified_targets": list(tier["external_unverified_targets"]),
        "estimates": {
            "estimate_scope": "lower_bound_only",
            "dense_params_lower_bound": params,
            "weight_memory_lower_bound_gb": (
                round(weight_gb, 3) if weight_gb is not None else None
            ),
            "kv_cache_lower_bound_gb": round(kv_gb, 3),
            "active_context_tokens": int(tier["active_context_tokens"]),
            "batch_size": int(batch_size),
            "weight_bytes_per_param": int(weight_bytes_per_param),
            "kv_bytes_per_value": int(kv_bytes_per_value),
        },
        "caveats": list(tier["caveats"]) + [
            "Lower-bound memory is not a full deployment requirement estimate.",
            "No throughput or latency claim is made by this report.",
            "A deployment profile does not prove training quality or benchmark capability.",
        ],
    }


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

    # --- Legacy/declared data-quality knobs ---
    # Current data.py does not implement n-gram dedup or quality-threshold
    # filtering. These fields remain for config compatibility until a real
    # data-quality pass wires them into an implemented pipeline.
    dedup_ngram_size  : int   = 13
    dedup_threshold   : float = 0.8
    min_doc_length    : int   = 50
    max_doc_length    : int   = 100000
    quality_threshold : float = 0.3

    # --- Data integrity ---
    # Exclude common benchmark source IDs from pretrain/tokenizer corpora.
    # This reduces source-overlap risk; it is not content contamination detection.
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


@dataclass
class AgentConfig:
    """Configuration for the verified coding-agent surface."""
    report_dir              : str = "./run_artifacts/agent"
    backend_kind            : str = "none"
    backend_script_path     : Optional[str] = None
    backend_model_id_or_path: Optional[str] = None
    backend_local_files_only: bool = True
    backend_trust_remote_code: bool = False
    backend_device          : str = "auto"
    backend_max_new_tokens  : int = 1024
    backend_temperature     : float = 0.0
    backend_prompt_max_chars: int = 12000
    max_file_excerpt_chars  : int = 4000
    default_retry_budget    : int = 3


model_cfg = ModelConfig()
train_cfg = TrainConfig()
sft_cfg = SFTConfig()
dpo_cfg = DPOConfig()
improver_cfg = ImproverConfig()
eval_cfg = EvalConfig()
distill_cfg = DistillConfig()
quant_cfg = QuantConfig()
hardware_profile_cfg = HardwareProfileConfig()
agent_cfg = AgentConfig()


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
        "agent": agent_cfg,
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
        "agent_cfg": agent_cfg,
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

            if target is model_cfg and "profile_name" in values:
                _apply_model_profile(target, str(values["profile_name"]))

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
        "agent": asdict(agent_cfg),
    }


validate_runtime_state()


def show_config():
    """Print configs. Called explicitly, never at import time."""
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("Configuration")
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("Model profile     : %s", model_cfg.profile_name)
    LOGGER.info("Architecture      : %sd x %sL x %sH", model_cfg.dim, model_cfg.n_layers, model_cfg.n_heads)
    LOGGER.info("Vocab size        : %s", f"{model_cfg.vocab_size:,}")
    LOGGER.info("Memory tokens     : %s", model_cfg.n_memory_tokens)
    LOGGER.info("Recursive loops   : %s", model_cfg.n_loops)
    LOGGER.info("Max refinement    : %s", model_cfg.max_refinement_loops)
    if model_cfg.use_moe:
        LOGGER.info(
            "MoE               : %sE / %sA every %s layers",
            model_cfg.n_experts,
            model_cfg.n_experts_active,
            model_cfg.moe_freq,
        )
    else:
        LOGGER.info("MoE               : disabled")
    LOGGER.info("MoD routing       : %s", model_cfg.use_mod_routing)
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
