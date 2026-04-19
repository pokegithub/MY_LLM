"""Hardware optimization and backend selection helpers."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch

from core.ops import is_flash_attention_available


def _is_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    check = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(check):
        return bool(check())
    return False


def detect_profile_name() -> str:
    """Detect default hardware profile from active CUDA device."""
    if not torch.cuda.is_available():
        return "cpu_profile.json"

    name = torch.cuda.get_device_name(0).lower()
    if "t4" in name:
        return "t4_profile.json"
    if "a10" in name:
        return "a10_profile.json"
    return "t4_profile.json"


def resolve_attention_backend(
    requested_backend: str,
    *,
    use_flashattention: bool,
    cuda_available: bool,
    device_name: str,
    flash_available: bool,
) -> str:
    """Resolve backend policy with a conservative auto mode."""
    backend = str(requested_backend).strip().lower()
    if backend == "sdpa":
        return "sdpa"
    if backend == "flash2":
        return "flash2" if (cuda_available and flash_available) else "sdpa"

    if backend != "auto":
        return "sdpa"

    if not cuda_available or not use_flashattention or not flash_available:
        return "sdpa"

    normalized_name = str(device_name).lower()
    if "t4" in normalized_name or "a10" in normalized_name:
        return "flash2"

    return "flash2"


def apply_profile_attention_policy(
    profile: Mapping[str, Any],
    *,
    current_backend: str,
    current_use_flash: bool,
) -> Tuple[str, bool]:
    """Apply profile-level attention preferences and return resolved values."""
    requested_backend = str(
        profile.get("attention_backend", current_backend)
    ).strip().lower()

    flash_pref_raw = profile.get("use_flashattention", current_use_flash)
    use_flash = bool(flash_pref_raw) if isinstance(flash_pref_raw, bool) else current_use_flash

    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    backend = resolve_attention_backend(
        requested_backend,
        use_flashattention=use_flash,
        cuda_available=cuda_available,
        device_name=device_name,
        flash_available=is_flash_attention_available(),
    )
    return backend, use_flash


def resolve_profile_dtype_policy(
    profile: Mapping[str, Any],
    *,
    current_dtype: str,
    current_amp_grad_scaler: bool,
) -> Tuple[str, bool]:
    """Resolve mixed-precision dtype/scaler policy from profile and hardware."""
    requested_dtype = str(
        profile.get("preferred_dtype", current_dtype)
    ).strip().lower()
    if requested_dtype not in {"bfloat16", "float16"}:
        requested_dtype = str(current_dtype).strip().lower()

    if requested_dtype == "bfloat16" and not _is_bf16_supported():
        return "float16", True

    if requested_dtype == "float16" and torch.cuda.is_available():
        return "float16", True

    return requested_dtype, bool(current_amp_grad_scaler)
