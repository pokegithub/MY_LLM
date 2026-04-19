"""Core tensor ops for attention backends and numerical stability."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from core.logging import get_logger

LOGGER = get_logger("core.ops")

try:
    from flash_attn import flash_attn_func  # type: ignore

    _HAS_FLASH_ATTN = True
except Exception:
    flash_attn_func = None
    _HAS_FLASH_ATTN = False


def is_flash_attention_available() -> bool:
    """Return whether flash-attn kernel bindings are importable."""
    return bool(_HAS_FLASH_ATTN)


def qk_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS-style normalization on the last dim in float32, then cast back."""
    x_f32 = x.float()
    denom = x_f32.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f32 * denom).to(dtype=x.dtype)


def apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Q/K tensors in pre-RoPE space before attention logits."""
    return qk_norm(q, eps=eps), qk_norm(k, eps=eps)


def should_use_flash_attention(
    *,
    backend: str,
    use_flashattention: bool,
    local_attention: bool,
    q: torch.Tensor,
    head_dim: int,
    attn_mask: Optional[torch.Tensor],
) -> bool:
    """Return True when flash attention is safe and requested."""
    normalized_backend = str(backend).strip().lower()
    if normalized_backend == "sdpa":
        return False
    if normalized_backend not in {"auto", "flash2"}:
        return False
    if not use_flashattention:
        return False
    if local_attention:
        return False
    if attn_mask is not None:
        # Custom masks (e.g., local windows) are routed through SDPA.
        return False
    if not _HAS_FLASH_ATTN:
        return False
    if not q.is_cuda:
        return False
    if q.dtype not in {torch.float16, torch.bfloat16}:
        return False
    if head_dim % 8 != 0:
        return False
    return True


def run_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Run flash_attn_func with [B,H,T,D] tensors."""
    qf = q.transpose(1, 2).contiguous()
    kf = k.transpose(1, 2).contiguous()
    vf = v.transpose(1, 2).contiguous()
    out = flash_attn_func(
        qf,
        kf,
        vf,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
    )
    return out.transpose(1, 2)


def run_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
) -> torch.Tensor:
    """Run torch SDPA with explicit mask/casual flags."""
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=is_causal,
    )


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: str,
    use_flashattention: bool,
    local_attention: bool,
    head_dim: int,
    attn_mask: Optional[torch.Tensor],
    decode_single_token: bool,
) -> torch.Tensor:
    """Dispatch attention kernel with conservative safety fallbacks."""
    if local_attention and attn_mask is None:
        raise ValueError(
            "Local attention requires an explicit attention mask."
        )

    if should_use_flash_attention(
        backend=backend,
        use_flashattention=use_flashattention,
        local_attention=local_attention,
        q=q,
        head_dim=head_dim,
        attn_mask=attn_mask,
    ):
        try:
            return run_flash_attention(
                q,
                k,
                v,
                causal=(not decode_single_token),
            )
        except Exception as e:
            LOGGER.warning(
                "FlashAttention failed, falling back to SDPA: %s",
                e,
            )

    return run_sdpa_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=(
            (attn_mask is None)
            and (not local_attention)
            and (not decode_single_token)
        ),
    )
