"""Hardware-aware optimizer and AMP helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch

from core.logging import get_logger


LOGGER = get_logger("optimizer_factory")


def _is_bf16_supported_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    check = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(check):
        return bool(check())
    return False


@dataclass
class OptimizerSettings:
    lr: float
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    use_fused: bool = True


@dataclass
class AmpPolicySettings:
    dtype_name: str = "bfloat16"
    device_type: str = "cuda"
    use_grad_scaler: bool = True
    init_scale: float = 65536.0
    growth_interval: int = 2000


def _supports_autocast(device_type: str, dtype: torch.dtype) -> bool:
    if device_type == "cuda":
        return torch.cuda.is_available() and dtype in {
            torch.float16,
            torch.bfloat16,
        }
    if device_type == "cpu":
        return dtype is torch.bfloat16
    return False


class AmpPolicy:
    """Wraps autocast + optional gradient scaling behind one interface."""

    def __init__(self, settings: AmpPolicySettings):
        self.device_type = str(settings.device_type).lower()
        self.dtype = get_amp_dtype(settings.dtype_name)
        force_grad_scaler = False

        if self.device_type == "cuda" and self.dtype == torch.bfloat16:
            if not _is_bf16_supported_cuda():
                self.dtype = torch.float16
                force_grad_scaler = True
                LOGGER.warning(
                    "CUDA bfloat16 unsupported on this device; forcing float16 autocast with grad scaler."
                )

        self._autocast_enabled = _supports_autocast(
            self.device_type,
            self.dtype,
        )

        scaler_enabled = (
            (bool(settings.use_grad_scaler) or force_grad_scaler)
            and self.device_type == "cuda"
            and torch.cuda.is_available()
            and self.dtype == torch.float16
        )

        if self.device_type == "cuda":
            self._scaler = torch.cuda.amp.GradScaler(
                enabled=scaler_enabled,
                init_scale=float(settings.init_scale),
                growth_interval=max(1, int(settings.growth_interval)),
            )
        else:
            self._scaler = None

    @property
    def uses_grad_scaler(self) -> bool:
        return bool(self._scaler is not None and self._scaler.is_enabled())

    def autocast(self):
        if self._autocast_enabled:
            return torch.autocast(device_type=self.device_type, dtype=self.dtype)
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        if self.uses_grad_scaler:
            self._scaler.scale(loss).backward()  # type: ignore[union-attr]
            return
        loss.backward()

    def clip_grad_norm_(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.nn.Parameter],
        max_norm: float,
    ) -> float:
        if self.uses_grad_scaler:
            self._scaler.unscale_(optimizer)  # type: ignore[union-attr]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        if torch.is_tensor(grad_norm):
            return float(grad_norm.item())
        return float(grad_norm)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.uses_grad_scaler:
            self._scaler.step(optimizer)  # type: ignore[union-attr]
            return
        optimizer.step()

    def update(self) -> None:
        if self.uses_grad_scaler:
            self._scaler.update()  # type: ignore[union-attr]


def build_amp_policy(settings: AmpPolicySettings) -> AmpPolicy:
    return AmpPolicy(settings)


def split_decay_parameters(model: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Split parameters into decay/non-decay groups by dimensionality."""
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return decay, no_decay


def build_adamw(
    model: torch.nn.Module,
    settings: OptimizerSettings,
) -> torch.optim.AdamW:
    """Build AdamW with fused kernel when available and enabled."""
    decay, no_decay = split_decay_parameters(model)
    fused = bool(settings.use_fused and torch.cuda.is_available())

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": settings.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=settings.lr,
        betas=(settings.beta1, settings.beta2),
        eps=settings.eps,
        fused=fused,
    )


def get_amp_dtype(dtype_name: str) -> torch.dtype:
    lowered = dtype_name.lower()
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float16":
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {dtype_name}")


def autocast_context(dtype_name: str):
    """Return device-aware autocast context manager."""
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = get_amp_dtype(dtype_name)
    return torch.autocast(device_type="cuda", dtype=dtype)
