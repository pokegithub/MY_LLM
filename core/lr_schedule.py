"""Shared learning-rate schedule helpers."""

from __future__ import annotations

import math


def cosine_warmup_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine decay with linear warmup.

    Args:
        step: Current optimization step.
        total_steps: Planned total optimization steps.
        warmup_steps: Number of warmup steps.
        max_lr: Peak learning rate.
        min_lr: Final learning rate floor.
    """
    current_step = max(int(step), 0)
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    if current_step < warmup_steps:
        return max_lr * current_step / max(1, warmup_steps)

    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def cosine_warmup_lr_from_tokens(
    step: int,
    tokens_per_step: int,
    total_tokens: int,
    warmup_tokens: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Token-budget version of cosine warmup schedule."""
    tps = max(int(tokens_per_step), 1)
    total_steps = max(int(total_tokens) // tps, 1)
    warmup_steps = max(int(warmup_tokens) // tps, 0)
    return cosine_warmup_lr(
        step=step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        max_lr=max_lr,
        min_lr=min_lr,
    )
