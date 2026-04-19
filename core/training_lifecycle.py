"""Shared control-flow helpers for training loops."""

from __future__ import annotations

import torch


def should_sync_step(iteration_index: int, grad_accum: int) -> bool:
    """Return True when an optimizer step should occur."""
    accum = max(int(grad_accum), 1)
    return (int(iteration_index) + 1) % accum == 0


def should_run_interval(
    step: int,
    interval: int,
    *,
    include_zero: bool = False,
) -> bool:
    """Return True when `step` hits an interval boundary."""
    every = int(interval)
    if every <= 0:
        return False
    current_step = int(step)
    if not include_zero and current_step <= 0:
        return False
    return current_step % every == 0


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Apply a single LR value to all optimizer parameter groups."""
    lr_value = float(lr)
    for group in optimizer.param_groups:
        group["lr"] = lr_value
