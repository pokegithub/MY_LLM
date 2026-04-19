"""Shared sequence tensor helpers for next-token training/evaluation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def shift_for_next_token(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shift logits/labels to next-token aligned tensors."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = token_ids[:, 1:].contiguous()
    return shift_logits, shift_labels


def gather_token_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather per-token log-probabilities for provided token ids."""
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, 2, token_ids.unsqueeze(2)).squeeze(2)


def sequence_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-sequence summed token log-probabilities."""
    token_lp = gather_token_log_probs(logits, token_ids)
    if token_mask is not None:
        token_lp = token_lp * token_mask
    return token_lp.sum(dim=-1)


def next_token_cross_entropy(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    ignore_index: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy for next-token prediction with internal shift."""
    shift_logits, shift_labels = shift_for_next_token(logits, token_ids)
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )


def masked_next_token_sequence_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence log-probabilities with next-token shift and mask."""
    shift_logits, shift_labels = shift_for_next_token(logits, token_ids)
    shift_mask = token_mask[:, 1:].contiguous()
    return sequence_log_probs(shift_logits, shift_labels, shift_mask)
