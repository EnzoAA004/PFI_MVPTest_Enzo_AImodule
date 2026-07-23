"""Configurable losses for axial v3 low-cost experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_index: int,
    alpha: float = 0.7,
    beta: float = 0.3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Tversky loss for one class.

    Convention: alpha weights false positives and beta weights false negatives.
    Increasing alpha penalizes over-prediction more strongly.
    """

    if logits.ndim != 4:
        raise ValueError(f"logits must be [N,C,H,W], got {tuple(logits.shape)}")
    probs = torch.softmax(logits, dim=1)[:, class_index]
    truth = (target == class_index).to(dtype=probs.dtype)
    dims = tuple(range(1, probs.ndim))
    tp = (probs * truth).sum(dim=dims)
    fp = (probs * (1 - truth)).sum(dim=dims)
    fn = ((1 - probs) * truth).sum(dim=dims)
    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - score.mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_index: int,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 1.333,
) -> torch.Tensor:
    return torch.pow(tversky_loss(logits, target, class_index=class_index, alpha=alpha, beta=beta), gamma)


def raw0_false_positive_penalty(logits: torch.Tensor, target: torch.Tensor, *, raw0_index: int = 1) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, raw0_index]
    absent = (target != raw0_index).to(dtype=probs.dtype)
    return (probs * absent).mean()


def combined_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    raw0_tversky_weight: float = 0.0,
    raw0_fp_penalty_weight: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=class_weights)
    loss = ce
    if raw0_tversky_weight:
        loss = loss + raw0_tversky_weight * tversky_loss(logits, target, class_index=1)
    if raw0_fp_penalty_weight:
        loss = loss + raw0_fp_penalty_weight * raw0_false_positive_penalty(logits, target, raw0_index=1)
    return loss
