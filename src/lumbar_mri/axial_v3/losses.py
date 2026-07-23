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


def soft_dice_loss_multiclass(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    include_background: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Foreground soft Dice loss matching the axial v2 baseline convention."""

    if logits.ndim != 4:
        raise ValueError(f"logits must be [N,C,H,W], got {tuple(logits.shape)}")
    if target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError(f"target shape {tuple(target.shape)} incompatible with logits {tuple(logits.shape)}")
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).to(dtype=probs.dtype)
    start = 0 if include_background else 1
    dims = (0, 2, 3)
    intersection = (probs[:, start:] * one_hot[:, start:]).sum(dim=dims)
    denominator = probs[:, start:].sum(dim=dims) + one_hot[:, start:].sum(dim=dims)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def v2_baseline_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, CE and soft-Dice for B0 v2 reproduction."""

    ce = F.cross_entropy(logits, target.long(), weight=class_weights)
    dice = soft_dice_loss_multiclass(logits, target, include_background=False)
    return ce + dice, ce, dice
