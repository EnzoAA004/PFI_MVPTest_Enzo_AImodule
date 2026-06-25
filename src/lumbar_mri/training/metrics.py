"""Métricas de segmentación para máscaras multiclase."""

from __future__ import annotations

import numpy as np


def dice_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_id: int,
    eps: float = 1e-8,
) -> float:
    """Calcula Dice para una clase específica."""

    true_mask = np.asarray(y_true) == class_id
    pred_mask = np.asarray(y_pred) == class_id
    intersection = np.logical_and(true_mask, pred_mask).sum()
    denominator = true_mask.sum() + pred_mask.sum()
    if denominator == 0:
        return 1.0
    return float((2.0 * intersection + eps) / (denominator + eps))


def iou_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_id: int,
    eps: float = 1e-8,
) -> float:
    """Calcula IoU/Jaccard para una clase específica."""

    true_mask = np.asarray(y_true) == class_id
    pred_mask = np.asarray(y_pred) == class_id
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def dice_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: list[int] | tuple[int, ...],
) -> dict[int, float]:
    """Calcula Dice para varias clases."""

    return {class_id: dice_score(y_true, y_pred, class_id) for class_id in class_ids}


def mean_dice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: list[int] | tuple[int, ...],
) -> float:
    """Calcula Dice promedio sobre las clases indicadas."""

    values = list(dice_per_class(y_true, y_pred, class_ids).values())
    return float(np.mean(values)) if values else 0.0
