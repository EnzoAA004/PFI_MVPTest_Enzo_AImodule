"""Validation-only raw_0 calibration helpers."""

from __future__ import annotations

import numpy as np


def apply_raw0_threshold(
    probabilities: np.ndarray,
    *,
    raw0_index: int = 1,
    min_probability: float = 0.0,
    min_margin: float = 0.0,
) -> np.ndarray:
    """Apply a raw_0 gate and reassign rejected raw_0 pixels to the second class."""

    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim < 3:
        raise ValueError("probabilities must have class dimension followed by image dimensions")
    if not 0 <= raw0_index < probs.shape[0]:
        raise ValueError(f"raw0_index out of range: {raw0_index}")
    order = np.argsort(probs, axis=0)
    top = order[-1]
    second = order[-2]
    top_prob = np.take_along_axis(probs, top[None, ...], axis=0)[0]
    second_prob = np.take_along_axis(probs, second[None, ...], axis=0)[0]
    pred = top.astype(np.int64)
    raw0_top = pred == raw0_index
    accepted = (top_prob >= min_probability) & ((top_prob - second_prob) >= min_margin)
    pred[raw0_top & ~accepted] = second[raw0_top & ~accepted]
    return pred


def raw0_presence_metrics(prediction: np.ndarray, target: np.ndarray, *, raw0_index: int = 1) -> dict[str, int | float]:
    pred = np.asarray(prediction) == raw0_index
    truth = np.asarray(target) == raw0_index
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    pred_present = bool(pred.any())
    gt_present = bool(truth.any())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "truePositivePixels": tp,
        "falsePositivePixels": fp,
        "falseNegativePixels": fn,
        "predictedInGtAbsentCases": int(pred_present and not gt_present),
        "gtPresent": int(gt_present),
        "predPresent": int(pred_present),
        "precision": precision,
        "recall": recall,
    }
