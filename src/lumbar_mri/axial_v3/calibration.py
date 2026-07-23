"""Validation-only raw_0 calibration helpers."""

from __future__ import annotations

import numpy as np

from .metrics import metrics_from_predictions


def _probabilities_with_batch(probabilities: np.ndarray) -> tuple[np.ndarray, bool]:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim == 3:
        return probs[None, ...], False
    if probs.ndim == 4:
        return probs, True
    raise ValueError(f"probabilities must be [C,H,W] or [N,C,H,W], got {probs.shape}")


def apply_raw0_threshold(
    probabilities: np.ndarray,
    *,
    raw0_index: int = 1,
    min_probability: float = 0.0,
    min_margin: float = 0.0,
) -> np.ndarray:
    """Apply a raw_0 gate and reassign rejected raw_0 pixels to the second class."""

    probs, had_batch = _probabilities_with_batch(probabilities)
    if not 0 <= raw0_index < probs.shape[1]:
        raise ValueError(f"raw0_index out of range: {raw0_index}")
    if min_probability < 0 or min_probability > 1:
        raise ValueError("min_probability must be between 0 and 1")
    if min_margin < 0:
        raise ValueError("min_margin must be non-negative")
    order = np.argsort(probs, axis=1)
    top = order[:, -1]
    second = order[:, -2]
    top_prob = np.take_along_axis(probs, top[:, None, ...], axis=1)[:, 0]
    second_prob = np.take_along_axis(probs, second[:, None, ...], axis=1)[:, 0]
    pred = top.astype(np.int64)
    raw0_top = pred == raw0_index
    accepted = (top_prob >= min_probability) & ((top_prob - second_prob) >= min_margin)
    pred[raw0_top & ~accepted] = second[raw0_top & ~accepted]
    return pred if had_batch else pred[0]


def apply_raw0_slice_presence_gate(
    probabilities: np.ndarray,
    presence_scores: np.ndarray,
    presence_threshold: float,
    *,
    raw0_index: int = 1,
) -> np.ndarray:
    probs, had_batch = _probabilities_with_batch(probabilities)
    scores = np.asarray(presence_scores, dtype=np.float32).reshape(-1)
    if probs.shape[0] != scores.shape[0]:
        raise ValueError(f"presence_scores length {scores.shape[0]} does not match batch {probs.shape[0]}")
    if presence_threshold < 0 or presence_threshold > 1:
        raise ValueError("presence_threshold must be between 0 and 1")
    order = np.argsort(probs, axis=1)
    top = order[:, -1]
    second = order[:, -2]
    pred = top.astype(np.int64)
    absent = scores < presence_threshold
    raw0_top = pred == raw0_index
    pred[absent[:, None, None] & raw0_top] = second[absent[:, None, None] & raw0_top]
    return pred if had_batch else pred[0]


def raw0_presence_metrics(prediction: np.ndarray, target: np.ndarray, *, raw0_index: int = 1) -> dict[str, int | float]:
    metrics = metrics_from_predictions(prediction, target)
    raw0 = metrics["perClass"]["raw_0"]
    return {
        "truePositivePixels": raw0["truePositivePixels"],
        "falsePositivePixels": raw0["falsePositivePixels"],
        "falseNegativePixels": raw0["falseNegativePixels"],
        "predictedInGtAbsentCases": raw0["predictedInGtAbsentCases"],
        "gtPresent": raw0["gtPresentCases"],
        "predPresent": raw0["predPresentCases"],
        "precision": raw0["precision"] if raw0["precision"] is not None else 0.0,
        "recall": raw0["recall"] if raw0["recall"] is not None else 0.0,
    }


def raw0_metrics_from_predictions(prediction: np.ndarray, target: np.ndarray) -> dict[str, int | float | None]:
    raw0 = metrics_from_predictions(prediction, target)["perClass"]["raw_0"]
    return dict(raw0)
