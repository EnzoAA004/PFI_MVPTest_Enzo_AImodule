"""Iteration B low-cost experiment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SelectionGuardrail:
    max_other_class_dice_drop: float = 0.05
    protected_classes: tuple[str, ...] = ("raw_50", "raw_100", "raw_150", "raw_200")


def cap_class_weight_ratio(weights: np.ndarray, *, max_ratio: float | None) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if max_ratio is None:
        return values
    if max_ratio <= 0:
        raise ValueError("max_ratio must be positive")
    positive = values[values > 0]
    if positive.size == 0:
        return values
    floor = float(positive.min())
    ceiling = floor * max_ratio
    return np.clip(values, floor, ceiling)


def apply_raw0_effective_weight(weights: np.ndarray, *, raw0_index: int = 1, multiplier: float = 1.0) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64).copy()
    if multiplier <= 0:
        raise ValueError("raw0 multiplier must be positive")
    values[raw0_index] *= multiplier
    return values


def validation_ranking_key(metrics: Mapping[str, float | int | None]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("dice_macro_foreground") or float("-inf")),
        -float(metrics.get("raw0PredictedInGtAbsentCases") or float("inf")),
        float(metrics.get("raw0Precision") or float("-inf")),
        float(metrics.get("dice_macro_excluding_raw0") or float("-inf")),
    )


def passes_other_class_guardrail(
    candidate_per_class: Mapping[str, Mapping[str, float | None]],
    baseline_per_class: Mapping[str, Mapping[str, float | None]],
    guardrail: SelectionGuardrail,
) -> bool:
    for class_name in guardrail.protected_classes:
        candidate_dice = candidate_per_class.get(class_name, {}).get("dice")
        baseline_dice = baseline_per_class.get(class_name, {}).get("dice")
        if candidate_dice is None or baseline_dice is None:
            continue
        if float(baseline_dice) - float(candidate_dice) > guardrail.max_other_class_dice_drop:
            return False
    return True
