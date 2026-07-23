"""Iteration B low-cost experiment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SelectionGuardrail:
    max_other_class_dice_drop: float = 0.05
    protected_classes: tuple[str, ...] = ("raw_50", "raw_100", "raw_150", "raw_200")
    fail_on_missing: bool = True


def metric_or_default(metrics: Mapping[str, float | int | None], key: str, default: float) -> float:
    value = metrics.get(key)
    return default if value is None else float(value)


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
        metric_or_default(metrics, "dice_macro_foreground", float("-inf")),
        -metric_or_default(metrics, "raw0PredictedInGtAbsentCases", float("inf")),
        metric_or_default(metrics, "raw0Precision", float("-inf")),
        metric_or_default(metrics, "dice_macro_excluding_raw0", float("-inf")),
    )


def passes_other_class_guardrail(
    candidate_per_class: Mapping[str, Mapping[str, float | None]],
    baseline_per_class: Mapping[str, Mapping[str, float | None]],
    guardrail: SelectionGuardrail,
) -> bool:
    return bool(evaluate_other_class_guardrail(candidate_per_class, baseline_per_class, guardrail)["passed"])


def evaluate_other_class_guardrail(
    candidate_per_class: Mapping[str, Mapping[str, float | None]],
    baseline_per_class: Mapping[str, Mapping[str, float | None]],
    guardrail: SelectionGuardrail,
) -> dict[str, object]:
    per_class: dict[str, dict[str, float | None]] = {}
    missing: list[str] = []
    violated: list[str] = []
    reasons: list[str] = []
    for class_name in guardrail.protected_classes:
        candidate_dice = candidate_per_class.get(class_name, {}).get("dice")
        baseline_dice = baseline_per_class.get(class_name, {}).get("dice")
        if candidate_dice is None or baseline_dice is None:
            missing.append(class_name)
            per_class[class_name] = {"baselineDice": baseline_dice, "candidateDice": candidate_dice, "drop": None}
            continue
        drop = float(baseline_dice) - float(candidate_dice)
        per_class[class_name] = {
            "baselineDice": float(baseline_dice),
            "candidateDice": float(candidate_dice),
            "drop": drop,
        }
        if drop > guardrail.max_other_class_dice_drop:
            violated.append(class_name)
    if missing and guardrail.fail_on_missing:
        reasons.append(f"missing protected Dice metrics: {missing}")
    if violated:
        reasons.append(f"protected class Dice drop exceeded: {violated}")
    return {
        "passed": not reasons,
        "maxAllowedDrop": guardrail.max_other_class_dice_drop,
        "perClass": per_class,
        "missingMetrics": missing,
        "violatedClasses": violated,
        "reasons": reasons,
    }
