"""Experiment expansion and validation-only ranking for axial v3 Iteration B."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .guards import find_forbidden_test_references, require_train_val_only
from .low_cost import SelectionGuardrail, evaluate_other_class_guardrail, validation_ranking_key


@dataclass(frozen=True)
class LowCostExperiment:
    experimentId: str
    iteration: str
    experimentType: str
    runId: str
    seed: int = 2026
    architecture: str = "axial_unet2d"
    raw0EffectiveWeightMultiplier: float = 1.0
    maxClassWeightRatio: float | None = None
    lossName: str = "cross_entropy_plus_soft_dice"
    tverskyAlpha: float | None = None
    tverskyBeta: float | None = None
    focalGamma: float | None = None
    raw0TverskyWeight: float = 1.0
    raw0FpPenaltyWeight: float = 0.0
    minProbability: float | None = None
    minMargin: float | None = None
    presenceHeadEnabled: bool = False
    lambdaPresence: float = 0.0
    raw0BalancedSamplerEnabled: bool = False
    positiveFraction: float | None = None
    monitorMetric: str = "dice_macro_foreground"
    maxEpochs: int = 80
    earlyStoppingPatience: int = 12
    parentExperimentId: str | None = None
    parentRunId: str | None = None


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def validate_low_cost_grid(config: dict[str, Any]) -> None:
    require_train_val_only(config.get("developmentSplits", []), context="low_cost_grid")
    text = json.dumps(config)
    forbidden = find_forbidden_test_references(text)
    if forbidden:
        raise ValueError(f"low-cost config contains forbidden test references: {forbidden}")
    if not config.get("selectionPolicy"):
        raise ValueError("selectionPolicy is required")
    if not config.get("guardrails"):
        raise ValueError("guardrails are required")
    ids = [item["id"] for item in config.get("experiments", [])]
    if ids != sorted(set(ids)) or set(ids) != {f"B{i}" for i in range(7)}:
        raise ValueError("expected unique B0-B6 experiment IDs")
    for item in config["experiments"]:
        for value in item.get("raw0EffectiveWeightMultiplierGrid", []):
            if value <= 0:
                raise ValueError("raw0 multipliers must be positive")
        calib = item.get("raw0Calibration", {})
        for value in calib.get("minProbabilityGrid", []):
            if value < 0 or value > 1:
                raise ValueError("thresholds must be between 0 and 1")
        for value in calib.get("minMarginGrid", []):
            if value < 0:
                raise ValueError("margins must be non-negative")
        for value in item.get("presenceHead", {}).get("lambdaPresenceGrid", []):
            if value < 0:
                raise ValueError("lambdaPresence must be non-negative")
        for value in item.get("raw0BalancedSampler", {}).get("positiveFractionGrid", []):
            if value <= 0 or value >= 1:
                raise ValueError("positiveFraction must be between 0 and 1")


def load_low_cost_grid(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_low_cost_grid(payload)
    return payload


def expand_low_cost_experiments(config_json: dict[str, Any]) -> list[LowCostExperiment]:
    validate_low_cost_grid(config_json)
    expanded: list[LowCostExperiment] = []
    by_id = {item["id"]: item for item in config_json["experiments"]}
    expanded.append(LowCostExperiment("B0", "B", "B0", "axial-v3-B0"))
    for value in by_id["B1"]["raw0EffectiveWeightMultiplierGrid"]:
        expanded.append(LowCostExperiment(f"B1-raw0w-{_slug(value)}", "B", "B1", f"axial-v3-B1-raw0w-{_slug(value)}", raw0EffectiveWeightMultiplier=float(value)))
    for value in by_id["B2"]["maxClassWeightRatioGrid"]:
        expanded.append(LowCostExperiment(f"B2-cap-{_slug(value)}", "B", "B2", f"axial-v3-B2-cap-{_slug(value)}", maxClassWeightRatio=float(value)))
    tv = by_id["B3"]["tversky"]
    expanded.append(LowCostExperiment("B3-tversky-a0p7-b0p3", "B", "B3", "axial-v3-B3-tversky-a0p7-b0p3", lossName="tversky_raw0", tverskyAlpha=float(tv["alpha"]), tverskyBeta=float(tv["beta"]), raw0TverskyWeight=1.0, raw0FpPenaltyWeight=0.0))
    calib = by_id["B4"]["raw0Calibration"]
    for threshold in calib["minProbabilityGrid"]:
        for margin in calib["minMarginGrid"]:
                expanded.append(LowCostExperiment(f"B4-thr-{_slug(threshold)}-margin-{_slug(margin)}", "B", "B4", f"axial-v3-B4-thr-{_slug(threshold)}-margin-{_slug(margin)}", minProbability=float(threshold), minMargin=float(margin), parentExperimentId="B0", parentRunId="axial-v3-B0"))
    for value in by_id["B5"]["presenceHead"]["lambdaPresenceGrid"]:
        expanded.append(LowCostExperiment(f"B5-presence-lambda-{_slug(value)}", "B", "B5", f"axial-v3-B5-presence-lambda-{_slug(value)}", presenceHeadEnabled=True, lambdaPresence=float(value)))
    for value in by_id["B6"]["raw0BalancedSampler"]["positiveFractionGrid"]:
        expanded.append(LowCostExperiment(f"B6-balanced-{_slug(value)}", "B", "B6", f"axial-v3-B6-balanced-{_slug(value)}", raw0BalancedSamplerEnabled=True, positiveFraction=float(value)))
    ids = [item.experimentId for item in expanded]
    if len(ids) != len(set(ids)):
        raise ValueError("expanded experiments contain duplicate IDs")
    return expanded


def estimate_run_count(config_json: dict[str, Any]) -> int:
    return len(expand_low_cost_experiments(config_json))


def select_experiment(config_json: dict[str, Any], experiment_id: str) -> LowCostExperiment:
    for experiment in expand_low_cost_experiments(config_json):
        if experiment.experimentId == experiment_id:
            return experiment
    raise KeyError(f"unknown axial v3 experiment ID: {experiment_id}")


def rank_validation_experiments(rows: list[dict[str, Any]], baseline_per_class: dict[str, Any], guardrail: SelectionGuardrail) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if row.get("smokeOnly"):
            reasons.append("smoke run is not eligible")
        if row.get("trainingStatus") != "completed":
            reasons.append("run is not completed")
        metrics = row.get("validationMetrics", {})
        key = validation_ranking_key(metrics)
        if any(value != value or value in {float("inf"), float("-inf")} for value in key):
            reasons.append("ranking metrics missing or non-finite")
        guard = evaluate_other_class_guardrail(row.get("perClass", {}), baseline_per_class, guardrail)
        if not guard["passed"]:
            reasons.extend(str(reason) for reason in guard["reasons"])
        payload = {**row, "rankingKey": key, "guardrail": guard, "discardReasons": reasons}
        if reasons:
            discarded.append(payload)
        else:
            accepted.append(payload)
    accepted.sort(key=lambda row: row["rankingKey"], reverse=True)
    if accepted:
        accepted[0]["selectionLabel"] = "validation_candidate"
    return {"accepted": accepted, "discarded": discarded}
