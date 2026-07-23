from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lumbar_mri.axial_v3.guards import find_forbidden_test_references, reject_test_paths, require_train_val_only
from lumbar_mri.axial_v3.experiments import estimate_run_count, expand_low_cost_experiments, validate_low_cost_grid
from lumbar_mri.axial_v3.low_cost import SelectionGuardrail, apply_raw0_effective_weight, cap_class_weight_ratio, evaluate_other_class_guardrail, passes_other_class_guardrail, validation_ranking_key
from lumbar_mri.axial_v3.registry import REGISTRY_COLUMNS, ExperimentRegistryRow, append_registry_row, registry_row_from_config


def test_require_train_val_only_rejects_test() -> None:
    assert require_train_val_only(["train", "validation"]) == {"train", "val"}
    with pytest.raises(ValueError, match="only train/val"):
        require_train_val_only(["train", "test"])
    with pytest.raises(ValueError, match="test artifact"):
        reject_test_paths(["outputs/axial_final_v2/test_metrics.json"])


def test_static_forbidden_references() -> None:
    text = "metrics = read_json('test_metrics.json')\nAXIAL_FINAL_TEST_CONFIRMATION = 'x'"
    hits = find_forbidden_test_references(text)
    assert "test_metrics.json" in hits
    assert "AXIAL_FINAL_TEST_CONFIRMATION" in hits
    assert find_forbidden_test_references("require_train_val_only(['train', 'val'])") == []


def test_registry_schema_and_append(tmp_path: Path) -> None:
    payload = {
        "experimentId": "B0-001",
        "iteration": "B",
        "experimentType": "B0",
        "runId": "axial-v3-b0",
        "createdAtUtc": "2026-07-23T00:00:00Z",
        "updatedAtUtc": "2026-07-23T00:00:00Z",
        "gitCommit": "abc123",
        "aiServiceCommit": "abc123",
        "seed": 42,
        "configPath": "config/axial_v3_low_cost_grid.json",
        "configSha256": "sha-config",
        "splitSha256": "sha",
        "trainingStatus": "planned",
        "smokeOnly": False,
        "selectedEpoch": None,
        "monitorMetric": "dice_macro_foreground",
        "notes": "synthetic",
    }
    row = registry_row_from_config(payload)
    path = tmp_path / "experiment_registry.csv"
    append_registry_row(path, row)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["experimentId"] == "B0-001"
    assert rows[0]["notes"] == "synthetic"
    assert set(REGISTRY_COLUMNS).issubset(rows[0].keys())


def test_registry_requires_core_fields() -> None:
    with pytest.raises(ValueError, match="missing registry fields"):
        registry_row_from_config({"experimentId": "bad"})


def test_low_cost_weighting_ranking_and_guardrail() -> None:
    weights = cap_class_weight_ratio(apply_raw0_effective_weight([1, 10, 2], raw0_index=1, multiplier=0.5), max_ratio=3)
    assert weights.tolist() == [1.0, 3.0, 2.0]
    assert validation_ranking_key({"dice_macro_foreground": 0.8, "raw0PredictedInGtAbsentCases": 2, "raw0Precision": 0.4, "dice_macro_excluding_raw0": 0.9}) == (0.8, -2.0, 0.4, 0.9)
    assert validation_ranking_key({"dice_macro_foreground": 0.0, "raw0PredictedInGtAbsentCases": 0, "raw0Precision": 0.0, "dice_macro_excluding_raw0": 0.0}) == (0.0, -0.0, 0.0, 0.0)
    baseline = {"raw_50": {"dice": 0.9}, "raw_100": {"dice": 0.8}}
    candidate_ok = {"raw_50": {"dice": 0.86}, "raw_100": {"dice": 0.8}}
    candidate_bad = {"raw_50": {"dice": 0.7}, "raw_100": {"dice": 0.8}}
    guardrail = SelectionGuardrail(max_other_class_dice_drop=0.05, protected_classes=("raw_50", "raw_100"))
    assert passes_other_class_guardrail(candidate_ok, baseline, guardrail) is True
    assert passes_other_class_guardrail(candidate_bad, baseline, guardrail) is False
    missing = evaluate_other_class_guardrail({"raw_50": {"dice": 0.9}}, baseline, guardrail)
    assert missing["passed"] is False
    assert "raw_100" in missing["missingMetrics"]
    assert evaluate_other_class_guardrail({"raw_50": {"dice": 0.9}}, baseline, SelectionGuardrail(protected_classes=("raw_50", "raw_100"), fail_on_missing=False))["passed"] is True


def test_experiment_expansion_from_config() -> None:
    config = {
        "developmentSplits": ["train", "val"],
        "selectionPolicy": ["validation.dice_macro_foreground desc"],
        "guardrails": {"maxOtherClassDiceDrop": 0.05},
        "experiments": [
            {"id": "B0"},
            {"id": "B1", "raw0EffectiveWeightMultiplierGrid": [0.25, 1.0]},
            {"id": "B2", "maxClassWeightRatioGrid": [2.0]},
            {"id": "B3", "tversky": {"alpha": 0.7, "beta": 0.3}},
            {"id": "B4", "raw0Calibration": {"minProbabilityGrid": [0.5], "minMarginGrid": [0.1]}},
            {"id": "B5", "presenceHead": {"lambdaPresenceGrid": [0.25]}},
            {"id": "B6", "raw0BalancedSampler": {"positiveFractionGrid": [0.5]}},
        ],
    }
    validate_low_cost_grid(config)
    experiments = expand_low_cost_experiments(config)
    assert estimate_run_count(config) == len(experiments)
    assert len({item.experimentId for item in experiments}) == len(experiments)
    assert any(item.experimentId == "B0" for item in experiments)
