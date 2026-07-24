from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lumbar_mri.axial_v3.audit import audit_raw0_slice, summarize_raw0_by_patient
from lumbar_mri.axial_v3.audit import write_audit_outputs
from lumbar_mri.axial_v3.architectures import ArchitectureConfig, build_axial_v3_model
from lumbar_mri.axial_v3.calibration import apply_raw0_slice_presence_gate, apply_raw0_threshold, raw0_metrics_from_predictions, raw0_presence_metrics
from lumbar_mri.axial_v3.labels import CLASS_INDEX_TO_NAME, RAW_TO_CLASS_INDEX, mask_to_class_indices, map_raw_mask_to_class_indices, validate_indexed_mask_labels, validate_raw_mask_labels
from lumbar_mri.axial_v3.iteration_a import AxialV3AuditConfig, optional_env_path, run_iteration_a, validate_v2_checkpoint_metadata
from lumbar_mri.axial_v3.registry import read_registry
from lumbar_mri.axial_v3.training import AxialV3TrainConfig, _load_run_report_per_class, run_calibration, sha256_file


def test_raw0_slice_audit_positive_negative_border_and_components() -> None:
    mask = np.zeros((6, 6), dtype=np.int64)
    mask[0, 1:3] = 1
    mask[4:6, 5] = 1
    mask[3, 3] = 2
    row = audit_raw0_slice(
        mask=mask,
        split="val",
        patient_id="p1",
        study_id="s1",
        slice_id="z003",
        raw0_value=1,
        background_value=0,
    )
    assert row.raw0Present is True
    assert row.raw0PixelCount == 4
    assert row.connectedComponents == 2
    assert row.largestComponentArea == 2
    assert row.largestComponentRatio == pytest.approx(0.5)
    assert row.minDistanceToBorder == 0
    assert row.touchesBorder is True
    assert row.boundingBox == {"minRow": 0, "minCol": 1, "maxRow": 5, "maxCol": 5}
    assert row.otherClassesPresent == [2]

    absent = audit_raw0_slice(
        mask=np.zeros((4, 4), dtype=np.int64),
        split="train",
        patient_id="p2",
        study_id="s1",
        slice_id="z001",
    )
    assert absent.raw0Present is False
    assert absent.connectedComponents == 0
    assert absent.minDistanceToBorder is None


def test_raw0_patient_summary() -> None:
    rows = [
        audit_raw0_slice(mask=np.eye(4, dtype=np.int64), split="train", patient_id="p1", study_id="s", slice_id="1"),
        audit_raw0_slice(mask=np.zeros((4, 4), dtype=np.int64), split="train", patient_id="p1", study_id="s", slice_id="2"),
    ]
    summary = summarize_raw0_by_patient(rows)
    assert summary[0]["patientId"] == "p1"
    assert summary[0]["sliceCount"] == 2
    assert summary[0]["raw0SliceCount"] == 1
    assert summary[0]["raw0SliceRatio"] == pytest.approx(0.5)


def test_write_audit_outputs(tmp_path) -> None:
    rows = [
        audit_raw0_slice(mask=np.eye(4, dtype=np.int64), split="val", patient_id="p1", study_id="s", slice_id="1")
    ]
    summary = write_audit_outputs(rows, tmp_path)
    assert summary["schemaVersion"] == "axial-v3-raw0-audit-v1"
    assert (tmp_path / "tables" / "raw0_slice_audit.csv").exists()
    assert (tmp_path / "tables" / "raw0_patient_audit.csv").exists()
    assert (tmp_path / "metrics" / "raw0_audit_summary.json").exists()


def test_raw0_threshold_reassigns_to_second_class_and_metrics() -> None:
    probs = np.zeros((3, 2, 2), dtype=np.float32)
    probs[1] = [[0.51, 0.9], [0.4, 0.2]]
    probs[2] = [[0.49, 0.05], [0.5, 0.7]]
    probs[0] = [[0.0, 0.05], [0.1, 0.1]]

    pred = apply_raw0_threshold(probs, raw0_index=1, min_probability=0.5, min_margin=0.05)
    assert pred[0, 0] == 2
    assert pred[0, 1] == 1
    assert pred[1, 0] == 2
    assert pred[1, 1] == 2

    target = np.array([[2, 1], [2, 2]], dtype=np.int64)
    metrics = raw0_presence_metrics(pred, target, raw0_index=1)
    assert metrics["truePositivePixels"] == 1
    assert metrics["falsePositivePixels"] == 0
    assert metrics["falseNegativePixels"] == 0
    assert metrics["predictedInGtAbsentCases"] == 0


def test_label_mapping_requires_explicit_mode() -> None:
    raw = np.array([[250, 0, 50], [100, 150, 200]], dtype=np.int16)
    mapped = map_raw_mask_to_class_indices(raw)
    assert mapped.tolist() == [[0, 1, 2], [3, 4, 5]]
    validate_raw_mask_labels(raw)
    validate_indexed_mask_labels(mapped)
    assert mask_to_class_indices(raw, mode="raw")[0, 1] == 1
    assert mask_to_class_indices(mapped, mode="indexed")[0, 1] == 1
    with pytest.raises(ValueError, match="unexpected raw"):
        validate_raw_mask_labels(np.array([[999]]))
    with pytest.raises(ValueError, match="unknown mask label mode"):
        mask_to_class_indices(raw, mode="auto")


def test_optional_checkpoint_env_empty_is_none(monkeypatch) -> None:
    monkeypatch.delenv("AXIAL_V2_CHECKPOINT_PATH", raising=False)
    assert optional_env_path("AXIAL_V2_CHECKPOINT_PATH") is None
    cfg = AxialV3AuditConfig()
    assert cfg.V2_CHECKPOINT_PATH is None


def test_v2_checkpoint_metadata_is_strict_by_default() -> None:
    incomplete = {"model_state_dict": {}, "runId": "axial-final-v2"}
    cfg = AxialV3AuditConfig(PFI_ALLOW_INCOMPLETE_V2_CHECKPOINT_METADATA=False)
    with pytest.raises(ValueError, match="metadata is incomplete"):
        validate_v2_checkpoint_metadata(incomplete, cfg)
    permissive = AxialV3AuditConfig(PFI_ALLOW_INCOMPLETE_V2_CHECKPOINT_METADATA=True)
    assert "smokeOnly" in validate_v2_checkpoint_metadata(incomplete, permissive)["missingMetadataFields"]


def test_v2_checkpoint_nested_label_mapping_validates(tmp_path: Path) -> None:
    manifest = tmp_path / "split.csv"
    manifest.write_text("image_file_path,final_label_file_path,case_id_norm,split\n", encoding="utf-8")
    checkpoint = {
        "model_state_dict": {},
        "runId": "axial-final-v2",
        "smokeOnly": False,
        "num_classes": 6,
        "base_channels": 16,
        "target_size": (256, 256),
        "monitorMetric": "dice_macro_foreground",
        "raw0Boost": 1.0,
        "architecture": "AxialUNet2D",
        "aiServiceCommit": "285159abcdef",
        "splitSha256": sha256_file(manifest),
        "rawToClassIndex": RAW_TO_CLASS_INDEX,
        "labelMapping": {
            "rawToClassIndex": RAW_TO_CLASS_INDEX,
            "classIndexToName": CLASS_INDEX_TO_NAME,
        },
        "preprocessingConfig": {"targetSize": (256, 256)},
    }
    cfg = AxialV3AuditConfig(SPLIT_MANIFEST_PATH=manifest)
    result = validate_v2_checkpoint_metadata(checkpoint, cfg)
    assert "labelMapping" in result["validatedMetadataFields"]


def test_guardrail_per_class_prefers_gated_metrics(tmp_path: Path) -> None:
    report = tmp_path / "run_report.json"
    report.write_text(
        json.dumps(
            {
                "finalValidationMetrics": {
                    "perClass": {"raw_50": {"dice": 0.1}},
                    "presenceGatedSegmentation": {"perClass": {"raw_50": {"dice": 0.9}}},
                }
            }
        ),
        encoding="utf-8",
    )
    row = {"artifactPath": str(report), "validationGatedDiceMacroForeground": 0.8}
    assert _load_run_report_per_class(row)["raw_50"]["dice"] == 0.9
    assert _load_run_report_per_class({**row, "validationGatedDiceMacroForeground": None})["raw_50"]["dice"] == 0.1


def test_raw0_threshold_batch_and_presence_gate() -> None:
    probs = np.zeros((2, 3, 2, 2), dtype=np.float32)
    probs[:, 1] = 0.6
    probs[:, 2] = 0.4
    gated = apply_raw0_threshold(probs, min_probability=0.5, min_margin=0.3)
    assert gated.shape == (2, 2, 2)
    assert np.all(gated == 2)
    presence_gated = apply_raw0_slice_presence_gate(probs, np.array([0.2, 0.9]), 0.5)
    assert np.all(presence_gated[0] == 2)
    assert np.all(presence_gated[1] == 1)


def test_raw0_metrics_counts_cases_per_slice() -> None:
    pred = np.zeros((2, 3, 3), dtype=np.int64)
    target = np.zeros((2, 3, 3), dtype=np.int64)
    pred[0, 0, 0] = 1
    target[1, 0, 0] = 1
    raw0 = raw0_metrics_from_predictions(pred, target)
    assert raw0["evaluableCases"] == 2
    assert raw0["gtPresentCases"] == 1
    assert raw0["gtAbsentCases"] == 1
    assert raw0["predPresentCases"] == 1
    assert raw0["predictedInGtAbsentCases"] == 1


def test_iteration_a_writes_review_outputs_and_real_previews(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = []
    for split, patient, raw0 in [("train", "p1", True), ("train", "p2", False), ("val", "p3", True), ("val", "p4", False)]:
        image = np.linspace(0, 1, 144, dtype=np.float32).reshape(12, 12)
        mask = np.full((8, 8), 250, dtype=np.uint8)
        if raw0:
            mask[0:2, 1:3] = 0
        image_path = data_root / f"{patient}_img.npy"
        mask_path = data_root / f"{patient}_mask.npy"
        np.save(image_path, image)
        np.save(mask_path, mask)
        rows.append(
            {
                "image_file_path": image_path.name,
                "final_label_file_path": mask_path.name,
                "case_id_norm": patient,
                "study_id": patient,
                "split": split,
            }
        )
    manifest = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    cfg = AxialV3AuditConfig(
        RUN_ID="synthetic-a",
        DATASET_ROOT=data_root,
        SPLIT_MANIFEST_PATH=manifest,
        OUTPUT_ROOT=tmp_path / "outputs",
        MAX_PREVIEW_CASES=2,
        V2_CHECKPOINT_PATH=None,
    )
    result = run_iteration_a(cfg)
    output_dir = cfg.OUTPUT_ROOT / cfg.RUN_ID
    assert result["status"] == "ready_outputs_created"
    assert (output_dir / "tables" / "raw0_human_review_candidates.csv").exists()
    preview = output_dir / "previews" / "raw0_positive_examples.png"
    assert preview.exists()
    assert preview.stat().st_size > 1000
    probability_summary = json.loads((output_dir / "metrics" / "raw0_validation_probability_summary.json").read_text(encoding="utf-8"))
    assert probability_summary["status"] == "skipped"
    manifest_rows = json.loads((output_dir / "manifests" / "iteration_a_artifacts.json").read_text(encoding="utf-8"))
    assert any(item["relativePath"] == "tables/raw0_human_review_candidates.csv" for item in manifest_rows)


def test_run_calibration_registers_b4_without_smoke_name_error(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = []
    for split, patient, raw0 in [("train", "p1", True), ("train", "p2", False), ("val", "p3", True), ("val", "p4", False)]:
        image = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        mask = np.full((8, 8), 250, dtype=np.uint8)
        if raw0:
            mask[2:4, 2:4] = 0
        image_path = data_root / f"{patient}_img.npy"
        mask_path = data_root / f"{patient}_mask.npy"
        np.save(image_path, image)
        np.save(mask_path, mask)
        rows.append(
            {
                "image_file_path": image_path.name,
                "final_label_file_path": mask_path.name,
                "case_id_norm": patient,
                "study_id": patient,
                "split": split,
            }
        )
    manifest = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    parent = tmp_path / "parent.pt"
    model = build_axial_v3_model(ArchitectureConfig(base_channels=2))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "experimentId": "B0",
            "runId": "axial-v3-B0",
            "smokeOnly": False,
            "splitSha256": sha256_file(manifest),
            "config": {"BASE_CHANNELS": 2},
        },
        parent,
    )
    cfg = AxialV3TrainConfig(
        RUN_ID="axial-v3-B4-test",
        EXPERIMENT_ID="B4-test",
        EXPERIMENT_TYPE="B4",
        DATASET_ROOT=data_root,
        SPLIT_MANIFEST_PATH=manifest,
        OUTPUT_ROOT=tmp_path / "outputs",
        REGISTRY_PATH=tmp_path / "registry.csv",
        BASE_CHANNELS=2,
        BATCH_SIZE=1,
        MIN_PROBABILITY=0.5,
        MIN_MARGIN=0.0,
        PARENT_EXPERIMENT_ID="B0",
        PARENT_RUN_ID="axial-v3-B0",
    )
    result = run_calibration(cfg, parent_checkpoint_path=parent)
    assert result["status"] == "calibration_completed"
    assert (cfg.OUTPUT_ROOT / "axial-v3-B4-test" / "reports" / "calibration_report.json").exists()
    registry = read_registry(cfg.REGISTRY_PATH)
    assert registry[0]["trainingStatus"] == "completed"
    assert registry[0]["smokeOnly"] is False
