from __future__ import annotations

import numpy as np
import pytest

from lumbar_mri.axial_v3.audit import audit_raw0_slice, summarize_raw0_by_patient
from lumbar_mri.axial_v3.audit import write_audit_outputs
from lumbar_mri.axial_v3.calibration import apply_raw0_threshold, raw0_presence_metrics


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
    assert (tmp_path / "raw0_slice_audit.csv").exists()
    assert (tmp_path / "raw0_patient_audit.csv").exists()
    assert (tmp_path / "raw0_audit_summary.json").exists()


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
