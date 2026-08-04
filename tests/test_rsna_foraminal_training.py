from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
AI_SERVICE = ROOT / "ai_service"
if AI_SERVICE.is_dir():
    sys.path.insert(0, str(AI_SERVICE))
else:
    # Allows running this copied test beside a standalone module during local validation.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfi_ai_service.training import rsna_foraminal_training as module  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(split: str, study_prefix: str) -> pd.DataFrame:
    rows = []
    classes = [("normal_mild", 0), ("moderate", 1), ("severe", 2)]
    for index, (severity, severity_code) in enumerate(classes):
        rows.append(
            {
                "study_id": f"{study_prefix}{index}",
                "side": "left" if index % 2 == 0 else "right",
                "level": module.LEVELS[index],
                "severity": severity,
                "severity_code": severity_code,
                "coordinate_series_id": f"series_{study_prefix}_{index}",
                "coordinate_instance_number": 10 + index,
                "coordinate_x": 128.0,
                "coordinate_y": 128.0,
                "coordinate_series_description": "Sagittal T1",
                "split": split,
                "internal_test_sealed": split == "internal_test",
                "human_review_required": True,
                "not_clinical_diagnosis": True,
                "official_test_accessed": False,
            }
        )
    return pd.DataFrame(rows)


def _write_split(tmp_path: Path, overlap: bool = False) -> Path:
    train = _frame("train", "train_")
    validation = _frame("validation", "validation_")
    if overlap:
        validation.loc[0, "study_id"] = train.loc[0, "study_id"]
    train_path = tmp_path / "train_manifest.csv"
    validation_path = tmp_path / "validation_manifest.csv"
    internal_path = tmp_path / "internal_test_manifest.csv"
    summary_path = tmp_path / "split_summary.json"
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    internal_path.write_bytes(b"sealed and deliberately not parsed")
    summary = {
        "approved": True,
        "nextNotebook": 59,
        "splits": {
            "train": {"rows": len(train)},
            "validation": {"rows": len(validation)},
        },
        "governance": {
            "internalTestSealed": True,
            "officialTestAccessed": False,
        },
        "gateResults": {"noStudyLeakage": True},
        "outputSha256": {
            "trainManifest": _sha(train_path),
            "validationManifest": _sha(validation_path),
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return tmp_path


def test_load_manifests_validates_hashes_and_keeps_test_sealed(tmp_path: Path):
    root = _write_split(tmp_path)
    train, validation, summary, hashes = module.load_manifests(root)

    assert len(train) == 3
    assert len(validation) == 3
    assert summary["approved"] is True
    assert hashes["train_manifest"] == _sha(root / "train_manifest.csv")
    assert set(train["severity"]) == set(module.CLASS_NAMES)
    assert "sample_id" in train.columns


def test_load_manifests_rejects_study_leakage(tmp_path: Path):
    root = _write_split(tmp_path, overlap=True)
    with pytest.raises(RuntimeError, match="Fuga train-validation"):
        module.load_manifests(root)


def test_sampling_audit_weights_rare_severe_more_than_common_normal():
    frame = pd.concat([_frame("train", "base_")] * 5, ignore_index=True)
    frame.loc[:, "study_id"] = [f"study_{index}" for index in range(len(frame))]
    frame.loc[:, "stratum"] = (
        frame["side"] + "__" + frame["level"] + "__" + frame["severity"]
    )
    frame = pd.concat(
        [frame, frame.loc[frame["severity"] == "normal_mild"].copy()],
        ignore_index=True,
    )
    weights, audit = module.sampling_audit(frame)

    severe_mean = weights[frame["severity_code"].to_numpy() == 2].mean()
    normal_mean = weights[frame["severity_code"].to_numpy() == 0].mean()
    assert severe_mean > normal_mean
    assert audit["classCounts"]["severe"] > 0


def test_crop_padding_and_uint8_stack_are_stable():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    crop = module._crop(image, x=0, y=0, size=6)
    stack = module._uint8_stack([crop, crop + 1, crop + 2])

    assert crop.shape == (6, 6)
    assert stack.shape == (3, 6, 6)
    assert stack.dtype == np.uint8
    assert int(stack.min()) >= 0
    assert int(stack.max()) <= 255


def test_metrics_are_perfect_for_perfect_predictions():
    targets = [0, 1, 2, 0, 1, 2]
    probabilities = np.eye(3, dtype=np.float64)[targets]
    result = module._metrics(targets, targets, probabilities)

    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["severe_recall"] == pytest.approx(1.0)
    assert result["moderate_recall"] == pytest.approx(1.0)
