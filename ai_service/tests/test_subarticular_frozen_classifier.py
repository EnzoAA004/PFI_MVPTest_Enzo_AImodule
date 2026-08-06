from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from pfi_ai_service import subarticular_frozen_classifier as runtime
from pfi_ai_service.subarticular_frozen_classifier import (
    CheckpointHashMismatchError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    InvalidSubarticularInputError,
    SubarticularFrozenClassifier,
    SubarticularFrozenClassifierConfig,
    SubarticularRoi,
    preprocess_prepared_2p5d,
)


class _TinyBackbone(nn.Module):
    num_features = 3

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image.mean(dim=(2, 3))


def _install_fake_timm(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.SimpleNamespace(
        create_model=lambda *args, **kwargs: _TinyBackbone()
    )
    monkeypatch.setitem(sys.modules, "timm", fake)


def _checkpoint_payload() -> dict:
    cfg = runtime.RuntimeTrainConfig(pretrained=False, num_workers=0, batch_size=2)
    model = runtime.RuntimeSubarticularClassifierModel(cfg)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.01)
        model.head[-1].bias.copy_(torch.tensor([0.05, 0.15, 0.45]))
    return {
        "schemaVersion": "pfi.rsna-subarticular-training-checkpoint.v1",
        "modelStateDict": model.state_dict(),
        "epoch": 6,
        "config": asdict(cfg),
        "classNames": list(runtime.CLASS_NAMES),
        "displayClassNames": list(runtime.DISPLAY_CLASS_NAMES),
        "sideToIndex": dict(runtime.SIDE_TO_INDEX),
        "levelToIndex": dict(runtime.LEVEL_TO_INDEX),
        "task": "subarticular_stenosis_left_right",
        "sequence": "Axial T2",
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "officialTestAccessed": False,
        "internalTestAccessed": False,
    }


def _write_checkpoint(tmp_path: Path, payload: dict | None = None) -> tuple[Path, str]:
    path = tmp_path / "frozen_subarticular_checkpoint.pt"
    torch.save(payload or _checkpoint_payload(), path)
    return path, runtime.sha256_file(path)


def _classifier(path: Path, sha: str) -> SubarticularFrozenClassifier:
    config = SubarticularFrozenClassifierConfig(
        checkpoint_path=path,
        expected_checkpoint_sha256=sha,
        map_location="cpu",
    )
    return SubarticularFrozenClassifier(config)


def _sample() -> np.ndarray:
    return np.arange(3 * 224 * 224, dtype=np.float32).reshape(3, 224, 224) % 255


def test_loader_accepts_expected_hash_and_freezes_eval_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_timm(monkeypatch)
    path, sha = _write_checkpoint(tmp_path)

    classifier = _classifier(path, sha).load()

    assert classifier.is_loaded is True
    assert classifier.model_metadata()["checkpointSha256"] == sha
    assert classifier.model_metadata()["humanReviewRequired"] is True
    assert classifier.model_metadata()["notClinicalDiagnosis"] is True
    assert classifier.model_metadata()["autonomousDiagnosis"] is False
    assert classifier._model is not None
    assert classifier._model.training is False
    assert all(not parameter.requires_grad for parameter in classifier._model.parameters())


def test_loader_rejects_missing_hash_and_incompatible_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_timm(monkeypatch)
    missing = tmp_path / "missing.pt"
    with pytest.raises(CheckpointNotFoundError):
        _classifier(missing, "0" * 64).load()

    path, sha = _write_checkpoint(tmp_path)
    with pytest.raises(CheckpointHashMismatchError):
        _classifier(path, "1" * 64).load()

    bad_payload = _checkpoint_payload()
    bad_payload["task"] = "other"
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad_path, bad_sha = _write_checkpoint(bad_dir, bad_payload)
    with pytest.raises(CheckpointIncompatibleError):
        _classifier(bad_path, bad_sha).load()


def test_preprocessing_preserves_training_shape_dtype_and_determinism() -> None:
    sample = _sample()
    config = SubarticularFrozenClassifierConfig(expected_checkpoint_sha256="0" * 64)

    first = preprocess_prepared_2p5d(sample, config)
    second = preprocess_prepared_2p5d(sample.copy(), config)

    assert tuple(first.shape) == (3, 224, 224)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_preprocessing_rejects_invalid_shapes_channels_and_non_finite_values() -> None:
    config = SubarticularFrozenClassifierConfig(expected_checkpoint_sha256="0" * 64)
    with pytest.raises(InvalidSubarticularInputError, match="expected_three_dimensional"):
        preprocess_prepared_2p5d(np.zeros((224, 224), dtype=np.float32), config)
    with pytest.raises(InvalidSubarticularInputError, match="expected_three_input_channels"):
        preprocess_prepared_2p5d(np.zeros((2, 224, 224), dtype=np.float32), config)
    bad = np.zeros((3, 224, 224), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(InvalidSubarticularInputError, match="nan_or_infinite"):
        preprocess_prepared_2p5d(bad, config)


def test_predict_preprocessed_uses_inference_mode_and_outputs_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_timm(monkeypatch)
    path, sha = _write_checkpoint(tmp_path)
    classifier = _classifier(path, sha).load()
    called = {"inference_mode": False}
    original = runtime.torch.inference_mode

    def wrapped_inference_mode(*args, **kwargs):
        called["inference_mode"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.torch, "inference_mode", wrapped_inference_mode)

    prediction = classifier.predict_preprocessed(_sample(), side="right", level="L4-L5", source_position=7)

    assert called["inference_mode"] is True
    assert prediction.findingType == "subarticular_stenosis"
    assert set(prediction.probabilities) == set(runtime.CLASS_NAMES)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert prediction.predictedSeverity == max(prediction.probabilities, key=prediction.probabilities.get)
    assert prediction.humanReviewRequired is True
    assert prediction.notClinicalDiagnosis is True
    assert prediction.autonomousDiagnosis is False
    assert "diagnosis" not in json.dumps(prediction.degenerativeFindings).replace("notClinicalDiagnosis", "")


def test_contract_mapping_is_pfi_degenerative_findings_v1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_timm(monkeypatch)
    path, sha = _write_checkpoint(tmp_path)
    payload = _classifier(path, sha).load().predict_preprocessed(
        _sample(), side="left", level="L5-S1", source_position=3
    ).degenerativeFindings
    finding = payload["findings"][0]

    assert payload["schemaVersion"] == "pfi.degenerative-findings.v1"
    assert finding["findingType"] == "subarticular_stenosis"
    assert finding["classification"]["label"] in runtime.CLASS_NAMES
    assert finding["review"] == {"required": True, "status": "pending"}
    assert finding["notClinicalDiagnosis"] is True
    assert finding["sourceSeries"] == {"role": "axial_t2", "position": 3}
    assert finding["model"] == {"modelId": runtime.MODEL_ID, "modelSha256": sha}
    forbidden = {"SeriesInstanceUID", "StudyInstanceUID", "SOPInstanceUID", "PatientID", "patientId"}
    text = json.dumps(payload)
    assert not any(item in text for item in forbidden)


def test_predict_from_roi_validates_side_level_coordinates_and_preserves_safe_roi_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_timm(monkeypatch)
    path, sha = _write_checkpoint(tmp_path)
    classifier = _classifier(path, sha).load()
    files = tuple(tmp_path / f"{index}.dcm" for index in (8, 9, 10))
    monkeypatch.setattr(runtime, "_dicom_files", lambda _: files)
    monkeypatch.setattr(runtime, "_read_dicom", lambda _: np.ones((64, 64), dtype=np.float32))

    prediction = classifier.predict_from_roi(
        SubarticularRoi(
            series_path=tmp_path / "series",
            instance_number=9,
            x=32.0,
            y=31.0,
            side="left",
            level="L3-L4",
        )
    )

    finding = prediction.degenerativeFindings["findings"][0]
    assert prediction.roiSource == "operator_provided_external_coordinate"
    assert prediction.warnings == ("roi_requires_external_anatomical_coordinate",)
    assert finding["localization"] == {"source": "external_coordinate", "researchOnly": True}
    assert finding["anatomy"] == {"level": "L3-L4", "side": "left"}

    with pytest.raises(InvalidSubarticularInputError, match="roi_coordinates_out_of_range"):
        classifier.predict_from_roi(
            SubarticularRoi(tmp_path / "series", 9, 99.0, 31.0, "left", "L3-L4")
        )
    with pytest.raises(InvalidSubarticularInputError, match="invalid_side"):
        classifier.predict_from_roi(
            SubarticularRoi(tmp_path / "series", 9, 32.0, 31.0, "bilateral", "L3-L4")
        )
    with pytest.raises(InvalidSubarticularInputError, match="invalid_level"):
        classifier.predict_from_roi(
            SubarticularRoi(tmp_path / "series", 9, 32.0, 31.0, "left", "T12-L1")
        )


def test_cuda_request_fails_closed_when_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_timm(monkeypatch)
    path, sha = _write_checkpoint(tmp_path)
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    config = SubarticularFrozenClassifierConfig(
        checkpoint_path=path,
        expected_checkpoint_sha256=sha,
        map_location="cuda",
    )

    with pytest.raises(runtime.SubarticularClassifierError, match="cuda_requested_but_not_available"):
        SubarticularFrozenClassifier(config).load()


def test_loader_rejects_unexpected_checkpoint_schema_epoch_or_governance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_timm(monkeypatch)
    mutations = {
        "schemaVersion": "pfi.other.v1",
        "epoch": 7,
        "officialTestAccessed": True,
    }
    for key, value in mutations.items():
        payload = _checkpoint_payload()
        payload[key] = value
        case_dir = tmp_path / key
        case_dir.mkdir()
        path, sha = _write_checkpoint(case_dir, payload)
        with pytest.raises(CheckpointIncompatibleError):
            _classifier(path, sha).load()
