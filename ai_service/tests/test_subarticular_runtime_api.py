from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from pfi_ai_service import subarticular_frozen_classifier as classifier_module
from pfi_ai_service import subarticular_runtime_service as service
from pfi_ai_service.api import app
from pfi_ai_service.input_registry import _INPUT_REGISTRY, register_existing_path


class _TinyBackbone(nn.Module):
    num_features = 3

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image.mean(dim=(2, 3))


def _install_fake_timm(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.SimpleNamespace(create_model=lambda *args, **kwargs: _TinyBackbone())
    monkeypatch.setitem(sys.modules, "timm", fake)


def _checkpoint_payload() -> dict:
    cfg = classifier_module.RuntimeTrainConfig(pretrained=False, num_workers=0, batch_size=2)
    model = classifier_module.RuntimeSubarticularClassifierModel(cfg)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.01)
        model.head[-1].bias.copy_(torch.tensor([0.02, 0.11, 0.42]))
    return {
        "schemaVersion": "pfi.rsna-subarticular-training-checkpoint.v1",
        "modelStateDict": model.state_dict(),
        "epoch": 6,
        "config": asdict(cfg),
        "classNames": list(classifier_module.CLASS_NAMES),
        "displayClassNames": list(classifier_module.DISPLAY_CLASS_NAMES),
        "sideToIndex": dict(classifier_module.SIDE_TO_INDEX),
        "levelToIndex": dict(classifier_module.LEVEL_TO_INDEX),
        "task": "subarticular_stenosis_left_right",
        "sequence": "Axial T2",
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "officialTestAccessed": False,
        "internalTestAccessed": False,
    }


def _write_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: dict | None = None) -> Path:
    _install_fake_timm(monkeypatch)
    path = tmp_path / "frozen_subarticular_checkpoint.pt"
    torch.save(payload or _checkpoint_payload(), path)
    sha = classifier_module.sha256_file(path)
    monkeypatch.setenv("PFI_SUBARTICULAR_CHECKPOINT_PATH", str(path))
    monkeypatch.setenv("PFI_SUBARTICULAR_DEVICE", "cpu")
    monkeypatch.setattr(service, "EXPECTED_CHECKPOINT_SHA256", sha)
    return path


def _register_axial_series(tmp_path: Path) -> str:
    series_dir = tmp_path / "axial-series"
    series_dir.mkdir()
    metadata = register_existing_path(
        case_id="CASE-SUBARTICULAR",
        plane="axial",
        path=series_dir,
        source_key="test-series",
        suffix=".dcm",
    )
    return str(metadata["inputId"])


@pytest.fixture(autouse=True)
def _clean_registry_and_cache(monkeypatch: pytest.MonkeyPatch):
    _INPUT_REGISTRY.clear()
    service.clear_subarticular_classifier_cache()
    monkeypatch.delenv("PFI_SUBARTICULAR_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("PFI_SUBARTICULAR_DEVICE", raising=False)
    yield
    _INPUT_REGISTRY.clear()
    service.clear_subarticular_classifier_cache()


def _patch_dicom(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shape: tuple[int, int] = (64, 64)) -> None:
    files = tuple(tmp_path / "axial-series" / f"{index}.dcm" for index in (8, 9, 10))
    monkeypatch.setattr(classifier_module, "_dicom_files", lambda _: files)
    monkeypatch.setattr(classifier_module, "_read_dicom", lambda _: np.ones(shape, dtype=np.float32))


def _assert_public_payload_has_no_internal_details(payload: dict) -> None:
    text = json.dumps(payload)
    forbidden = [
        "SeriesInstanceUID",
        "StudyInstanceUID",
        "SOPInstanceUID",
        "PatientID",
        "patientId",
        "C:/",
        "C:\\",
        "/tmp/",
        "/app/",
        "frozen_subarticular_checkpoint.pt",
        "Traceback",
    ]
    assert not any(item in text for item in forbidden)


def test_runtime_status_does_not_load_or_expose_checkpoint_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(monkeypatch, tmp_path)

    status = service.get_subarticular_runtime_status()

    assert status["configured"] is True
    assert status["artifactPresent"] is True
    assert status["loaded"] is False
    assert status["status"] == "available"
    assert status["humanReviewRequired"] is True
    assert status["notClinicalDiagnosis"] is True
    assert status["autonomousDiagnosis"] is False
    assert str(checkpoint) not in json.dumps(status)


def test_lazy_cache_and_clear(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_checkpoint(monkeypatch, tmp_path)

    first = service.get_subarticular_classifier()
    second = service.get_subarticular_classifier()
    assert first is second
    assert first.is_loaded is True
    assert all(not parameter.requires_grad for parameter in first._model.parameters())
    assert first._model.training is False

    service.clear_subarticular_classifier_cache()
    third = service.get_subarticular_classifier()
    assert third is not first


def test_runtime_reports_missing_and_hash_errors_without_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_SUBARTICULAR_CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    with pytest.raises(classifier_module.CheckpointNotFoundError):
        service.get_subarticular_classifier()
    status_code, code = service.public_error_status(classifier_module.CheckpointNotFoundError("x"))
    assert (status_code, code) == (503, "SUBARTICULAR_CHECKPOINT_UNAVAILABLE")

    _write_checkpoint(monkeypatch, tmp_path)
    monkeypatch.setattr(service, "EXPECTED_CHECKPOINT_SHA256", "0" * 64)
    with pytest.raises(classifier_module.CheckpointHashMismatchError):
        service.get_subarticular_classifier()


def test_health_readiness_models_and_runtime_publish_degenerative_model_without_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint(monkeypatch, tmp_path)
    client = TestClient(app)

    for route in ["/health", "/warmup", "/models", "/readiness", "/models/runtime"]:
        response = client.get(route)
        assert response.status_code == 200
        body = response.json()
        assert "degenerativeFindingModels" in body
        text = json.dumps(body)
        assert "rsna_subarticular_axial_t2_2p5d" in text
        assert str(checkpoint) not in text
        assert "segmentationModels" in body or route in {"/health", "/warmup", "/readiness"}
    assert service.get_subarticular_runtime_status()["loaded"] is False


def test_subarticular_endpoint_success_with_registered_axial_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_checkpoint(monkeypatch, tmp_path)
    input_id = _register_axial_series(tmp_path)
    _patch_dicom(monkeypatch, tmp_path)

    response = TestClient(app).post(
        "/degenerative-findings/subarticular/predict",
        json={
            "inputId": input_id,
            "instanceNumber": 9,
            "x": 32.0,
            "y": 31.0,
            "side": "left",
            "level": "L4-L5",
        },
    )

    assert response.status_code == 200
    body = response.json()
    finding = body["degenerativeFindings"]["findings"][0]
    assert body["degenerativeFindings"]["schemaVersion"] == "pfi.degenerative-findings.v1"
    assert finding["findingType"] == "subarticular_stenosis"
    assert finding["anatomy"] == {"level": "L4-L5", "side": "left"}
    assert finding["localization"] == {"source": "external_coordinate", "researchOnly": True}
    assert finding["review"] == {"required": True, "status": "pending"}
    assert finding["notClinicalDiagnosis"] is True
    assert body["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True
    assert body["autonomousDiagnosis"] is False
    assert body["warnings"] == ["roi_requires_external_anatomical_coordinate"]
    probabilities = finding["classification"]["probabilities"]
    assert all(np.isfinite(value) for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert finding["classification"]["label"] == max(probabilities, key=probabilities.get)
    _assert_public_payload_has_no_internal_details(body)


def test_subarticular_endpoint_rejects_unknown_non_axial_and_incompatible_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_checkpoint(monkeypatch, tmp_path)
    client = TestClient(app)
    payload = {"inputId": "missing", "instanceNumber": 1, "x": 1.0, "y": 1.0, "side": "left", "level": "L4-L5"}

    missing = client.post("/degenerative-findings/subarticular/predict", json=payload)
    assert missing.status_code == 404

    sagittal_dir = tmp_path / "sagittal"
    sagittal_dir.mkdir()
    sagittal = register_existing_path(case_id="CASE", plane="sagittal", path=sagittal_dir, source_key="test", suffix=".dcm")
    non_axial = client.post("/degenerative-findings/subarticular/predict", json={**payload, "inputId": sagittal["inputId"]})
    assert non_axial.status_code == 409

    axial_png = tmp_path / "axial.png"
    axial_png.write_bytes(b"not-a-dicom")
    incompatible = register_existing_path(case_id="CASE", plane="axial", path=axial_png, source_key="test", suffix=".png")
    response = client.post("/degenerative-findings/subarticular/predict", json={**payload, "inputId": incompatible["inputId"]})
    assert response.status_code == 422


def test_subarticular_endpoint_rejects_arbitrary_paths_and_invalid_roi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_checkpoint(monkeypatch, tmp_path)
    input_id = _register_axial_series(tmp_path)
    _patch_dicom(monkeypatch, tmp_path)
    client = TestClient(app)
    base = {"inputId": input_id, "instanceNumber": 9, "x": 32.0, "y": 31.0, "side": "left", "level": "L4-L5"}

    arbitrary_path = client.post("/degenerative-findings/subarticular/predict", json={**base, "inputPath": "C:/secret/input.dcm"})
    assert arbitrary_path.status_code == 422

    invalid_side = client.post("/degenerative-findings/subarticular/predict", json={**base, "side": "bilateral"})
    assert invalid_side.status_code == 400
    _assert_public_payload_has_no_internal_details(invalid_side.json())

    invalid_level = client.post("/degenerative-findings/subarticular/predict", json={**base, "level": "T12-L1"})
    assert invalid_level.status_code == 400

    out_of_range = client.post("/degenerative-findings/subarticular/predict", json={**base, "x": 900.0})
    assert out_of_range.status_code == 400


def test_subarticular_endpoint_returns_503_when_checkpoint_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFI_SUBARTICULAR_CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    input_id = _register_axial_series(tmp_path)

    response = TestClient(app).post(
        "/degenerative-findings/subarticular/predict",
        json={"inputId": input_id, "instanceNumber": 9, "x": 32.0, "y": 31.0, "side": "left", "level": "L4-L5"},
    )

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "SUBARTICULAR_CHECKPOINT_UNAVAILABLE"
    assert body["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True
    _assert_public_payload_has_no_internal_details(body)


@pytest.mark.real_checkpoint
def test_real_checkpoint_smoke_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = service._configured_checkpoint_path()
    if checkpoint is None:
        pytest.skip("PFI_SUBARTICULAR_CHECKPOINT_PATH no configurado")
    classifier = service.get_subarticular_classifier()
    prediction = classifier.predict_preprocessed(
        np.arange(3 * 224 * 224, dtype=np.float32).reshape(3, 224, 224) % 255,
        side="left",
        level="L4-L5",
        source_position=0,
    )
    assert prediction.degenerativeFindings["schemaVersion"] == "pfi.degenerative-findings.v1"
