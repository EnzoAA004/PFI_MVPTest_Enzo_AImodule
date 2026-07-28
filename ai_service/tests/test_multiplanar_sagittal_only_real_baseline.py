from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.real_inference_runtime import clear_model_cache


def configure_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()


def test_sagittal_only_input_id_does_not_call_axial_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_runtime(monkeypatch, tmp_path)
    calls = []

    def fake_run_pipeline(request):
        calls.append(request)
        return {
            "runId": "sag-run-1",
            "traceId": request.metadata["traceId"],
            "plane": request.plane,
            "inputId": request.input_id,
            "aiOutput": {
                "inferenceMode": "real_baseline",
                "requestedInferenceMode": "real_baseline",
            },
            "modelArtifact": {
                "baselineReady": True,
                "availableForRealInference": True,
            },
            "quality": {
                "maskCount": 2,
                "landmarkCount": 2,
                "measurementCount": 3,
            },
            "metadata": request.metadata,
        }

    monkeypatch.setattr("pfi_ai_service.multiplanar_run.run_pipeline", fake_run_pipeline)

    response = TestClient(app).post(
        "/multiplanar/run",
        headers={"X-Trace-Id": "trace-sag-only-mock"},
        json={
            "caseId": "CASE-SAG-ONLY-MOCK",
            "sagittalInputId": "inp_sag_mock",
            "sagittalModelKey": "sagittal_spider",
            "axialModelKey": "axial_t2_alkafri",
            "metadata": {
                "inferenceMode": "real_baseline",
                "requestedInferenceMode": "real_baseline",
                "allowContractFallback": False,
                "axialMode": "optional_not_provided",
            },
        },
    )

    assert response.status_code == 200, response.text
    assert len(calls) == 1
    assert calls[0].plane == "sagittal"
    assert calls[0].input_id == "inp_sag_mock"
    assert calls[0].metadata["allowContractFallback"] is False
    body = response.json()
    assert body["effectiveInferenceMode"] == "real_baseline"
    assert body["sagittalRunReady"] is True
    assert body["axialRunReady"] is False
    assert body["dualRunReady"] is False
    assert body["planes"]["axial"] is None
    assert body["threeD"]["status"] == "blocked_missing_axial"
    assert body["threeD"]["enabled"] is False
    assert body["threeD"]["sourcePlaneRunIds"]["axial"] is None
    assert body["quality"]["planeCount"] == 1
    report = tmp_path / "outputs" / "multiplanar_reports" / f"{body['runId']}.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["planes"]["axial"] is None


def test_sagittal_only_uploaded_mha_input_id_runs_real_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sitk = pytest.importorskip("SimpleITK")
    configure_runtime(monkeypatch, tmp_path)
    client = TestClient(app)
    case_id = "CASE-SAG-ONLY-MHA"
    mha_path = tmp_path / "sagittal_fixture.mha"
    array = np.random.default_rng(2026).normal(size=(9, 64, 64)).astype(np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.2, 1.2, 3.0))
    sitk.WriteImage(image, str(mha_path))

    with mha_path.open("rb") as handle:
        upload = client.post(
            "/inputs",
            data={"caseId": case_id, "plane": "sagittal"},
            files={"file": ("sagittal_fixture.mha", handle, "application/octet-stream")},
        )
    assert upload.status_code == 200, upload.text
    input_id = upload.json()["inputId"]

    response = client.post(
        "/multiplanar/run",
        headers={"X-Trace-Id": "trace-sag-only-mha"},
        json={
            "caseId": case_id,
            "sagittalInputId": input_id,
            "sagittalModelKey": "sagittal_spider",
            "axialModelKey": "axial_t2_alkafri",
            "metadata": {
                "inferenceMode": "real_baseline",
                "requestedInferenceMode": "real_baseline",
                "allowContractFallback": False,
                "axialMode": "optional_not_provided",
                "traceId": "trace-sag-only-mha",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    sagittal = body["planes"]["sagittal"]
    assert body["effectiveInferenceMode"] == "real_baseline"
    assert body["sagittalRunReady"] is True
    assert body["axialRunReady"] is False
    assert body["dualRunReady"] is False
    assert body["planes"]["axial"] is None
    assert body["threeD"]["status"] == "blocked_missing_axial"
    assert sagittal["inputId"] == input_id
    assert sagittal["aiOutput"]["inferenceMode"] == "real_baseline"
    assert sagittal["metadata"]["allowContractFallback"] is False
    assert sagittal["metadata"]["inputFormat"] == ".mha"
    assert sagittal["metadata"]["outputFiles"]["maskPath"] == {"generated": True, "fileName": "mask.npy"}
    assert sagittal["metadata"]["outputFiles"]["overlayPath"] == {"generated": True, "fileName": "overlay.png"}


def test_multiplanar_with_axial_input_runs_dual_real_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_runtime(monkeypatch, tmp_path)
    client = TestClient(app)
    case_id = "CASE-STRICT-AXIAL-PROVIDED"
    sagittal_id = client.post(
        "/inputs",
        json={"caseId": case_id, "plane": "sagittal", "sourceKey": "fixture:sagittal_sample"},
    ).json()["inputId"]
    axial_id = client.post(
        "/inputs",
        json={"caseId": case_id, "plane": "axial", "sourceKey": "fixture:axial_sample"},
    ).json()["inputId"]

    response = client.post(
        "/multiplanar/run",
        headers={"X-Trace-Id": "trace-strict-axial-provided"},
        json={
            "caseId": case_id,
            "sagittalInputId": sagittal_id,
            "axialInputId": axial_id,
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-strict-axial-provided",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["traceId"] == "trace-strict-axial-provided"
    assert body["effectiveInferenceMode"] == "real_baseline"
    assert body["sagittalRunReady"] is True
    assert body["axialRunReady"] is True
    assert body["dualRunReady"] is True
    assert body["planes"]["sagittal"]["aiOutput"]["inferenceMode"] == "real_baseline"
    assert body["planes"]["axial"]["aiOutput"]["inferenceMode"] == "real_baseline"
    assert body["planes"]["axial"]["quality"]["maskCount"] > 0
    assert body["planes"]["axial"]["quality"]["landmarkCount"] > 0
    assert body["planes"]["axial"]["quality"]["measurementCount"] > 0


def test_sagittal_real_failure_without_fallback_returns_controlled_500(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_runtime(monkeypatch, tmp_path)

    def fail_run_pipeline(request):
        raise RuntimeError("synthetic sagittal failure with internal/path/hidden")

    monkeypatch.setattr("pfi_ai_service.multiplanar_run.run_pipeline", fail_run_pipeline)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/multiplanar/run",
        headers={"X-Trace-Id": "trace-sag-failure"},
        json={
            "caseId": "CASE-SAG-FAILURE",
            "sagittalInputId": "inp_sag_failure",
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
            },
        },
    )

    assert response.status_code == 500
    body = response.json()
    assert body["traceId"] == "trace-sag-failure"
    assert body["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True
    assert body["message"] == "Fallo interno controlado del AI Module"
    assert "internal/path" not in json.dumps(body)
