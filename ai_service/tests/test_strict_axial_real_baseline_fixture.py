from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.input_registry import InputRegistrationRequest, register_server_side_input
from pfi_ai_service.real_inference_runtime import clear_model_cache


def test_pipeline_run_strict_axial_real_baseline_fixture(monkeypatch, tmp_path) -> None:
    fixture = Path("ai_service/tests/fixtures/real_baseline/axial_sample_input.npy")
    assert fixture.exists()

    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()

    response = TestClient(app).post(
        "/pipeline/run",
        headers={"X-Trace-Id": "trace-ai008-axial-fixture"},
        json={
            "caseId": "CASE-AI008-AXIAL-FIXTURE",
            "plane": "axial",
            "modelKey": "axial_t2_alkafri",
            "inputPath": str(fixture),
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-ai008-axial-fixture",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["traceId"] == "trace-ai008-axial-fixture"
    assert body["plane"] == "axial"
    assert body["aiOutput"]["inferenceMode"] == "real_baseline"
    assert body["aiOutput"]["requestedInferenceMode"] == "real_baseline"
    assert body["metadata"]["inferenceMode"] == "real_baseline"
    assert body["synthetic"] is False
    assert body.get("fallbackReason") is None
    assert body["allowContractFallback"] is False
    assert body["agentDecision"]["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True
    assert body["modelKey"] == "axial_t2_alkafri"
    assert body["modelVersion"] == "axial-final-v2"
    assert body["artifactHash"] == "a48cbddd858b5615010fd809412f3d17dae6871fbe12a38f4720e6f6bc70f739"
    assert body["modelArtifact"]["availableForRealInference"] is True
    assert body["modelArtifact"]["baselineReady"] is False
    assert body["modelArtifact"]["manifestBaselineReady"] is False
    assert body["modelArtifact"]["readiness"] == "real_candidate_ready"
    assert body["modelArtifact"]["runtimeQualification"] == "axial_candidate_runtime_ready"
    assert body["modelArtifact"]["qualityGatePassed"] is False
    assert body["modelArtifact"]["trainingStatus"] == "candidate_below_quality_gate"
    assert body["modelArtifact"]["heldOutReuseWarning"]
    assert body["metadata"]["selectedSlice"] is not None
    assert body["metadata"]["sliceCount"] >= 1
    assert body["metadata"]["processedShape"] == [256, 256]
    assert body["quality"]["measurementsDerivedFromPredictionMask"] is True
    assert body["quality"]["maskCount"] > 0
    assert body["quality"]["landmarkCount"] > 0
    assert body["quality"]["measurementCount"] > 0
    assert body["masks"]
    assert body["landmarks"]
    assert body["measurementValues"]
    assert body["assets"]["overlay.png"]["assetName"] == "overlay.png"
    assert body["assets"]["overlay.png"]["size"] > 0
    assert body["assets"]["mask-preview.png"]["assetName"] == "mask-preview.png"
    assert body["assets"]["mask-preview.png"]["size"] > 0
    assert "inputPath" not in body
    assert "input_path" not in body
    assert "sourcePath" not in body["metadata"]
    assert "outputFiles" not in body["metadata"]
    assert "ai_service/tests/fixtures/real_baseline" not in response.text
    assert str(tmp_path) not in response.text


def test_strict_axial_real_baseline_rejects_model_plane_mismatch(monkeypatch, tmp_path) -> None:
    fixture = Path("ai_service/tests/fixtures/real_baseline/axial_sample_input.npy")
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()

    response = TestClient(app).post(
        "/pipeline/run",
        headers={"X-Trace-Id": "trace-ai008-axial-mismatch"},
        json={
            "caseId": "CASE-AI008-AXIAL-MISMATCH",
            "plane": "sagittal",
            "modelKey": "axial_t2_alkafri",
            "inputPath": str(fixture),
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-ai008-axial-mismatch",
            },
        },
    )

    assert response.status_code == 409
    body = response.json()
    assert body["traceId"] == "trace-ai008-axial-mismatch"
    assert "Modelo incompatible con plano real_baseline" in body["message"]


def test_axial_input_id_cannot_be_used_as_sagittal(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()
    input_metadata = register_server_side_input(
        InputRegistrationRequest(caseId="CASE-AI008-INPUT-PLANE", plane="axial", sourceKey="fixture:axial_sample")
    )

    response = TestClient(app).post(
        "/pipeline/run",
        headers={"X-Trace-Id": "trace-ai008-input-plane"},
        json={
            "caseId": "CASE-AI008-INPUT-PLANE",
            "plane": "sagittal",
            "modelKey": "sagittal_spider",
            "inputId": input_metadata["inputId"],
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-ai008-input-plane",
            },
        },
    )

    assert response.status_code == 409
    body = response.json()
    assert body["traceId"] == "trace-ai008-input-plane"
    assert body["message"] == "inputId no pertenece al plano solicitado"
