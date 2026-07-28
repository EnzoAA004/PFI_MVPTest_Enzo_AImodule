from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.real_inference_runtime import clear_model_cache


def configure_local_final_models(monkeypatch, tmp_path: Path) -> None:
    models_root = Path("models/final").resolve()
    monkeypatch.setenv("PFI_MODEL_DIR", str(models_root))
    monkeypatch.setenv(
        "PFI_SAGITTAL_MODEL_PATH",
        str(models_root / "sagittal_spider_multiclass_final_best.pt"),
    )
    monkeypatch.setenv(
        "PFI_AXIAL_MODEL_PATH",
        str(models_root / "axial_t2_alkafri_final_v2_candidate.pt"),
    )
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()


def test_axial_candidate_is_qualified_for_strict_real_baseline(monkeypatch, tmp_path: Path) -> None:
    configure_local_final_models(monkeypatch, tmp_path)
    client = TestClient(app)

    models_response = client.get("/models")
    assert models_response.status_code == 200, models_response.text
    models_body = models_response.json()
    sagittal = models_body["models"]["sagittal_spider"]
    axial = models_body["models"]["axial_t2_alkafri"]

    assert sagittal["artifact"]["exists"] is True
    assert sagittal["manifest"]["valid"] is True
    assert sagittal["baselineReady"] is True
    assert sagittal["availableForRealInference"] is True
    assert sagittal["inferenceModes"]["real_baseline"] is True
    assert sagittal["readiness"] == "real_baseline_ready"

    assert axial["artifact"]["exists"] is True
    assert axial["manifest"]["trainingStatus"] == "candidate_below_quality_gate"
    assert axial["manifest"]["valid"] is True
    assert axial["baselineReady"] is False
    assert axial["manifestBaselineReady"] is False
    assert axial["availableForRealInference"] is True
    assert axial["readiness"] == "real_candidate_ready"
    assert axial["runtimeQualification"] == "axial_candidate_runtime_ready"
    assert axial["qualityGatePassed"] is False
    assert axial["trainingStatus"] == "candidate_below_quality_gate"
    assert axial["heldOutReuseWarning"]
    assert axial["qualityGate"]["heldOutReuseWarning"] == axial["heldOutReuseWarning"]
    assert axial["metrics"]["test"]["dice_macro_excluding_raw0"] >= 0.80
    assert axial["inferenceModes"]["real_baseline"] is True
    assert axial["manifest"]["content"]["artifactFile"] == "axial_t2_alkafri_final_v2_candidate.pt"

    verify_response = client.get("/models/verify")
    assert verify_response.status_code == 200, verify_response.text
    verify_body = verify_response.json()
    assert verify_body["status"] == "real_candidate_available"
    assert verify_body["baselineModelsReady"] == 1
    assert verify_body["artifactsAvailable"] == 2
    assert verify_body["readyForRealInference"] is True
    assert len(verify_body["runtimeCandidateModels"]) == 1
    assert verify_body["runtimeCandidateModels"][0]["modelKey"] == "axial_t2_alkafri"
    assert verify_body["runtimeCandidateModels"][0]["baselineReady"] is False
    assert verify_body["runtimeCandidateModels"][0]["qualityGatePassed"] is False

    sagittal_fixture = Path("ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy")
    sagittal_response = client.post(
        "/pipeline/run",
        json={
            "caseId": "CASE-PARTIAL-SAGITTAL",
            "plane": "sagittal",
            "modelKey": "sagittal_spider",
            "inputPath": str(sagittal_fixture),
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-partial-sagittal",
            },
        },
    )
    assert sagittal_response.status_code == 200, sagittal_response.text
    sagittal_body = sagittal_response.json()
    assert sagittal_body["aiOutput"]["inferenceMode"] == "real_baseline"
    assert sagittal_body["modelArtifact"]["baselineReady"] is True
    assert sagittal_body["modelArtifact"]["availableForRealInference"] is True

    axial_fixture = Path("ai_service/tests/fixtures/real_baseline/axial_sample_input.npy")
    axial_response = client.post(
        "/pipeline/run",
        json={
            "caseId": "CASE-PARTIAL-AXIAL",
            "plane": "axial",
            "modelKey": "axial_t2_alkafri",
            "inputPath": str(axial_fixture),
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-partial-axial",
            },
        },
    )
    assert axial_response.status_code == 200, axial_response.text
    axial_body = axial_response.json()
    assert axial_body["aiOutput"]["inferenceMode"] == "real_baseline"
    assert axial_body["modelArtifact"]["runtimeQualification"] == "axial_candidate_runtime_ready"
    assert axial_body["modelArtifact"]["baselineReady"] is False
    assert axial_body["modelArtifact"]["availableForRealInference"] is True
    assert axial_body["modelArtifact"]["readiness"] == "real_candidate_ready"
    assert axial_body["quality"]["maskCount"] > 0
    assert axial_body["quality"]["landmarkCount"] > 0
    assert axial_body["quality"]["measurementCount"] > 0


def test_multiplanar_readiness_flags_allow_sagittal_only_progress(monkeypatch, tmp_path: Path) -> None:
    configure_local_final_models(monkeypatch, tmp_path)
    client = TestClient(app)
    sagittal_fixture = Path("ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy")
    axial_fixture = Path("ai_service/tests/fixtures/real_baseline/axial_sample_input.npy")

    response = client.post(
        "/multiplanar/run",
        json={
            "caseId": "CASE-PARTIAL-MULTI",
            "sagittalInputPath": str(sagittal_fixture),
            "axialInputPath": str(axial_fixture),
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": True,
                "traceId": "trace-partial-multi",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sagittalRunReady"] is True
    assert body["axialRunReady"] is True
    assert body["dualRunReady"] is True
    assert body["effectiveInferenceMode"] == "real_baseline"
    assert body["planes"]["sagittal"]["aiOutput"]["inferenceMode"] == "real_baseline"
    assert body["planes"]["axial"]["aiOutput"]["inferenceMode"] == "real_baseline"
    assert body["planes"]["sagittal"]["synthetic"] is False
    assert body["planes"]["axial"]["synthetic"] is False
