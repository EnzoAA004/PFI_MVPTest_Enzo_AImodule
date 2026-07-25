from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.real_inference_runtime import clear_model_cache


def test_multiplanar_run_allows_partial_real_baseline_with_axial_candidate(monkeypatch, tmp_path) -> None:
    sagittal_fixture = Path("ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy")
    axial_fixture = Path("ai_service/tests/fixtures/real_baseline/axial_sample_input.npy")
    assert sagittal_fixture.exists()
    assert axial_fixture.exists()

    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()

    response = TestClient(app).post(
        "/multiplanar/run",
        headers={"X-Trace-Id": "trace-ai012-multiplanar-fixture"},
        json={
            "caseId": "CASE-AI012-MULTI-FIXTURE",
            "sagittalInputPath": str(sagittal_fixture),
            "axialInputPath": str(axial_fixture),
            "sagittalModelKey": "sagittal_spider",
            "axialModelKey": "axial_t2_alkafri",
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": True,
                "traceId": "trace-ai012-multiplanar-fixture",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["runId"].startswith("multi-")
    assert body["metadata"]["multiplanarRunId"] == body["runId"]
    assert body["traceId"] == "trace-ai012-multiplanar-fixture"
    assert body["effectiveInferenceMode"] == "mixed"
    assert body["requestedInferenceMode"] == "real_baseline"
    assert body["sagittalRunReady"] is True
    assert body["axialRunReady"] is False
    assert body["dualRunReady"] is False

    sagittal = body["planes"]["sagittal"]
    axial = body["planes"]["axial"]
    assert sagittal["runId"]
    assert axial["runId"]
    assert sagittal["runId"] != axial["runId"]
    assert body["threeD"]["sourcePlaneRunIds"] == {
        "sagittal": sagittal["runId"],
        "axial": axial["runId"],
    }

    assert sagittal["aiOutput"]["inferenceMode"] == "real_baseline"
    assert sagittal["metadata"]["inferenceMode"] == "real_baseline"
    assert sagittal["traceId"] == "trace-ai012-multiplanar-fixture"
    flags = sagittal["aiOutput"]["agentDecision"].get("flags", [])
    assert "contract_fallback_after_real_inference_failure" not in flags
    assert all("contract_mode_used" not in flag for flag in flags)
    output_files = sagittal["metadata"]["outputFiles"]
    for key, suffix in {
        "imagePath": "input.png",
        "maskPath": "mask.npy",
        "confidencePath": "confidence.npy",
        "overlayPath": "overlay.png",
    }.items():
        output_path = Path(output_files[key])
        assert output_path.name == suffix
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    assert axial["aiOutput"]["inferenceMode"] == "contract"
    assert axial["modelArtifact"]["baselineReady"] is False
    assert axial["modelArtifact"]["manifest"]["trainingStatus"] == "candidate_below_quality_gate"
