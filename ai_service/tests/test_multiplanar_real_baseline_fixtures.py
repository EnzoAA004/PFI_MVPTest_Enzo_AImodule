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
    assert body["effectiveInferenceMode"] == "contract"
    assert body["requestedInferenceMode"] == "real_baseline"
    assert body["sagittalRunReady"] is True
    assert body["axialRunReady"] is True
    assert body["dualRunReady"] is True

    sagittal = body["planes"]["sagittal"]
    axial = body["planes"]["axial"]
    assert sagittal["runId"]
    assert axial["runId"]
    assert sagittal["runId"] != axial["runId"]
    assert body["threeD"]["sourcePlaneRunIds"] == {
        "sagittal": sagittal["runId"],
        "axial": axial["runId"],
    }

    assert sagittal["aiOutput"]["inferenceMode"] == "contract"
    assert sagittal["metadata"]["inferenceMode"] == "contract"
    assert sagittal["traceId"] == "trace-ai012-multiplanar-fixture"
    assert sagittal["synthetic"] is True
    assert sagittal["fallbackReason"] == "real_baseline_model_not_ready"
    output_files = sagittal["metadata"]["outputFiles"]
    assert not output_files

    assert axial["aiOutput"]["inferenceMode"] == "contract"
    assert axial["modelArtifact"]["baselineReady"] is False
    assert axial["modelArtifact"]["manifest"]["trainingStatus"] == "candidate_below_quality_gate"
