from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.real_inference_runtime import clear_model_cache


def test_multiplanar_run_strict_real_baseline_with_real_fixtures(monkeypatch, tmp_path) -> None:
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
                "allowContractFallback": False,
                "traceId": "trace-ai012-multiplanar-fixture",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["runId"].startswith("multi-")
    assert body["metadata"]["multiplanarRunId"] == body["runId"]
    assert body["traceId"] == "trace-ai012-multiplanar-fixture"
    assert body["effectiveInferenceMode"] == "real_baseline"
    assert body["requestedInferenceMode"] == "real_baseline"

    sagittal = body["planes"]["sagittal"]
    axial = body["planes"]["axial"]
    assert sagittal["runId"]
    assert axial["runId"]
    assert sagittal["runId"] != axial["runId"]
    assert body["threeD"]["sourcePlaneRunIds"] == {
        "sagittal": sagittal["runId"],
        "axial": axial["runId"],
    }

    for plane_name, plane in (("sagittal", sagittal), ("axial", axial)):
        assert plane["aiOutput"]["inferenceMode"] == "real_baseline", plane_name
        assert plane["metadata"]["inferenceMode"] == "real_baseline", plane_name
        assert plane["traceId"] == "trace-ai012-multiplanar-fixture", plane_name
        flags = plane["aiOutput"]["agentDecision"].get("flags", [])
        assert "contract_fallback_after_real_inference_failure" not in flags
        assert all("contract_mode_used" not in flag for flag in flags)
        output_files = plane["metadata"]["outputFiles"]
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
