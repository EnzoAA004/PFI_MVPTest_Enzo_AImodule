from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.real_inference_runtime import clear_model_cache


def test_pipeline_run_strict_sagittal_real_baseline_fixture(monkeypatch, tmp_path) -> None:
    fixture = Path("ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy")
    assert fixture.exists()

    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()

    response = TestClient(app).post(
        "/pipeline/run",
        json={
            "caseId": "CASE-AI007-SAGITTAL-FIXTURE",
            "plane": "sagittal",
            "modelKey": "sagittal_spider",
            "inputPath": str(fixture),
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-ai007-sagittal-fixture",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["aiOutput"]["inferenceMode"] == "real_baseline"
    assert body["aiOutput"]["requestedInferenceMode"] == "real_baseline"
    assert body["metadata"]["inferenceMode"] == "real_baseline"
    assert body["runId"]
    assert body["traceId"] == "trace-ai007-sagittal-fixture"

    output_files = body["metadata"]["outputFiles"]
    expected = {
        "imagePath": "input.png",
        "maskPath": "mask.npy",
        "confidencePath": "confidence.npy",
        "overlayPath": "overlay.png",
    }
    for key, suffix in expected.items():
        output_path = Path(output_files[key])
        assert output_path.name == suffix
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    flags = body["aiOutput"]["agentDecision"].get("flags", [])
    assert "contract_fallback_after_real_inference_failure" not in flags
    assert all("contract_mode_used" not in flag for flag in flags)
