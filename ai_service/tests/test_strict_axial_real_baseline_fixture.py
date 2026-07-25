from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.real_inference_runtime import clear_model_cache


def test_pipeline_run_rejects_strict_axial_candidate_real_baseline(monkeypatch, tmp_path) -> None:
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

    assert response.status_code == 409, response.text
    body = response.json()
    assert body["traceId"] == "trace-ai008-axial-fixture"
    assert "Modelo no habilitado para real_baseline: axial_t2_alkafri" in body["message"]
