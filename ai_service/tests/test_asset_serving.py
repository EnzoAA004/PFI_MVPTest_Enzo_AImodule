from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.asset_registry import clear_asset_registry
from pfi_ai_service.real_inference_runtime import clear_model_cache


client = TestClient(app)


def run_sagittal_fixture(monkeypatch, tmp_path) -> dict:
    fixture = Path("ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy")
    assert fixture.exists()
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_asset_registry()
    clear_model_cache()
    response = client.post(
        "/pipeline/run",
        json={
            "caseId": "CASE-AI016-ASSET-SERVE",
            "plane": "sagittal",
            "modelKey": "sagittal_spider",
            "inputPath": str(fixture),
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-ai016-assets",
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metadata"]["requestedInferenceMode"] == "real_baseline"
    assert body["metadata"]["runtime"] == "pytorch"
    assert set(body["assets"]) == {"input.png", "mask.npy", "confidence.npy", "overlay.png"}
    return body


def test_get_asset_serves_allowed_png(monkeypatch, tmp_path) -> None:
    body = run_sagittal_fixture(monkeypatch, tmp_path)

    response = client.get(f"/assets/{body['runId']}/sagittal/overlay.png")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/png")
    assert len(response.content) > 0


def test_get_asset_rejects_traversal_and_disallowed_asset(monkeypatch, tmp_path) -> None:
    body = run_sagittal_fixture(monkeypatch, tmp_path)

    traversal = client.get(f"/assets/{body['runId']}/sagittal/..%5Coverlay.png")
    not_allowed = client.get(f"/assets/{body['runId']}/sagittal/report.json")

    assert traversal.status_code == 403
    assert not_allowed.status_code == 403
    assert "outputs" not in traversal.text
    assert "outputs" not in not_allowed.text


def test_get_asset_returns_404_for_unknown_run_or_missing_asset(monkeypatch, tmp_path) -> None:
    body = run_sagittal_fixture(monkeypatch, tmp_path)

    unknown_run = client.get("/assets/missing-run/sagittal/input.png")
    missing_asset = client.get(f"/assets/{body['runId']}/sagittal/mask-preview.png")

    assert unknown_run.status_code == 404
    assert missing_asset.status_code == 404
    assert "outputs" not in unknown_run.text
    assert "outputs" not in missing_asset.text


def test_get_asset_does_not_serve_raw_arrays_or_model_artifacts(monkeypatch, tmp_path) -> None:
    body = run_sagittal_fixture(monkeypatch, tmp_path)

    mask_response = client.get(f"/assets/{body['runId']}/sagittal/mask.npy")
    confidence_response = client.get(f"/assets/{body['runId']}/sagittal/confidence.npy")
    pt_response = client.get(f"/assets/{body['runId']}/sagittal/model.pt")
    pth_response = client.get(f"/assets/{body['runId']}/sagittal/model.pth")

    assert mask_response.status_code == 403
    assert confidence_response.status_code == 403
    assert pt_response.status_code == 403
    assert pth_response.status_code == 403
    assert "application/octet-stream" not in mask_response.headers.get("content-type", "")
    assert "application/octet-stream" not in confidence_response.headers.get("content-type", "")
