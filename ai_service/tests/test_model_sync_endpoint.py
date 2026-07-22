from __future__ import annotations

from fastapi.testclient import TestClient

from pfi_ai_service.api import app


def test_models_sync_endpoint_returns_release_materialization_payload(monkeypatch) -> None:
    from pfi_ai_service import api

    payload = {
        "status": "models_sync_completed",
        "items": [{
            "modelKey": "sagittal_spider",
            "source": "gcs_verified_release",
            "releaseId": "sagittal_spider_final_v1",
            "status": "existing_release_verified",
            "artifactSynced": True,
            "manifestSynced": True,
            "modelCardSynced": True,
            "filesReplaced": 0,
            "gcsReadOnly": True,
        }],
        "readyForRealInference": True,
        "defaultInferenceMode": "real_baseline",
    }
    monkeypatch.setattr(api, "sync_model_artifacts", lambda force=False: payload | {"force": force})

    response = TestClient(app).post("/models/sync?force=true")

    assert response.status_code == 200
    body = response.json()
    assert body["force"] is True
    assert body["items"][0]["source"] == "gcs_verified_release"
    assert body["items"][0]["gcsReadOnly"] is True
