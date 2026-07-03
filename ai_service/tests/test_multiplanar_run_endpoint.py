from fastapi.testclient import TestClient

from pfi_ai_service.api import app


def test_multiplanar_run_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("PFI_MODEL_DIR", str(tmp_path / "models"))
    client = TestClient(app)
    response = client.post(
        "/multiplanar/run",
        headers={"X-Trace-Id": "trace-multi-test"},
        json={"caseId": "CASE-TEST-002", "metadata": {"inferenceMode": "contract"}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "multiplanar_run_ready"
    assert body["traceId"] == "trace-multi-test"
    assert body["planes"]["sagittal"]["plane"] == "sagittal"
    assert body["planes"]["axial"]["plane"] == "axial"
