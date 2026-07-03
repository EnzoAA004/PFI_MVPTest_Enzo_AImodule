from fastapi.testclient import TestClient

from pfi_ai_service.api import app


def test_real_inference_runtime_endpoint():
    response = TestClient(app).get("/models/runtime")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pytorch_runtime_ready"
    assert "torchVersion" in body
    assert body["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True
