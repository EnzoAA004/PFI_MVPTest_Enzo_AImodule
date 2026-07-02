from __future__ import annotations

from fastapi.testclient import TestClient

from pfi_ai_service.api import TRACE_ID_HEADER, app


def test_http_exception_uses_standard_error_shape(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    client = TestClient(app)

    response = client.get("/agent/report/missing-run", headers={TRACE_ID_HEADER: "demo-ai-error"})

    assert response.status_code == 404
    assert response.headers[TRACE_ID_HEADER] == "demo-ai-error"
    body = response.json()
    assert body["status"] == "error"
    assert body["code"] == "NOT_FOUND"
    assert body["traceId"] == "demo-ai-error"
    assert body["path"] == "/agent/report/missing-run"
    assert body["method"] == "GET"
    assert body["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True
    assert "No existe reporte" in body["message"]
