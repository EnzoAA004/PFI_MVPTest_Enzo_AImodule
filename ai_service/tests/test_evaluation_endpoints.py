from __future__ import annotations

from fastapi.testclient import TestClient

from pfi_ai_service.api import app


def test_evaluation_summary_endpoint_returns_governance() -> None:
    client = TestClient(app)

    response = client.get("/evaluation/summary")

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "reportCount" in body
    assert body["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True


def test_evaluation_evidence_endpoint_returns_governance() -> None:
    client = TestClient(app)

    response = client.get("/evaluation/evidence")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "evidence_summary_ready"
    assert "latestRunId" in body
    assert body["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True
