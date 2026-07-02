from __future__ import annotations

from fastapi.testclient import TestClient

from pfi_ai_service.api import TRACE_ID_HEADER, app


def test_agent_report_summary_endpoint_returns_compact_traceability(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    client = TestClient(app)
    run_response = client.post(
        "/pipeline/run",
        headers={TRACE_ID_HEADER: "trace-summary-endpoint"},
        json={
            "caseId": "CASE-SUMMARY-001",
            "plane": "sagittal",
            "modelKey": "sagittal_spider",
            "inputPath": "demo/CASE-SUMMARY-001",
            "metadata": {"source": "summary-endpoint-test"},
        },
    )
    assert run_response.status_code == 200
    run_id = run_response.json()["runId"]

    summary_response = client.get(f"/agent/report/{run_id}/summary", headers={TRACE_ID_HEADER: "trace-summary-endpoint"})

    assert summary_response.status_code == 200
    assert summary_response.headers[TRACE_ID_HEADER] == "trace-summary-endpoint"
    summary = summary_response.json()
    assert summary["runId"] == run_id
    assert summary["traceId"] == "trace-summary-endpoint"
    assert summary["caseId"] == "CASE-SUMMARY-001"
    assert summary["modelKey"] == "sagittal_spider"
    assert summary["inferenceMode"] == "contract"
    assert summary["humanReviewRequired"] is True
    assert summary["notClinicalDiagnosis"] is True
    assert summary["diagnosisGenerated"] is False
    assert summary["measurementCount"] >= 1
