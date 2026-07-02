from __future__ import annotations

from fastapi.testclient import TestClient

from pfi_ai_service.api import TRACE_ID_HEADER, app


def test_pipeline_endpoint_attaches_trace_id_to_response_and_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    client = TestClient(app)

    response = client.post(
        "/pipeline/run",
        headers={TRACE_ID_HEADER: "demo-endpoint-trace"},
        json={
            "caseId": "CASE-TRACE-HTTP",
            "plane": "sagittal",
            "modelKey": "sagittal_spider",
            "inputPath": "demo/CASE-TRACE-HTTP",
            "metadata": {"source": "endpoint-test"},
        },
    )

    assert response.status_code == 200
    assert response.headers[TRACE_ID_HEADER] == "demo-endpoint-trace"
    body = response.json()
    assert body["traceId"] == "demo-endpoint-trace"
    assert body["metadata"]["traceId"] == "demo-endpoint-trace"
    assert body["metadata"]["aiTraceId"] == "demo-endpoint-trace"
    assert body["metadata"]["correlationId"] == "demo-endpoint-trace"
    assert body["metadata"]["source"] == "endpoint-test"
