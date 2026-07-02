from __future__ import annotations

from fastapi.testclient import TestClient

from pfi_ai_service.api import TRACE_ID_HEADER, app, resolve_trace_id


def test_resolve_trace_id_sanitizes_and_limits_length() -> None:
    trace_id = resolve_trace_id(" demo trace! " * 20)

    assert " " not in trace_id
    assert "!" not in trace_id
    assert len(trace_id) <= 96


def test_health_keeps_incoming_trace_id_header() -> None:
    client = TestClient(app)

    response = client.get("/health", headers={TRACE_ID_HEADER: "demo-trace-123"})

    assert response.status_code == 200
    assert response.headers[TRACE_ID_HEADER] == "demo-trace-123"


def test_health_generates_trace_id_when_missing() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers[TRACE_ID_HEADER].startswith("trace-")
