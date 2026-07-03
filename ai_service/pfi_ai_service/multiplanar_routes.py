from __future__ import annotations

from fastapi import FastAPI, Request

from .multiplanar_run import MultiplanarRunRequest, run_multiplanar_pipeline


def register_multiplanar_routes(app: FastAPI) -> None:
    @app.post("/multiplanar/run")
    def multiplanar_run(request: MultiplanarRunRequest, http_request: Request):
        trace_id = request_trace_id(http_request)
        metadata = dict(request.metadata or {})
        metadata.setdefault("traceId", trace_id)
        metadata.setdefault("aiTraceId", trace_id)
        metadata.setdefault("correlationId", trace_id)
        traced_request = request.model_copy(update={"metadata": metadata})
        return run_multiplanar_pipeline(traced_request)


def request_trace_id(request: Request) -> str:
    value = getattr(request.state, "trace_id", None)
    if isinstance(value, str) and value.strip():
        return value
    header = request.headers.get("X-Trace-Id")
    return header.strip() if header and header.strip() else "unavailable"
