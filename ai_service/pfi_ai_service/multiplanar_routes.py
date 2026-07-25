from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .input_registry import InputRegistryError
from .multiplanar_run import MultiplanarRunRequest, run_multiplanar_pipeline

LOGGER = logging.getLogger(__name__)


def register_multiplanar_routes(app: FastAPI) -> None:
    @app.post("/multiplanar/run")
    def multiplanar_run(request: MultiplanarRunRequest, http_request: Request):
        trace_id = request_trace_id(http_request)
        metadata = dict(request.metadata or {})
        metadata.setdefault("traceId", trace_id)
        metadata.setdefault("aiTraceId", trace_id)
        metadata.setdefault("correlationId", trace_id)
        traced_request = request.model_copy(update={"metadata": metadata})
        try:
            return run_multiplanar_pipeline(traced_request)
        except (HTTPException, InputRegistryError):
            raise
        except Exception as exc:
            LOGGER.exception(
                "multiplanar_run_unhandled_exception",
                extra={
                    "traceId": trace_id,
                    "caseId": request.case_id,
                    "sagittalInputIdPresent": bool(request.sagittal_input_id),
                    "axialInputIdPresent": bool(request.axial_input_id),
                    "exceptionType": type(exc).__name__,
                    "errorMessage": str(exc)[:240],
                },
            )
            return JSONResponse(
                status_code=500,
                content=error_body(
                    status_code=500,
                    message="Fallo interno controlado del AI Module",
                    trace_id=trace_id,
                    path=str(http_request.url.path),
                    method=http_request.method,
                ),
                headers={"X-Trace-Id": trace_id},
            )


def request_trace_id(request: Request) -> str:
    value = getattr(request.state, "trace_id", None)
    if isinstance(value, str) and value.strip():
        return value
    header = request.headers.get("X-Trace-Id")
    return header.strip() if header and header.strip() else "unavailable"


def error_body(*, status_code: int, message: str, trace_id: str, path: str, method: str) -> dict[str, Any]:
    return {
        "status": "error",
        "code": "AI_MODULE_ERROR" if status_code >= 500 else "CLIENT_ERROR",
        "message": message,
        "traceId": trace_id,
        "path": path,
        "method": method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
    }
