from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .input_registry import InputRegistryError
from .multiplanar_run import MultiplanarRunRequest, run_multiplanar_pipeline
from .multiplanar_v2_executor import (
    CanonicalMultiplanarExecutor,
    MultiplanarV2Error,
    exception_to_v2,
    http_exception_to_v2,
    structured_validation_error,
)
from .multiplanar_v2_models import MultiplanarRunV2Request

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

    @app.post("/v2/multiplanar/run")
    async def multiplanar_run_v2(http_request: Request):
        trace_id = request_trace_id(http_request)
        payload = await http_request.json()
        try:
            request = MultiplanarRunV2Request.model_validate(payload)
        except ValidationError as exc:
            error = structured_validation_error(payload, exc, trace_id)
            return JSONResponse(status_code=error.status_code, content=error.body(), headers={"X-Trace-Id": trace_id})
        trace_id = request.traceId or trace_id
        try:
            response = CanonicalMultiplanarExecutor().run(request, trace_id=trace_id)
            return response.model_dump(mode="json")
        except MultiplanarV2Error as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.body(), headers={"X-Trace-Id": trace_id})
        except HTTPException as exc:
            error = http_exception_to_v2(exc, trace_id=trace_id, case_id=request.caseId, requested_planes=requested_planes(request))
            return JSONResponse(status_code=error.status_code, content=error.body(), headers={"X-Trace-Id": trace_id})
        except Exception as exc:
            LOGGER.exception(
                "multiplanar_run_v2_unhandled_exception",
                extra={
                    "traceId": trace_id,
                    "caseId": request.caseId,
                    "exceptionType": type(exc).__name__,
                    "errorMessage": str(exc)[:240],
                },
            )
            error = exception_to_v2(exc, trace_id=trace_id, case_id=request.caseId, requested_planes=requested_planes(request))
            return JSONResponse(status_code=error.status_code, content=error.body(), headers={"X-Trace-Id": trace_id})


def request_trace_id(request: Request) -> str:
    value = getattr(request.state, "trace_id", None)
    if isinstance(value, str) and value.strip():
        return value
    header = request.headers.get("X-Trace-Id")
    return header.strip() if header and header.strip() else "unavailable"


def requested_planes(request: MultiplanarRunV2Request) -> list[str]:
    planes: list[str] = []
    if request.planes.sagittal is not None:
        planes.append("sagittal")
    if request.planes.axial is not None:
        planes.append("axial")
    return planes


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
