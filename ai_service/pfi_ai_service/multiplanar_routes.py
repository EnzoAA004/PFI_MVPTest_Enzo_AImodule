from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from .input_registry import InputRegistryError
from .multiplanar_run import MultiplanarRunRequest
from .multiplanar_v2_executor import (
    CanonicalMultiplanarExecutor,
    LegacyMultiplanarV1Adapter,
    LegacyMultiplanarV1RequestMapper,
    MultiplanarV2Error,
    exception_to_v2,
    http_exception_to_v2,
    structured_validation_error,
)
from .multiplanar_v2_models import MultiplanarRunV2Request
from .reporting import write_json
from .security import safe_exception_type
from .settings import get_settings

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
            v2_request = LegacyMultiplanarV1RequestMapper().to_v2(traced_request, trace_id=trace_id)
            v2_response = CanonicalMultiplanarExecutor().run(v2_request, trace_id=trace_id, allow_unregistered_inputs=True)
            legacy_response = LegacyMultiplanarV1Adapter().from_v2(v2_response)
            write_json(get_settings().output_dir / "multiplanar_reports" / f"{v2_response.runId}.json", legacy_response)
            return legacy_response
        except MultiplanarV2Error as exc:
            return JSONResponse(status_code=exc.status_code, content=legacy_error_body(exc, http_request), headers={"X-Trace-Id": trace_id})
        except (HTTPException, InputRegistryError):
            raise
        except Exception as exc:
            LOGGER.error(
                "multiplanar_run_unhandled_exception",
                extra={
                    "traceId": trace_id,
                    "caseId": request.case_id,
                    "sagittalInputIdPresent": bool(request.sagittal_input_id),
                    "axialInputIdPresent": bool(request.axial_input_id),
                    "exceptionType": safe_exception_type(exc),
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
        try:
            payload = await http_request.json()
            if not isinstance(payload, dict):
                raise ValueError("body_must_be_object")
            request = MultiplanarRunV2Request.model_validate(payload)
        except ValidationError as exc:
            error = structured_validation_error(payload, exc, trace_id)
            return JSONResponse(status_code=error.status_code, content=error.body(), headers={"X-Trace-Id": trace_id})
        except (ValueError, ClientDisconnect) as exc:
            error = invalid_request_error(trace_id, details={"reason": type(exc).__name__})
            return JSONResponse(status_code=400, content=error.body(), headers={"X-Trace-Id": trace_id})
        try:
            response = CanonicalMultiplanarExecutor().run(request, trace_id=trace_id)
            return response.model_dump(mode="json")
        except MultiplanarV2Error as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.body(), headers={"X-Trace-Id": trace_id})
        except HTTPException as exc:
            error = http_exception_to_v2(exc, trace_id=trace_id, case_id=request.caseId, requested_planes=requested_planes(request))
            return JSONResponse(status_code=error.status_code, content=error.body(), headers={"X-Trace-Id": trace_id})
        except Exception as exc:
            LOGGER.error(
                "multiplanar_run_v2_unhandled_exception",
                extra={
                    "traceId": trace_id,
                    "caseId": request.caseId,
                    "exceptionType": safe_exception_type(exc),
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


def invalid_request_error(trace_id: str, *, details: dict[str, Any]) -> MultiplanarV2Error:
    return MultiplanarV2Error(
        "INVALID_MULTIPLANAR_REQUEST",
        "Request multiplanar v2 invalido.",
        status_code=400,
        trace_id=trace_id,
        case_id=None,
        requested_planes=[],
        details=details,
    )


def legacy_error_body(exc: MultiplanarV2Error, request: Request) -> dict[str, Any]:
    message = "Fallo interno controlado del AI Module" if exc.code == "REAL_INFERENCE_FAILED" else exc.message
    return {
        "status": "error",
        "code": exc.code,
        "message": message,
        "traceId": exc.trace_id,
        "path": str(request.url.path),
        "method": request.method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
    }


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
