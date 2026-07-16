from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .input_registry import InputRegistryError
from .multiplanar_routes import register_multiplanar_routes
from .real_inference_routes import register_real_inference_routes

TRACE_ID_HEADER = "X-Trace-Id"


def register_error_handlers(app: FastAPI) -> None:
    register_multiplanar_routes(app)
    register_real_inference_routes(app)

    @app.exception_handler(InputRegistryError)
    async def input_registry_exception_handler(request: Request, exc: InputRegistryError) -> JSONResponse:
        trace_id = trace_id_from_request(request)
        body: dict[str, Any] = {
            "status": "error",
            "code": code_for_status(exc.status_code),
            "message": exc.message,
            "traceId": trace_id,
            "path": request.url.path,
            "method": request.method,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
        }
        return JSONResponse(status_code=exc.status_code, content=body, headers={TRACE_ID_HEADER: trace_id})


    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        trace_id = trace_id_from_request(request)
        detail = exc.detail if isinstance(exc.detail, str) else "Error HTTP controlado"
        body: dict[str, Any] = {
            "status": "error",
            "code": code_for_status(exc.status_code),
            "message": detail,
            "traceId": trace_id,
            "path": request.url.path,
            "method": request.method,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
        }
        return JSONResponse(status_code=exc.status_code, content=body, headers={TRACE_ID_HEADER: trace_id})


def trace_id_from_request(request: Request) -> str:
    value = getattr(request.state, "trace_id", None)
    if isinstance(value, str) and value.strip():
        return value
    header = request.headers.get(TRACE_ID_HEADER)
    return header if header else "unavailable"


def code_for_status(status_code: int) -> str:
    if status_code == 400:
        return "BAD_REQUEST"
    if status_code == 401:
        return "UNAUTHORIZED"
    if status_code == 403:
        return "FORBIDDEN"
    if status_code == 404:
        return "NOT_FOUND"
    if status_code == 409:
        return "CONFLICT"
    if 400 <= status_code < 500:
        return "CLIENT_ERROR"
    return "AI_MODULE_ERROR"
