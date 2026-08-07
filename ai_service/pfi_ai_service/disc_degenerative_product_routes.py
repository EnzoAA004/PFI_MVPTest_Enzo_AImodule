from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from .disc_degenerative_product_runtime import (
    ProductDiscMultitaskPredictRequest,
    predict_product_disc_degenerative,
    product_runtime_status,
)
from .disc_degenerative_runtime import (
    DiscDegenerativeRuntimeError,
    public_error_message,
    public_error_status,
)
from .full_series_segmentation import (
    FullSeriesSegmentationError,
    FullSeriesSegmentationRequest,
    resolve_full_series_asset,
    run_full_series_segmentation,
)
from .security import sanitize_public_payload

TRACE_ID_HEADER = "X-Trace-Id"


def _trace_id(request: Request) -> str:
    value = getattr(request.state, "trace_id", None)
    if isinstance(value, str) and value.strip():
        return value
    return request.headers.get(TRACE_ID_HEADER) or "unavailable"


def _series_error(exc: FullSeriesSegmentationError, request: Request) -> JSONResponse:
    trace_id = _trace_id(request)
    return JSONResponse(
        status_code=exc.status_code,
        headers={TRACE_ID_HEADER: trace_id},
        content=sanitize_public_payload(
            {
                "status": "error",
                "code": "FULL_SERIES_SEGMENTATION_ERROR",
                "message": exc.message,
                "traceId": trace_id,
                "path": request.url.path,
                "method": request.method,
                "humanReviewRequired": True,
                "notClinicalDiagnosis": True,
                "autonomousDiagnosis": False,
            }
        ),
    )


def register_disc_degenerative_product_routes(app: FastAPI) -> None:
    @app.get("/v2/degenerative-findings/disc-multitask/runtime")
    def disc_multitask_product_runtime():
        return sanitize_public_payload(product_runtime_status())

    @app.post("/v2/degenerative-findings/disc-multitask/predict")
    def disc_multitask_product_predict(
        payload: ProductDiscMultitaskPredictRequest,
        request: Request,
    ):
        try:
            return sanitize_public_payload(predict_product_disc_degenerative(payload))
        except DiscDegenerativeRuntimeError as exc:
            status_code, code = public_error_status(exc)
            trace_id = _trace_id(request)
            return JSONResponse(
                status_code=status_code,
                headers={TRACE_ID_HEADER: trace_id},
                content=sanitize_public_payload(
                    {
                        "status": "error",
                        "code": code,
                        "message": public_error_message(code),
                        "traceId": trace_id,
                        "path": request.url.path,
                        "method": request.method,
                        "humanReviewRequired": True,
                        "notClinicalDiagnosis": True,
                        "autonomousDiagnosis": False,
                    }
                ),
            )

    @app.post("/v2/series-segmentation/run")
    def full_series_segmentation_run(
        payload: FullSeriesSegmentationRequest,
        request: Request,
    ):
        try:
            return sanitize_public_payload(run_full_series_segmentation(payload))
        except FullSeriesSegmentationError as exc:
            return _series_error(exc, request)

    @app.get(
        "/v2/series-segmentation/{run_id}/{plane}/slices/{index}/{asset_name}"
    )
    def full_series_segmentation_asset(
        run_id: str,
        plane: str,
        index: int,
        asset_name: str,
        request: Request,
    ):
        try:
            path = resolve_full_series_asset(run_id, plane, index, asset_name)
        except FullSeriesSegmentationError as exc:
            return _series_error(exc, request)
        return FileResponse(
            path,
            media_type="image/png",
            filename=asset_name,
        )
