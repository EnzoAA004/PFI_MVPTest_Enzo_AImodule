from __future__ import annotations

import logging
from hashlib import sha256
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .axial_level import axial_slice_level, axial_slice_levels
from .input_registry import InputRegistryError, register_existing_path, resolve_input_id
from .model_artifacts import model_status
from .multiplanar_3d_reconstruction import build_lumbar_3d_status
from . import multiplanar_run as legacy_multiplanar_module
from .pipeline import PipelineRunRequest
from .reporting import write_json
from .settings import MODEL_REGISTRY, get_settings
from .multiplanar_v2_models import (
    CoordinateSpaceV2,
    GovernanceV2,
    MultiplanarReadinessV2,
    MultiplanarRunV2Request,
    MultiplanarRunV2Response,
    PlaneAssetV2,
    PlaneExecutionV2Request,
    PlaneInputV2,
    PlaneLandmarkV2,
    PlaneMaskV2,
    PlaneMeasurementV2,
    PlaneModelV2,
    PlaneNameV2,
    PlaneQualityV2,
    PlaneSegmentationV2,
    SegmentationInstanceV2,
    PlaneRunV2Result,
    PlaneSeriesV2,
    ReviewPolicyV2,
    StructuredAiErrorV2,
    ThreeDStatusV2,
    WorkspaceQualityV2,
    WorkspaceModeV2,
)
from .multiplanar_run import MultiplanarRunRequest

log = logging.getLogger(__name__)

MODEL_NOT_READY_MESSAGE = "Modelo no habilitado para real_baseline"


def run_pipeline(request: PipelineRunRequest) -> dict[str, Any]:
    return legacy_multiplanar_module.run_pipeline(request)


class MultiplanarV2Error(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        trace_id: str,
        case_id: str | None,
        requested_planes: list[PlaneNameV2] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.trace_id = trace_id
        self.case_id = case_id
        self.requested_planes = requested_planes or []
        self.details = details or {}

    def body(self) -> dict[str, Any]:
        return StructuredAiErrorV2(
            status="error",
            schemaVersion="pfi.error.v2",
            code=self.code,  # type: ignore[arg-type]
            message=self.message,
            traceId=self.trace_id,
            caseId=self.case_id,
            requestedPlanes=self.requested_planes,
            details=self.details,
            governance=GovernanceV2(
                humanReviewRequired=HUMAN_REVIEW_REQUIRED,
                notClinicalDiagnosis=NOT_CLINICAL_DIAGNOSIS,
            ),
        ).model_dump(mode="json")


def structured_validation_error(payload: Any, exc: ValidationError, trace_id: str) -> MultiplanarV2Error:
    case_id = payload.get("caseId") if isinstance(payload, dict) else None
    requested = requested_planes_from_payload(payload)
    no_plane = "NO_PLANE_REQUESTED" in str(exc)
    return MultiplanarV2Error(
        "NO_PLANE_REQUESTED" if no_plane else "INVALID_MULTIPLANAR_REQUEST",
        "Debe solicitar al menos un plano." if no_plane else "Request multiplanar v2 invalido.",
        status_code=400,
        trace_id=trace_id,
        case_id=str(case_id) if case_id else None,
        requested_planes=requested,
        details={"validation": scrub_validation_errors(exc.errors())},
    )


def scrub_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scrubbed = []
    for error in errors:
        clean = {key: value for key, value in error.items() if key not in {"input", "url", "ctx"}}
        scrubbed.append(clean)
    return scrubbed


def requested_planes_from_payload(payload: Any) -> list[PlaneNameV2]:
    if not isinstance(payload, dict) or not isinstance(payload.get("planes"), dict):
        return []
    planes: list[PlaneNameV2] = []
    if payload["planes"].get("sagittal") is not None:
        planes.append("sagittal")
    if payload["planes"].get("axial") is not None:
        planes.append("axial")
    return planes


class CanonicalMultiplanarExecutor:
    def run(
        self,
        request: MultiplanarRunV2Request,
        *,
        trace_id: str,
        allow_unregistered_inputs: bool = False,
    ) -> MultiplanarRunV2Response:
        requested_planes = planes_requested(request)
        workspace_mode = workspace_mode_for(requested_planes)
        run_id = deterministic_multiplanar_run_id(request, requested_planes)
        preflight = self._preflight(
            request,
            trace_id,
            requested_planes,
            allow_global_fallback=request.allowContractFallback,
            allow_unregistered_inputs=allow_unregistered_inputs,
        )

        plane_results: dict[PlaneNameV2, PlaneRunV2Result | None] = {"sagittal": None, "axial": None}
        if request.inferenceMode == "real_baseline" and request.allowContractFallback and any(
            not item["model"].get("availableForRealInference") for item in preflight.values()
        ):
            reason = sanitize_fallback_reason("real_baseline_model_not_ready")
            for plane in requested_planes:
                plane_spec = getattr(request.planes, plane)
                assert plane_spec is not None
                plane_results[plane] = synthetic_plane_result(plane, plane_spec, preflight[plane], "contract", reason)
            return self._persist_response(request, trace_id, run_id, requested_planes, workspace_mode, plane_results, preflight)

        if request.inferenceMode in {"contract", "mock"}:
            for plane in requested_planes:
                plane_spec = getattr(request.planes, plane)
                assert plane_spec is not None
                plane_results[plane] = synthetic_plane_result(
                    plane,
                    plane_spec,
                    preflight[plane],
                    request.inferenceMode,
                    f"explicit_{request.inferenceMode}_mode",
                )
            return self._persist_response(request, trace_id, run_id, requested_planes, workspace_mode, plane_results, preflight)

        for plane in requested_planes:
            plane_spec = getattr(request.planes, plane)
            assert plane_spec is not None
            try:
                pipeline_response = run_pipeline(self._pipeline_request(request, plane, plane_spec, run_id, trace_id, workspace_mode))
                plane_results[plane] = normalize_plane_result_v2(
                    plane,
                    plane_spec,
                    pipeline_response,
                    preflight[plane],
                    strict=not allow_unregistered_inputs,
                )
            except MultiplanarV2Error:
                raise
            except Exception as exc:
                if not request.allowContractFallback:
                    raise MultiplanarV2Error(
                        "REAL_INFERENCE_FAILED",
                        "Fallo de inferencia real baseline.",
                        status_code=500,
                        trace_id=trace_id,
                        case_id=request.caseId,
                        requested_planes=requested_planes,
                        details={"plane": plane, "type": type(exc).__name__},
                    ) from exc
                # El unico rastro que sobrevivia de una inferencia real fallida era el
                # nombre de la clase de la excepcion, dentro de un `fallbackReason`
                # sanitizado. Con eso no se diagnostica nada: la corrida se degrada a modo
                # contract, responde 200 y el motivo real se pierde. El log es interno y
                # no sale al cliente, asi que aca si va la traza completa.
                log.exception(
                    "event=real_inference_failed plane=%s caseId=%s traceId=%s runId=%s "
                    "action=degraded_to_contract",
                    plane, request.caseId, trace_id, run_id,
                )
                reason = sanitize_fallback_reason(f"real_baseline_failed:{type(exc).__name__}")
                for fallback_plane in requested_planes:
                    fallback_spec = getattr(request.planes, fallback_plane)
                    assert fallback_spec is not None
                    plane_results[fallback_plane] = synthetic_plane_result(
                        fallback_plane,
                        fallback_spec,
                        preflight[fallback_plane],
                        "contract",
                        reason,
                    )
                break
        assign_axial_levels(plane_results)
        return self._persist_response(request, trace_id, run_id, requested_planes, workspace_mode, plane_results, preflight)

    def _persist_response(
        self,
        request: MultiplanarRunV2Request,
        trace_id: str,
        run_id: str,
        requested_planes: list[PlaneNameV2],
        workspace_mode: WorkspaceModeV2,
        plane_results: dict[PlaneNameV2, PlaneRunV2Result | None],
        preflight: dict[PlaneNameV2, dict[str, Any]],
    ) -> MultiplanarRunV2Response:
        response = build_workspace_response(
            request=request,
            trace_id=trace_id,
            run_id=run_id,
            requested_planes=requested_planes,
            workspace_mode=workspace_mode,
            planes=plane_results,
            readiness=workspace_readiness(preflight),
        )
        write_json(get_settings().output_dir / "multiplanar_reports_v2" / f"{run_id}.json", response.model_dump(mode="json"))
        return response

    def _preflight(
        self,
        request: MultiplanarRunV2Request,
        trace_id: str,
        requested_planes: list[PlaneNameV2],
        *,
        allow_global_fallback: bool,
        allow_unregistered_inputs: bool,
    ) -> dict[PlaneNameV2, dict[str, Any]]:
        preflight: dict[PlaneNameV2, dict[str, Any]] = {}
        for plane in requested_planes:
            plane_spec = getattr(request.planes, plane)
            assert plane_spec is not None
            model_info = MODEL_REGISTRY.get(plane_spec.modelKey)
            if model_info is None:
                raise MultiplanarV2Error(
                    "MODEL_NOT_FOUND",
                    f"Modelo no registrado: {plane_spec.modelKey}",
                    status_code=404,
                    trace_id=trace_id,
                    case_id=request.caseId,
                    requested_planes=requested_planes,
                    details={"plane": plane, "modelKey": plane_spec.modelKey},
                )
            if model_info.get("plane") != plane:
                raise MultiplanarV2Error(
                    "MODEL_PLANE_MISMATCH",
                    "El modelKey no corresponde al plano solicitado.",
                    status_code=409,
                    trace_id=trace_id,
                    case_id=request.caseId,
                    requested_planes=requested_planes,
                    details={"plane": plane, "modelKey": plane_spec.modelKey, "expectedPlane": model_info.get("plane")},
                )
            input_record = None
            try:
                input_record = resolve_input_id(plane_spec.inputId, case_id=request.caseId, plane=plane)
            except InputRegistryError as exc:
                if request.inferenceMode == "real_baseline" and not allow_unregistered_inputs:
                    raise MultiplanarV2Error(
                        "INPUT_NOT_FOUND",
                        exc.message,
                        status_code=exc.status_code,
                        trace_id=trace_id,
                        case_id=request.caseId,
                        requested_planes=requested_planes,
                        details={"plane": plane, "inputId": plane_spec.inputId},
                    ) from exc

            artifact = model_status(plane_spec.modelKey, dict(model_info))
            if request.inferenceMode == "real_baseline" and not artifact.get("availableForRealInference"):
                if allow_global_fallback:
                    preflight[plane] = {"model": artifact, "input": input_record}
                    continue
                raise MultiplanarV2Error(
                    "MODEL_NOT_READY",
                    f"{MODEL_NOT_READY_MESSAGE}: {plane_spec.modelKey}",
                    status_code=409,
                    trace_id=trace_id,
                    case_id=request.caseId,
                    requested_planes=requested_planes,
                    details={
                        "plane": plane,
                        "modelKey": plane_spec.modelKey,
                        "readiness": artifact.get("readiness"),
                        "trainingStatus": artifact.get("trainingStatus"),
                    },
                )
            preflight[plane] = {"model": artifact, "input": input_record}
        return preflight

    def _pipeline_request(
        self,
        request: MultiplanarRunV2Request,
        plane: PlaneNameV2,
        plane_spec: PlaneExecutionV2Request,
        run_id: str,
        trace_id: str,
        workspace_mode: WorkspaceModeV2,
    ) -> PipelineRunRequest:
        options = request.options.model_dump(mode="json")
        metadata = {
            "traceId": trace_id,
            "correlationId": trace_id,
            "multiplanarRunId": run_id,
            "workspaceMode": workspace_mode,
            "workspacePlane": plane,
            "inferenceMode": request.inferenceMode,
            "requestedInferenceMode": request.inferenceMode,
            "allowContractFallback": request.allowContractFallback,
            **{key: value for key, value in options.items() if value is not None},
        }
        return PipelineRunRequest(
            caseId=request.caseId,
            plane=plane,
            modelKey=plane_spec.modelKey,
            inputId=plane_spec.inputId,
            inputPath=None,
            metadata=metadata,
        )


def planes_requested(request: MultiplanarRunV2Request) -> list[PlaneNameV2]:
    planes: list[PlaneNameV2] = []
    if request.planes.sagittal is not None:
        planes.append("sagittal")
    if request.planes.axial is not None:
        planes.append("axial")
    return planes


def workspace_mode_for(planes: list[PlaneNameV2]) -> WorkspaceModeV2:
    if planes == ["sagittal"]:
        return "sagittal_only"
    if planes == ["axial"]:
        return "axial_only"
    return "dual_plane"


def deterministic_multiplanar_run_id(request: MultiplanarRunV2Request, planes: list[PlaneNameV2]) -> str:
    parts = [request.caseId, request.inferenceMode]
    for plane in planes:
        spec = getattr(request.planes, plane)
        assert spec is not None
        parts.extend([plane, spec.inputId, spec.modelKey])
    return "multi-" + sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def normalize_plane_result_v2(
    plane: PlaneNameV2,
    plane_spec: PlaneExecutionV2Request,
    response: dict[str, Any],
    preflight: dict[str, Any],
    *,
    strict: bool = True,
) -> PlaneRunV2Result:
    metadata = response.get("metadata") if isinstance(response.get("metadata"), dict) else {}
    model = plane_model_v2(preflight["model"])
    selected_slice = int_or_none(metadata.get("selectedSlice"))
    selected_axis = int_or_none(metadata.get("selectedAxis"))
    native_shape = int_list_or_none(metadata.get("inputShapeNative"))
    canonical_shape = int_list_or_none(metadata.get("inputShapeCanonical"))
    processed_shape = int_list_or_none(metadata.get("processedShape"))
    height, width = shape_to_height_width(processed_shape or canonical_shape)
    if width <= 0 or height <= 0:
        if not strict:
            width, height = 1, 1
        else:
            raise MultiplanarV2Error(
                "INVALID_MULTIPLANAR_RESPONSE",
                "Respuesta real_baseline sin coordinateSpace valido.",
                status_code=500,
                trace_id=str(response.get("traceId") or metadata.get("traceId") or "unavailable"),
                case_id=str(response.get("caseId") or response.get("case_id") or ""),
                requested_planes=[plane],
                details={"plane": plane, "reason": "invalid_processed_shape"},
            )
    coordinate_space = CoordinateSpaceV2(
        name=f"model_{width}x{height}",
        width=width,
        height=height,
        units="pixel",
        origin="top_left",
        xDirection="right",
        yDirection="down",
        sourceSliceIndex=selected_slice,
        sourceAxis=selected_axis,
    )
    quality_raw = response.get("quality")
    quality = plane_quality_v2(quality_raw)
    segmentation = plane_segmentation_v2((quality_raw or {}).get("segmentation") if isinstance(quality_raw, dict) else None)
    return PlaneRunV2Result(
        status="ready",
        plane=plane,
        runId=required_text(response.get("runId") or response.get("run_id"), "planeRunId", plane),
        effectiveInferenceMode=normalize_inference_mode(response.get("inferenceMode") or metadata.get("inferenceMode")),
        model=model,
        input=PlaneInputV2(
            inputId=plane_spec.inputId,
            format=strip_dot(metadata.get("inputFormat")),
            sizeBytes=int_or_none(metadata.get("inputSize")) or getattr(preflight.get("input"), "size", None),
            nativeShape=native_shape,
            canonicalShape=canonical_shape,
            orientationTransform=metadata.get("inputOrientationTransform"),
            spacingXyzMm=float_list_or_none(metadata.get("spacingXyz")),
            canonicalAxisSpacingMm=float_list_or_none(metadata.get("arrayAxisSpacingCanonical")),
            selectedSliceIndex=selected_slice,
            sliceCount=int_or_none(metadata.get("sliceCount")),
            selectedAxis=selected_axis,
            inPlaneSpacingMm=float_list_or_none(metadata.get("inPlaneSpacing")),
        ),
        coordinateSpace=coordinate_space,
        series=series_v2(plane, response.get("series"), selected_slice, strict=strict),
        assets=assets_v2(plane, str(response.get("runId") or response.get("run_id")), metadata),
        masks=masks_v2(plane, response.get("masks"), coordinate_space.name),
        landmarks=landmarks_v2(plane, response.get("landmarks"), coordinate_space.name),
        measurements=measurements_v2(plane, response.get("measurementValues") or (response.get("measurements") or {}).get("values"), metadata),
        segmentation=segmentation,
        quality=quality,
        synthetic=False,
        fallbackReason=None,
    )


def plane_model_v2(artifact: dict[str, Any]) -> PlaneModelV2:
    manifest = artifact.get("manifest") if isinstance(artifact.get("manifest"), dict) else {}
    return PlaneModelV2(
        key=str(artifact.get("key")),
        version=artifact.get("version"),
        readiness=str(artifact.get("readiness")),
        trainingStatus=artifact.get("trainingStatus") or manifest.get("trainingStatus"),
        artifactHash=artifact.get("artifactHash"),
        baselineReady=bool(artifact.get("baselineReady")),
        availableForRealInference=bool(artifact.get("availableForRealInference")),
        runtimeQualification=artifact.get("runtimeQualification"),
        qualityGatePassed=artifact.get("qualityGatePassed"),
        manifestStatus=manifest.get("status"),
        manifestValid=bool(manifest.get("valid")),
    )


def series_v2(plane: PlaneNameV2, raw: Any, selected_slice: int | None, *, strict: bool = True) -> list[PlaneSeriesV2]:
    source = raw if isinstance(raw, list) else []
    series: list[PlaneSeriesV2] = []
    for item in source:
        if not isinstance(item, dict) or item.get("plane") != plane:
            continue
        series.append(PlaneSeriesV2(
            id=required_text(item.get("id"), "series.id", plane),
            plane=plane,
            sequence=None,
            selectedSliceIndex=int_or_none(item.get("selectedSlice")) if item.get("selectedSlice") is not None else selected_slice,
            sliceCount=int_or_none(item.get("sliceCount")),
            status="served_single_slice" if item.get("status") == "real_baseline_ready" else str(item.get("status") or "served_single_slice"),
        ))
    if not series:
        if not strict:
            return []
        raise MultiplanarV2Error(
            "INVALID_MULTIPLANAR_RESPONSE",
            "Respuesta real_baseline sin serie procesada.",
            status_code=500,
            trace_id="unavailable",
            case_id=None,
            requested_planes=[plane],
            details={"plane": plane, "reason": "missing_real_series"},
        )
    return series


def assets_v2(plane: PlaneNameV2, run_id: str, metadata: dict[str, Any]) -> list[PlaneAssetV2]:
    names = {
        "input.png": ("input_preview", "image/png"),
        "overlay.png": ("overlay", "image/png"),
        "mask.npy": ("mask_array", "application/octet-stream"),
        "confidence.npy": ("confidence_array", "application/octet-stream"),
        "mask-preview.png": ("mask_preview", "image/png"),
    }
    generated: set[str] = set()
    output_files = metadata.get("outputFiles") if isinstance(metadata.get("outputFiles"), dict) else {}
    for value in output_files.values():
        if isinstance(value, dict) and value.get("generated") and value.get("fileName"):
            generated.add(str(value["fileName"]))
    assets = metadata.get("assets") if isinstance(metadata.get("assets"), dict) else {}
    for name in assets:
        generated.add(str(name))
    return [
        PlaneAssetV2(
            assetName=name,
            role=role,  # type: ignore[arg-type]
            contentType=content_type,
            generated=True,
            relativePath=f"/assets/{run_id}/{plane}/{name}",
        )
        for name, (role, content_type) in names.items()
        if name in generated
    ]


def masks_v2(plane: PlaneNameV2, raw: Any, coordinate_space: str) -> list[PlaneMaskV2]:
    masks = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        class_key = required_text(item.get("className") or item.get("label"), "mask.classKey", plane)
        masks.append(PlaneMaskV2(
            id=required_text(item.get("id"), "mask.id", plane),
            classKey=class_key,
            classId=int_or_none(item.get("classId")),
            level=text_or_none(item.get("level")),
            color=text_or_none(item.get("color")),
            confidence=float_or_none(item.get("confidence")),
            enabled=bool(item.get("enabled", True)),
            # El contorno viaja, asi que la mascara deja de ser una imagen fija y
            # pasa a ser una propuesta que el revisor puede corregir.
            editable=True,
            coordinateSpace=coordinate_space,
            geometry=mask_geometry(item.get("contours")),
        ))
    return masks


def landmarks_v2(plane: PlaneNameV2, raw: Any, coordinate_space: str) -> list[PlaneLandmarkV2]:
    landmarks = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or item.get("x") is None or item.get("y") is None:
            continue
        landmarks.append(PlaneLandmarkV2(
            id=required_text(item.get("id"), "landmark.id", plane),
            labelKey=required_text(item.get("label"), "landmark.labelKey", plane),
            x=float(item["x"]),
            y=float(item["y"]),
            confidence=float_or_none(item.get("confidence")),
            coordinateSpace=coordinate_space,
            source="derived_from_mask",
        ))
    return landmarks


def measurements_v2(plane: PlaneNameV2, raw: Any, metadata: dict[str, Any]) -> list[PlaneMeasurementV2]:
    physical = bool(metadata.get("inPlaneSpacing"))
    values = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or item.get("value") is None:
            continue
        unit = str(item.get("unit") or ("mm" if physical else "px")).replace("²", "2")
        if not physical and unit in {"mm", "mm2"}:
            unit = "px2" if unit == "mm2" else "px"
        values.append(PlaneMeasurementV2(
            id=required_text(item.get("id"), "measurement.id", plane),
            labelKey=required_text(item.get("label"), "measurement.labelKey", plane),
            value=float(item["value"]),
            unit=unit,
            confidence=float_or_none(item.get("confidence")),
            source="ai",
            status="pending_review",
            plane=plane,
            # El runtime asigna el nivel lumbar por disco (L1-L2 ... L5-S1) cuando
            # el encuadre lo permite, y deja None cuando no puede afirmarlo. Fijarlo
            # en None aca descartaba esa asignacion y todas las mediciones llegaban
            # al revisor agrupadas como "sin nivel".
            level=text_or_none(item.get("level")),
            levelScope="study" if item.get("levelScope") == "study" else "level",
            points=measurement_points(item.get("points")),
            experimental=bool(item.get("experimental", False)),
            detail=text_or_none(item.get("detail")),
            sliceIndex=int_or_none(item.get("sliceIndex")),
            measurementBasis="physical_spacing" if physical else "pixel_space",
            linkedLandmarkIds=list_string(item.get("linkedLandmarks")),
        ))
    return values


def measurement_points(raw: Any) -> list[dict[str, float]]:
    """Los puntos que definen la figura de una medicion, o ninguno.

    La cantidad depende de que se midio: dos una distancia, tres una listesis, cuatro
    un angulo. Un punto a medias no viaja: dibujar una figura con un vertice inventado
    ubicaria la medicion sobre anatomia que no es la que se midio, que es peor que no
    dibujarla. Por eso se exige que sobrevivan todos o ninguno.
    """
    if not isinstance(raw, list) or not 2 <= len(raw) <= 4:
        return []
    points = [
        {"x": float(point["x"]), "y": float(point["y"])}
        for point in raw
        if isinstance(point, dict) and isinstance(point.get("x"), (int, float)) and isinstance(point.get("y"), (int, float))
    ]
    return points if len(points) == len(raw) else []


def mask_geometry(contours: Any) -> dict[str, Any] | None:
    """Contorno de la instancia en la forma que consume el visor.

    Se toma el del corte inferido, que es el unico que la corrida produjo. Un punto
    sin coordenadas numericas se descarta en vez de viajar como null: un poligono a
    medias dibuja una figura que no es la que la IA segmento.
    """
    if not isinstance(contours, list) or not contours:
        return None
    first = contours[0]
    if not isinstance(first, dict):
        return None
    points = [
        {"x": float(point["x"]), "y": float(point["y"])}
        for point in (first.get("points") or [])
        if isinstance(point, dict) and isinstance(point.get("x"), (int, float)) and isinstance(point.get("y"), (int, float))
    ]
    if len(points) < 3:
        return None
    return {"kind": "polygon", "sliceIndex": first.get("sliceIndex"), "points": points}


def plane_segmentation_v2(raw: Any) -> PlaneSegmentationV2 | None:
    """Mapa de instancias tal como lo produjo el runtime.

    Se descarta entero si viene incompleto en vez de publicar un mapa a medias: un
    RLE truncado pinta regiones donde no las hay, que es peor que no pintar nada.
    """
    segmentation = raw if isinstance(raw, dict) else None
    if not segmentation or segmentation.get("encoding") != "rle-v1":
        return None
    data = segmentation.get("data")
    width = int_or_none(segmentation.get("width"))
    height = int_or_none(segmentation.get("height"))
    if not isinstance(data, list) or not data or len(data) % 2 != 0 or not width or not height:
        return None
    if sum(data[1::2]) != width * height:
        return None
    instances = [
        SegmentationInstanceV2(
            index=int(item["index"]),
            id=str(item["id"]),
            label=str(item.get("label") or item.get("classKey") or ""),
            classKey=str(item.get("classKey") or ""),
            level=text_or_none(item.get("level")),
        )
        for item in (segmentation.get("instances") or [])
        if isinstance(item, dict) and int_or_none(item.get("index")) is not None and item.get("id")
    ]
    return PlaneSegmentationV2(
        encoding="rle-v1",
        width=width,
        height=height,
        data=[int(value) for value in data],
        instances=instances,
    )


def plane_quality_v2(raw: Any) -> PlaneQualityV2:
    quality = raw if isinstance(raw, dict) else {}
    return PlaneQualityV2(
        maskCount=int(quality.get("maskCount", 0) or 0),
        landmarkCount=int(quality.get("landmarkCount", 0) or 0),
        measurementCount=int(quality.get("measurementCount", 0) or 0),
        meanConfidence=float_or_none(quality.get("meanConfidence")),
        meanForegroundConfidence=float_or_none(quality.get("meanForegroundConfidence")),
        foregroundRatio=float_or_none(quality.get("foregroundRatio")),
        slicePreviewCount=int(quality.get("slicePreviewCount", 0) or 0),
        volumeGeometry=quality.get("volumeGeometry") if isinstance(quality.get("volumeGeometry"), dict) else None,
        slicePixels=quality.get("slicePixels") if isinstance(quality.get("slicePixels"), dict) else None,
        discLevels=quality.get("discLevels") if isinstance(quality.get("discLevels"), list) else None,
        warnings=[],
    )


def assign_axial_levels(plane_results: dict[PlaneNameV2, PlaneRunV2Result | None]) -> None:
    """Le pone nivel a las mediciones axiales, cruzando la geometria con el sagital.

    Va aca y no dentro de la corrida de cada plano porque es lo unico que necesita los
    dos a la vez: el nivel lo conoce el sagital, que ve la serie completa de espacios
    discales, y la altura a la que esta el corte la conoce el axial. Cada plano por
    separado tiene la mitad de la respuesta.

    No hace nada si falta alguno de los dos, o si el corte axial no cae en un espacio
    discal. Que las mediciones sigan sin nivel es una respuesta, no una falla.
    """
    sagittal, axial = plane_results.get("sagittal"), plane_results.get("axial")
    if sagittal is None or axial is None or axial.synthetic or sagittal.synthetic:
        return
    sagittal_quality = sagittal.quality.model_dump() if sagittal.quality else None
    axial_quality = axial.quality.model_dump() if axial.quality else None

    # El nivel de cada corte, no solo el del que analizo el modelo. El visor lo necesita
    # para nombrar el corte que el medico esta mirando y para no mandar una clasificacion
    # subarticular bajo un nivel que no es. Ver axial_slice_levels.
    per_slice = axial_slice_levels(sagittal_quality, axial_quality)
    if per_slice and axial.quality is not None:
        axial.quality.sliceLevels = [
            {"index": index, "level": level} for index, level in sorted(per_slice.items())
        ]

    level = axial_slice_level(sagittal_quality, axial_quality)
    if not level:
        return
    for measurement in axial.measurements:
        # `levelScope` ya es "level" en estas mediciones: describen una estructura de un
        # nivel, y lo unico que faltaba era saber cual.
        if not measurement.level:
            measurement.level = level


def build_workspace_response(
    *,
    request: MultiplanarRunV2Request,
    trace_id: str,
    run_id: str,
    requested_planes: list[PlaneNameV2],
    workspace_mode: WorkspaceModeV2,
    planes: dict[PlaneNameV2, PlaneRunV2Result | None],
    readiness: MultiplanarReadinessV2,
) -> MultiplanarRunV2Response:
    completed = [plane for plane in requested_planes if planes.get(plane) is not None]
    quality = workspace_quality_v2(planes)
    effective_mode = workspace_effective_mode(planes)
    synthetic = any(bool(plane.synthetic) for plane in planes.values() if plane is not None)
    fallback_reason = first_fallback_reason(planes)
    return MultiplanarRunV2Response(
        status="completed",
        schemaVersion="pfi.multiplanar-run.v2",
        runId=run_id,
        traceId=trace_id,
        caseId=request.caseId,
        workspaceMode=workspace_mode,
        requestedInferenceMode=request.inferenceMode,
        effectiveInferenceMode=effective_mode,
        requestedPlanes=requested_planes,
        completedPlanes=completed,
        readiness=readiness,
        planes=planes,
        threeD=three_d_status_v2(run_id, workspace_mode, planes),
        quality=quality,
        review=ReviewPolicyV2(status="pending", required=True, approvalRequiresHumanConfirmation=True),
        governance=GovernanceV2(humanReviewRequired=True, notClinicalDiagnosis=True),
        synthetic=synthetic,
        fallbackReason=fallback_reason,
    )


def workspace_readiness(preflight: dict[PlaneNameV2, dict[str, Any]]) -> MultiplanarReadinessV2:
    sagittal_ready = bool((preflight.get("sagittal") or {}).get("model", {}).get("availableForRealInference"))
    axial_ready = bool((preflight.get("axial") or {}).get("model", {}).get("availableForRealInference"))
    return MultiplanarReadinessV2(sagittal=sagittal_ready, axial=axial_ready, dual=sagittal_ready and axial_ready)


def workspace_quality_v2(planes: dict[PlaneNameV2, PlaneRunV2Result | None]) -> WorkspaceQualityV2:
    by_plane = {"sagittal": planes.get("sagittal").quality if planes.get("sagittal") else None, "axial": planes.get("axial").quality if planes.get("axial") else None}
    qualities = [quality for quality in by_plane.values() if quality is not None]
    return WorkspaceQualityV2(
        planeCount=len(qualities),
        maskCount=sum(quality.maskCount for quality in qualities),
        landmarkCount=sum(quality.landmarkCount for quality in qualities),
        measurementCount=sum(quality.measurementCount for quality in qualities),
        byPlane=by_plane,  # type: ignore[arg-type]
    )


def three_d_status_v2(run_id: str, workspace_mode: WorkspaceModeV2, planes: dict[PlaneNameV2, PlaneRunV2Result | None]) -> ThreeDStatusV2:
    if workspace_mode == "sagittal_only":
        status = "blocked_missing_axial"
        required = ["axial_masks", "spacing", "slice_index_mapping"]
    elif workspace_mode == "axial_only":
        status = "blocked_missing_sagittal"
        required = ["sagittal_masks", "spacing", "slice_index_mapping"]
    else:
        return build_lumbar_3d_status(run_id, planes)  # type: ignore[arg-type]
    return ThreeDStatusV2(
        enabled=False,
        status=status,  # type: ignore[arg-type]
        sourcePlaneRunIds={
            "sagittal": planes.get("sagittal").runId if planes.get("sagittal") else None,
            "axial": planes.get("axial").runId if planes.get("axial") else None,
        },
        requiredInputs=required,
    )


def normalize_inference_mode(value: Any) -> str:
    normalized = str(value or "contract").strip().lower()
    return "real_baseline" if normalized == "real" else normalized if normalized in {"contract", "mock", "real_baseline"} else "contract"


def workspace_effective_mode(planes: dict[PlaneNameV2, PlaneRunV2Result | None]) -> str:
    modes = {plane.effectiveInferenceMode for plane in planes.values() if plane is not None}
    if not modes:
        return "contract"
    return modes.pop() if len(modes) == 1 else "mixed"


def first_fallback_reason(planes: dict[PlaneNameV2, PlaneRunV2Result | None]) -> str | None:
    for plane in planes.values():
        if plane is not None and plane.fallbackReason:
            return plane.fallbackReason
    return None


def shape_to_height_width(shape: list[int] | None) -> tuple[int, int]:
    if shape and len(shape) >= 2:
        return int(shape[-2]), int(shape[-1])
    return 0, 0


def required_text(value: Any, field: str, plane: PlaneNameV2) -> str:
    if value is None:
        raise MultiplanarV2Error(
            "INVALID_MULTIPLANAR_RESPONSE",
            "Respuesta real_baseline incompleta.",
            status_code=500,
            trace_id="unavailable",
            case_id=None,
            requested_planes=[plane],
            details={"plane": plane, "field": field},
        )
    normalized = str(value).strip()
    if not normalized or normalized == "None":
        raise MultiplanarV2Error(
            "INVALID_MULTIPLANAR_RESPONSE",
            "Respuesta real_baseline incompleta.",
            status_code=500,
            trace_id="unavailable",
            case_id=None,
            requested_planes=[plane],
            details={"plane": plane, "field": field},
        )
    return normalized


def synthetic_plane_result(
    plane: PlaneNameV2,
    plane_spec: PlaneExecutionV2Request,
    preflight: dict[str, Any],
    mode: str,
    fallback_reason: str,
) -> PlaneRunV2Result:
    return PlaneRunV2Result(
        status="ready",
        plane=plane,
        runId=synthetic_plane_run_id(plane, plane_spec, mode),
        effectiveInferenceMode=mode,  # type: ignore[arg-type]
        model=plane_model_v2(preflight["model"]),
        input=PlaneInputV2(
            inputId=plane_spec.inputId,
            format=getattr(preflight.get("input"), "format", None),
            sizeBytes=getattr(preflight.get("input"), "size", None),
            nativeShape=None,
            canonicalShape=None,
            orientationTransform=None,
            spacingXyzMm=None,
            canonicalAxisSpacingMm=None,
            selectedSliceIndex=None,
            sliceCount=None,
            selectedAxis=None,
            inPlaneSpacingMm=None,
        ),
        coordinateSpace=CoordinateSpaceV2(
            name="synthetic_empty_pixel_space",
            width=1,
            height=1,
            units="pixel",
            origin="top_left",
            xDirection="right",
            yDirection="down",
            sourceSliceIndex=None,
            sourceAxis=None,
        ),
        series=[],
        assets=[],
        masks=[],
        landmarks=[],
        measurements=[],
        quality=PlaneQualityV2(maskCount=0, landmarkCount=0, measurementCount=0, warnings=["synthetic_empty_result"]),
        synthetic=True,
        fallbackReason=fallback_reason,
    )


def synthetic_plane_run_id(plane: PlaneNameV2, plane_spec: PlaneExecutionV2Request, mode: str) -> str:
    raw = f"{plane}|{plane_spec.inputId}|{plane_spec.modelKey}|{mode}|synthetic"
    return "synthetic-" + sha256(raw.encode("utf-8")).hexdigest()[:16]


def sanitize_fallback_reason(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._:-" else "_" for char in value)[:120]


def int_or_none(value: Any) -> int | None:
    # `bool` es subclase de `int`, asi que sin el chequeo un True se convertiria en 1 y
    # un campo booleano mal tipeado pasaria como un indice de corte.
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def text_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_list_or_none(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def float_list_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def list_string(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def strip_dot(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).lstrip(".")


class LegacyMultiplanarV1Adapter:
    """Deprecated adapter for P8 backend compatibility until P9-B/P9-C."""

    def from_v2(self, response: MultiplanarRunV2Response) -> dict[str, Any]:
        payload = response.model_dump(mode="json")
        planes = {
            "sagittal": self._plane_from_v2(response.planes["sagittal"], response.traceId, response.requestedInferenceMode),
            "axial": self._plane_from_v2(response.planes["axial"], response.traceId, response.requestedInferenceMode),
        }
        return {
            "status": "multiplanar_run_ready",
            "schemaVersion": "multiplanar-run-v1",
            "runId": response.runId,
            "traceId": response.traceId,
            "caseId": response.caseId,
            "workspaceMode": response.workspaceMode,
            "requestedInferenceMode": response.requestedInferenceMode,
            "effectiveInferenceMode": response.effectiveInferenceMode,
            "sagittalRunReady": response.planes["sagittal"] is not None,
            "axialRunReady": response.planes["axial"] is not None,
            "dualRunReady": response.planes["sagittal"] is not None and response.planes["axial"] is not None,
            "planes": planes,
            "threeD": payload["threeD"],
            "quality": payload["quality"],
            "review": {"status": "pendiente", "professionalReviewRequired": True, "approvalRequiresHumanConfirmation": True},
            "metadata": {"multiplanarRunId": response.runId, "traceId": response.traceId, "workspaceMode": response.workspaceMode, "deidentified": True, "diagnosisGenerated": False},
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
        }

    def _plane_from_v2(
        self,
        plane: PlaneRunV2Result | None,
        trace_id: str,
        requested_inference_mode: str,
    ) -> dict[str, Any] | None:
        if plane is None:
            return None
        values = [
            {
                "id": measurement.id,
                "label": measurement.labelKey,
                "value": measurement.value,
                "unit": measurement.unit,
                "confidence": measurement.confidence,
                "status": "pendiente" if measurement.status == "pending_review" else measurement.status,
                "level": measurement.level,
            }
            for measurement in plane.measurements
        ]
        return {
            "runId": plane.runId,
            "traceId": trace_id,
            "plane": plane.plane,
            "status": plane.status,
            "inferenceMode": plane.effectiveInferenceMode,
            "requestedInferenceMode": requested_inference_mode,
            "effectiveInferenceMode": plane.effectiveInferenceMode,
            "allowContractFallback": legacy_allow_contract_fallback(plane),
            "artifactHash": plane.model.artifactHash,
            "degradedMode": plane.synthetic,
            "modelKey": plane.model.key,
            "modelVersion": plane.model.version,
            "inputId": plane.input.inputId,
            "series": [
                {
                    "id": series.id,
                    "plane": series.plane,
                    "sequence": series.sequence,
                    "sliceCount": series.sliceCount,
                    "selectedSlice": series.selectedSliceIndex,
                    "status": series.status,
                }
                for series in plane.series
            ],
            "masks": [
                {
                    "id": mask.id,
                    "label": mask.classKey,
                    "className": mask.classKey,
                    "confidence": mask.confidence,
                    "editable": mask.editable,
                    "enabled": mask.enabled,
                    "contours": [],
                    "synthetic": plane.synthetic,
                }
                for mask in plane.masks
            ],
            "landmarks": [
                {
                    "id": landmark.id,
                    "label": landmark.labelKey,
                    "x": landmark.x,
                    "y": landmark.y,
                    "editable": False,
                    "synthetic": plane.synthetic,
                }
                for landmark in plane.landmarks
            ],
            "measurements": {
                "status": "synthetic_ready" if plane.synthetic else "real_baseline_ready",
                "values": values,
                "source": "canonical_multiplanar_v2",
                "synthetic": plane.synthetic,
                "fallbackReason": plane.fallbackReason,
            },
            "measurementValues": values,
            "quality": plane.quality.model_dump(mode="json"),
            "assets": legacy_assets_map(plane.assets),
            "modelArtifact": legacy_model_artifact(plane.model),
            "metadata": {
                "inferenceMode": plane.effectiveInferenceMode,
                "allowContractFallback": legacy_allow_contract_fallback(plane),
                "inputFormat": f".{plane.input.format}" if plane.input.format else None,
                "outputFiles": legacy_output_files(plane.assets),
                "synthetic": plane.synthetic,
                "fallbackReason": plane.fallbackReason,
            },
            "aiOutput": {
                "status": "synthetic_ready" if plane.synthetic else "real_baseline_ready",
                "inferenceMode": plane.effectiveInferenceMode,
                "artifactHash": plane.model.artifactHash,
                "realInferenceAvailable": bool(
                    plane.model.availableForRealInference
                    and plane.effectiveInferenceMode == "real_baseline"
                    and not plane.synthetic
                    and plane.fallbackReason is None
                ),
                "modelKey": plane.model.key,
                "modelVersion": plane.model.version,
                "synthetic": plane.synthetic,
                "fallbackReason": plane.fallbackReason,
                "humanReviewRequired": True,
                "notClinicalDiagnosis": True,
            },
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
            "synthetic": plane.synthetic,
            "fallbackReason": plane.fallbackReason,
        }


class LegacyMultiplanarV1RequestMapper:
    def to_v2(self, request: MultiplanarRunRequest, *, trace_id: str) -> MultiplanarRunV2Request:
        metadata = dict(request.metadata or {})
        planes: dict[str, Any] = {"sagittal": None, "axial": None}
        if request.sagittal_input_id or request.sagittal_input_path:
            planes["sagittal"] = {
                "inputId": request.sagittal_input_id or register_legacy_input_path(request.case_id, "sagittal", request.sagittal_input_path),
                "modelKey": request.sagittal_model_key,
            }
        if request.axial_input_id or request.axial_input_path:
            planes["axial"] = {
                "inputId": request.axial_input_id or register_legacy_input_path(request.case_id, "axial", request.axial_input_path),
                "modelKey": request.axial_model_key,
            }
        if planes["sagittal"] is None and planes["axial"] is None:
            synthetic_id = f"legacy_contract_{sha256(request.case_id.encode('utf-8')).hexdigest()[:12]}"
            planes["sagittal"] = {"inputId": synthetic_id, "modelKey": request.sagittal_model_key}
        options = {
            key: metadata[key]
            for key in ("sliceIndex", "sliceAxis", "sliceWindowRadius", "inputOrientationTransform")
            if key in metadata
        }
        return MultiplanarRunV2Request.model_validate({
            "caseId": request.case_id,
            "traceId": trace_id,
            "inferenceMode": normalize_inference_mode(metadata.get("inferenceMode", metadata.get("mode", "contract"))),
            "allowContractFallback": bool_value(metadata.get("allowContractFallback"), default=True),
            "planes": planes,
            "options": options,
        })


def bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", "off"}


def legacy_path_input_id(case_id: str, plane: str, input_path: str | None) -> str:
    raw = f"{case_id}|{plane}|{input_path or ''}"
    return "legacy_path_" + sha256(raw.encode("utf-8")).hexdigest()[:16]


def register_legacy_input_path(case_id: str, plane: str, input_path: str | None) -> str:
    if not input_path:
        return legacy_path_input_id(case_id, plane, input_path)
    metadata = register_existing_path(
        case_id=case_id,
        plane=plane,
        path=Path(input_path),
        source_key="legacy_input_path",
    )
    return str(metadata["inputId"])


def legacy_allow_contract_fallback(plane: PlaneRunV2Result) -> bool:
    return not (
        plane.effectiveInferenceMode == "real_baseline"
        and not plane.synthetic
        and plane.fallbackReason is None
    )


def legacy_assets_map(assets: list[PlaneAssetV2]) -> dict[str, dict[str, Any]]:
    return {
        asset.assetName: {
            "role": asset.role,
            "contentType": asset.contentType,
            "generated": asset.generated,
            "relativePath": asset.relativePath,
        }
        for asset in assets
    }


def legacy_output_files(assets: list[PlaneAssetV2]) -> dict[str, dict[str, Any]]:
    key_for_name = {
        "input.png": "imagePath",
        "overlay.png": "overlayPath",
        "mask.npy": "maskPath",
        "confidence.npy": "confidencePath",
        "mask-preview.png": "maskPreviewPath",
    }
    return {
        key_for_name[asset.assetName]: {"generated": asset.generated, "fileName": asset.assetName}
        for asset in assets
        if asset.assetName in key_for_name
    }


def legacy_model_artifact(model: PlaneModelV2) -> dict[str, Any]:
    return {
        **model.model_dump(mode="json"),
        "manifest": {
            "status": model.manifestStatus,
            "valid": model.manifestValid,
            "trainingStatus": model.trainingStatus,
        },
    }


def http_exception_to_v2(exc: HTTPException, *, trace_id: str, case_id: str | None, requested_planes: list[PlaneNameV2]) -> MultiplanarV2Error:
    return MultiplanarV2Error(
        "REAL_INFERENCE_FAILED",
        "Fallo de inferencia real baseline.",
        status_code=exc.status_code,
        trace_id=trace_id,
        case_id=case_id,
        requested_planes=requested_planes,
        details={"reason": str(exc.detail)[:240]},
    )


def exception_to_v2(exc: Exception, *, trace_id: str, case_id: str | None, requested_planes: list[PlaneNameV2]) -> MultiplanarV2Error:
    return MultiplanarV2Error(
        "REAL_INFERENCE_FAILED",
        "Fallo de inferencia real baseline.",
        status_code=500,
        trace_id=trace_id,
        case_id=case_id,
        requested_planes=requested_planes,
        details={"type": type(exc).__name__},
    )
