from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS, build_agent_decision
from .contract_geometry import build_landmarks_from_masks, build_measurements_from_contract, contract_quality_summary
from .input_registry import InputRegistryError, resolve_input_id
from .model_artifacts import model_status
from .real_inference_runtime import run_real_inference
from .reporting import write_json
from .settings import MODEL_REGISTRY, get_settings


class PipelineRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_by_name=True)

    case_id: str = Field(..., alias="caseId")
    plane: Literal["sagittal", "axial"]
    model_key: str = Field(..., alias="modelKey")
    input_path: str | None = Field(default=None, alias="inputPath")
    input_id: str | None = Field(default=None, alias="inputId")
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _run_id_for(request: PipelineRunRequest) -> str:
    input_ref = request.input_id or request.input_path or ""
    raw = f"{request.case_id}|{request.plane}|{request.model_key}|{input_ref}"
    return sha256(raw.encode("utf-8")).hexdigest()[:16]


def _overlay_path_for(run_id: str) -> Optional[str]:
    settings = get_settings()
    overlay_path = settings.output_dir / "overlays" / f"{run_id}.png"
    if Path(overlay_path).exists():
        return str(overlay_path)
    return None


def _requested_inference_mode(request: PipelineRunRequest) -> str:
    value = request.metadata.get("inferenceMode", request.metadata.get("mode", "contract"))
    normalized = str(value).strip().lower()
    return normalized if normalized in {"contract", "mock", "real", "real_baseline"} else "contract"


def _effective_inference_mode(request: PipelineRunRequest) -> str:
    requested = _requested_inference_mode(request)
    return "contract" if requested in {"real", "real_baseline"} else requested


def _allow_contract_fallback(request: PipelineRunRequest) -> bool:
    value = request.metadata.get("allowContractFallback", True)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", "off"}


def _trace_id(request: PipelineRunRequest) -> str | None:
    value = request.metadata.get("traceId") or request.metadata.get("correlationId") or request.metadata.get("backendTraceId")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _resolve_input_reference(request: PipelineRunRequest) -> tuple[PipelineRunRequest, bool]:
    if request.input_id:
        record = resolve_input_id(request.input_id, case_id=request.case_id, plane=request.plane)
        metadata = {
            **request.metadata,
            "inputId": record.input_id,
            "inputFormat": record.format,
            "inputSize": record.size,
            "inputResolutionMode": "server_side_input_id",
        }
        return request.model_copy(update={"input_path": str(record.path), "metadata": metadata}), True
    if not request.input_path:
        raise InputRegistryError("inputPath o inputId requerido", status_code=400)
    return request, False


def _public_output_files(output_files: Any) -> dict[str, Any]:
    if not isinstance(output_files, dict):
        return {}
    public: dict[str, Any] = {}
    for key, value in output_files.items():
        name = Path(str(value)).name if value else None
        public[key] = {"generated": bool(value), "fileName": name}
    return public


def _strip_internal_path_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_internal_path_keys(item)
            for key, item in value.items()
            if key not in {"path", "sourcePath"}
        }
    if isinstance(value, list):
        return [_strip_internal_path_keys(item) for item in value]
    return value


def _sanitize_input_id_response(response: Dict[str, Any], request: PipelineRunRequest) -> Dict[str, Any]:
    if not request.input_id:
        return response
    sanitized = dict(response)
    sanitized.pop("input_path", None)
    sanitized.pop("inputPath", None)
    sanitized["inputId"] = request.input_id
    sanitized["input_id"] = request.input_id

    for series in sanitized.get("series", []) if isinstance(sanitized.get("series"), list) else []:
        if isinstance(series, dict):
            series.pop("imagePath", None)
            series.pop("overlayPath", None)

    sanitized.pop("overlay_path", None)
    sanitized.pop("overlayPath", None)

    if "modelArtifact" in sanitized:
        sanitized["modelArtifact"] = _strip_internal_path_keys(sanitized["modelArtifact"])

    metadata = dict(sanitized.get("metadata") or {})
    metadata.pop("sourcePath", None)
    metadata.pop("inputPath", None)
    metadata["inputId"] = request.input_id
    if "outputFiles" in metadata:
        metadata["outputFiles"] = _public_output_files(metadata["outputFiles"])
    sanitized["metadata"] = metadata
    return sanitized


def _contour(series_id: str, slice_index: int, points: list[tuple[float, float]]) -> Dict[str, Any]:
    return {
        "seriesId": series_id,
        "sliceIndex": slice_index,
        "points": [{"x": x, "y": y} for x, y in points],
    }


def _series_payload(request: PipelineRunRequest, overlay_path: Optional[str]) -> list[Dict[str, Any]]:
    primary_series = "series-sag-t2" if request.plane == "sagittal" else "series-ax-t2"
    return [
        {
            "id": "series-sag-t2",
            "name": "Sagittal T2",
            "plane": "sagittal",
            "sequence": "T2",
            "sliceCount": 96,
            "selectedSlice": 58,
            "imageUrl": None,
            "overlayUrl": overlay_path if primary_series == "series-sag-t2" else None,
            "overlayOpacity": 0.74,
            "status": "contract_ready",
        },
        {
            "id": "series-sag-t1",
            "name": "Sagittal T1",
            "plane": "sagittal",
            "sequence": "T1",
            "sliceCount": 96,
            "selectedSlice": 58,
            "imageUrl": None,
            "overlayUrl": None,
            "overlayOpacity": 0.6,
            "status": "reference_only",
        },
        {
            "id": "series-ax-t2",
            "name": "Axial T2 L4-L5",
            "plane": "axial",
            "sequence": "T2",
            "sliceCount": 48,
            "selectedSlice": 24,
            "imageUrl": None,
            "overlayUrl": overlay_path if primary_series == "series-ax-t2" else None,
            "overlayOpacity": 0.74,
            "status": "contract_ready",
        },
    ]


def _masks_payload() -> list[Dict[str, Any]]:
    return [
        {
            "id": "mask-vertebral-body-l4",
            "label": "Vertebral body L4",
            "className": "vertebral_body",
            "color": "#c8b28a",
            "confidence": 0.86,
            "editable": True,
            "enabled": True,
            "contours": [
                _contour("series-sag-t2", 58, [(184, 122), (260, 124), (270, 205), (190, 212)]),
            ],
        },
        {
            "id": "mask-disc-l45",
            "label": "Intervertebral disc L4-L5",
            "className": "disc",
            "color": "#2563eb",
            "confidence": 0.82,
            "editable": True,
            "enabled": True,
            "contours": [
                _contour("series-sag-t2", 58, [(178, 214), (272, 208), (285, 236), (181, 244)]),
                _contour("series-ax-t2", 24, [(190, 165), (246, 158), (275, 205), (220, 238), (176, 212)]),
            ],
        },
        {
            "id": "mask-canal-l45",
            "label": "Spinal canal L4-L5",
            "className": "spinal_canal",
            "color": "#16a34a",
            "confidence": 0.79,
            "editable": True,
            "enabled": True,
            "contours": [
                _contour("series-sag-t2", 58, [(292, 116), (329, 132), (326, 248), (289, 260), (278, 187)]),
                _contour("series-ax-t2", 24, [(220, 118), (257, 128), (272, 166), (246, 194), (208, 188), (195, 148)]),
            ],
        },
    ]


def _landmarks_payload(masks: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return build_landmarks_from_masks(masks)


def _measurement_values(masks: list[Dict[str, Any]], landmarks: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return build_measurements_from_contract(masks, landmarks)


def _measurements_payload(masks: list[Dict[str, Any]], landmarks: list[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "status": "contract_ready",
        "values": _measurement_values(masks, landmarks),
        "source": "contract_visual_pipeline",
        "description": "Salida tecnica estable: mediciones derivadas de contornos, editables y siempre revisables por profesional.",
    }


def _ai_output_payload(agent_decision: Dict[str, Any], inference_mode: str, requested_mode: str, artifact: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "contract_ready",
        "label": "Contrato visual listo",
        "description": "Pipeline preparado con series, masks, landmarks y measurements.values para revision profesional.",
        "inferenceMode": inference_mode,
        "requestedInferenceMode": requested_mode,
        "realInferenceAvailable": bool(artifact.get("availableForRealInference", False)),
        "modelReadiness": artifact.get("readiness"),
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
        "agentDecision": agent_decision,
    }


def _as_backend_response(
    *,
    run_id: str,
    request: PipelineRunRequest,
    overlay_path: Optional[str],
    agent_decision: Dict[str, Any],
) -> Dict[str, Any]:
    requested_mode = _requested_inference_mode(request)
    inference_mode = _effective_inference_mode(request)
    model_info = dict(MODEL_REGISTRY.get(request.model_key, {}))
    artifact = model_status(request.model_key, model_info)
    series = _series_payload(request, overlay_path)
    masks = _masks_payload()
    landmarks = _landmarks_payload(masks)
    measurements = _measurements_payload(masks, landmarks)
    ai_output = _ai_output_payload(agent_decision, inference_mode, requested_mode, artifact)
    quality = contract_quality_summary(masks, landmarks, measurements["values"])
    trace_id = _trace_id(request)
    response = {
        "run_id": run_id,
        "runId": run_id,
        "traceId": trace_id,
        "case_id": request.case_id,
        "caseId": request.case_id,
        "studyId": f"STUDY-{request.case_id.replace('CASE-', '')}",
        "patientId": request.metadata.get("patientId", "PAT-DEMO-0087"),
        "studyDate": request.metadata.get("studyDate", "2026-07-01"),
        "modality": "MRI",
        "bodyRegion": "Lumbar Spine",
        "reviewStatus": "pendiente",
        "plane": request.plane,
        "model_key": request.model_key,
        "modelKey": request.model_key,
        "modelVersion": artifact.get("version", "contract-v1"),
        "input_path": request.input_path,
        "inputPath": request.input_path,
        "inputId": request.input_id,
        "series": series,
        "masks": masks,
        "landmarks": landmarks,
        "measurements": measurements,
        "measurementValues": measurements["values"],
        "overlay_path": overlay_path,
        "overlayPath": overlay_path,
        "aiOutput": ai_output,
        "agent_decision": agent_decision,
        "agentDecision": agent_decision,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "not_clinical_diagnosis": NOT_CLINICAL_DIAGNOSIS,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
        "quality": quality,
        "modelArtifact": artifact,
        "metadata": {
            **request.metadata,
            "traceId": trace_id,
            "contractMode": "visual_review_v1",
            "inferenceMode": inference_mode,
            "requestedInferenceMode": requested_mode,
            "modelReadiness": artifact.get("readiness"),
            "modelArtifact": artifact.get("artifact"),
            "quality": quality,
            "deidentified": True,
            "diagnosisGenerated": False,
        },
    }
    return response


def run_pipeline(request: PipelineRunRequest) -> Dict[str, Any]:
    request, uses_input_id = _resolve_input_reference(request)
    flags: list[str] = []
    model_info = MODEL_REGISTRY.get(request.model_key)
    artifact = model_status(request.model_key, dict(model_info or {}))
    model_matches_plane = model_info is not None and model_info.get("plane") == request.plane
    if model_info is None:
        flags.append(f"unknown_model_key:{request.model_key}")
    elif not model_matches_plane:
        flags.append(f"model_plane_mismatch:expected={model_info.get('plane')},received={request.plane}")

    requested_mode = _requested_inference_mode(request)
    run_id = _run_id_for(request)
    wants_real = requested_mode in {"real", "real_baseline"}
    can_run_real = wants_real and model_matches_plane and bool(artifact.get("availableForRealInference", False))

    if can_run_real:
        try:
            response = run_real_inference(request, run_id)
            response = _sanitize_input_id_response(response, request)
            write_json(get_settings().output_dir / "agent_reports" / f"{run_id}.json", response)
            return response
        except Exception as exc:
            if not _allow_contract_fallback(request):
                raise
            fallback_metadata = dict(request.metadata)
            fallback_metadata["realInferenceFailure"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:240],
            }
            fallback_metadata["realInferenceAttempted"] = True
            request = request.model_copy(update={"metadata": fallback_metadata})
            flags.append(f"real_inference_failed:{type(exc).__name__}")
            flags.append("contract_fallback_after_real_inference_failure")
    elif wants_real and not artifact.get("availableForRealInference", False):
        flags.append("model_artifact_missing_for_real_inference")
    elif wants_real and not model_matches_plane:
        flags.append("real_inference_not_started_model_plane_mismatch")

    if wants_real:
        flags.append(f"{requested_mode}_requested_but_contract_mode_used")

    overlay_path = _overlay_path_for(run_id)
    agent_decision = build_agent_decision(
        plane=request.plane,
        model_key=request.model_key,
        flags=flags or ["contract_visual_pipeline_requires_review"],
    )

    response = _as_backend_response(
        run_id=run_id,
        request=request,
        overlay_path=overlay_path,
        agent_decision=agent_decision,
    )
    response = _sanitize_input_id_response(response, request)
    write_json(get_settings().output_dir / "agent_reports" / f"{run_id}.json", response)
    return response
