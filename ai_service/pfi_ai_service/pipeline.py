from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS, build_agent_decision
from .contract_geometry import build_landmarks_from_masks, build_measurements_from_contract, contract_quality_summary
from .model_artifacts import model_status
from .reporting import write_json
from .settings import MODEL_REGISTRY, get_settings


class PipelineRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_by_name=True)

    case_id: str = Field(..., alias="caseId")
    plane: Literal["sagittal", "axial"]
    model_key: str = Field(..., alias="modelKey")
    input_path: str = Field(..., alias="inputPath")
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _run_id_for(request: PipelineRunRequest) -> str:
    raw = f"{request.case_id}|{request.plane}|{request.model_key}|{request.input_path}"
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
    return normalized if normalized in {"contract", "mock", "real"} else "contract"


def _effective_inference_mode(request: PipelineRunRequest) -> str:
    # Real inference remains behind the dedicated /inference/* endpoints until model/runtime validation is complete.
    requested = _requested_inference_mode(request)
    return "contract" if requested == "real" else requested


def _trace_id(request: PipelineRunRequest) -> str | None:
    value = request.metadata.get("traceId") or request.metadata.get("correlationId") or request.metadata.get("backendTraceId")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


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
    flags = []
    model_info = MODEL_REGISTRY.get(request.model_key)
    artifact = model_status(request.model_key, dict(model_info or {}))
    if model_info is None:
        flags.append(f"unknown_model_key:{request.model_key}")
    elif model_info.get("plane") != request.plane:
        flags.append(f"model_plane_mismatch:expected={model_info.get('plane')},received={request.plane}")

    requested_mode = _requested_inference_mode(request)
    inference_mode = _effective_inference_mode(request)
    if requested_mode == "real" and inference_mode != "real":
        flags.append("real_inference_requested_but_contract_mode_used")
    if requested_mode == "real" and not artifact.get("availableForRealInference", False):
        flags.append("model_artifact_missing_for_real_inference")

    run_id = _run_id_for(request)
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
    write_json(get_settings().output_dir / "agent_reports" / f"{run_id}.json", response)
    return response
