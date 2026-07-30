from __future__ import annotations

from hashlib import sha256
import logging
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .input_registry import InputRegistryError
from .pipeline import PipelineRunRequest, run_pipeline
from .security import sanitize_public_text
from .reporting import write_json
from .settings import get_settings

LOGGER = logging.getLogger(__name__)


class MultiplanarRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_by_name=True)

    case_id: str = Field(..., alias="caseId")
    sagittal_input_path: str | None = Field(default=None, alias="sagittalInputPath")
    axial_input_path: str | None = Field(default=None, alias="axialInputPath")
    sagittal_input_id: str | None = Field(default=None, alias="sagittalInputId")
    axial_input_id: str | None = Field(default=None, alias="axialInputId")
    sagittal_model_key: str = Field(default="sagittal_spider", alias="sagittalModelKey")
    axial_model_key: str = Field(default="axial_t2_alkafri", alias="axialModelKey")
    metadata: Dict[str, Any] = Field(default_factory=dict)


def run_multiplanar_pipeline(request: MultiplanarRunRequest) -> Dict[str, Any]:
    run_id = shared_run_id(request)
    trace_id = trace_id_from(request.metadata)
    log_multiplanar_event(request, trace_id, "start")
    sagittal: Dict[str, Any]
    axial: Dict[str, Any] | None = None

    try:
        sagittal_request = plane_request(request, "sagittal", run_id, trace_id)
        log_multiplanar_event(request, trace_id, "run_sagittal")
        sagittal = run_pipeline(sagittal_request)

        if has_axial_input(request):
            axial_request = plane_request(request, "axial", run_id, trace_id)
            log_multiplanar_event(request, trace_id, "run_axial")
            axial = run_pipeline(axial_request)
        else:
            log_multiplanar_event(request, trace_id, "skip_axial_not_requested")
    except Exception as exc:
        log_multiplanar_event(
            request,
            trace_id,
            "failed",
            exc_type=type(exc).__name__,
            exc_message=compact_message(exc),
        )
        raise

    response = {
        "status": "multiplanar_run_ready",
        "schemaVersion": "multiplanar-run-v1",
        "runId": run_id,
        "traceId": trace_id,
        "caseId": request.case_id,
        "workspaceMode": "dual_plane_with_3d_context",
        "requestedInferenceMode": requested_mode(request.metadata),
        "effectiveInferenceMode": effective_workspace_mode(sagittal, axial),
        "sagittalRunReady": plane_real_ready(sagittal),
        "axialRunReady": plane_real_ready(axial),
        "dualRunReady": axial is not None and plane_real_ready(sagittal) and plane_real_ready(axial),
        "planes": {
            "sagittal": sagittal,
            "axial": axial,
        },
        "threeD": three_d_status(sagittal, axial),
        "quality": workspace_quality(sagittal, axial),
        "review": {
            "status": "pendiente",
            "professionalReviewRequired": HUMAN_REVIEW_REQUIRED,
            "approvalRequiresHumanConfirmation": True,
        },
        "metadata": {
            **request.metadata,
            "multiplanarRunId": run_id,
            "traceId": trace_id,
            "workspaceMode": "dual_plane_with_3d_context",
            "deidentified": True,
            "diagnosisGenerated": False,
        },
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }
    write_json(get_settings().output_dir / "multiplanar_reports" / f"{run_id}.json", response)
    log_multiplanar_event(request, trace_id, "completed")
    return response


def plane_request(request: MultiplanarRunRequest, plane: str, run_id: str, trace_id: str | None) -> PipelineRunRequest:
    is_sagittal = plane == "sagittal"
    input_path = request.sagittal_input_path if is_sagittal else request.axial_input_path
    input_id = request.sagittal_input_id if is_sagittal else request.axial_input_id
    model_key = request.sagittal_model_key if is_sagittal else request.axial_model_key
    metadata = {
        **request.metadata,
        "multiplanarRunId": run_id,
        "workspaceMode": "dual_plane_with_3d_context",
        "workspacePlane": plane,
    }
    if trace_id:
        metadata.setdefault("traceId", trace_id)
        metadata.setdefault("correlationId", trace_id)
    if not input_path and not input_id and requested_mode(request.metadata) in {"real", "real_baseline"}:
        raise InputRegistryError(f"{plane}InputId o {plane}InputPath requerido para real_baseline", status_code=400)
    return PipelineRunRequest(
        caseId=request.case_id,
        plane=plane,
        modelKey=model_key,
        inputPath=input_path or (None if input_id else f"demo/{request.case_id}/{plane}"),
        inputId=input_id,
        metadata=metadata,
    )


def shared_run_id(request: MultiplanarRunRequest) -> str:
    raw = "|".join([
        request.case_id,
        request.sagittal_input_id or request.sagittal_input_path or "",
        request.axial_input_id or request.axial_input_path or "",
        request.sagittal_model_key,
        request.axial_model_key,
    ])
    return "multi-" + sha256(raw.encode("utf-8")).hexdigest()[:16]


def trace_id_from(metadata: Dict[str, Any]) -> str | None:
    value = metadata.get("traceId") or metadata.get("correlationId") or metadata.get("backendTraceId")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def requested_mode(metadata: Dict[str, Any]) -> str:
    value = str(metadata.get("inferenceMode", metadata.get("mode", "contract"))).strip().lower()
    return value if value in {"contract", "mock", "real", "real_baseline"} else "contract"


def effective_workspace_mode(sagittal: Dict[str, Any], axial: Dict[str, Any] | None) -> str:
    sagittal_mode = str((sagittal.get("aiOutput") or {}).get("inferenceMode", "contract"))
    if axial is None:
        return sagittal_mode
    modes = {sagittal_mode, str((axial.get("aiOutput") or {}).get("inferenceMode", "contract"))}
    return modes.pop() if len(modes) == 1 else "mixed"


def plane_real_ready(plane_response: Dict[str, Any] | None) -> bool:
    if plane_response is None:
        return False
    ai_output = plane_response.get("aiOutput") if isinstance(plane_response.get("aiOutput"), dict) else {}
    artifact = plane_response.get("modelArtifact") if isinstance(plane_response.get("modelArtifact"), dict) else {}
    return (
        ai_output.get("inferenceMode") == "real_baseline"
        and bool(artifact.get("baselineReady"))
        and bool(artifact.get("availableForRealInference"))
    )


def workspace_quality(sagittal: Dict[str, Any], axial: Dict[str, Any] | None) -> Dict[str, Any]:
    sagittal_quality = sagittal.get("quality") if isinstance(sagittal.get("quality"), dict) else {}
    axial_quality = axial.get("quality") if axial is not None and isinstance(axial.get("quality"), dict) else {}
    return {
        "sagittal": sagittal_quality,
        "axial": axial_quality if axial is not None else None,
        "planeCount": 1 if axial is None else 2,
        "maskCount": int(sagittal_quality.get("maskCount", 0)) + int(axial_quality.get("maskCount", 0)),
        "landmarkCount": int(sagittal_quality.get("landmarkCount", 0)) + int(axial_quality.get("landmarkCount", 0)),
        "measurementCount": int(sagittal_quality.get("measurementCount", 0)) + int(axial_quality.get("measurementCount", 0)),
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
    }


def has_axial_input(request: MultiplanarRunRequest) -> bool:
    return bool(request.axial_input_id or request.axial_input_path)


def three_d_status(sagittal: Dict[str, Any], axial: Dict[str, Any] | None) -> Dict[str, Any]:
    if axial is None:
        return {
            "status": "blocked_missing_axial",
            "enabled": False,
            "sourcePlaneRunIds": {
                "sagittal": sagittal.get("runId"),
                "axial": None,
            },
            "requiredInputs": ["axial_masks", "spacing", "slice_index_mapping"],
        }
    return {
        "status": "experimental_blocked_missing_anatomical_mapping",
        "enabled": False,
        "sourcePlaneRunIds": {
            "sagittal": sagittal.get("runId"),
            "axial": axial.get("runId"),
        },
        "requiredInputs": ["sagittal_masks", "axial_masks", "spacing", "slice_index_mapping"],
    }


def compact_message(exc: Exception) -> str:
    return sanitize_public_text(str(exc).replace("\n", " ")[:240])


def log_multiplanar_event(
    request: MultiplanarRunRequest,
    trace_id: str | None,
    phase: str,
    *,
    exc_type: str | None = None,
    exc_message: str | None = None,
) -> None:
    payload = {
        "event": "multiplanar_run",
        "phase": phase,
        "traceId": trace_id,
        "caseId": request.case_id,
        "sagittalInputIdPresent": bool(request.sagittal_input_id),
        "sagittalInputPathPresent": bool(request.sagittal_input_path),
        "axialInputIdPresent": bool(request.axial_input_id),
        "axialInputPathPresent": bool(request.axial_input_path),
    }
    if exc_type:
        payload["exceptionType"] = exc_type
    if exc_message:
        payload["message"] = exc_message
    LOGGER.info("multiplanar_run", extra={"multiplanar": payload})
