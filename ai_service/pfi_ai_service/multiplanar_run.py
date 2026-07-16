from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .pipeline import PipelineRunRequest, run_pipeline
from .reporting import write_json
from .settings import get_settings


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
    sagittal_request = plane_request(request, "sagittal", run_id, trace_id)
    axial_request = plane_request(request, "axial", run_id, trace_id)

    sagittal = run_pipeline(sagittal_request)
    axial = run_pipeline(axial_request)

    response = {
        "status": "multiplanar_run_ready",
        "schemaVersion": "multiplanar-run-v1",
        "runId": run_id,
        "traceId": trace_id,
        "caseId": request.case_id,
        "workspaceMode": "dual_plane_with_3d_context",
        "requestedInferenceMode": requested_mode(request.metadata),
        "effectiveInferenceMode": effective_workspace_mode(sagittal, axial),
        "planes": {
            "sagittal": sagittal,
            "axial": axial,
        },
        "threeD": {
            "status": "pending_registered_reconstruction",
            "enabled": False,
            "sourcePlaneRunIds": {
                "sagittal": sagittal.get("runId"),
                "axial": axial.get("runId"),
            },
            "requiredInputs": ["sagittal_masks", "axial_masks", "spacing", "slice_index_mapping"],
        },
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


def effective_workspace_mode(sagittal: Dict[str, Any], axial: Dict[str, Any]) -> str:
    modes = {
        str((sagittal.get("aiOutput") or {}).get("inferenceMode", "contract")),
        str((axial.get("aiOutput") or {}).get("inferenceMode", "contract")),
    }
    return modes.pop() if len(modes) == 1 else "mixed"


def workspace_quality(sagittal: Dict[str, Any], axial: Dict[str, Any]) -> Dict[str, Any]:
    sagittal_quality = sagittal.get("quality") if isinstance(sagittal.get("quality"), dict) else {}
    axial_quality = axial.get("quality") if isinstance(axial.get("quality"), dict) else {}
    return {
        "sagittal": sagittal_quality,
        "axial": axial_quality,
        "planeCount": 2,
        "maskCount": int(sagittal_quality.get("maskCount", 0)) + int(axial_quality.get("maskCount", 0)),
        "landmarkCount": int(sagittal_quality.get("landmarkCount", 0)) + int(axial_quality.get("landmarkCount", 0)),
        "measurementCount": int(sagittal_quality.get("measurementCount", 0)) + int(axial_quality.get("measurementCount", 0)),
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
    }
