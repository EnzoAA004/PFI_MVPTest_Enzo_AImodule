from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS, build_agent_decision
from .settings import MODEL_REGISTRY, get_settings


class PipelineRunRequest(BaseModel):
    case_id: str = Field(..., alias="caseId")
    plane: Literal["sagittal", "axial"]
    model_key: str = Field(..., alias="modelKey")
    input_path: str = Field(..., alias="inputPath")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True


def _run_id_for(request: PipelineRunRequest) -> str:
    raw = f"{request.case_id}|{request.plane}|{request.model_key}|{request.input_path}"
    return sha256(raw.encode("utf-8")).hexdigest()[:16]


def _overlay_path_for(run_id: str) -> Optional[str]:
    settings = get_settings()
    overlay_path = settings.output_dir / "overlays" / f"{run_id}.png"
    if Path(overlay_path).exists():
        return str(overlay_path)
    return None


def _as_backend_response(
    *,
    run_id: str,
    request: PipelineRunRequest,
    measurements: Dict[str, Any],
    overlay_path: Optional[str],
    agent_decision: Dict[str, Any],
) -> Dict[str, Any]:
    response = {
        "run_id": run_id,
        "runId": run_id,
        "case_id": request.case_id,
        "caseId": request.case_id,
        "plane": request.plane,
        "model_key": request.model_key,
        "modelKey": request.model_key,
        "input_path": request.input_path,
        "inputPath": request.input_path,
        "measurements": measurements,
        "overlay_path": overlay_path,
        "overlayPath": overlay_path,
        "agent_decision": agent_decision,
        "agentDecision": agent_decision,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "not_clinical_diagnosis": NOT_CLINICAL_DIAGNOSIS,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
        "metadata": request.metadata,
    }
    return response


def run_pipeline(request: PipelineRunRequest) -> Dict[str, Any]:
    flags = []
    model_info = MODEL_REGISTRY.get(request.model_key)
    if model_info is None:
        flags.append(f"unknown_model_key:{request.model_key}")
    elif model_info.get("plane") != request.plane:
        flags.append(f"model_plane_mismatch:expected={model_info.get('plane')},received={request.plane}")

    run_id = _run_id_for(request)
    overlay_path = _overlay_path_for(run_id)
    measurements: Dict[str, Any] = {
        "status": "pending_real_inference",
        "values": [],
        "source": "contract_smoke_pipeline",
        "description": "No se calcularon mediciones clinicas; pendiente conectar inferencia real.",
    }
    agent_decision = build_agent_decision(
        plane=request.plane,
        model_key=request.model_key,
        flags=flags or ["contract_smoke_pipeline_requires_review"],
    )

    return _as_backend_response(
        run_id=run_id,
        request=request,
        measurements=measurements,
        overlay_path=overlay_path,
        agent_decision=agent_decision,
    )
