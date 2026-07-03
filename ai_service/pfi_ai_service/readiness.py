from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .contract_schema import contract_verification
from .model_artifacts import verify_model_artifacts
from .report_summary import recent_agent_report_summaries


def build_readiness(output_dir: Path) -> Dict[str, Any]:
    contract = contract_verification()
    artifacts = verify_model_artifacts()
    reports = recent_agent_report_summaries(output_dir / "agent_reports", limit=1)
    contract_valid = bool(contract.get("valid"))
    ready_for_real = contract_valid and bool(artifacts.get("readyForRealInference")) and bool(artifacts.get("valid"))
    ready_for_demo = contract_valid and bool(artifacts.get("defaultInferenceMode"))
    status = "ready_for_real_inference" if ready_for_real else "contract_ready"
    if not contract_valid:
        status = "degraded_contract_invalid"

    completion = mvp_completion(contract_valid, artifacts, reports, ready_for_demo, ready_for_real)
    return {
        "status": status,
        "service": "pfi-ai-module",
        "readyForDemo": ready_for_demo,
        "readyForRealInference": ready_for_real,
        "defaultInferenceMode": artifacts.get("defaultInferenceMode", "contract"),
        "recommendedInferenceMode": "real" if ready_for_real else "contract",
        "recommendedAction": recommended_action(contract_valid, artifacts),
        "mvpCompletion": completion,
        "contract": {
            "valid": contract_valid,
            "schemaVersion": contract.get("schemaVersion"),
            "schemaHash": contract.get("schemaHash"),
            "hashValid": contract.get("hashValid"),
            "governanceValid": contract.get("governanceValid"),
            "missingRootFields": contract.get("missingRootFields", []),
        },
        "modelArtifacts": {
            "valid": artifacts.get("valid"),
            "status": artifacts.get("status"),
            "modelsRegistered": artifacts.get("modelsRegistered"),
            "artifactsAvailable": artifacts.get("artifactsAvailable"),
            "artifactsMissing": artifacts.get("artifactsMissing"),
            "artifactsHashed": artifacts.get("artifactsHashed"),
            "missingArtifacts": artifacts.get("missingArtifacts", []),
            "unverifiedArtifacts": artifacts.get("unverifiedArtifacts", []),
        },
        "reports": {
            "count": reports.get("count", 0),
            "hasRecentReports": bool(reports.get("count", 0)),
        },
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def mvp_completion(
    contract_valid: bool,
    artifacts: Dict[str, Any],
    reports: Dict[str, Any],
    ready_for_demo: bool,
    ready_for_real: bool,
) -> Dict[str, Any]:
    items = [
        completion_item("technical_contract", contract_valid),
        completion_item("human_review_governance", True),
        completion_item("artifact_verification", artifacts.get("status") is not None),
        completion_item("run_reports", bool(reports.get("count", 0))),
        completion_item("demo_readiness", ready_for_demo),
        completion_item("real_inference_readiness", ready_for_real),
    ]
    complete = sum(1 for item in items if item["status"] == "complete")
    return {
        "completionPercent": round(complete * 100 / len(items)),
        "completeItems": complete,
        "totalItems": len(items),
        "items": items,
    }


def completion_item(key: str, complete: bool) -> Dict[str, Any]:
    return {"key": key, "status": "complete" if complete else "pending"}


def recommended_action(contract_valid: bool, artifacts: Dict[str, Any]) -> str:
    if not contract_valid:
        return "Revisar contrato tecnico: schema hash, campos minimos y reglas de gobernanza."
    if artifacts.get("valid"):
        return "Artifact verificado: el entorno puede evaluar inferencia real bajo revision profesional."
    missing = artifacts.get("artifactsMissing", 0)
    if isinstance(missing, int) and missing > 0:
        return "Faltan artifacts .pt: mantener modo contrato hasta subir y verificar modelos reales."
    return "Mantener modo contrato y revisar integridad de artifacts antes de inferencia real."
