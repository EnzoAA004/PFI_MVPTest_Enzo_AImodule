from __future__ import annotations

from typing import Any, Dict, List


HUMAN_REVIEW_REQUIRED = True
NOT_CLINICAL_DIAGNOSIS = True


def build_agent_decision(
    *,
    plane: str,
    model_key: str,
    flags: List[str] | None = None,
) -> Dict[str, Any]:
    """Build a technical review decision, never a clinical diagnosis."""
    reasons = list(flags or [])
    if not reasons:
        reasons.append("resultado tecnico generado para revision profesional")

    status = "requires_professional_review"
    priority = "standard"
    if any("missing" in reason or "unknown" in reason for reason in reasons):
        priority = "high"

    frontend_priority = "alta" if priority == "high" else "media"
    frontend_status = "requiere_revision"

    return {
        "agent_status": status,
        "review_priority": priority,
        "agent_reasons": reasons,
        "recommended_action": "Revisar overlays, mediciones y trazabilidad antes de usar el resultado.",
        "plane": plane,
        "model_key": model_key,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
        "not_clinical_diagnosis": NOT_CLINICAL_DIAGNOSIS,
        "status": frontend_status,
        "priority": frontend_priority,
        "flags": reasons,
        "reasons": reasons,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def regression_test_report() -> Dict[str, Any]:
    decision = build_agent_decision(
        plane="sagittal",
        model_key="sagittal_spider",
        flags=["smoke_contract_only_no_medical_input"],
    )
    return {
        "status": "ok",
        "checks": {
            "human_review_required": decision["human_review_required"],
            "not_clinical_diagnosis": decision["not_clinical_diagnosis"],
            "agent_policy_loaded": True,
        },
        "agent_decision": decision,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
        "not_clinical_diagnosis": NOT_CLINICAL_DIAGNOSIS,
    }
