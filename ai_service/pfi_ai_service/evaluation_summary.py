from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .report_summary import recent_agent_report_summaries


def evaluation_summary(output_dir: Path) -> Dict[str, Any]:
    reports = recent_agent_report_summaries(output_dir / "agent_reports", limit=100)
    items = reports.get("items", [])
    measurement_counts = [item.get("measurementCount", 0) for item in items]
    mask_counts = [item.get("maskCount", 0) for item in items]
    contract_runs = sum(1 for item in items if item.get("inferenceMode") == "contract")
    real_runs = sum(1 for item in items if item.get("inferenceMode") == "real")
    return {
        "status": "evaluation_evidence_available" if items else "evaluation_evidence_empty",
        "reportCount": len(items),
        "contractRunCount": contract_runs,
        "realRunCount": real_runs,
        "averageMeasurementCount": average(measurement_counts),
        "averageMaskCount": average(mask_counts),
        "hasProfessionalReviewEvidence": any(item.get("reviewStatus") not in (None, "pendiente") for item in items),
        "hasRealInferenceEvidence": real_runs > 0,
        "items": items,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def average(values: list[Any]) -> float:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return round(sum(numeric) / len(numeric), 2) if numeric else 0.0
