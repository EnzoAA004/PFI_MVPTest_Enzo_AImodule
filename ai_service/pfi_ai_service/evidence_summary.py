from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .evaluation_summary import evaluation_summary


def evidence_summary(output_dir: Path) -> Dict[str, Any]:
    evaluation = evaluation_summary(output_dir)
    return {
        "status": "evidence_summary_ready",
        "latestRunId": evaluation.get("latestRunId", ""),
        "reportCount": evaluation.get("reportCount", 0),
        "hasReports": evaluation.get("hasReports", False),
        "hasRealInferenceEvidence": evaluation.get("hasRealInferenceEvidence", False),
        "hasProfessionalReviewEvidence": evaluation.get("hasProfessionalReviewEvidence", False),
        "humanReviewRequired": evaluation.get("humanReviewRequired", True),
        "notClinicalDiagnosis": evaluation.get("notClinicalDiagnosis", True),
    }
