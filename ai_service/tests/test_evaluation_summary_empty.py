from __future__ import annotations

from pfi_ai_service.evaluation_summary import evaluation_summary


def test_evaluation_summary_empty_has_safe_latest_run_id(tmp_path) -> None:
    result = evaluation_summary(tmp_path)

    assert result["status"] == "evaluation_evidence_empty"
    assert result["reportCount"] == 0
    assert result["hasReports"] is False
    assert result["latestRunId"] == ""
    assert result["humanReviewRequired"] is True
    assert result["notClinicalDiagnosis"] is True
