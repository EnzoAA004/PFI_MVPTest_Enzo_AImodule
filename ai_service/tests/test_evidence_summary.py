from __future__ import annotations

from pfi_ai_service.evidence_summary import evidence_summary


def test_evidence_summary_empty_outputs(tmp_path) -> None:
    result = evidence_summary(tmp_path)

    assert result["status"] == "evidence_summary_ready"
    assert result["reportCount"] == 0
    assert result["hasReports"] is False
    assert result["latestRunId"] == ""
    assert result["humanReviewRequired"] is True
