from __future__ import annotations

import json

from pfi_ai_service.evidence_summary import evidence_summary


def test_evidence_summary_reports_latest_run(tmp_path) -> None:
    reports = tmp_path / "agent_reports"
    reports.mkdir()
    (reports / "run-latest.json").write_text(json.dumps({
        "runId": "run-latest",
        "caseId": "CASE-1",
        "aiOutput": {"inferenceMode": "contract"},
        "quality": {"maskCount": 2, "measurementCount": 3},
        "metadata": {"diagnosisGenerated": False},
        "reviewStatus": "aceptado"
    }), encoding="utf-8")

    result = evidence_summary(tmp_path)

    assert result["status"] == "evidence_summary_ready"
    assert result["hasReports"] is True
    assert result["latestRunId"] == "run-latest"
    assert result["hasProfessionalReviewEvidence"] is True
