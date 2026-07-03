from __future__ import annotations

import json

from pfi_ai_service.evaluation_summary import evaluation_summary


def test_evaluation_summary_exposes_latest_run_id(tmp_path) -> None:
    reports = tmp_path / "agent_reports"
    reports.mkdir()
    (reports / "run-one.json").write_text(json.dumps({
        "runId": "run-one",
        "caseId": "CASE-1",
        "aiOutput": {"inferenceMode": "contract"},
        "quality": {"maskCount": 2, "measurementCount": 3},
        "metadata": {"diagnosisGenerated": False}
    }), encoding="utf-8")

    result = evaluation_summary(tmp_path)

    assert result["hasReports"] is True
    assert result["latestRunId"] == "run-one"
    assert result["reportCount"] == 1
