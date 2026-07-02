from __future__ import annotations

import json

from pfi_ai_service.report_summary import recent_agent_report_summaries


def test_recent_agent_report_summaries_lists_latest_reports(tmp_path) -> None:
    reports_dir = tmp_path / "agent_reports"
    reports_dir.mkdir()
    first = reports_dir / "run-a.json"
    second = reports_dir / "run-b.json"
    first.write_text(json.dumps(report("run-a", "CASE-A", "trace-a")), encoding="utf-8")
    second.write_text(json.dumps(report("run-b", "CASE-B", "trace-b")), encoding="utf-8")

    result = recent_agent_report_summaries(reports_dir, limit=10)

    assert result["status"] == "ok"
    assert result["count"] == 2
    assert result["skipped"] == 0
    assert result["humanReviewRequired"] is True
    assert result["notClinicalDiagnosis"] is True
    run_ids = {item["runId"] for item in result["items"]}
    assert run_ids == {"run-a", "run-b"}
    assert all("reportFile" in item for item in result["items"])
    assert all("reportModifiedAt" in item for item in result["items"])


def test_recent_agent_report_summaries_handles_missing_directory(tmp_path) -> None:
    result = recent_agent_report_summaries(tmp_path / "missing", limit=10)

    assert result["status"] == "ok"
    assert result["count"] == 0
    assert result["items"] == []


def report(run_id: str, case_id: str, trace_id: str) -> dict:
    return {
        "runId": run_id,
        "traceId": trace_id,
        "caseId": case_id,
        "plane": "sagittal",
        "modelKey": "sagittal_spider",
        "aiOutput": {"inferenceMode": "contract", "modelReadiness": "contract_only_missing_artifact"},
        "quality": {"maskCount": 3, "landmarkCount": 3, "measurementCount": 3},
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "metadata": {"diagnosisGenerated": False, "deidentified": True},
    }
