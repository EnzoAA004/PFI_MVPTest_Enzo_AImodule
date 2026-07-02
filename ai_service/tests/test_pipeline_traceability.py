from __future__ import annotations

import json

from pfi_ai_service.pipeline import PipelineRunRequest, run_pipeline


def test_pipeline_run_exposes_trace_id_and_persists_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    request = PipelineRunRequest(
        caseId="CASE-TRACE-001",
        plane="sagittal",
        modelKey="sagittal_spider",
        inputPath="demo/CASE-TRACE-001",
        metadata={"traceId": "demo-trace-pipeline", "source": "test"},
    )

    response = run_pipeline(request)

    assert response["traceId"] == "demo-trace-pipeline"
    assert response["metadata"]["traceId"] == "demo-trace-pipeline"
    assert response["metadata"]["source"] == "test"
    report_path = tmp_path / "outputs" / "agent_reports" / f"{response['runId']}.json"
    assert report_path.exists()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["traceId"] == "demo-trace-pipeline"
    assert persisted["metadata"]["traceId"] == "demo-trace-pipeline"
