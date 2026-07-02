from __future__ import annotations

from pfi_ai_service.pipeline import PipelineRunRequest, run_pipeline


def test_pipeline_run_returns_visual_review_contract() -> None:
    response = run_pipeline(PipelineRunRequest(
        caseId="CASE-DEMO-0142",
        plane="sagittal",
        modelKey="sagittal_spider",
        inputPath="demo/CASE-DEMO-0142",
        metadata={"patientId": "PAT-0087", "studyDate": "2026-07-01"},
    ))

    assert response["runId"]
    assert response["caseId"] == "CASE-DEMO-0142"
    assert response["aiOutput"]["status"] == "contract_ready"
    assert response["aiOutput"]["humanReviewRequired"] is True
    assert response["aiOutput"]["notClinicalDiagnosis"] is True
    assert len(response["series"]) >= 3
    assert len(response["masks"]) >= 3
    assert len(response["landmarks"]) >= 3
    assert len(response["measurementValues"]) >= 3
    assert response["quality"]["measurementCount"] == len(response["measurementValues"])
    assert response["quality"]["measurementsDerivedFromContours"] is True
    assert all("aiValue" in item for item in response["measurementValues"])
    assert all("reviewerValue" in item for item in response["measurementValues"])


def test_real_inference_request_degrades_to_contract_mode() -> None:
    response = run_pipeline(PipelineRunRequest(
        caseId="CASE-DEMO-0142",
        plane="sagittal",
        modelKey="sagittal_spider",
        inputPath="demo/CASE-DEMO-0142",
        metadata={"inferenceMode": "real"},
    ))

    assert response["aiOutput"]["requestedInferenceMode"] == "real"
    assert response["aiOutput"]["inferenceMode"] == "contract"
    assert response["aiOutput"]["realInferenceAvailable"] is False
    assert "real_inference_requested_but_contract_mode_used" in response["agentDecision"]["flags"]
