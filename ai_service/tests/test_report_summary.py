from __future__ import annotations

from pfi_ai_service.report_summary import summarize_agent_report


def test_summarize_agent_report_extracts_traceability_and_counts() -> None:
    report = {
        "runId": "run-123",
        "traceId": "trace-abc",
        "caseId": "CASE-001",
        "studyId": "STUDY-001",
        "patientId": "PAT-001",
        "studyDate": "2026-07-01",
        "plane": "sagittal",
        "modelKey": "sagittal_spider",
        "modelVersion": "contract-v1",
        "reviewStatus": "pendiente",
        "masks": [{"id": "m1"}],
        "landmarks": [{"id": "l1"}, {"id": "l2"}],
        "measurementValues": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "aiOutput": {
            "inferenceMode": "contract",
            "requestedInferenceMode": "real",
            "modelReadiness": "contract_only_missing_artifact",
            "realInferenceAvailable": False,
        },
        "quality": {
            "measurementsDerivedFromContours": True,
        },
        "modelArtifact": {
            "artifactHash": None,
            "artifactIntegrityStatus": "missing_artifact",
            "artifact": {"exists": False},
        },
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "metadata": {"diagnosisGenerated": False, "deidentified": True, "source": "test"},
    }

    summary = summarize_agent_report(report)

    assert summary["runId"] == "run-123"
    assert summary["traceId"] == "trace-abc"
    assert summary["caseId"] == "CASE-001"
    assert summary["inferenceMode"] == "contract"
    assert summary["requestedInferenceMode"] == "real"
    assert summary["maskCount"] == 1
    assert summary["landmarkCount"] == 2
    assert summary["measurementCount"] == 3
    assert summary["humanReviewRequired"] is True
    assert summary["notClinicalDiagnosis"] is True
    assert summary["diagnosisGenerated"] is False
