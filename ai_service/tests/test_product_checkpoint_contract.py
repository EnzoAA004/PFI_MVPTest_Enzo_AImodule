from fastapi.testclient import TestClient

from pfi_ai_service.api import app


def test_product_checkpoint_contract_separates_green_synthetic_gates_from_real_e2e() -> None:
    response = TestClient(app).get("/v2/product-checkpoint/contract")
    assert response.status_code == 200
    body = response.json()

    assert body["schemaVersion"] == "pfi.p10-9.ai-product-checkpoint.v1"
    assert body["frontendRequiredForValidation"] is False
    assert body["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True
    assert body["autonomousDiagnosis"] is False

    assert body["study"]["segmentationScope"] == "supported_analyzable_series_all_slices"
    assert body["study"]["allStudySeriesAutomaticallySegmented"] is False
    assert body["fullSeriesSegmentation"]["syntheticCoverageTested"] is True
    assert body["anatomyPresentation"]["severityEncodedByColor"] is False

    gates = body["e2eGates"]
    assert gates["p10_6Regression"] is True
    assert gates["p10_7PreprocessingParity"] is True
    assert gates["fullSeriesSyntheticCoverage"] is True
    assert gates["p10_7LocalizationOrchestrationSynthetic"] is True
    assert gates["p10_7RealCheckpointSmoke"] is False
    assert gates["automaticDiscLocalizationRealStudy"] is False
    assert gates["aiBackendHttp"] is False
    assert gates["checkpointReadyForFrontendHandoff"] is False
