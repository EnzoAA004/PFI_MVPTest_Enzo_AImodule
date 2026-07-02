from __future__ import annotations

from pfi_ai_service.model_artifacts import verify_model_artifacts


def test_verify_model_artifacts_reports_missing_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PFI_MODEL_DIR", str(tmp_path / "models"))

    verification = verify_model_artifacts()

    assert verification["status"] == "degraded_contract_mode"
    assert verification["valid"] is False
    assert verification["defaultInferenceMode"] == "contract"
    assert verification["artifactsMissing"] >= 1
    assert len(verification["missingArtifacts"]) >= 1
    assert verification["humanReviewRequired"] is True
    assert verification["notClinicalDiagnosis"] is True
