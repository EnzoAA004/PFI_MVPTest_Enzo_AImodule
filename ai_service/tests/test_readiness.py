from __future__ import annotations

from pfi_ai_service.readiness import build_readiness


def test_build_readiness_defaults_to_contract_mode_when_artifacts_are_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PFI_MODEL_DIR", str(tmp_path / "models"))

    readiness = build_readiness(tmp_path / "outputs")

    assert readiness["service"] == "pfi-ai-module"
    assert readiness["status"] == "contract_ready"
    assert readiness["readyForDemo"] is True
    assert readiness["readyForRealInference"] is False
    assert readiness["defaultInferenceMode"] == "contract"
    assert readiness["recommendedInferenceMode"] == "contract"
    assert readiness["contract"]["valid"] is True
    assert readiness["modelArtifacts"]["valid"] is False
    assert readiness["humanReviewRequired"] is True
    assert readiness["notClinicalDiagnosis"] is True
