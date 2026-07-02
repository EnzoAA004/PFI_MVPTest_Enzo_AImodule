from __future__ import annotations

from pfi_ai_service.model_artifacts import artifact_summary, registry_with_artifact_status
from pfi_ai_service.settings import MODEL_REGISTRY


def test_registry_with_artifact_status_exposes_all_models() -> None:
    models = registry_with_artifact_status()

    assert set(models.keys()) == set(MODEL_REGISTRY.keys())
    for model_key, model in models.items():
        assert model["key"] == model_key
        assert model["artifact"]["path"]
        assert model["artifact"]["extension"] == ".pt"
        assert model["readiness"] in {"contract_only_missing_artifact", "real_artifact_available"}
        assert model["inferenceModes"]["contract"] is True
        assert model["inferenceModes"]["mock"] is True
        assert model["inferenceModes"]["real"] == model["availableForRealInference"]
        assert model["humanReviewRequired"] is True
        assert model["notClinicalDiagnosis"] is True


def test_artifact_summary_is_safe_for_diagnostics() -> None:
    summary = artifact_summary()

    assert summary["modelsRegistered"] == len(MODEL_REGISTRY)
    assert summary["artifactsAvailable"] + summary["artifactsMissing"] == len(MODEL_REGISTRY)
    assert summary["defaultInferenceMode"] in {"contract", "real"}
    assert summary["humanReviewRequired"] is True
    assert summary["notClinicalDiagnosis"] is True
