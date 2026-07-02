from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .settings import MODEL_REGISTRY, get_settings


MODEL_PATH_KEYS = {
    "sagittal_spider": "sagittal_model_path",
    "axial_t2_alkafri": "axial_model_path",
}


def _artifact_status(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    size_bytes = path.stat().st_size if exists and path.is_file() else 0
    return {
        "path": str(path),
        "exists": exists,
        "sizeBytes": size_bytes,
        "sizeMb": round(size_bytes / 1024 / 1024, 2) if size_bytes else 0,
        "extension": path.suffix,
    }


def model_artifact_path(model_key: str) -> Path | None:
    settings = get_settings()
    attr = MODEL_PATH_KEYS.get(model_key)
    return getattr(settings, attr) if attr else None


def model_status(model_key: str, info: Dict[str, Any]) -> Dict[str, Any]:
    path = model_artifact_path(model_key)
    artifact = _artifact_status(path) if path is not None else {
        "path": None,
        "exists": False,
        "sizeBytes": 0,
        "sizeMb": 0,
        "extension": None,
    }
    real_ready = bool(artifact["exists"])
    return {
        **info,
        "key": model_key,
        "version": info.get("version", "contract-v1"),
        "artifact": artifact,
        "readiness": "real_artifact_available" if real_ready else "contract_only_missing_artifact",
        "inferenceModes": {
            "contract": True,
            "mock": True,
            "real": real_ready,
        },
        "availableForRealInference": real_ready,
        "enabled": True,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def registry_with_artifact_status() -> Dict[str, Dict[str, Any]]:
    return {model_key: model_status(model_key, dict(info)) for model_key, info in MODEL_REGISTRY.items()}


def artifact_summary() -> Dict[str, Any]:
    models = registry_with_artifact_status()
    available = sum(1 for model in models.values() if model["availableForRealInference"])
    missing = len(models) - available
    return {
        "modelsRegistered": len(models),
        "artifactsAvailable": available,
        "artifactsMissing": missing,
        "readyForRealInference": available == len(models) and len(models) > 0,
        "defaultInferenceMode": "real" if available == len(models) and len(models) > 0 else "contract",
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }
