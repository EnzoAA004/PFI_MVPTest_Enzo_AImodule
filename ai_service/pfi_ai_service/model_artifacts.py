from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .settings import MODEL_REGISTRY, get_settings


MODEL_PATH_KEYS = {
    "sagittal_spider": "sagittal_model_path",
    "axial_t2_alkafri": "axial_model_path",
}


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _last_modified(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _artifact_status(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    size_bytes = path.stat().st_size if exists and path.is_file() else 0
    sha256 = _sha256(path) if exists and path.is_file() else None
    return {
        "path": str(path),
        "exists": exists,
        "sizeBytes": size_bytes,
        "sizeMb": round(size_bytes / 1024 / 1024, 2) if size_bytes else 0,
        "extension": path.suffix,
        "hashAlgorithm": "sha256",
        "sha256": sha256,
        "lastModified": _last_modified(path),
        "integrityStatus": "hashed" if sha256 else "missing_artifact",
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
        "hashAlgorithm": "sha256",
        "sha256": None,
        "lastModified": None,
        "integrityStatus": "missing_artifact",
    }
    real_ready = bool(artifact["exists"])
    return {
        **info,
        "key": model_key,
        "version": info.get("version", "contract-v1"),
        "artifact": artifact,
        "artifactHash": artifact["sha256"],
        "artifactIntegrityStatus": artifact["integrityStatus"],
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
    hashed = sum(1 for model in models.values() if model.get("artifactHash"))
    return {
        "modelsRegistered": len(models),
        "artifactsAvailable": available,
        "artifactsMissing": missing,
        "artifactsHashed": hashed,
        "readyForRealInference": available == len(models) and len(models) > 0,
        "defaultInferenceMode": "real" if available == len(models) and len(models) > 0 else "contract",
        "hashAlgorithm": "sha256",
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }
