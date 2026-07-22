from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .model_manifest import read_model_manifest
from .settings import MODEL_REGISTRY, get_settings


MODEL_PATH_KEYS = {
    "sagittal_spider": "sagittal_model_path",
    "axial_t2_alkafri": "axial_model_path",
}

MODEL_URI_KEYS = {
    "sagittal_spider": "sagittal_model_uri",
    "axial_t2_alkafri": "axial_model_uri",
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


def model_artifact_uri(model_key: str) -> str | None:
    settings = get_settings()
    if model_key == "sagittal_spider" and settings.sagittal_release_uri:
        return settings.sagittal_release_uri
    attr = MODEL_URI_KEYS.get(model_key)
    return getattr(settings, attr) if attr else None


def model_status(model_key: str, info: Dict[str, Any]) -> Dict[str, Any]:
    path = model_artifact_path(model_key)
    external_uri = model_artifact_uri(model_key)
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
    artifact["externalUriConfigured"] = bool(external_uri)
    manifest = read_model_manifest(path)
    artifact_ready = bool(artifact["exists"] and artifact["sha256"])
    baseline_ready = artifact_ready and bool(manifest.get("baselineReady"))
    readiness = "real_baseline_ready" if baseline_ready else "real_artifact_missing_manifest" if artifact_ready else "external_artifact_configured" if external_uri else "contract_only_missing_artifact"
    return {
        **info,
        "key": model_key,
        "version": manifest.get("version") or info.get("version", "contract-v1"),
        "artifact": artifact,
        "manifest": manifest,
        "artifactHash": artifact["sha256"],
        "artifactIntegrityStatus": artifact["integrityStatus"],
        "readiness": readiness,
        "inferenceModes": {
            "contract": True,
            "mock": True,
            "real": baseline_ready,
            "real_baseline": baseline_ready,
        },
        "availableForRealInference": baseline_ready,
        "baselineReady": baseline_ready,
        "externalArtifactConfigured": bool(external_uri),
        "enabled": True,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def registry_with_artifact_status() -> Dict[str, Dict[str, Any]]:
    return {model_key: model_status(model_key, dict(info)) for model_key, info in MODEL_REGISTRY.items()}


def artifact_summary() -> Dict[str, Any]:
    models = registry_with_artifact_status()
    available = sum(1 for model in models.values() if model["availableForRealInference"])
    missing = sum(1 for model in models.values() if not model.get("artifact", {}).get("exists"))
    hashed = sum(1 for model in models.values() if model.get("artifactHash"))
    baseline_ready = sum(1 for model in models.values() if model.get("baselineReady"))
    external_configured = sum(1 for model in models.values() if model.get("externalArtifactConfigured"))
    return {
        "modelsRegistered": len(models),
        "artifactsAvailable": sum(1 for model in models.values() if model.get("artifact", {}).get("exists")),
        "artifactsMissing": missing,
        "artifactsHashed": hashed,
        "baselineModelsReady": baseline_ready,
        "externalArtifactsConfigured": external_configured,
        "readyForRealInference": available == len(models) and len(models) > 0,
        "defaultInferenceMode": "real_baseline" if available == len(models) and len(models) > 0 else "contract",
        "hashAlgorithm": "sha256",
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def verify_model_artifacts() -> Dict[str, Any]:
    models = registry_with_artifact_status()
    missing = []
    missing_manifest = []
    unverified = []
    verified = []
    for model_key, status in models.items():
        artifact = status.get("artifact", {})
        manifest = status.get("manifest", {})
        exists = bool(artifact.get("exists"))
        sha256 = artifact.get("sha256")
        integrity = artifact.get("integrityStatus")
        baseline_ready = bool(status.get("baselineReady"))
        model_result = {
            "modelKey": model_key,
            "plane": status.get("plane"),
            "version": status.get("version"),
            "path": artifact.get("path"),
            "exists": exists,
            "externalArtifactConfigured": bool(status.get("externalArtifactConfigured")),
            "hashAlgorithm": artifact.get("hashAlgorithm", "sha256"),
            "sha256": sha256,
            "integrityStatus": integrity,
            "manifestStatus": manifest.get("status"),
            "manifestValid": bool(manifest.get("valid")),
            "baselineReady": baseline_ready,
            "readiness": status.get("readiness"),
            "availableForRealInference": bool(status.get("availableForRealInference")),
            "verified": exists and bool(sha256) and integrity == "hashed" and baseline_ready,
        }
        if not exists:
            missing.append(model_result)
        elif not baseline_ready:
            missing_manifest.append(model_result)
        elif not model_result["verified"]:
            unverified.append(model_result)
        else:
            verified.append(model_result)

    summary = artifact_summary()
    valid = not missing and not missing_manifest and not unverified and bool(models)
    return {
        "status": "real_baseline_verified" if valid else "degraded_contract_mode",
        "valid": valid,
        "readyForRealInference": summary["readyForRealInference"],
        "defaultInferenceMode": summary["defaultInferenceMode"],
        "modelsRegistered": summary["modelsRegistered"],
        "artifactsAvailable": summary["artifactsAvailable"],
        "artifactsMissing": summary["artifactsMissing"],
        "artifactsHashed": summary["artifactsHashed"],
        "baselineModelsReady": summary["baselineModelsReady"],
        "externalArtifactsConfigured": summary["externalArtifactsConfigured"],
        "hashAlgorithm": "sha256",
        "verifiedModels": verified,
        "missingArtifacts": missing,
        "missingManifestOrBaselineEvidence": missing_manifest,
        "unverifiedArtifacts": unverified,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }
