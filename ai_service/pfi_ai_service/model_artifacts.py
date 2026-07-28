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
    axial_runtime_ready = artifact_ready and axial_candidate_runtime_ready(model_key, manifest)
    available_for_real = baseline_ready or axial_runtime_ready
    quality_gate = manifest_quality_gate(manifest)
    quality_gate_passed = quality_gate.get("qualityGatePassed")
    held_out_reuse_warning = manifest.get("heldOutReuseWarning") or quality_gate.get("heldOutReuseWarning")
    readiness = (
        "real_baseline_ready"
        if baseline_ready
        else "real_candidate_ready"
        if axial_runtime_ready
        else "real_artifact_missing_manifest"
        if artifact_ready
        else "external_artifact_configured"
        if external_uri
        else "contract_only_missing_artifact"
    )
    training_status = manifest.get("trainingStatus")
    return {
        **info,
        "key": model_key,
        "version": manifest.get("version") or info.get("version", "contract-v1"),
        "artifact": artifact,
        "manifest": manifest,
        "artifactHash": artifact["sha256"],
        "artifactIntegrityStatus": artifact["integrityStatus"],
        "readiness": readiness,
        "trainingStatus": training_status,
        "qualityGatePassed": quality_gate_passed,
        "heldOutReuseWarning": held_out_reuse_warning,
        "qualityGate": quality_gate,
        "metrics": manifest.get("metrics"),
        "inferenceModes": {
            "contract": True,
            "mock": True,
            "real": available_for_real,
            "real_baseline": available_for_real,
        },
        "availableForRealInference": available_for_real,
        "baselineReady": baseline_ready,
        "manifestBaselineReady": baseline_ready,
        "runtimeQualification": "axial_candidate_runtime_ready" if axial_runtime_ready and not baseline_ready else None,
        "externalArtifactConfigured": bool(external_uri),
        "enabled": True,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def manifest_quality_gate(manifest: Dict[str, Any]) -> Dict[str, Any]:
    content = manifest.get("content") if isinstance(manifest.get("content"), dict) else {}
    quality_gate = content.get("qualityGate") if isinstance(content.get("qualityGate"), dict) else {}
    return quality_gate


def axial_candidate_runtime_ready(model_key: str, manifest: Dict[str, Any]) -> bool:
    if model_key != "axial_t2_alkafri":
        return False
    if not manifest.get("valid"):
        return False
    if manifest.get("trainingStatus") != "candidate_below_quality_gate":
        return False
    if manifest.get("sha256Status") != "MATCH":
        return False
    content = manifest.get("content") if isinstance(manifest.get("content"), dict) else {}
    quality_gate = manifest_quality_gate(manifest)
    runtime_verification = quality_gate.get("runtimeVerification") if isinstance(quality_gate.get("runtimeVerification"), dict) else {}
    if runtime_verification.get("finite") is not True:
        return False
    metrics = content.get("metrics") if isinstance(content.get("metrics"), dict) else {}
    test_metrics = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}
    dice_excluding_raw0 = test_metrics.get("dice_macro_excluding_raw0") or quality_gate.get("diceMacroExcludingRaw0")
    try:
        return float(dice_excluding_raw0) >= 0.80
    except (TypeError, ValueError):
        return False


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
    runtime_candidates = []
    for model_key, status in models.items():
        artifact = status.get("artifact", {})
        manifest = status.get("manifest", {})
        exists = bool(artifact.get("exists"))
        sha256 = artifact.get("sha256")
        integrity = artifact.get("integrityStatus")
        baseline_ready = bool(status.get("baselineReady"))
        available_for_real = bool(status.get("availableForRealInference"))
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
            "manifestBaselineReady": bool(status.get("manifestBaselineReady")),
            "readiness": status.get("readiness"),
            "runtimeQualification": status.get("runtimeQualification"),
            "availableForRealInference": available_for_real,
            "trainingStatus": status.get("trainingStatus"),
            "qualityGatePassed": status.get("qualityGatePassed"),
            "heldOutReuseWarning": status.get("heldOutReuseWarning"),
            "qualityGate": status.get("qualityGate"),
            "metrics": status.get("metrics"),
            "verified": exists and bool(sha256) and integrity == "hashed" and baseline_ready,
        }
        if not exists:
            missing.append(model_result)
        elif not baseline_ready and available_for_real:
            runtime_candidates.append(model_result)
        elif not baseline_ready:
            missing_manifest.append(model_result)
        elif not model_result["verified"]:
            unverified.append(model_result)
        else:
            verified.append(model_result)

    summary = artifact_summary()
    valid = not missing and not missing_manifest and not unverified and not runtime_candidates and bool(models)
    candidate_available = (
        not valid
        and not missing
        and not missing_manifest
        and not unverified
        and bool(runtime_candidates)
        and all(bool(status.get("availableForRealInference")) for status in models.values())
    )
    return {
        "status": "real_baseline_verified" if valid else "real_candidate_available" if candidate_available else "degraded_contract_mode",
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
        "runtimeCandidateModels": runtime_candidates,
        "missingArtifacts": missing,
        "missingManifestOrBaselineEvidence": missing_manifest,
        "unverifiedArtifacts": unverified,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }
