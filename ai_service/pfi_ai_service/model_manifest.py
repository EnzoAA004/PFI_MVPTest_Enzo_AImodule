from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from .settings import MODEL_REGISTRY

REQUIRED_MANIFEST_FIELDS = [
    "modelKey",
    "version",
    "artifactFile",
    "dataset",
    "task",
    "inputPlane",
    "classes",
    "metrics",
    "trainingStatus",
]

REQUIRED_METRICS = ["dice", "iou"]
OPTIONAL_SHA_FIELDS = ["sha256", "artifactSha256", "artifactHash", "checkpointSha256"]


def manifest_path_for_artifact(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")


def read_model_manifest(artifact_path: Path | None) -> Dict[str, Any]:
    if artifact_path is None:
        return missing_manifest(None)
    manifest_path = manifest_path_for_artifact(artifact_path)
    if not manifest_path.exists() or not manifest_path.is_file():
        return missing_manifest(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": str(manifest_path),
            "exists": True,
            "valid": False,
            "status": "invalid_manifest_json",
            "message": str(exc),
            "missingFields": REQUIRED_MANIFEST_FIELDS,
            "missingMetrics": REQUIRED_METRICS,
            "validationErrors": ["manifest_json_invalid"],
        }
    validation = validate_manifest(manifest, artifact_path=artifact_path)
    return {
        "path": str(manifest_path),
        "exists": True,
        **validation,
        "content": manifest,
    }


def validate_manifest(manifest: Dict[str, Any], artifact_path: Path | None = None) -> Dict[str, Any]:
    missing_fields = [field for field in REQUIRED_MANIFEST_FIELDS if field not in manifest]
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
    missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in metrics]
    validation_errors: list[str] = []

    if missing_fields:
        validation_errors.append("missing_required_fields:" + ",".join(missing_fields))
    if missing_metrics:
        validation_errors.append("missing_required_metrics:" + ",".join(missing_metrics))

    model_key = manifest.get("modelKey")
    registry_entry = MODEL_REGISTRY.get(str(model_key)) if model_key is not None else None
    if registry_entry is None:
        validation_errors.append("modelKey_not_registered")

    version = manifest.get("version")
    if not isinstance(version, str) or not version.strip():
        validation_errors.append("version_empty")

    dataset = manifest.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        validation_errors.append("dataset_empty")

    artifact_file = manifest.get("artifactFile")
    if not isinstance(artifact_file, str) or not artifact_file.strip():
        validation_errors.append("artifactFile_empty")
    elif artifact_path is not None:
        if artifact_file != artifact_path.name:
            validation_errors.append(f"artifactFile_mismatch:expected={artifact_path.name};actual={artifact_file}")
        if not artifact_path.exists() or not artifact_path.is_file():
            validation_errors.append("artifactFile_missing_on_disk")

    input_plane = manifest.get("inputPlane")
    if registry_entry is not None and input_plane != registry_entry.get("plane"):
        validation_errors.append(
            f"inputPlane_mismatch:expected={registry_entry.get('plane')};actual={input_plane}"
        )

    classes = manifest.get("classes")
    if not isinstance(classes, list) or not all(isinstance(item, str) and item for item in classes):
        validation_errors.append("classes_invalid")
    elif registry_entry is not None:
        registry_classes = [name for _, name in sorted(registry_entry.get("class_names", {}).items())]
        if classes != registry_classes:
            validation_errors.append("classes_mismatch_registry")

    metric_range_errors = []
    for metric_name in REQUIRED_METRICS:
        value = metrics.get(metric_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 1:
            metric_range_errors.append(metric_name)
    if metric_range_errors:
        validation_errors.append("metrics_out_of_range:" + ",".join(metric_range_errors))

    expected_sha = manifest_sha256(manifest)
    actual_sha = sha256_file(artifact_path) if artifact_path is not None and artifact_path.is_file() else None
    sha_status = "AUSENTE"
    if expected_sha:
        if actual_sha is None:
            sha_status = "ARTIFACT_AUSENTE"
            validation_errors.append("sha256_present_but_artifact_missing")
        elif expected_sha.lower() == actual_sha.lower():
            sha_status = "MATCH"
        else:
            sha_status = "MISMATCH"
            validation_errors.append(f"sha256_mismatch:expected={expected_sha.lower()};actual={actual_sha.lower()}")

    valid = not missing_fields and not missing_metrics and not validation_errors
    training_status = str(manifest.get("trainingStatus", "unknown"))
    baseline_ready = valid and training_status in {"baseline_trained", "baseline_evaluated", "validated_baseline"}
    return {
        "valid": valid,
        "baselineReady": baseline_ready,
        "status": "baseline_manifest_ready" if baseline_ready else "baseline_manifest_incomplete",
        "missingFields": missing_fields,
        "missingMetrics": missing_metrics,
        "validationErrors": validation_errors,
        "trainingStatus": training_status,
        "modelKey": manifest.get("modelKey"),
        "version": manifest.get("version"),
        "dataset": manifest.get("dataset"),
        "task": manifest.get("task"),
        "inputPlane": manifest.get("inputPlane"),
        "classes": manifest.get("classes", []),
        "metrics": metrics,
        "sha256": expected_sha,
        "artifactSha256": actual_sha,
        "sha256Status": sha_status,
    }


def manifest_sha256(manifest: Dict[str, Any]) -> str | None:
    for field in OPTIONAL_SHA_FIELDS:
        value = manifest.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    checksum = manifest.get("checksum")
    if isinstance(checksum, dict):
        value = checksum.get("sha256")
        if isinstance(value, str) and value.strip():
            return value.strip()
    artifact = manifest.get("artifact")
    if isinstance(artifact, dict):
        value = artifact.get("sha256")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing_manifest(path: Path | None) -> Dict[str, Any]:
    return {
        "path": str(path) if path else None,
        "exists": False,
        "valid": False,
        "baselineReady": False,
        "status": "missing_manifest",
        "missingFields": REQUIRED_MANIFEST_FIELDS,
        "missingMetrics": REQUIRED_METRICS,
        "validationErrors": ["missing_manifest"],
        "trainingStatus": "missing",
    }
