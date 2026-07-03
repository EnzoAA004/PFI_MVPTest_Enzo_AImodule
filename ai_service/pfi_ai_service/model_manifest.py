from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

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
        }
    validation = validate_manifest(manifest)
    return {
        "path": str(manifest_path),
        "exists": True,
        **validation,
        "content": manifest,
    }


def validate_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    missing_fields = [field for field in REQUIRED_MANIFEST_FIELDS if field not in manifest]
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
    missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in metrics]
    valid = not missing_fields and not missing_metrics
    training_status = str(manifest.get("trainingStatus", "unknown"))
    baseline_ready = valid and training_status in {"baseline_trained", "baseline_evaluated", "validated_baseline"}
    return {
        "valid": valid,
        "baselineReady": baseline_ready,
        "status": "baseline_manifest_ready" if baseline_ready else "baseline_manifest_incomplete",
        "missingFields": missing_fields,
        "missingMetrics": missing_metrics,
        "trainingStatus": training_status,
        "modelKey": manifest.get("modelKey"),
        "version": manifest.get("version"),
        "dataset": manifest.get("dataset"),
        "task": manifest.get("task"),
        "inputPlane": manifest.get("inputPlane"),
        "classes": manifest.get("classes", []),
        "metrics": metrics,
    }


def missing_manifest(path: Path | None) -> Dict[str, Any]:
    return {
        "path": str(path) if path else None,
        "exists": False,
        "valid": False,
        "baselineReady": False,
        "status": "missing_manifest",
        "missingFields": REQUIRED_MANIFEST_FIELDS,
        "missingMetrics": REQUIRED_METRICS,
        "trainingStatus": "missing",
    }
