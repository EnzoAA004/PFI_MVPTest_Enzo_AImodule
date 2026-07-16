from __future__ import annotations

import hashlib

from pfi_ai_service.model_manifest import validate_manifest


def valid_manifest(artifact_name: str, digest: str) -> dict:
    return {
        "modelKey": "sagittal_spider",
        "version": "real-baseline-v1",
        "artifactFile": artifact_name,
        "dataset": "spider_sagittal",
        "task": "lumbar_mri_multiclass_segmentation",
        "inputPlane": "sagittal",
        "classes": ["background", "vertebra_group", "canal", "disc_group"],
        "metrics": {"dice": 0.83, "iou": 0.73, "scope": "synthetic_test"},
        "trainingStatus": "baseline_evaluated",
        "sha256": digest,
    }


def test_manifest_validation_accepts_valid_manifest(tmp_path) -> None:
    artifact = tmp_path / "sagittal_spider_multiclass_final_best.pt"
    artifact.write_bytes(b"synthetic-checkpoint")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    result = validate_manifest(valid_manifest(artifact.name, digest), artifact_path=artifact)

    assert result["valid"] is True
    assert result["baselineReady"] is True
    assert result["sha256Status"] == "MATCH"
    assert result["validationErrors"] == []


def test_manifest_validation_rejects_missing_field_and_metric_out_of_range(tmp_path) -> None:
    artifact = tmp_path / "sagittal_spider_multiclass_final_best.pt"
    artifact.write_bytes(b"synthetic-checkpoint")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = valid_manifest(artifact.name, digest)
    manifest.pop("dataset")
    manifest["metrics"]["dice"] = 1.5

    result = validate_manifest(manifest, artifact_path=artifact)

    assert result["valid"] is False
    assert "dataset" in result["missingFields"]
    assert "missing_required_fields:dataset" in result["validationErrors"]
    assert "metrics_out_of_range:dice" in result["validationErrors"]


def test_manifest_validation_rejects_sha_or_classes_mismatch(tmp_path) -> None:
    artifact = tmp_path / "sagittal_spider_multiclass_final_best.pt"
    artifact.write_bytes(b"synthetic-checkpoint")
    manifest = valid_manifest(artifact.name, "0" * 64)
    manifest["classes"] = ["background", "wrong_class"]

    result = validate_manifest(manifest, artifact_path=artifact)

    assert result["valid"] is False
    assert result["sha256Status"] == "MISMATCH"
    assert any(error.startswith("sha256_mismatch:") for error in result["validationErrors"])
    assert "classes_mismatch_registry" in result["validationErrors"]
