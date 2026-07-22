from __future__ import annotations

import hashlib

from pfi_ai_service.model_artifacts import artifact_summary, model_status


def test_model_status_hashes_existing_artifact(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact_path = model_dir / "sagittal_spider_multiclass_final_best.pt"
    artifact_bytes = b"pfi-demo-artifact"
    artifact_path.write_bytes(artifact_bytes)
    monkeypatch.setenv("PFI_MODEL_DIR", str(model_dir))

    status = model_status("sagittal_spider", {"plane": "sagittal"})

    assert status["availableForRealInference"] is False
    assert status["readiness"] == "real_artifact_missing_manifest"
    assert status["artifactHash"] == hashlib.sha256(artifact_bytes).hexdigest()
    assert status["artifactIntegrityStatus"] == "hashed"
    assert status["artifact"]["sha256"] == status["artifactHash"]
    assert status["artifact"]["hashAlgorithm"] == "sha256"
    assert status["artifact"]["lastModified"] is not None


def test_artifact_summary_counts_hashed_artifacts(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "sagittal_spider_multiclass_final_best.pt").write_bytes(b"sagittal")
    monkeypatch.setenv("PFI_MODEL_DIR", str(model_dir))

    summary = artifact_summary()

    assert summary["modelsRegistered"] >= 1
    assert summary["artifactsAvailable"] == 1
    assert summary["artifactsHashed"] == 1
    assert summary["hashAlgorithm"] == "sha256"
    assert summary["defaultInferenceMode"] == "contract"
