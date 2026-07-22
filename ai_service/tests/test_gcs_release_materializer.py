from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pfi_ai_service.gcs_release_materializer import (
    EXPECTED_PUBLICATION_ARTIFACT_COUNT,
    GcsReleaseConfig,
    ReleaseValidationError,
    SAGITTAL_MANIFEST_FILE,
    SAGITTAL_MODELCARD_FILE,
    SAGITTAL_MODEL_FILE,
    materialize_sagittal_gcs_release,
    parse_gs_uri,
)


class FakeBlob:
    def __init__(self, data: bytes | None) -> None:
        self.data = data

    def exists(self) -> bool:
        return self.data is not None

    def download_to_filename(self, filename: str) -> None:
        if self.data is None:
            raise FileNotFoundError(filename)
        Path(filename).write_bytes(self.data)


class FakeBucket:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.blobs = blobs

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self.blobs.get(name))


class FakeClient:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.blobs = blobs
        self.bucket_calls = 0

    def bucket(self, name: str) -> FakeBucket:
        self.bucket_calls += 1
        return FakeBucket(self.blobs)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def runtime_manifest(model_sha: str) -> dict:
    return {
        "modelKey": "sagittal_spider",
        "version": "sagittal-spider-final-v1",
        "artifactFile": SAGITTAL_MODEL_FILE,
        "dataset": "SPIDER",
        "task": "lumbar_mri_multiclass_segmentation",
        "inputPlane": "sagittal",
        "classes": ["background", "vertebra_group", "canal", "disc_group"],
        "metrics": {"dice": 0.876, "iou": 0.781},
        "trainingStatus": "baseline_evaluated",
        "sha256": model_sha,
        "numClasses": 4,
        "baseChannels": 16,
        "targetSize": [256, 256],
    }


def release_fixture(tmp_path, mutate: str | None = None):
    prefix = "models/releases/sagittal_spider_final_v1"
    model = b"synthetic-final-sagittal-model"
    manifest = json_bytes(runtime_manifest(digest(model)))
    modelcard = b"# Sagittal model card\n"
    files = {
        SAGITTAL_MODEL_FILE: model,
        SAGITTAL_MANIFEST_FILE: manifest,
        SAGITTAL_MODELCARD_FILE: modelcard,
    }
    content_artifact_count = EXPECTED_PUBLICATION_ARTIFACT_COUNT - 1
    for index in range(content_artifact_count - len(files)):
        files[f"evidence_{index}.json"] = json_bytes({"index": index})
    artifacts = [
        {"fileName": name, "sizeBytes": len(data), "sha256": digest(data)}
        for name, data in sorted(files.items())
    ]
    release_manifest = {
        "releaseId": "sagittal_spider_final_v1",
        "releaseContentSha256": "content-sha",
        "artifacts": artifacts,
    }
    release_manifest_bytes = json_bytes(release_manifest)
    receipt = {
        "releaseId": "sagittal_spider_final_v1",
        "releaseContentSha256": "content-sha",
        "remoteVerificationPassed": True,
    }
    receipt_bytes = json_bytes(receipt)
    success = {
        "status": "published_and_verified",
        "releaseId": "sagittal_spider_final_v1",
        "remoteVerificationPassed": True,
        "publicationArtifactCount": EXPECTED_PUBLICATION_ARTIFACT_COUNT,
        "releaseContentSha256": "content-sha",
        "releaseManifestSha256": digest(release_manifest_bytes),
        "receiptSha256": digest(receipt_bytes),
    }
    if mutate == "bad_success_status":
        success["status"] = "draft"
    if mutate == "bad_release_id":
        success["releaseId"] = "other"
    if mutate == "bad_content_sha":
        success["releaseContentSha256"] = "wrong"
    if mutate == "bad_receipt_sha":
        success["receiptSha256"] = "0" * 64
    if mutate == "bad_artifact_size":
        artifacts[0]["sizeBytes"] += 1
        release_manifest_bytes = json_bytes(release_manifest)
        success["releaseManifestSha256"] = digest(release_manifest_bytes)
    if mutate == "bad_artifact_sha":
        artifacts[0]["sha256"] = "0" * 64
        release_manifest_bytes = json_bytes(release_manifest)
        success["releaseManifestSha256"] = digest(release_manifest_bytes)
    if mutate == "bad_model_sha":
        model_sha = "0" * 64
    else:
        model_sha = digest(model)
    if mutate == "bad_runtime_manifest":
        manifest_dict = runtime_manifest(digest(model))
        manifest_dict["classes"] = ["background", "wrong"]
        files[SAGITTAL_MANIFEST_FILE] = json_bytes(manifest_dict)
        for entry in artifacts:
            if entry["fileName"] == SAGITTAL_MANIFEST_FILE:
                entry["sizeBytes"] = len(files[SAGITTAL_MANIFEST_FILE])
                entry["sha256"] = digest(files[SAGITTAL_MANIFEST_FILE])
        release_manifest_bytes = json_bytes(release_manifest)
        success["releaseManifestSha256"] = digest(release_manifest_bytes)
    if mutate == "missing_artifact":
        files.pop(SAGITTAL_MODEL_FILE)
    blobs = {
        f"{prefix}/_SUCCESS.json": json_bytes(success),
        f"{prefix}/publish_receipt.json": receipt_bytes,
        f"{prefix}/release_manifest.json": release_manifest_bytes,
    }
    blobs.update({f"{prefix}/{name}": data for name, data in files.items()})
    config = GcsReleaseConfig(
        project_id="pfi-asplanatti-fabrello-v1",
        release_uri=f"gs://pfi-rm-lumbar-artifacts-2026-ef/{prefix}/",
        release_content_sha256="content-sha",
        release_manifest_sha256=success["releaseManifestSha256"],
        model_sha256=model_sha,
        destination_dir=tmp_path / "models" / "final",
    )
    return config, FakeClient(blobs), files


def provenance_dir(config: GcsReleaseConfig) -> Path:
    return config.destination_dir / ".releases" / "sagittal_spider_final_v1"


def install_valid_release(tmp_path):
    config, client, files = release_fixture(tmp_path)
    result = materialize_sagittal_gcs_release(config, client_factory=lambda _: client)
    assert result["status"] == "synced_verified"
    return config, client, files


def test_parse_gs_uri_accepts_valid_uri() -> None:
    parsed = parse_gs_uri("gs://bucket/a/b/")
    assert parsed.bucket == "bucket"
    assert parsed.prefix == "a/b"


@pytest.mark.parametrize("uri", ["http://bucket/a", "gs:///a", "gs://bucket/../x"])
def test_parse_gs_uri_rejects_invalid_uri(uri: str) -> None:
    with pytest.raises(ValueError):
        parse_gs_uri(uri)


def test_valid_release_downloads_and_replaces_atomically(tmp_path) -> None:
    config, client, _ = release_fixture(tmp_path)
    result = materialize_sagittal_gcs_release(config, client_factory=lambda _: client)
    assert result["status"] == "synced_verified"
    assert result["filesReplaced"] == 3
    assert result["releaseMetadataReplaced"] == 3
    assert result["releaseMetadataVerified"] is True
    assert (config.destination_dir / SAGITTAL_MODEL_FILE).exists()
    assert (config.destination_dir / SAGITTAL_MANIFEST_FILE).exists()
    assert (config.destination_dir / SAGITTAL_MODELCARD_FILE).exists()
    assert (provenance_dir(config) / "_SUCCESS.json").exists()
    assert (provenance_dir(config) / "publish_receipt.json").exists()
    assert (provenance_dir(config) / "release_manifest.json").exists()


def test_existing_release_is_idempotent(tmp_path) -> None:
    config, client, _ = release_fixture(tmp_path)
    materialize_sagittal_gcs_release(config, client_factory=lambda _: client)
    result = materialize_sagittal_gcs_release(config, client_factory=lambda _: client)
    assert result["status"] == "existing_release_verified"
    assert result["filesReplaced"] == 0
    assert result["releaseMetadataReplaced"] == 0
    assert result["releaseMetadataVerified"] is True


def test_idempotent_release_does_not_create_gcs_client(tmp_path) -> None:
    config, client, _ = install_valid_release(tmp_path)

    def fail_client(_project_id):
        raise AssertionError("GCS client should not be created for verified local release")

    result = materialize_sagittal_gcs_release(config, client_factory=fail_client)

    assert result["status"] == "existing_release_verified"
    assert client.bucket_calls == 1


@pytest.mark.parametrize(
    ("file_name", "replacement"),
    [
        (SAGITTAL_MODELCARD_FILE, b"# Tampered card\n"),
        (SAGITTAL_MODELCARD_FILE, b""),
        (SAGITTAL_MANIFEST_FILE, b"{}"),
    ],
)
def test_local_runtime_artifact_tampering_requires_force(tmp_path, file_name: str, replacement: bytes) -> None:
    config, _client, _files = install_valid_release(tmp_path)
    (config.destination_dir / file_name).write_bytes(replacement)

    result = materialize_sagittal_gcs_release(config, client_factory=lambda _project_id: (_ for _ in ()).throw(AssertionError("no GCS")))

    assert result["status"] == "local_release_mismatch_requires_force"
    assert result["filesReplaced"] == 0
    assert result["releaseMetadataVerified"] is False


@pytest.mark.parametrize("file_name", ["release_manifest.json", "publish_receipt.json", "_SUCCESS.json"])
def test_local_release_metadata_tampering_requires_force(tmp_path, file_name: str) -> None:
    config, _client, _files = install_valid_release(tmp_path)
    (provenance_dir(config) / file_name).write_text("{}", encoding="utf-8")

    result = materialize_sagittal_gcs_release(config, client_factory=lambda _project_id: (_ for _ in ()).throw(AssertionError("no GCS")))

    assert result["status"] == "local_release_mismatch_requires_force"
    assert result["releaseMetadataVerified"] is False


def test_missing_local_metadata_requires_force_when_artifacts_exist(tmp_path) -> None:
    config, _client, _files = install_valid_release(tmp_path)
    (provenance_dir(config) / "_SUCCESS.json").unlink()

    result = materialize_sagittal_gcs_release(config, client_factory=lambda _project_id: (_ for _ in ()).throw(AssertionError("no GCS")))

    assert result["status"] == "local_release_mismatch_requires_force"


@pytest.mark.parametrize(
    "mutate",
    [
        "bad_success_status",
        "bad_release_id",
        "bad_content_sha",
        "bad_receipt_sha",
        "missing_artifact",
        "bad_artifact_size",
        "bad_artifact_sha",
        "bad_model_sha",
        "bad_runtime_manifest",
    ],
)
def test_invalid_release_fails_before_replacing_destination(tmp_path, mutate: str) -> None:
    config, client, _ = release_fixture(tmp_path, mutate=mutate)
    config.destination_dir.mkdir(parents=True)
    existing = config.destination_dir / SAGITTAL_MODEL_FILE
    existing.write_bytes(b"keep-me")
    with pytest.raises(ReleaseValidationError):
        materialize_sagittal_gcs_release(config, force=True, client_factory=lambda _: client)
    assert existing.read_bytes() == b"keep-me"


def test_release_manifest_sha_configuration_mismatch_fails(tmp_path) -> None:
    config, client, _ = release_fixture(tmp_path)
    bad_config = GcsReleaseConfig(
        project_id=config.project_id,
        release_uri=config.release_uri,
        release_content_sha256=config.release_content_sha256,
        release_manifest_sha256="0" * 64,
        model_sha256=config.model_sha256,
        destination_dir=config.destination_dir,
    )

    with pytest.raises(ReleaseValidationError, match="releaseManifestSha256"):
        materialize_sagittal_gcs_release(bad_config, client_factory=lambda _: client)


def test_local_mismatch_requires_force_and_force_replaces(tmp_path) -> None:
    config, client, _ = release_fixture(tmp_path)
    config.destination_dir.mkdir(parents=True)
    (config.destination_dir / SAGITTAL_MODEL_FILE).write_bytes(b"bad")
    (config.destination_dir / SAGITTAL_MANIFEST_FILE).write_text("{}", encoding="utf-8")
    (config.destination_dir / SAGITTAL_MODELCARD_FILE).write_text("bad", encoding="utf-8")

    blocked = materialize_sagittal_gcs_release(config, client_factory=lambda _: client)
    assert blocked["status"] == "local_release_mismatch_requires_force"

    replaced = materialize_sagittal_gcs_release(config, force=True, client_factory=lambda _: client)
    assert replaced["status"] == "synced_verified"
    assert replaced["filesReplaced"] == 3
    assert replaced["releaseMetadataReplaced"] == 3


def test_force_restores_tampered_model_card_and_metadata(tmp_path) -> None:
    config, client, files = install_valid_release(tmp_path)
    (config.destination_dir / SAGITTAL_MODELCARD_FILE).write_bytes(b"bad")
    (provenance_dir(config) / "release_manifest.json").write_text("{}", encoding="utf-8")

    result = materialize_sagittal_gcs_release(config, force=True, client_factory=lambda _: client)

    assert result["status"] == "synced_verified"
    assert (config.destination_dir / SAGITTAL_MODELCARD_FILE).read_bytes() == files[SAGITTAL_MODELCARD_FILE]
    assert json.loads((provenance_dir(config) / "release_manifest.json").read_text(encoding="utf-8"))["releaseId"] == "sagittal_spider_final_v1"


def test_replacement_error_restores_runtime_and_metadata(monkeypatch, tmp_path) -> None:
    config, client, _files = install_valid_release(tmp_path)
    original_card = (config.destination_dir / SAGITTAL_MODELCARD_FILE).read_bytes()
    original_success = (provenance_dir(config) / "_SUCCESS.json").read_bytes()
    config2, client2, files2 = release_fixture(tmp_path)
    files2[SAGITTAL_MODELCARD_FILE] = b"# New card\n"
    # Make the remote model card different and rebuild the fake release consistently.
    config2, client2, _files2 = release_fixture(tmp_path)
    replace_count = {"count": 0}
    from pfi_ai_service import gcs_release_materializer as materializer
    real_replace = materializer.os.replace

    def fail_after_first(source, destination):
        replace_count["count"] += 1
        if replace_count["count"] == 3:
            raise OSError("synthetic replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(materializer.os, "replace", fail_after_first)

    with pytest.raises(OSError, match="synthetic replacement failure"):
        materialize_sagittal_gcs_release(config2, force=True, client_factory=lambda _: client2)

    assert (config.destination_dir / SAGITTAL_MODELCARD_FILE).read_bytes() == original_card
    assert (provenance_dir(config) / "_SUCCESS.json").read_bytes() == original_success


def test_model_materializer_keeps_axial_legacy_path(monkeypatch, tmp_path) -> None:
    from pfi_ai_service import model_materializer

    settings = SimpleNamespace(
        models_root=tmp_path,
        sagittal_model_path=tmp_path / SAGITTAL_MODEL_FILE,
        axial_model_path=tmp_path / "axial_t2_alkafri_final_best.pt",
        sagittal_model_uri=None,
        axial_model_uri=None,
        sagittal_manifest_uri=None,
        axial_manifest_uri=None,
        sagittal_release_uri=None,
        sagittal_release_content_sha256=None,
        sagittal_release_manifest_sha256=None,
        sagittal_model_sha256=None,
        gcp_project_id=None,
        model_download_token=None,
    )
    seen: list[str] = []
    monkeypatch.setattr(model_materializer, "get_settings", lambda: settings)
    monkeypatch.setattr(model_materializer, "verify_model_artifacts", lambda: {"readyForRealInference": False, "defaultInferenceMode": "contract"})
    monkeypatch.setattr(
        model_materializer,
        "sync_one_model",
        lambda model_key, *_args, **_kwargs: seen.append(model_key) or {"modelKey": model_key, "status": "legacy"},
    )

    result = model_materializer.sync_model_artifacts()

    assert [item["modelKey"] for item in result["items"]] == ["sagittal_spider", "axial_t2_alkafri"]
    assert seen == ["sagittal_spider", "axial_t2_alkafri"]
