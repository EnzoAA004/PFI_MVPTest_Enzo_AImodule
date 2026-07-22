from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .model_manifest import manifest_path_for_artifact, read_model_manifest, validate_manifest

SAGITTAL_RELEASE_ID = "sagittal_spider_final_v1"
SAGITTAL_MODEL_KEY = "sagittal_spider"
SAGITTAL_VERSION = "sagittal-spider-final-v1"
SAGITTAL_MODEL_FILE = "sagittal_spider_multiclass_final_best.pt"
SAGITTAL_MANIFEST_FILE = f"{SAGITTAL_MODEL_FILE}.manifest.json"
SAGITTAL_MODELCARD_FILE = f"{SAGITTAL_MODEL_FILE}.modelcard.md"
EXPECTED_BUCKET = "pfi-rm-lumbar-artifacts-2026-ef"
EXPECTED_PUBLICATION_ARTIFACT_COUNT = 12
EXPECTED_CLASSES = ["background", "vertebra_group", "canal", "disc_group"]
REQUIRED_RELEASE_FILES = {
    "_SUCCESS.json",
    "publish_receipt.json",
    "release_manifest.json",
    SAGITTAL_MODEL_FILE,
    SAGITTAL_MANIFEST_FILE,
    SAGITTAL_MODELCARD_FILE,
}


class ReleaseValidationError(RuntimeError):
    """Raised when a remote release cannot be trusted."""


@dataclass(frozen=True)
class GcsReleaseConfig:
    project_id: str | None
    release_uri: str
    release_content_sha256: str
    release_manifest_sha256: str
    model_sha256: str
    destination_dir: Path
    release_id: str = SAGITTAL_RELEASE_ID
    model_key: str = SAGITTAL_MODEL_KEY
    version: str = SAGITTAL_VERSION


@dataclass(frozen=True)
class GsUri:
    bucket: str
    prefix: str


def parse_gs_uri(uri: str) -> GsUri:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"URI GCS invalida: {uri}")
    prefix = parsed.path.strip("/")
    if not prefix or ".." in prefix.split("/"):
        raise ValueError(f"Prefijo GCS invalido: {uri}")
    return GsUri(bucket=parsed.netloc, prefix=prefix)


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseValidationError(f"JSON invalido en {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"JSON raiz invalido en {path.name}")
    return value


def _client_factory(project_id: str | None) -> Any:
    from google.cloud import storage

    return storage.Client(project=project_id)


def _object_name(prefix: str, file_name: str) -> str:
    return f"{prefix.rstrip('/')}/{file_name}"


def _safe_artifact_name(entry: dict[str, Any]) -> str | None:
    for key in ("fileName", "filename", "name", "artifactFile", "relativePath", "objectName", "destination"):
        raw = entry.get(key)
        if isinstance(raw, str) and raw.strip():
            name = raw.strip().replace("\\", "/").rstrip("/")
            return name.rsplit("/", 1)[-1]
    uri = entry.get("destinationUri") or entry.get("gcsUri") or entry.get("uri")
    if isinstance(uri, str) and uri.strip():
        return uri.strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return None


def _entry_sha256(entry: dict[str, Any]) -> str | None:
    for key in ("sha256", "artifactSha256", "contentSha256", "localSha256"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    checksum = entry.get("checksum")
    if isinstance(checksum, dict):
        value = checksum.get("sha256")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _entry_size(entry: dict[str, Any]) -> int | None:
    for key in ("sizeBytes", "size", "bytes"):
        value = entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _release_artifacts(release_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = release_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ReleaseValidationError("release_manifest sin lista artifacts")
    by_name: dict[str, dict[str, Any]] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ReleaseValidationError("artifact invalido en release_manifest")
        if "sourcePath" in entry:
            raise ReleaseValidationError("release_manifest expone sourcePath")
        name = _safe_artifact_name(entry)
        if not name:
            raise ReleaseValidationError("artifact sin nombre resoluble")
        by_name[name] = entry
    return by_name


def _download_blob(bucket: Any, prefix: str, file_name: str, destination: Path) -> None:
    blob = bucket.blob(_object_name(prefix, file_name))
    exists = blob.exists()
    if not exists:
        raise ReleaseValidationError(f"Artifact remoto faltante: {file_name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        blob.download_to_filename(str(tmp))
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _validate_downloaded_artifact(name: str, entry: dict[str, Any], path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise ReleaseValidationError(f"Artifact descargado faltante: {name}")
    expected_size = _entry_size(entry)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ReleaseValidationError(
            f"Tamanio invalido en {name}: expected={expected_size};actual={path.stat().st_size}"
        )
    expected_sha = _entry_sha256(entry)
    actual_sha = sha256_file(path)
    if expected_sha and expected_sha != actual_sha:
        raise ReleaseValidationError(f"SHA invalido en {name}: expected={expected_sha};actual={actual_sha}")


def _validate_success(config: GcsReleaseConfig, success: dict[str, Any]) -> None:
    if success.get("status") != "published_and_verified":
        raise ReleaseValidationError(f"_SUCCESS.status invalido: {success.get('status')}")
    if success.get("releaseId") != config.release_id:
        raise ReleaseValidationError(f"_SUCCESS.releaseId invalido: {success.get('releaseId')}")
    if success.get("remoteVerificationPassed") is not True:
        raise ReleaseValidationError("_SUCCESS.remoteVerificationPassed no es true")
    if success.get("publicationArtifactCount") != EXPECTED_PUBLICATION_ARTIFACT_COUNT:
        raise ReleaseValidationError(
            f"_SUCCESS.publicationArtifactCount invalido: {success.get('publicationArtifactCount')}"
        )
    if str(success.get("releaseContentSha256", "")).lower() != config.release_content_sha256.lower():
        raise ReleaseValidationError("_SUCCESS.releaseContentSha256 no coincide")
    if str(success.get("releaseManifestSha256", "")).lower() != config.release_manifest_sha256.lower():
        raise ReleaseValidationError("_SUCCESS.releaseManifestSha256 no coincide")


def _validate_receipt(config: GcsReleaseConfig, receipt: dict[str, Any]) -> None:
    if receipt.get("releaseId") != config.release_id:
        raise ReleaseValidationError("publish_receipt releaseId no coincide")
    if str(receipt.get("releaseContentSha256", "")).lower() != config.release_content_sha256.lower():
        raise ReleaseValidationError("publish_receipt releaseContentSha256 no coincide")
    if receipt.get("remoteVerificationPassed") is not True:
        raise ReleaseValidationError("publish_receipt remoteVerificationPassed no es true")


def _validate_release_manifest(config: GcsReleaseConfig, release_manifest: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    actual_manifest_sha = sha256_file(path)
    if actual_manifest_sha != config.release_manifest_sha256.lower():
        raise ReleaseValidationError(
            f"release_manifest SHA no coincide: expected={config.release_manifest_sha256};actual={actual_manifest_sha}"
        )
    if release_manifest.get("releaseId") != config.release_id:
        raise ReleaseValidationError("release_manifest releaseId no coincide")
    if str(release_manifest.get("releaseContentSha256", "")).lower() != config.release_content_sha256.lower():
        raise ReleaseValidationError("release_manifest releaseContentSha256 no coincide")
    artifacts = _release_artifacts(release_manifest)
    missing = sorted(REQUIRED_RELEASE_FILES - {"_SUCCESS.json", "publish_receipt.json", "release_manifest.json"} - set(artifacts))
    if missing:
        raise ReleaseValidationError("release_manifest no lista artifacts requeridos: " + ",".join(missing))
    return artifacts


def _validate_runtime_manifest(config: GcsReleaseConfig, manifest_path: Path, model_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    result = validate_manifest(manifest, artifact_path=model_path)
    if not result.get("valid"):
        raise ReleaseValidationError("runtime manifest invalido: " + ",".join(result.get("validationErrors", [])))
    if result.get("baselineReady") is not True:
        raise ReleaseValidationError("runtime manifest baselineReady no es true")
    if result.get("sha256Status") != "MATCH":
        raise ReleaseValidationError(f"runtime manifest sha256Status invalido: {result.get('sha256Status')}")
    if result.get("validationErrors"):
        raise ReleaseValidationError("runtime manifest contiene errores")
    if result.get("modelKey") != config.model_key:
        raise ReleaseValidationError(f"runtime manifest modelKey invalido: {result.get('modelKey')}")
    if result.get("version") != config.version:
        raise ReleaseValidationError(f"runtime manifest version invalida: {result.get('version')}")
    if result.get("classes") != EXPECTED_CLASSES:
        raise ReleaseValidationError("runtime manifest classes no coinciden")
    if int(manifest.get("numClasses", manifest.get("num_classes", 0))) != 4:
        raise ReleaseValidationError("runtime manifest numClasses invalido")
    if int(manifest.get("baseChannels", manifest.get("base_channels", 0))) != 16:
        raise ReleaseValidationError("runtime manifest baseChannels invalido")
    target = manifest.get("targetSize", manifest.get("target_size"))
    if list(target or []) != [256, 256]:
        raise ReleaseValidationError("runtime manifest targetSize invalido")
    return read_model_manifest(model_path)


def _final_paths(config: GcsReleaseConfig) -> dict[str, Path]:
    model_path = config.destination_dir / SAGITTAL_MODEL_FILE
    return {
        SAGITTAL_MODEL_FILE: model_path,
        SAGITTAL_MANIFEST_FILE: manifest_path_for_artifact(model_path),
        SAGITTAL_MODELCARD_FILE: config.destination_dir / SAGITTAL_MODELCARD_FILE,
    }


def _local_release_status(config: GcsReleaseConfig) -> dict[str, Any]:
    paths = _final_paths(config)
    model_path = paths[SAGITTAL_MODEL_FILE]
    manifest_path = paths[SAGITTAL_MANIFEST_FILE]
    modelcard_path = paths[SAGITTAL_MODELCARD_FILE]
    if not all(path.exists() and path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return {"complete": False, "reason": "missing_local_files"}
    model_sha = sha256_file(model_path)
    if model_sha != config.model_sha256.lower():
        return {"complete": False, "reason": "local_model_sha_mismatch", "actualSha256": model_sha}
    try:
        manifest = load_json(manifest_path)
        validation = validate_manifest(manifest, artifact_path=model_path)
    except Exception as exc:
        return {"complete": False, "reason": "local_manifest_invalid", "message": str(exc)}
    if validation.get("valid") and validation.get("baselineReady") and validation.get("sha256Status") == "MATCH":
        return {
            "complete": True,
            "modelSha256": model_sha,
            "manifestStatus": validation.get("status"),
            "modelCardSizeBytes": modelcard_path.stat().st_size,
        }
    return {"complete": False, "reason": "local_manifest_not_ready", "validationErrors": validation.get("validationErrors", [])}


def _replace_final_files(config: GcsReleaseConfig, staging: Path) -> int:
    final_paths = _final_paths(config)
    config.destination_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path | None]] = []
    replaced_destinations: list[Path] = []
    for name, destination in final_paths.items():
        source = staging / name
        tmp = destination.with_name(destination.name + ".replace")
        shutil.copyfile(source, tmp)
        prepared.append((tmp, destination))
    try:
        for tmp, destination in prepared:
            backup = destination.with_name(destination.name + ".backup")
            if destination.exists():
                os.replace(destination, backup)
                backups.append((destination, backup))
            else:
                backups.append((destination, None))
            os.replace(tmp, destination)
            replaced_destinations.append(destination)
    except Exception:
        for destination in replaced_destinations:
            destination.unlink(missing_ok=True)
        for destination, backup in reversed(backups):
            if backup is not None and backup.exists():
                os.replace(backup, destination)
        for tmp, _destination in prepared:
            tmp.unlink(missing_ok=True)
        raise
    for _destination, backup in backups:
        if backup is not None:
            backup.unlink(missing_ok=True)
    return len(final_paths)


def materialize_sagittal_gcs_release(
    config: GcsReleaseConfig,
    *,
    force: bool = False,
    client_factory: Callable[[str | None], Any] | None = None,
) -> dict[str, Any]:
    parsed = parse_gs_uri(config.release_uri)
    if parsed.bucket != EXPECTED_BUCKET:
        raise ReleaseValidationError(f"Bucket inesperado para release sagital: {parsed.bucket}")

    local = _local_release_status(config)
    if local.get("complete") and not force:
        return {
            "modelKey": config.model_key,
            "source": "gcs_verified_release",
            "releaseId": config.release_id,
            "releaseContentSha256": config.release_content_sha256,
            "releaseManifestSha256": config.release_manifest_sha256,
            "modelSha256": config.model_sha256,
            "status": "existing_release_verified",
            "artifactSynced": True,
            "manifestSynced": True,
            "modelCardSynced": True,
            "filesReplaced": 0,
            "gcsReadOnly": True,
            "localValidation": local,
        }
    if not force and local.get("reason") in {"local_model_sha_mismatch", "local_manifest_invalid", "local_manifest_not_ready"}:
        return {
            "modelKey": config.model_key,
            "source": "gcs_verified_release",
            "releaseId": config.release_id,
            "status": "local_release_mismatch_requires_force",
            "artifactSynced": False,
            "manifestSynced": False,
            "modelCardSynced": False,
            "filesReplaced": 0,
            "gcsReadOnly": True,
            "localValidation": local,
        }

    client = (client_factory or _client_factory)(config.project_id)
    bucket = client.bucket(parsed.bucket)
    staging_root = config.destination_dir.parent / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{config.release_id}-", dir=staging_root) as tmpdir:
        staging = Path(tmpdir)
        for name in ("_SUCCESS.json", "publish_receipt.json", "release_manifest.json"):
            _download_blob(bucket, parsed.prefix, name, staging / name)

        success = load_json(staging / "_SUCCESS.json")
        receipt = load_json(staging / "publish_receipt.json")
        release_manifest = load_json(staging / "release_manifest.json")
        _validate_success(config, success)
        _validate_receipt(config, receipt)
        receipt_sha = sha256_file(staging / "publish_receipt.json")
        if str(success.get("receiptSha256", "")).lower() != receipt_sha:
            raise ReleaseValidationError("_SUCCESS.receiptSha256 no coincide con publish_receipt real")

        artifacts = _validate_release_manifest(config, release_manifest, staging / "release_manifest.json")
        if success.get("publicationArtifactCount") != len(artifacts) + 1:
            raise ReleaseValidationError("publicationArtifactCount no coincide con artifacts mas release_manifest")

        for name in sorted(artifacts):
            _download_blob(bucket, parsed.prefix, name, staging / name)
            _validate_downloaded_artifact(name, artifacts[name], staging / name)

        model_path = staging / SAGITTAL_MODEL_FILE
        model_sha = sha256_file(model_path)
        if model_sha != config.model_sha256.lower():
            raise ReleaseValidationError(f"model SHA no coincide: expected={config.model_sha256};actual={model_sha}")
        runtime_manifest = _validate_runtime_manifest(config, staging / SAGITTAL_MANIFEST_FILE, model_path)
        if not (staging / SAGITTAL_MODELCARD_FILE).stat().st_size:
            raise ReleaseValidationError("model card vacia")

        files_replaced = _replace_final_files(config, staging)

    return {
        "modelKey": config.model_key,
        "source": "gcs_verified_release",
        "releaseId": config.release_id,
        "releaseContentSha256": config.release_content_sha256,
        "releaseManifestSha256": config.release_manifest_sha256,
        "modelSha256": config.model_sha256,
        "status": "synced_verified",
        "artifactSynced": True,
        "manifestSynced": True,
        "modelCardSynced": True,
        "filesReplaced": files_replaced,
        "gcsReadOnly": True,
        "runtimeManifest": runtime_manifest,
    }
