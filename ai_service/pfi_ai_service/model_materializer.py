from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .gcs_release_materializer import GcsReleaseConfig, materialize_sagittal_gcs_release
from .model_artifacts import verify_model_artifacts
from .model_manifest import manifest_path_for_artifact
from .settings import get_settings

MODEL_SYNC_SPECS = {
    "sagittal_spider": {
        "artifact_attr": "sagittal_model_path",
        "uri_attr": "sagittal_model_uri",
        "manifest_uri_attr": "sagittal_manifest_uri",
    },
    "axial_t2_alkafri": {
        "artifact_attr": "axial_model_path",
        "uri_attr": "axial_model_uri",
        "manifest_uri_attr": "axial_manifest_uri",
    },
}


def sync_model_artifacts(force: bool = False) -> Dict[str, Any]:
    settings = get_settings()
    settings.models_root.mkdir(parents=True, exist_ok=True)
    items = []
    for model_key, spec in MODEL_SYNC_SPECS.items():
        artifact_path: Path = getattr(settings, spec["artifact_attr"])
        artifact_uri: str | None = getattr(settings, spec["uri_attr"])
        manifest_uri: str | None = getattr(settings, spec["manifest_uri_attr"])
        if model_key == "sagittal_spider" and settings.sagittal_release_uri:
            if not (
                settings.sagittal_release_content_sha256
                and settings.sagittal_release_manifest_sha256
                and settings.sagittal_model_sha256
            ):
                items.append({
                    "modelKey": model_key,
                    "source": "gcs_verified_release",
                    "status": "missing_release_hash_configuration",
                    "artifactSynced": False,
                    "manifestSynced": False,
                    "modelCardSynced": False,
                    "filesReplaced": 0,
                    "gcsReadOnly": True,
                })
            else:
                try:
                    items.append(materialize_sagittal_gcs_release(
                        GcsReleaseConfig(
                            project_id=settings.gcp_project_id,
                            release_uri=settings.sagittal_release_uri,
                            release_content_sha256=settings.sagittal_release_content_sha256,
                            release_manifest_sha256=settings.sagittal_release_manifest_sha256,
                            model_sha256=settings.sagittal_model_sha256,
                            destination_dir=settings.models_root,
                        ),
                        force=force,
                    ))
                except Exception as exc:
                    items.append({
                        "modelKey": model_key,
                        "source": "gcs_verified_release",
                        "status": "sync_failed",
                        "message": str(exc),
                        "artifactSynced": False,
                        "manifestSynced": False,
                        "modelCardSynced": False,
                        "filesReplaced": 0,
                        "gcsReadOnly": True,
                    })
            continue
        items.append(sync_one_model(model_key, artifact_uri, artifact_path, manifest_uri, force, settings.model_download_token))
    verification = verify_model_artifacts()
    return {
        "status": "models_sync_completed",
        "items": items,
        "verification": verification,
        "readyForRealInference": verification.get("readyForRealInference", False),
        "defaultInferenceMode": verification.get("defaultInferenceMode", "contract"),
        "authenticatedDownloadConfigured": bool(settings.model_download_token),
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def sync_one_model(
    model_key: str,
    artifact_uri: str | None,
    artifact_path: Path,
    manifest_uri: str | None,
    force: bool,
    token: str | None = None,
) -> Dict[str, Any]:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path_for_artifact(artifact_path)
    result: Dict[str, Any] = {
        "modelKey": model_key,
        "artifactPath": str(artifact_path),
        "manifestPath": str(manifest_path),
        "artifactUriConfigured": bool(artifact_uri),
        "manifestUriConfigured": bool(manifest_uri),
    }
    if not artifact_uri:
        result.update({"status": "missing_artifact_uri", "artifactSynced": False, "manifestSynced": manifest_path.exists()})
        return result

    artifact_result = materialize_uri(artifact_uri, artifact_path, force, token)
    result["artifact"] = artifact_result
    if manifest_uri:
        result["manifest"] = materialize_uri(manifest_uri, manifest_path, force, token)
    else:
        result["manifest"] = {"status": "local_manifest", "path": str(manifest_path), "synced": manifest_path.exists()}

    result["artifactSynced"] = bool(result["artifact"].get("synced"))
    result["manifestSynced"] = bool(result["manifest"].get("synced"))
    result["status"] = "synced" if result["artifactSynced"] and result["manifestSynced"] else "partially_synced" if result["artifactSynced"] else "not_synced"
    return result


def materialize_uri(uri: str, destination: Path, force: bool, token: str | None = None) -> Dict[str, Any]:
    if destination.exists() and destination.is_file() and not force:
        return {"status": "already_exists", "path": str(destination), "synced": True, "sizeBytes": destination.stat().st_size}

    parsed = urlparse(uri)
    try:
        if parsed.scheme in {"http", "https"}:
            download_http(uri, destination, token)
        elif parsed.scheme == "file":
            copy_local(Path(parsed.path), destination)
        elif parsed.scheme == "":
            copy_local(Path(uri), destination)
        else:
            return {"status": "unsupported_uri_scheme", "scheme": parsed.scheme, "path": str(destination), "synced": False}
    except Exception as exc:
        return {"status": "sync_failed", "message": str(exc), "path": str(destination), "synced": False}

    return {"status": "synced", "path": str(destination), "synced": True, "sizeBytes": destination.stat().st_size}


def download_http(uri: str, destination: Path, token: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    headers = {"User-Agent": "pfi-ai-module/real-baseline"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(uri, headers=headers)
    try:
        with urlopen(request, timeout=300) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def copy_local(source: Path, destination: Path) -> None:
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, tmp_path)
    tmp_path.replace(destination)
