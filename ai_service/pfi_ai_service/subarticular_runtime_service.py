"""Lazy runtime bridge for the frozen RSNA subarticular classifier."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from .input_registry import InputRecord, InputRegistryError, resolve_registered_input
from .settings import get_settings
from .subarticular_frozen_classifier import (
    EXPECTED_CHECKPOINT_SHA256,
    MODEL_ID,
    CheckpointHashMismatchError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    InvalidSubarticularInputError,
    SubarticularClassifierError,
    SubarticularFrozenClassifier,
    SubarticularFrozenClassifierConfig,
    SubarticularPrediction,
    SubarticularRoi,
)

log = logging.getLogger(__name__)

ALLOWED_REGISTERED_FORMATS = {"dcm", "dicom", "dicom_series"}
PUBLIC_ROI_LIMITATION = "requires_external_anatomical_coordinate; no automatic ROI localizer is validated"


@dataclass(frozen=True)
class _CacheKey:
    path: str
    expected_sha256: str
    device: str
    mtime_ns: int | None
    size: int | None


_CACHE_LOCK = Lock()
_CLASSIFIER_CACHE: tuple[_CacheKey, SubarticularFrozenClassifier] | None = None


def _configured_checkpoint_path() -> Path | None:
    return get_settings().subarticular_checkpoint_path


def _runtime_device_name() -> str:
    configured = get_settings().subarticular_device or "cpu"
    configured = configured.strip().lower()
    if configured == "auto":
        return "cuda" if _torch_cuda_available() else "cpu"
    return configured


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _checkpoint_key(path: Path, expected_sha256: str, device: str) -> _CacheKey:
    if path.is_file():
        stat = path.stat()
        return _CacheKey(str(path.resolve()), expected_sha256, device, stat.st_mtime_ns, stat.st_size)
    return _CacheKey(str(path), expected_sha256, device, None, None)


def get_subarticular_runtime_status(*, verify_hash: bool = False) -> dict[str, Any]:
    path = _configured_checkpoint_path()
    configured = path is not None
    artifact_present = bool(path and path.is_file())
    loaded = bool(_CLASSIFIER_CACHE and _CLASSIFIER_CACHE[1].is_loaded)
    status = "not_configured"
    hash_status = "not_checked"
    if configured and not artifact_present:
        status = "artifact_missing"
    elif configured and artifact_present:
        status = "loaded" if loaded else "available"
        if verify_hash:
            try:
                from .subarticular_frozen_classifier import sha256_file

                hash_status = "match" if sha256_file(path) == EXPECTED_CHECKPOINT_SHA256 else "mismatch"
                if hash_status == "mismatch":
                    status = "invalid_hash"
            except Exception:
                # "unavailable" no distingue un archivo ilegible de un import roto, y
                # este chequeo es lo que decide si el checkpoint es el esperado. Sin el
                # rastro, el readiness dice "no se pudo verificar" y nadie puede saber
                # por que.
                log.exception("event=subarticular_hash_check_failed path=%s", path)
                hash_status = "unavailable"

    return {
        "modelId": MODEL_ID,
        "configured": configured,
        "artifactPresent": artifact_present,
        "loaded": loaded,
        "status": status,
        "checkpointHashExpected": EXPECTED_CHECKPOINT_SHA256,
        "checkpointHashStatus": hash_status,
        "device": _runtime_device_name(),
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "autonomousDiagnosis": False,
        "roiLimitation": PUBLIC_ROI_LIMITATION,
    }


def clear_subarticular_classifier_cache() -> None:
    global _CLASSIFIER_CACHE
    with _CACHE_LOCK:
        _CLASSIFIER_CACHE = None


def get_subarticular_classifier() -> SubarticularFrozenClassifier:
    global _CLASSIFIER_CACHE
    path = _configured_checkpoint_path()
    if path is None:
        raise CheckpointNotFoundError("subarticular_checkpoint_not_configured")
    device = _runtime_device_name()
    config = SubarticularFrozenClassifierConfig(
        checkpoint_path=path,
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        map_location=device,
    )
    key = _checkpoint_key(path, config.expected_checkpoint_sha256, device)
    with _CACHE_LOCK:
        if _CLASSIFIER_CACHE and _CLASSIFIER_CACHE[0] == key:
            return _CLASSIFIER_CACHE[1]
        classifier = SubarticularFrozenClassifier(config).load()
        _CLASSIFIER_CACHE = (key, classifier)
        return classifier


def predict_subarticular_prepared(sample, *, side: str, level: str, source_position: int = 0) -> SubarticularPrediction:
    return get_subarticular_classifier().predict_preprocessed(
        sample,
        side=side,
        level=level,
        source_position=source_position,
    )


def predict_subarticular_from_registered_roi(
    *,
    input_id: str,
    instance_number: int,
    x: float,
    y: float,
    side: str,
    level: str,
) -> SubarticularPrediction:
    record = _registered_axial_input(input_id)
    roi = SubarticularRoi(
        series_path=record.path,
        instance_number=int(instance_number),
        x=float(x),
        y=float(y),
        side=side,
        level=level,
    )
    return get_subarticular_classifier().predict_from_roi(roi)


def _registered_axial_input(input_id: str) -> InputRecord:
    record = resolve_registered_input(input_id)
    if record.plane != "axial":
        raise InputRegistryError("inputId no corresponde a una serie axial", status_code=409)
    if record.format not in ALLOWED_REGISTERED_FORMATS:
        raise InputRegistryError("input axial incompatible con clasificador subarticular", status_code=422)
    if not record.path.exists() or not (record.path.is_dir() or record.path.is_file()):
        raise InputRegistryError("input axial registrado no disponible", status_code=404)
    return record


def public_error_status(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, CheckpointNotFoundError):
        return 503, "SUBARTICULAR_CHECKPOINT_UNAVAILABLE"
    if isinstance(exc, CheckpointHashMismatchError):
        return 500, "SUBARTICULAR_CHECKPOINT_HASH_MISMATCH"
    if isinstance(exc, CheckpointIncompatibleError):
        return 500, "SUBARTICULAR_CHECKPOINT_INCOMPATIBLE"
    if isinstance(exc, InvalidSubarticularInputError):
        return 400, "SUBARTICULAR_INVALID_INPUT"
    if isinstance(exc, SubarticularClassifierError):
        return 500, "SUBARTICULAR_RUNTIME_ERROR"
    return 500, "AI_MODULE_ERROR"


def public_error_message(code: str) -> str:
    messages = {
        "SUBARTICULAR_CHECKPOINT_UNAVAILABLE": "Checkpoint subarticular no disponible.",
        "SUBARTICULAR_CHECKPOINT_HASH_MISMATCH": "Checkpoint subarticular invalido.",
        "SUBARTICULAR_CHECKPOINT_INCOMPATIBLE": "Checkpoint subarticular incompatible.",
        "SUBARTICULAR_INVALID_INPUT": "Solicitud subarticular invalida.",
        "SUBARTICULAR_RUNTIME_ERROR": "Fallo controlado del clasificador subarticular.",
    }
    return messages.get(code, "Fallo interno controlado del AI Module")
