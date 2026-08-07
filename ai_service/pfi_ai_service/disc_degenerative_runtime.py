"""Runtime bridge for P10.7 SPIDER disc-level degenerative multitask export."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping
from uuid import uuid5, NAMESPACE_URL

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    import torch
    from torch import nn
    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only in dependency-broken environments.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

from .contracts.disc_degenerative_findings import (
    BINARY_LABELS,
    BINARY_TASKS,
    CATEGORICAL_TASKS,
    DEPLOYMENT_STATUS_BY_TASK,
    EXPECTED_CHECKPOINT_SHA256,
    FINDING_TYPES,
    MODEL_ID,
    MODIC_LABELS,
    PFIRRMANN_LABELS,
    SCHEMA_VERSION,
    SERIES_ROLES,
    SPINE_LEVELS,
    TASK_ORDER,
    classification_kind,
    classification_labels,
    level_to_ivd_mapping,
    validate_disc_degenerative_findings_envelope,
)
from .settings import get_settings

EXPECTED_CHECKPOINT_SCHEMA_VERSION = "pfi.p10-7-research-export.v1"
PREPROCESSING_PARITY_VALIDATED = False
AUTOMATIC_DISC_LOCALIZATION_VALIDATED = False
MODEL_SHA256 = EXPECTED_CHECKPOINT_SHA256


class DiscDegenerativeRuntimeError(Exception):
    status_code = 500
    public_code = "DISC_DEGENERATIVE_RUNTIME_ERROR"


class DiscDegenerativeCheckpointNotConfigured(DiscDegenerativeRuntimeError):
    status_code = 503
    public_code = "DISC_DEGENERATIVE_CHECKPOINT_NOT_CONFIGURED"


class DiscDegenerativeCheckpointMissing(DiscDegenerativeRuntimeError):
    status_code = 503
    public_code = "DISC_DEGENERATIVE_CHECKPOINT_MISSING"


class DiscDegenerativeCheckpointHashMismatch(DiscDegenerativeRuntimeError):
    status_code = 503
    public_code = "DISC_DEGENERATIVE_CHECKPOINT_HASH_MISMATCH"


class DiscDegenerativeCheckpointInvalid(DiscDegenerativeRuntimeError):
    status_code = 503
    public_code = "DISC_DEGENERATIVE_CHECKPOINT_INVALID"


class DiscDegenerativeDependencyUnavailable(DiscDegenerativeRuntimeError):
    status_code = 503
    public_code = "DISC_DEGENERATIVE_DEPENDENCY_UNAVAILABLE"


class DiscDegenerativeInputUnavailable(DiscDegenerativeRuntimeError):
    status_code = 422
    public_code = "DISC_DEGENERATIVE_PREPROCESSING_NOT_AVAILABLE"


class DiscDegenerativeInvalidRequest(DiscDegenerativeRuntimeError):
    status_code = 422
    public_code = "DISC_DEGENERATIVE_INVALID_REQUEST"


class DiscSourceSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    inputId: str | None = None
    available: bool = True
    positions: list[int] = Field(default_factory=list)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in SERIES_ROLES:
            raise ValueError("unsupported_source_series_role")
        return value


class DiscLevelPredictItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str
    localization: dict[str, Any] | None = None
    sourceSeries: list[DiscSourceSeries] = Field(default_factory=list)

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        if value not in SPINE_LEVELS:
            raise ValueError("unsupported_lumbar_level")
        return value


class DiscMultitaskPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    caseId: str | None = None
    levels: list[DiscLevelPredictItem] = Field(default_factory=list, min_length=1)


@dataclass(frozen=True)
class PreparedDiscLevel:
    level: str
    t1: np.ndarray | None
    t2: np.ndarray | None
    t1_positions: tuple[int, ...] = ()
    t2_positions: tuple[int, ...] = ()
    localization_source: str = "external_disc_roi"


@dataclass(frozen=True)
class _CacheKey:
    path: str
    expected_sha256: str
    device: str
    mtime_ns: int | None
    size: int | None


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise DiscDegenerativeDependencyUnavailable("torch_runtime_unavailable") from _TORCH_IMPORT_ERROR
    return torch


_ModuleBase = nn.Module if nn is not None else object


class P10_7DiscClassifier(_ModuleBase):
    def __init__(self, backbone: str = "efficientnet_b0", dropout: float = 0.25, pretrained: bool = False) -> None:
        _require_torch()
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=3,
        )
        feature_dim = int(self.encoder.num_features)
        self.ivd_embedding = nn.Embedding(25, 16)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2 + 16, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.heads = nn.ModuleDict(
            {
                "pfirrmann_grade": nn.Linear(256, 5),
                "modic_change": nn.Linear(256, 4),
                **{task: nn.Linear(256, 1) for task in BINARY_TASKS},
            }
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor, availability: torch.Tensor, ivd_index: torch.Tensor) -> dict[str, torch.Tensor]:
        t1_features = self.encoder(t1) * availability[:, 0:1]
        t2_features = self.encoder(t2) * availability[:, 1:2]
        ivd_features = self.ivd_embedding(ivd_index.clamp(0, 24))
        shared = self.trunk(torch.cat([t1_features, t2_features, availability, ivd_features], dim=1))
        return {name: head(shared) for name, head in self.heads.items()}


_CACHE_LOCK = Lock()
_CLASSIFIER_CACHE: tuple[_CacheKey, "DiscDegenerativeClassifier"] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configured_checkpoint_path() -> Path | None:
    return get_settings().p10_7_checkpoint_path


def _runtime_device_name() -> str:
    configured = get_settings().p10_7_device or "cpu"
    configured = configured.strip().lower()
    if configured == "auto":
        if torch is None:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured


def _checkpoint_key(path: Path, expected_sha256: str, device: str) -> _CacheKey:
    if path.is_file():
        stat = path.stat()
        return _CacheKey(str(path.resolve()), expected_sha256, device, stat.st_mtime_ns, stat.st_size)
    return _CacheKey(str(path), expected_sha256, device, None, None)


class DiscDegenerativeClassifier:
    def __init__(self, checkpoint_path: Path, *, expected_sha256: str = EXPECTED_CHECKPOINT_SHA256, map_location: str = "cpu") -> None:
        self.checkpoint_path = checkpoint_path
        self.expected_sha256 = expected_sha256
        self.map_location = map_location
        self.model: P10_7DiscClassifier | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> "DiscDegenerativeClassifier":
        torch_runtime = _require_torch()
        if not self.checkpoint_path.is_file():
            raise DiscDegenerativeCheckpointMissing("disc_degenerative_checkpoint_missing")
        actual = sha256_file(self.checkpoint_path)
        if actual != self.expected_sha256:
            raise DiscDegenerativeCheckpointHashMismatch("disc_degenerative_checkpoint_hash_mismatch")

        checkpoint = torch_runtime.load(self.checkpoint_path, map_location=self.map_location, weights_only=False)
        self._validate_checkpoint_contract(checkpoint)
        cfg = checkpoint["trainConfig"]
        model = P10_7DiscClassifier(
            backbone=str(cfg["backbone"]),
            dropout=float(cfg["dropout"]),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["modelStateDict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model.to(torch_runtime.device(self.map_location))
        self.metadata = {
            "modelId": MODEL_ID,
            "modelSha256": actual,
            "schemaVersion": checkpoint["schemaVersion"],
            "taskOrder": list(checkpoint["taskOrder"]),
            "device": self.map_location,
        }
        return self

    def _validate_checkpoint_contract(self, checkpoint: Any) -> None:
        if not isinstance(checkpoint, Mapping):
            raise DiscDegenerativeCheckpointInvalid("checkpoint_payload_not_mapping")
        required = ["schemaVersion", "modelStateDict", "trainConfig", "taskOrder", "categoricalTasks", "binaryTasks"]
        missing = [key for key in required if key not in checkpoint]
        if missing:
            raise DiscDegenerativeCheckpointInvalid("checkpoint_missing_required_fields")
        if checkpoint.get("schemaVersion") != EXPECTED_CHECKPOINT_SCHEMA_VERSION:
            raise DiscDegenerativeCheckpointInvalid("checkpoint_schema_version_mismatch")
        if list(checkpoint.get("taskOrder", [])) != list(TASK_ORDER):
            raise DiscDegenerativeCheckpointInvalid("checkpoint_task_order_mismatch")
        if dict(checkpoint.get("categoricalTasks", {})) != dict(CATEGORICAL_TASKS):
            raise DiscDegenerativeCheckpointInvalid("checkpoint_categorical_tasks_mismatch")
        if list(checkpoint.get("binaryTasks", [])) != list(BINARY_TASKS):
            raise DiscDegenerativeCheckpointInvalid("checkpoint_binary_tasks_mismatch")
        cfg = checkpoint.get("trainConfig")
        if not isinstance(cfg, Mapping) or not {"backbone", "dropout"}.issubset(set(cfg.keys())):
            raise DiscDegenerativeCheckpointInvalid("checkpoint_train_config_incomplete")

    def predict_preprocessed(self, samples: list[PreparedDiscLevel]) -> dict[str, Any]:
        torch_runtime = _require_torch()
        if self.model is None:
            raise DiscDegenerativeCheckpointInvalid("classifier_not_loaded")
        if not samples:
            raise DiscDegenerativeInvalidRequest("levels_required")

        findings: list[dict[str, Any]] = []
        with torch_runtime.no_grad():
            for sample in samples:
                t1, t2, availability = _prepare_tensors(sample)
                mapping = level_to_ivd_mapping(sample.level)
                device = next(self.model.parameters()).device
                outputs = self.model(
                    t1.to(device),
                    t2.to(device),
                    availability.to(device),
                    torch_runtime.tensor([mapping.ivd_index], dtype=torch_runtime.long, device=device),
                )
                findings.extend(_findings_from_outputs(sample, outputs))

        envelope = {
            "discDegenerativeFindings": {
                "schemaVersion": SCHEMA_VERSION,
                "findings": findings,
            },
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
            "autonomousDiagnosis": False,
        }
        validate_disc_degenerative_findings_envelope(envelope)
        return envelope


def _prepare_tensors(sample: PreparedDiscLevel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch_runtime = _require_torch()
    if sample.t1 is None and sample.t2 is None:
        raise DiscDegenerativeInvalidRequest("at_least_one_modality_required")
    t1 = _validate_crop(sample.t1, "t1") if sample.t1 is not None else np.zeros((3, 224, 224), dtype=np.float32)
    t2 = _validate_crop(sample.t2, "t2") if sample.t2 is not None else np.zeros((3, 224, 224), dtype=np.float32)
    availability = np.array([sample.t1 is not None, sample.t2 is not None], dtype=np.float32)
    return (
        torch_runtime.from_numpy(t1).float().unsqueeze(0),
        torch_runtime.from_numpy(t2).float().unsqueeze(0),
        torch_runtime.from_numpy(availability).float().unsqueeze(0),
    )


def _validate_crop(value: np.ndarray, role: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3, 224, 224):
        raise DiscDegenerativeInvalidRequest(f"{role}_crop_shape_invalid")
    if not np.isfinite(array).all():
        raise DiscDegenerativeInvalidRequest(f"{role}_crop_non_finite")
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def _findings_from_outputs(sample: PreparedDiscLevel, outputs: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        probabilities = _probabilities_for_task(task, outputs[task])
        label = max(probabilities, key=probabilities.get)
        result.append(
            {
                "findingId": str(uuid5(NAMESPACE_URL, f"{MODEL_ID}:{sample.level}:{task}")),
                "findingType": task,
                "anatomy": {"level": sample.level, "side": None},
                "classification": {
                    "kind": classification_kind(task),
                    "label": label,
                    "probabilities": probabilities,
                },
                "evidence": {
                    "deploymentStatus": DEPLOYMENT_STATUS_BY_TASK[task],
                    "evaluationDataset": "SPIDER_internal_test",
                    "externalValidationAvailable": False,
                },
                "evaluation": {"status": "evaluated"},
                "sourceSeries": [
                    {"role": "sagittal_t1", "available": sample.t1 is not None, "positions": list(sample.t1_positions)},
                    {"role": "sagittal_t2", "available": sample.t2 is not None, "positions": list(sample.t2_positions)},
                ],
                "localization": {
                    "source": sample.localization_source,
                    "researchOnly": True,
                    "automaticAnatomicalLocalizationValidated": False,
                },
                "model": {"modelId": MODEL_ID, "modelSha256": MODEL_SHA256},
                "review": {"required": True, "status": "pending"},
                "notClinicalDiagnosis": True,
            }
        )
    return result


def _probabilities_for_task(task: str, logits: torch.Tensor) -> dict[str, float]:
    torch_runtime = _require_torch()
    labels = classification_labels(task)
    if task in CATEGORICAL_TASKS:
        values = torch_runtime.softmax(logits[0], dim=0).detach().cpu().numpy().astype(float)
    else:
        present = float(torch_runtime.sigmoid(logits[0, 0]).detach().cpu())
        values = np.array([1.0 - present, present], dtype=float)
    values = values / max(float(values.sum()), 1e-12)
    return {label: float(value) for label, value in zip(labels, values)}


def get_disc_degenerative_runtime_status(*, verify_hash: bool = False) -> dict[str, Any]:
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
                hash_status = "match" if sha256_file(path) == EXPECTED_CHECKPOINT_SHA256 else "mismatch"
                if hash_status == "mismatch":
                    status = "invalid_hash"
            except Exception:
                hash_status = "unavailable"
    return {
        "modelId": MODEL_ID,
        "configured": configured,
        "artifactPresent": artifact_present,
        "loaded": loaded,
        "status": status,
        "dependencyAvailable": torch is not None,
        "checkpointHashExpected": EXPECTED_CHECKPOINT_SHA256,
        "checkpointHashStatus": hash_status,
        "device": _runtime_device_name(),
        "schemaVersion": SCHEMA_VERSION,
        "preprocessingParityValidated": PREPROCESSING_PARITY_VALIDATED,
        "automaticDiscLocalizationValidated": AUTOMATIC_DISC_LOCALIZATION_VALIDATED,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "autonomousDiagnosis": False,
    }


def clear_disc_degenerative_classifier_cache() -> None:
    global _CLASSIFIER_CACHE
    with _CACHE_LOCK:
        _CLASSIFIER_CACHE = None


def get_disc_degenerative_classifier() -> DiscDegenerativeClassifier:
    global _CLASSIFIER_CACHE
    path = _configured_checkpoint_path()
    if path is None:
        raise DiscDegenerativeCheckpointNotConfigured("disc_degenerative_checkpoint_not_configured")
    device = _runtime_device_name()
    key = _checkpoint_key(path, EXPECTED_CHECKPOINT_SHA256, device)
    with _CACHE_LOCK:
        if _CLASSIFIER_CACHE and _CLASSIFIER_CACHE[0] == key:
            return _CLASSIFIER_CACHE[1]
        classifier = DiscDegenerativeClassifier(path, expected_sha256=EXPECTED_CHECKPOINT_SHA256, map_location=device).load()
        _CLASSIFIER_CACHE = (key, classifier)
        return classifier


def predict_disc_degenerative_from_registered_request(request: DiscMultitaskPredictRequest) -> dict[str, Any]:
    _validate_registered_request_references(request)
    raise DiscDegenerativeInputUnavailable("runtime_preprocessing_parity_not_validated")


def _validate_registered_request_references(request: DiscMultitaskPredictRequest) -> None:
    for item in request.levels:
        available_roles = {series.role for series in item.sourceSeries if series.available}
        if not available_roles:
            raise DiscDegenerativeInvalidRequest("at_least_one_modality_required")
        for series in item.sourceSeries:
            if not series.inputId:
                continue
            try:
                from .input_registry import InputRegistryError, resolve_registered_input
            except Exception as exc:
                raise DiscDegenerativeInputUnavailable("input_registry_unavailable") from exc

            try:
                record = resolve_registered_input(series.inputId)
            except InputRegistryError as exc:
                error = DiscDegenerativeInvalidRequest("source_input_not_registered")
                error.status_code = exc.status_code
                raise error from exc
            if record.plane != "sagittal":
                raise DiscDegenerativeInvalidRequest("source_input_must_be_sagittal")


def public_error_status(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, DiscDegenerativeRuntimeError):
        return exc.status_code, exc.public_code
    return 500, "AI_MODULE_ERROR"


def public_error_message(code: str) -> str:
    messages = {
        "DISC_DEGENERATIVE_CHECKPOINT_NOT_CONFIGURED": "Checkpoint P10.7 no configurado.",
        "DISC_DEGENERATIVE_CHECKPOINT_MISSING": "Checkpoint P10.7 no disponible.",
        "DISC_DEGENERATIVE_CHECKPOINT_HASH_MISMATCH": "Checkpoint P10.7 invalido.",
        "DISC_DEGENERATIVE_CHECKPOINT_INVALID": "Checkpoint P10.7 incompatible.",
        "DISC_DEGENERATIVE_DEPENDENCY_UNAVAILABLE": "Runtime P10.7 no disponible.",
        "DISC_DEGENERATIVE_PREPROCESSING_NOT_AVAILABLE": "Preprocesamiento P10.7 no disponible para inferencia productiva.",
        "DISC_DEGENERATIVE_INVALID_REQUEST": "Solicitud P10.7 invalida.",
    }
    return messages.get(code, "Fallo interno controlado del AI Module")
