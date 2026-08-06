"""Runtime integration for the frozen RSNA subarticular Axial T2 classifier.

The classifier is research-only, requires human review, and does not perform
anatomical ROI detection. Runtime callers must provide either a prepared 2.5D
three-channel sample or an explicit ROI/coordinate that matches the training
pipeline assumptions from Notebook 63/64.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .contracts.degenerative_findings import (
    SCHEMA_VERSION,
    validate_degenerative_findings_payload,
)

MODEL_ID = "rsna_subarticular_axial_t2_2p5d"
EXPECTED_CHECKPOINT_SHA256 = "d41262d57b13c146a48ab15f5e183cc6a55fc92724b7d0c286cea1f2ce26e84a"
CHECKPOINT_ENV_VAR = "PFI_SUBARTICULAR_CHECKPOINT_PATH"
DEFAULT_DEV_CHECKPOINT = (
    Path("PFI_MVP")
    / "models"
    / "P10_6_rsna_findings"
    / "subarticular_axial_t2_2p5d"
    / "final_internal_test_evaluation"
    / "frozen_subarticular_checkpoint.pt"
)
CLASS_NAMES = ("normal_mild", "moderate", "severe")
DISPLAY_CLASS_NAMES = ("Normal/Mild", "Moderate", "Severe")
SIDES = ("left", "right")
SIDE_TO_INDEX = {name: index for index, name in enumerate(SIDES)}
LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
LEVEL_TO_INDEX = {name: index for index, name in enumerate(LEVELS)}
FINAL_INTERNAL_TEST_METRICS = {
    "macro_f1": 0.6284558720836352,
    "balanced_accuracy": 0.6565011404151501,
    "normal_mild_recall": 0.8455165692007798,
    "moderate_recall": 0.44582593250444047,
    "severe_recall": 0.6781609195402298,
    "weighted_log_loss": 0.6884576824836782,
    "support": 2876,
}


class SubarticularClassifierError(RuntimeError):
    """Base class for controlled runtime errors."""


class CheckpointNotFoundError(SubarticularClassifierError):
    pass


class CheckpointHashMismatchError(SubarticularClassifierError):
    pass


class CheckpointIncompatibleError(SubarticularClassifierError):
    pass


class InvalidSubarticularInputError(SubarticularClassifierError):
    pass


@dataclass(frozen=True)
class RuntimeTrainConfig:
    seed: int = 2026
    image_size: int = 224
    crop_size: int = 256
    batch_size: int = 32
    num_workers: int = 2
    max_epochs: int = 15
    patience: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    model_name: str = "efficientnet_b0"
    pretrained: bool = False
    side_embedding_dim: int = 8
    level_embedding_dim: int = 12
    dropout: float = 0.25
    label_smoothing: float = 0.05
    severe_loss_multiplier: float = 1.25
    max_grad_norm: float = 2.0
    minimum_macro_f1: float = 0.36
    minimum_balanced_accuracy: float = 0.45
    minimum_severe_recall: float = 0.30
    minimum_moderate_recall: float = 0.25


@dataclass(frozen=True)
class SubarticularFrozenClassifierConfig:
    model_name: str = "efficientnet_b0"
    image_size: int = 224
    crop_size: int = 256
    input_channels: int = 3
    task: str = "subarticular_stenosis_left_right"
    sequence: str = "Axial T2"
    classes: tuple[str, str, str] = CLASS_NAMES
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256
    checkpoint_path: Path | None = None
    map_location: str = "cpu"
    humanReviewRequired: bool = True
    notClinicalDiagnosis: bool = True
    autonomousDiagnosis: bool = False

    @classmethod
    def from_env(cls) -> "SubarticularFrozenClassifierConfig":
        configured = os.getenv(CHECKPOINT_ENV_VAR)
        path = Path(configured) if configured else DEFAULT_DEV_CHECKPOINT
        return cls(checkpoint_path=path)


@dataclass(frozen=True)
class SubarticularRoi:
    series_path: Path
    instance_number: int
    x: float
    y: float
    side: str
    level: str


@dataclass(frozen=True)
class SubarticularPrediction:
    findingType: str
    predictedSeverity: str
    probabilities: Mapping[str, float]
    confidence: float
    side: str
    level: str
    modelName: str
    checkpointSha256: str
    humanReviewRequired: bool
    notClinicalDiagnosis: bool
    autonomousDiagnosis: bool
    roiSource: str
    warnings: tuple[str, ...]
    degenerativeFindings: Mapping[str, Any]


class RuntimeSubarticularClassifierModel(nn.Module):
    """Runtime mirror of the Notebook 63 EfficientNet + side/level head."""

    def __init__(self, config: RuntimeTrainConfig) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            config.model_name,
            pretrained=False,
            in_chans=3,
            num_classes=0,
            global_pool="avg",
        )
        self.side_embedding = nn.Embedding(len(SIDES), config.side_embedding_dim)
        self.level_embedding = nn.Embedding(len(LEVELS), config.level_embedding_dim)
        feature_dim = int(getattr(self.backbone, "num_features"))
        combined_dim = feature_dim + config.side_embedding_dim + config.level_embedding_dim
        self.head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(config.dropout),
            nn.Linear(combined_dim, len(CLASS_NAMES)),
        )
        self.feature_dim = feature_dim

    def forward(self, image: torch.Tensor, side_index: torch.Tensor, level_index: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        combined = torch.cat(
            [features, self.side_embedding(side_index), self.level_embedding(level_index)],
            dim=1,
        )
        return self.head(combined)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(map_location: str) -> torch.device:
    requested = str(map_location or "cpu").strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SubarticularClassifierError("cuda_requested_but_not_available")
    if requested not in {"cpu", "cuda"} and not requested.startswith("cuda:"):
        raise SubarticularClassifierError("unsupported_runtime_device")
    return torch.device(requested)


def _checkpoint_path(config: SubarticularFrozenClassifierConfig) -> Path:
    return Path(config.checkpoint_path or DEFAULT_DEV_CHECKPOINT)


def _validate_side_level(side: str, level: str) -> tuple[str, str]:
    normalized_side = str(side).strip().lower()
    normalized_level = str(level).strip().upper().replace("/", "-")
    if normalized_side not in SIDES:
        raise InvalidSubarticularInputError("invalid_side")
    if normalized_level not in LEVELS:
        raise InvalidSubarticularInputError("invalid_level")
    return normalized_side, normalized_level


def _validate_checkpoint_payload(
    checkpoint: Any,
    config: SubarticularFrozenClassifierConfig,
) -> RuntimeTrainConfig:
    if not isinstance(checkpoint, Mapping):
        raise CheckpointIncompatibleError("checkpoint_payload_must_be_mapping")
    required = {"modelStateDict", "config", "classNames", "sideToIndex", "levelToIndex"}
    missing = sorted(required - set(checkpoint.keys()))
    if missing:
        raise CheckpointIncompatibleError("checkpoint_missing_required_fields")
    if checkpoint.get("task") != config.task:
        raise CheckpointIncompatibleError("checkpoint_task_mismatch")
    if checkpoint.get("sequence") != config.sequence:
        raise CheckpointIncompatibleError("checkpoint_sequence_mismatch")
    if tuple(checkpoint.get("classNames") or ()) != tuple(config.classes):
        raise CheckpointIncompatibleError("checkpoint_class_names_mismatch")
    if dict(checkpoint.get("sideToIndex") or {}) != dict(SIDE_TO_INDEX):
        raise CheckpointIncompatibleError("checkpoint_side_mapping_mismatch")
    if dict(checkpoint.get("levelToIndex") or {}) != dict(LEVEL_TO_INDEX):
        raise CheckpointIncompatibleError("checkpoint_level_mapping_mismatch")
    if checkpoint.get("humanReviewRequired") is not True:
        raise CheckpointIncompatibleError("checkpoint_human_review_required_mismatch")
    if checkpoint.get("notClinicalDiagnosis") is not True:
        raise CheckpointIncompatibleError("checkpoint_not_clinical_diagnosis_mismatch")

    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, Mapping):
        raise CheckpointIncompatibleError("checkpoint_config_must_be_mapping")
    for key, expected in {
        "model_name": config.model_name,
        "image_size": config.image_size,
        "crop_size": config.crop_size,
    }.items():
        if raw_config.get(key) != expected:
            raise CheckpointIncompatibleError(f"checkpoint_{key}_mismatch")
    return replace(RuntimeTrainConfig(**dict(raw_config)), pretrained=False)


def preprocess_prepared_2p5d(
    sample: np.ndarray,
    config: SubarticularFrozenClassifierConfig,
) -> torch.Tensor:
    array = np.asarray(sample)
    if array.size == 0:
        raise InvalidSubarticularInputError("empty_array")
    if array.ndim != 3:
        raise InvalidSubarticularInputError("expected_three_dimensional_2p5d_array")
    if array.shape[0] != config.input_channels and array.shape[-1] == config.input_channels:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] != config.input_channels:
        raise InvalidSubarticularInputError("expected_three_input_channels")
    if array.shape[1] <= 0 or array.shape[2] <= 0:
        raise InvalidSubarticularInputError("invalid_spatial_shape")
    if not np.isfinite(array).all():
        raise InvalidSubarticularInputError("array_contains_nan_or_infinite")

    value = array.astype(np.float32, copy=False)
    if float(value.max()) <= 1.0 and float(value.min()) >= 0.0:
        value = value * 255.0
    value = np.clip(value, 0.0, 255.0)
    tensor = torch.from_numpy(value).float()[None]
    if tuple(value.shape[1:]) != (config.image_size, config.image_size):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(config.image_size, config.image_size),
            mode="bilinear",
            align_corners=False,
        )
    image = tensor[0] / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (image - mean) / std


@lru_cache(maxsize=4096)
def _dicom_files(series_path_text: str) -> tuple[Path, ...]:
    series_path = Path(series_path_text)
    files = list(series_path.glob("*.dcm"))
    if not files:
        raise InvalidSubarticularInputError("dicom_series_incomplete")

    def order(path: Path) -> tuple[int, str]:
        if path.stem.isdigit():
            return int(path.stem), path.name
        try:
            import pydicom

            header = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["InstanceNumber"])
            return int(getattr(header, "InstanceNumber", 0)), path.name
        except Exception:
            return 0, path.name

    return tuple(sorted(files, key=order))


def _read_dicom(path: Path) -> np.ndarray:
    import pydicom

    dataset = pydicom.dcmread(path)
    image = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return image * slope + intercept


def _crop(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    if image.ndim != 2:
        image = np.squeeze(image)
    if image.ndim != 2:
        raise InvalidSubarticularInputError(f"expected_2d_slice_shape={image.shape}")
    height, width = image.shape
    half = size // 2
    left = int(round(float(x))) - half
    top = int(round(float(y))) - half
    right = left + size
    bottom = top + size
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    crop = image[top:bottom, left:right]
    if crop.shape != (size, size):
        raise InvalidSubarticularInputError("invalid_crop_shape")
    return crop


def _uint8_stack(stack: list[np.ndarray]) -> np.ndarray:
    joined = np.concatenate([array[np.isfinite(array)].ravel() for array in stack])
    if joined.size == 0:
        return np.zeros((len(stack), *stack[0].shape), dtype=np.uint8)
    low, high = np.percentile(joined, [1, 99])
    high = max(float(high), float(low) + 1.0)
    normalized = [
        (np.clip((array - low) / (high - low), 0.0, 1.0) * 255.0).round().astype(np.uint8)
        for array in stack
    ]
    return np.stack(normalized, axis=0)


def _resize_stack(stack: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(stack).float()[None]
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized.clamp(0, 255).round().byte().numpy()


def build_preprocessed_from_roi(
    roi: SubarticularRoi,
    config: SubarticularFrozenClassifierConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    side, level = _validate_side_level(roi.side, roi.level)
    if roi.instance_number < 0:
        raise InvalidSubarticularInputError("instance_number_must_be_non_negative")
    files = _dicom_files(str(roi.series_path))
    numeric_lookup = {int(path.stem): index for index, path in enumerate(files) if path.stem.isdigit()}
    if numeric_lookup:
        center = numeric_lookup.get(int(roi.instance_number))
        if center is None:
            nearest = min(numeric_lookup, key=lambda value: abs(value - int(roi.instance_number)))
            center = numeric_lookup[nearest]
    else:
        center = min(max(int(roi.instance_number) - 1, 0), len(files) - 1)
    indices = (max(0, center - 1), center, min(len(files) - 1, center + 1))
    crops: list[np.ndarray] = []
    for index in indices:
        image = _read_dicom(files[index])
        if image.ndim != 2:
            image = np.squeeze(image)
        if image.ndim != 2:
            raise InvalidSubarticularInputError("dicom_slice_must_be_2d")
        height, width = image.shape
        if roi.x < 0 or roi.y < 0 or roi.x >= width or roi.y >= height:
            raise InvalidSubarticularInputError("roi_coordinates_out_of_range")
        crops.append(_crop(image, roi.x, roi.y, config.crop_size))
    stack = _resize_stack(_uint8_stack(crops), config.image_size)
    return stack, {
        "side": side,
        "level": level,
        "sourceSeries": {"role": "axial_t2", "position": int(roi.instance_number)},
        "localization": {"source": "external_coordinate", "researchOnly": True},
        "roiSource": "operator_provided_external_coordinate",
        "warnings": ("roi_requires_external_anatomical_coordinate",),
    }


class SubarticularFrozenClassifier:
    def __init__(self, config: SubarticularFrozenClassifierConfig | None = None) -> None:
        self.config = config or SubarticularFrozenClassifierConfig.from_env()
        self._model: RuntimeSubarticularClassifierModel | None = None
        self._checkpoint_sha256: str | None = None
        self._device: torch.device | None = None
        self._checkpoint_metadata: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> "SubarticularFrozenClassifier":
        path = _checkpoint_path(self.config)
        if not path.is_file():
            raise CheckpointNotFoundError("subarticular_checkpoint_not_found")
        actual_sha = sha256_file(path)
        if actual_sha != self.config.expected_checkpoint_sha256:
            raise CheckpointHashMismatchError("subarticular_checkpoint_sha256_mismatch")
        device = _resolve_device(self.config.map_location)
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        train_config = _validate_checkpoint_payload(checkpoint, self.config)
        model = RuntimeSubarticularClassifierModel(train_config)
        state_dict = checkpoint.get("modelStateDict")
        if not isinstance(state_dict, Mapping):
            raise CheckpointIncompatibleError("checkpoint_state_dict_missing")
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        self._device = device
        self._checkpoint_sha256 = actual_sha
        self._checkpoint_metadata = {
            "schemaVersion": checkpoint.get("schemaVersion"),
            "epoch": checkpoint.get("epoch"),
            "config": asdict(train_config),
            "classNames": list(self.config.classes),
        }
        return self

    def model_metadata(self) -> dict[str, Any]:
        return {
            "modelId": MODEL_ID,
            "modelName": self.config.model_name,
            "task": self.config.task,
            "sequence": self.config.sequence,
            "imageSize": self.config.image_size,
            "cropSize": self.config.crop_size,
            "inputChannels": self.config.input_channels,
            "classes": list(self.config.classes),
            "checkpointSha256": self._checkpoint_sha256 or self.config.expected_checkpoint_sha256,
            "loaded": self.is_loaded,
            "device": str(self._device) if self._device else self.config.map_location,
            "humanReviewRequired": self.config.humanReviewRequired,
            "notClinicalDiagnosis": self.config.notClinicalDiagnosis,
            "autonomousDiagnosis": self.config.autonomousDiagnosis,
            "finalInternalTestMetrics": dict(FINAL_INTERNAL_TEST_METRICS),
            "roiLimitation": "requires_external_anatomical_coordinate; no automatic ROI localizer is validated",
        }

    def predict_preprocessed(
        self,
        sample: np.ndarray,
        *,
        side: str,
        level: str,
        source_position: int = 0,
        localization_source: str = "not_available",
        research_only: bool = True,
        roi_source: str = "prepared_2p5d_runtime_input",
        warnings: tuple[str, ...] = ("no_automatic_roi_localizer_validated",),
    ) -> SubarticularPrediction:
        model = self._require_model()
        device = self._device or torch.device("cpu")
        side, level = _validate_side_level(side, level)
        if source_position < 0:
            raise InvalidSubarticularInputError("source_position_must_be_non_negative")
        image = preprocess_prepared_2p5d(sample, self.config).to(device)
        side_index = torch.tensor([SIDE_TO_INDEX[side]], dtype=torch.long, device=device)
        level_index = torch.tensor([LEVEL_TO_INDEX[level]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(image[None], side_index, level_index)
            probabilities_tensor = torch.softmax(logits.float(), dim=1)[0]
        probabilities = probabilities_tensor.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(probabilities).all():
            raise SubarticularClassifierError("probabilities_not_finite")
        total = float(probabilities.sum())
        if total <= 0.0:
            raise SubarticularClassifierError("probabilities_invalid_sum")
        probabilities = probabilities / total
        class_index = int(np.argmax(probabilities))
        predicted = self.config.classes[class_index]
        probability_map = {
            label: float(probabilities[index]) for index, label in enumerate(self.config.classes)
        }
        payload = self._contract_payload(
            predicted=predicted,
            probabilities=probability_map,
            side=side,
            level=level,
            source_position=source_position,
            localization_source=localization_source,
            research_only=research_only,
        )
        validate_degenerative_findings_payload(payload)
        return SubarticularPrediction(
            findingType="subarticular_stenosis",
            predictedSeverity=predicted,
            probabilities=probability_map,
            confidence=float(probabilities[class_index]),
            side=side,
            level=level,
            modelName=self.config.model_name,
            checkpointSha256=self._checkpoint_sha256 or self.config.expected_checkpoint_sha256,
            humanReviewRequired=self.config.humanReviewRequired,
            notClinicalDiagnosis=self.config.notClinicalDiagnosis,
            autonomousDiagnosis=self.config.autonomousDiagnosis,
            roiSource=roi_source,
            warnings=warnings,
            degenerativeFindings=payload,
        )

    def predict_from_roi(self, roi: SubarticularRoi) -> SubarticularPrediction:
        stack, metadata = build_preprocessed_from_roi(roi, self.config)
        return self.predict_preprocessed(
            stack,
            side=metadata["side"],
            level=metadata["level"],
            source_position=metadata["sourceSeries"]["position"],
            localization_source=metadata["localization"]["source"],
            research_only=metadata["localization"]["researchOnly"],
            roi_source=metadata["roiSource"],
            warnings=metadata["warnings"],
        )

    def _require_model(self) -> RuntimeSubarticularClassifierModel:
        if self._model is None:
            self.load()
        if self._model is None:
            raise SubarticularClassifierError("subarticular_model_not_loaded")
        return self._model

    def _contract_payload(
        self,
        *,
        predicted: str,
        probabilities: Mapping[str, float],
        side: str,
        level: str,
        source_position: int,
        localization_source: str,
        research_only: bool,
    ) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "findings": [
                {
                    "findingId": f"subarticular-stenosis-{level.lower()}-{side}-axial-t2-{source_position}",
                    "findingType": "subarticular_stenosis",
                    "anatomy": {"level": level, "side": side},
                    "classification": {"label": predicted, "probabilities": dict(probabilities)},
                    "evaluation": {"status": "evaluated"},
                    "sourceSeries": {"role": "axial_t2", "position": int(source_position)},
                    "localization": {"source": localization_source, "researchOnly": bool(research_only)},
                    "model": {
                        "modelId": MODEL_ID,
                        "modelSha256": self._checkpoint_sha256 or self.config.expected_checkpoint_sha256,
                    },
                    "review": {"required": True, "status": "pending"},
                    "notClinicalDiagnosis": True,
                }
            ],
        }
