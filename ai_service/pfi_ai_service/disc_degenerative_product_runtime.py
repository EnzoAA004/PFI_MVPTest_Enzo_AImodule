"""P10.9 product bridge for P10.7 disc-level degenerative findings.

This module reproduces the Notebook 66 crop math from a segmentation-derived disc ROI.
It does not discover a disc by itself: the ROI must come from the segmentation pipeline
and is validated as such. Automatic disc localization therefore remains an independent
E2E gate even though preprocessing parity can be tested deterministically here.
"""
from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts.disc_degenerative_findings import SERIES_ROLES, SPINE_LEVELS
from .disc_degenerative_runtime import (
    DiscDegenerativeInputUnavailable,
    DiscDegenerativeInvalidRequest,
    PreparedDiscLevel,
    get_disc_degenerative_classifier,
)
from .input_registry import InputRegistryError, resolve_registered_input
from .real_inference_runtime import load_input

PREPROCESSING_SPEC = "pfi.p10-7.notebook-66-crop-2p5d.v1"
PREPROCESSING_PARITY_VALIDATED = True
AUTOMATIC_DISC_LOCALIZATION_VALIDATED = False
OUTPUT_SIZE = 224
MARGIN_RATIO = 0.80


class SegmentationDerivedDiscLocalization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["segmentation_derived_disc_level"]
    sliceIndex: int = Field(ge=0)
    bboxYx: tuple[float, float, float, float]
    coordinateSpaceWidth: int = Field(gt=0)
    coordinateSpaceHeight: int = Field(gt=0)
    maskId: str | None = None
    segmentationRunId: str | None = None

    @field_validator("bboxYx")
    @classmethod
    def valid_bbox(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        y0, y1, x0, x1 = (float(v) for v in value)
        if not all(math.isfinite(v) for v in (y0, y1, x0, x1)):
            raise ValueError("bbox_non_finite")
        if y1 <= y0 or x1 <= x0:
            raise ValueError("bbox_empty")
        return y0, y1, x0, x1


class ProductDiscSourceSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["sagittal_t1", "sagittal_t2"]
    inputId: str
    available: bool = True
    localization: SegmentationDerivedDiscLocalization

    @field_validator("inputId")
    @classmethod
    def non_empty_input(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("input_id_required")
        return value


class ProductDiscLevelPredictItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str
    sourceSeries: list[ProductDiscSourceSeries] = Field(min_length=1)

    @field_validator("level")
    @classmethod
    def valid_level(cls, value: str) -> str:
        if value not in SPINE_LEVELS:
            raise ValueError("unsupported_lumbar_level")
        return value

    @model_validator(mode="after")
    def unique_roles(self) -> "ProductDiscLevelPredictItem":
        roles = [series.role for series in self.sourceSeries if series.available]
        if not roles:
            raise ValueError("at_least_one_modality_required")
        if len(roles) != len(set(roles)):
            raise ValueError("duplicate_source_series_role")
        return self


class ProductDiscMultitaskPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caseId: str | None = None
    levels: list[ProductDiscLevelPredictItem] = Field(min_length=1)


class PreparedCrop(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    crop: Any
    positions: tuple[int, int, int]
    nativeBboxYx: tuple[int, int, int, int]


def normalize_volume_notebook66(array: np.ndarray) -> np.ndarray:
    """Notebook 66 normalization: whole-volume p1/p99 clipping to [0, 1]."""
    volume = np.asarray(array, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise DiscDegenerativeInvalidRequest("source_volume_non_finite")
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _native_bbox(
    localization: SegmentationDerivedDiscLocalization,
    rows: int,
    columns: int,
) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = localization.bboxYx
    scale_y = float(rows) / float(localization.coordinateSpaceHeight)
    scale_x = float(columns) / float(localization.coordinateSpaceWidth)

    y0i = max(0, min(rows - 1, int(math.floor(y0 * scale_y))))
    y1i = max(y0i + 1, min(rows, int(math.ceil(y1 * scale_y))))
    x0i = max(0, min(columns - 1, int(math.floor(x0 * scale_x))))
    x1i = max(x0i + 1, min(columns, int(math.ceil(x1 * scale_x))))
    return y0i, y1i, x0i, x1i


def crop_2p5d_from_segmentation_roi(
    volume: np.ndarray,
    localization: SegmentationDerivedDiscLocalization,
    *,
    output_size: int = OUTPUT_SIZE,
    margin_ratio: float = MARGIN_RATIO,
) -> PreparedCrop:
    """Reproduce Notebook 66 after the disc mask has supplied center slice + tight bbox."""
    torch_runtime = __import__("torch")
    functional = __import__("torch.nn.functional", fromlist=["interpolate"])

    image = np.asarray(volume, dtype=np.float32)
    if image.ndim != 3:
        raise DiscDegenerativeInvalidRequest("source_volume_must_be_3d")

    center_z = int(localization.sliceIndex)
    if center_z >= image.shape[0]:
        raise DiscDegenerativeInvalidRequest("localization_slice_out_of_range")

    y0, y1, x0, x1 = _native_bbox(localization, image.shape[1], image.shape[2])
    height, width = y1 - y0, x1 - x0
    margin_y = max(8, int(round(height * margin_ratio)))
    margin_x = max(8, int(round(width * margin_ratio)))
    y0, y1 = max(0, y0 - margin_y), min(image.shape[1], y1 + margin_y)
    x0, x1 = max(0, x0 - margin_x), min(image.shape[2], x1 + margin_x)

    z_indices = tuple(max(0, min(image.shape[0] - 1, center_z + delta)) for delta in (-1, 0, 1))
    normalized = normalize_volume_notebook66(image)
    crop = normalized[list(z_indices), y0:y1, x0:x1]
    if crop.size == 0:
        raise DiscDegenerativeInvalidRequest("localized_crop_empty")

    tensor = torch_runtime.from_numpy(crop).unsqueeze(0)
    resized = functional.interpolate(
        tensor,
        size=(int(output_size), int(output_size)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).numpy().astype(np.float32)

    return PreparedCrop(
        crop=resized,
        positions=z_indices,
        nativeBboxYx=(y0, y1, x0, x1),
    )


def _prepare_source(series: ProductDiscSourceSeries) -> PreparedCrop:
    if series.role not in SERIES_ROLES:
        raise DiscDegenerativeInvalidRequest("unsupported_source_series_role")
    try:
        record = resolve_registered_input(series.inputId)
    except InputRegistryError as exc:
        error = DiscDegenerativeInvalidRequest("source_input_not_registered")
        error.status_code = exc.status_code
        raise error from exc

    if record.plane != "sagittal":
        raise DiscDegenerativeInvalidRequest("source_input_must_be_sagittal")
    if not record.analyzable:
        raise DiscDegenerativeInvalidRequest("source_input_not_analyzable")

    try:
        loaded = load_input(str(record.path), "sagittal")
    except Exception as exc:
        raise DiscDegenerativeInputUnavailable("source_volume_load_failed") from exc

    return crop_2p5d_from_segmentation_roi(loaded.array, series.localization)


def prepare_product_request(request: ProductDiscMultitaskPredictRequest) -> list[PreparedDiscLevel]:
    prepared: list[PreparedDiscLevel] = []
    for item in request.levels:
        t1: np.ndarray | None = None
        t2: np.ndarray | None = None
        t1_positions: tuple[int, ...] = ()
        t2_positions: tuple[int, ...] = ()

        for series in item.sourceSeries:
            if not series.available:
                continue
            result = _prepare_source(series)
            if series.role == "sagittal_t1":
                t1 = np.asarray(result.crop, dtype=np.float32)
                t1_positions = tuple(result.positions)
            elif series.role == "sagittal_t2":
                t2 = np.asarray(result.crop, dtype=np.float32)
                t2_positions = tuple(result.positions)

        if t1 is None and t2 is None:
            raise DiscDegenerativeInvalidRequest("at_least_one_modality_required")

        prepared.append(
            PreparedDiscLevel(
                level=item.level,
                t1=t1,
                t2=t2,
                t1_positions=t1_positions,
                t2_positions=t2_positions,
                localization_source="segmentation_derived_disc_level",
            )
        )
    return prepared


def predict_product_disc_degenerative(request: ProductDiscMultitaskPredictRequest) -> dict[str, Any]:
    samples = prepare_product_request(request)
    envelope = get_disc_degenerative_classifier().predict_preprocessed(samples)
    envelope["runtime"] = {
        "preprocessingSpec": PREPROCESSING_SPEC,
        "preprocessingParityValidated": PREPROCESSING_PARITY_VALIDATED,
        "localizationSource": "segmentation_derived_disc_level",
        "automaticDiscLocalizationValidated": AUTOMATIC_DISC_LOCALIZATION_VALIDATED,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
    }
    return envelope


def product_runtime_status() -> dict[str, Any]:
    return {
        "preprocessingSpec": PREPROCESSING_SPEC,
        "preprocessingParityValidated": PREPROCESSING_PARITY_VALIDATED,
        "automaticDiscLocalizationValidated": AUTOMATIC_DISC_LOCALIZATION_VALIDATED,
        "acceptedLocalizationSource": "segmentation_derived_disc_level",
        "externalRoiAccepted": False,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
    }
