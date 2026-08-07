"""P10.9 full-series segmentation and disc-localization bridge.

The legacy real-baseline pipeline intentionally analyzes one representative slice.
For clinical review and P10.7 localization we also need a series-level pass: every
supported slice gets an original image, an overlay, a segmentation map and descriptive
measurements. Disc ROIs are aggregated across sagittal slices by maximum predicted disc
area, mirroring the slice-selection rule used by Notebook 66.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .input_registry import InputRegistryError, resolve_registered_input
from .model_artifacts import model_status
from .real_inference_runtime import (
    MODEL_REGISTRY,
    PALETTE,
    build_masks,
    build_measurements,
    build_segmentation,
    canonicalize_loaded_input,
    class_name,
    connected_instances,
    in_plane_spacing,
    load_input,
    load_model,
    lumbar_disc_levels,
    native_slice,
    prediction_grid_spacing,
    resize_image,
    slice_axis_for,
    upsample_labels,
)
from .settings import get_settings

SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]+$")
MAX_FULL_SERIES_SLICES = 512
DEFAULT_BATCH_SIZE = 8

SEMANTIC_COLORS = {
    "disc": "#3b82f6",
    "disc_group": "#3b82f6",
    "vertebra": "#f59e0b",
    "vertebra_group": "#f59e0b",
    "posterior_element": "#f97316",
    "canal": "#10b981",
    "raw_50": "#3b82f6",
    "raw_100": "#f97316",
    "raw_150": "#10b981",
    "raw_200": "#8b5cf6",
}


class FullSeriesSegmentationError(Exception):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class FullSeriesSegmentationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caseId: str
    inputId: str
    plane: Literal["sagittal", "axial"]
    modelKey: str

    @field_validator("caseId", "inputId", "modelKey")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must_not_be_empty")
        return value


def _run_id(request: FullSeriesSegmentationRequest) -> str:
    raw = f"{request.caseId}|{request.inputId}|{request.plane}|{request.modelKey}|full-series-v1"
    return "series-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _output_root(run_id: str, plane: str) -> Path:
    return get_settings().output_dir / "full_series_segmentation" / run_id / plane


def _semantic_color(mask: dict[str, Any]) -> str:
    label = str(mask.get("label") or "")
    class_name_value = str(mask.get("className") or "")
    return SEMANTIC_COLORS.get(label) or SEMANTIC_COLORS.get(class_name_value) or "#64748b"


def _anatomy_key(mask: dict[str, Any]) -> str:
    label = str(mask.get("label") or mask.get("className") or "structure")
    level = str(mask.get("level") or "").strip()
    if level:
        return f"{label}:{level}"
    return label


def _enrich_masks(masks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for mask in masks:
        item = dict(mask)
        item["anatomyKey"] = _anatomy_key(mask)
        item["semanticColor"] = _semantic_color(mask)
        item["colorPolicy"] = "stable_by_anatomical_role"
        result.append(item)
    return result


def _save_slice_assets(
    run_id: str,
    plane: str,
    index: int,
    render: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, str]:
    directory = _output_root(run_id, plane) / f"slice-{index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    original_path = directory / "original.png"
    overlay_path = directory / "overlay.png"

    render = np.asarray(render, dtype=np.float32)
    render_mask = upsample_labels(prediction, (int(render.shape[0]), int(render.shape[1])))
    Image.fromarray(np.clip(render * 255.0, 0, 255).astype(np.uint8)).save(original_path)

    overlay = np.stack([render, render, render], axis=-1)
    alpha = 0.42
    for class_id in sorted(int(v) for v in np.unique(prediction) if int(v) != 0):
        color = np.asarray(PALETTE.get(class_id, (255, 255, 0)), dtype=np.float32) / 255.0
        selected = render_mask == class_id
        overlay[selected] = (1.0 - alpha) * overlay[selected] + alpha * color
    Image.fromarray(np.clip(overlay * 255.0, 0, 255).astype(np.uint8)).save(overlay_path)

    return {
        "original": f"/v2/series-segmentation/{run_id}/{plane}/slices/{index}/original.png",
        "overlay": f"/v2/series-segmentation/{run_id}/{plane}/slices/{index}/overlay.png",
    }


def _disc_candidates(
    model_key: str,
    prediction: np.ndarray,
    slice_index: int,
) -> list[dict[str, Any]]:
    disc_class = next(
        (
            class_id
            for class_id in sorted(int(v) for v in np.unique(prediction) if int(v) != 0)
            if class_name(model_key, class_id) == "disc_group"
        ),
        None,
    )
    if disc_class is None:
        return []
    components = connected_instances(prediction == disc_class)
    levels = lumbar_disc_levels(len(components))
    candidates = []
    for component, level in zip(components, levels):
        if not level:
            continue
        rows, columns = np.where(component)
        if rows.size == 0:
            continue
        candidates.append(
            {
                "level": level,
                "sliceIndex": int(slice_index),
                "areaPixels": int(component.sum()),
                "bboxYx": [
                    int(rows.min()),
                    int(rows.max()) + 1,
                    int(columns.min()),
                    int(columns.max()) + 1,
                ],
                "coordinateSpaceWidth": int(prediction.shape[1]),
                "coordinateSpaceHeight": int(prediction.shape[0]),
            }
        )
    return candidates


def _aggregate_disc_localizations(candidates: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for item in candidates:
        level = str(item["level"])
        current = best.get(level)
        if current is None or int(item["areaPixels"]) > int(current["areaPixels"]):
            best[level] = item

    ordered = []
    for level in ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"):
        item = best.get(level)
        if item is None:
            continue
        ordered.append(
            {
                "source": "segmentation_derived_disc_level",
                "level": level,
                "sliceIndex": item["sliceIndex"],
                "bboxYx": item["bboxYx"],
                "coordinateSpaceWidth": item["coordinateSpaceWidth"],
                "coordinateSpaceHeight": item["coordinateSpaceHeight"],
                "maskId": f"full-series-disc-{level.lower()}",
                "segmentationRunId": run_id,
                "selectionRule": "maximum_predicted_disc_area_across_series",
                "automaticAnatomicalLocalizationValidated": False,
            }
        )
    return ordered


def run_full_series_segmentation(request: FullSeriesSegmentationRequest) -> dict[str, Any]:
    try:
        record = resolve_registered_input(request.inputId)
    except InputRegistryError as exc:
        raise FullSeriesSegmentationError("input_not_registered", exc.status_code) from exc

    if record.case_id != request.caseId:
        raise FullSeriesSegmentationError("input_case_mismatch", 409)
    if record.plane != request.plane:
        raise FullSeriesSegmentationError("input_plane_mismatch", 409)
    if not record.analyzable:
        raise FullSeriesSegmentationError("input_not_analyzable", 422)

    model_info = MODEL_REGISTRY.get(request.modelKey)
    if not isinstance(model_info, dict):
        raise FullSeriesSegmentationError("model_not_registered", 404)
    if model_info.get("plane") != request.plane:
        raise FullSeriesSegmentationError("model_plane_mismatch", 409)

    artifact = model_status(request.modelKey, dict(model_info))
    if not artifact.get("availableForRealInference", False):
        raise FullSeriesSegmentationError("model_not_ready", 409)

    cached = load_model(request.modelKey)
    loaded = canonicalize_loaded_input(load_input(str(record.path), request.plane), request.plane, {})
    if loaded.array.ndim != 3:
        raise FullSeriesSegmentationError("full_series_requires_3d_input", 422)

    axis = slice_axis_for(loaded, request.plane, cached.checkpoint, {})
    count = int(loaded.array.shape[axis])
    if count <= 0 or count > MAX_FULL_SERIES_SLICES:
        raise FullSeriesSegmentationError("slice_count_out_of_supported_range", 422)

    target = tuple(cached.runtime_metadata.get("targetSize", (256, 256)))
    target_size = (int(target[0]), int(target[1]))
    batch_size = max(1, int(os.getenv("PFI_FULL_SERIES_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))))
    spacing = in_plane_spacing(loaded, axis)

    import torch

    run_id = _run_id(request)
    slice_results: list[dict[str, Any]] = []
    disc_candidates: list[dict[str, Any]] = []

    for start in range(0, count, batch_size):
        indices = list(range(start, min(count, start + batch_size)))
        prepared = [
            resize_image(np.take(loaded.array, index, axis=axis), target_size)
            for index in indices
        ]
        tensor = torch.from_numpy(np.stack(prepared)[:, None]).float().to(cached.device)
        with torch.inference_mode():
            logits = cached.model(tensor)
            probabilities = torch.softmax(logits, dim=1)

        for offset, index in enumerate(indices):
            probs = probabilities[offset]
            prediction = torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.uint8)
            confidence = torch.max(probs, dim=0).values.detach().cpu().numpy().astype(np.float32)
            masks = _enrich_masks(
                build_masks(request.modelKey, request.plane, prediction, confidence, f"series-{request.plane}", index)
            )
            segmentation = build_segmentation(request.modelKey, request.plane, prediction)
            measurements = build_measurements(
                request.modelKey,
                request.plane,
                prediction,
                confidence,
                prediction_grid_spacing(spacing, loaded.array.shape, axis, prediction.shape),
                index,
            )
            assets = _save_slice_assets(
                run_id,
                request.plane,
                index,
                native_slice(loaded, axis, index),
                prediction,
            )
            if request.plane == "sagittal":
                disc_candidates.extend(_disc_candidates(request.modelKey, prediction, index))

            foreground = prediction > 0
            slice_results.append(
                {
                    "sliceIndex": index,
                    "status": "segmented",
                    "assets": assets,
                    "segmentation": segmentation,
                    "masks": masks,
                    "measurements": measurements,
                    "quality": {
                        "meanConfidence": round(float(confidence.mean()), 4),
                        "meanForegroundConfidence": round(
                            float(confidence[foreground].mean()) if foreground.any() else 0.0,
                            4,
                        ),
                        "foregroundRatio": round(float(foreground.mean()), 6),
                    },
                }
            )

    disc_localizations = _aggregate_disc_localizations(disc_candidates, run_id) if request.plane == "sagittal" else []
    report = {
        "schemaVersion": "pfi.full-series-segmentation.v1",
        "status": "completed",
        "runId": run_id,
        "caseId": request.caseId,
        "inputId": request.inputId,
        "plane": request.plane,
        "modelKey": request.modelKey,
        "sliceCount": count,
        "segmentedSliceCount": len(slice_results),
        "coverageComplete": len(slice_results) == count,
        "defaultPresentation": "original",
        "overlayPresentation": "on_demand",
        "anatomyColorPolicy": "stable_by_anatomical_role",
        "semanticColors": SEMANTIC_COLORS,
        "discLocalizations": disc_localizations,
        "automaticDiscLocalizationValidated": False,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "slices": slice_results,
    }

    destination = _output_root(run_id, request.plane) / "report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report


def resolve_full_series_asset(run_id: str, plane: str, index: int, asset_name: str) -> Path:
    if not SAFE_ID.fullmatch(run_id):
        raise FullSeriesSegmentationError("invalid_run_id", 400)
    if plane not in {"sagittal", "axial"}:
        raise FullSeriesSegmentationError("invalid_plane", 400)
    if index < 0 or index >= MAX_FULL_SERIES_SLICES:
        raise FullSeriesSegmentationError("invalid_slice_index", 400)
    if asset_name not in {"original.png", "overlay.png"}:
        raise FullSeriesSegmentationError("invalid_asset", 404)
    path = _output_root(run_id, plane) / f"slice-{index:03d}" / asset_name
    if not path.is_file():
        raise FullSeriesSegmentationError("asset_not_found", 404)
    return path
