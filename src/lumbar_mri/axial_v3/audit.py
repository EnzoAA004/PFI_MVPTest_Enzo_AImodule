"""Raw label audit helpers for axial v3 Iteration A."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Raw0SliceAudit:
    split: str
    patientId: str
    studyId: str
    sliceId: str
    raw0Present: bool
    raw0PixelCount: int
    raw0AreaRatio: float
    boundingBox: dict[str, int | None]
    centroid: dict[str, float | None]
    minDistanceToBorder: int | None
    touchesBorder: bool
    connectedComponents: int
    largestComponentArea: int
    largestComponentRatio: float | None
    imageHeight: int
    imageWidth: int
    maskHeight: int
    maskWidth: int
    otherClassesPresent: list[int]


def _connected_component_areas(binary_mask: np.ndarray) -> list[int]:
    mask = np.asarray(binary_mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    areas: list[int] = []
    height, width = mask.shape
    for row, col in zip(*np.where(mask & ~visited)):
        if visited[row, col]:
            continue
        area = 0
        queue: deque[tuple[int, int]] = deque([(int(row), int(col))])
        visited[row, col] = True
        while queue:
            r, c = queue.popleft()
            area += 1
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        areas.append(area)
    return areas


def audit_raw0_slice(
    *,
    mask: np.ndarray,
    split: str,
    patient_id: str,
    study_id: str,
    slice_id: str,
    image_shape: tuple[int, int] | None = None,
    raw0_value: int = 1,
    background_value: int = 0,
) -> Raw0SliceAudit:
    """Compute one train/validation raw_0 audit row without assigning anatomy."""

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask_array.shape}")
    image_height, image_width = image_shape or mask_array.shape
    raw0 = mask_array == raw0_value
    rows, cols = np.where(raw0)
    count = int(raw0.sum())
    area = int(mask_array.size)
    other = sorted(
        int(value)
        for value in np.unique(mask_array)
        if int(value) not in {int(raw0_value), int(background_value)}
    )
    if count == 0:
        bbox = {"minRow": None, "minCol": None, "maxRow": None, "maxCol": None}
        centroid = {"row": None, "col": None}
        min_distance = None
        touches = False
        component_areas: list[int] = []
    else:
        bbox = {
            "minRow": int(rows.min()),
            "minCol": int(cols.min()),
            "maxRow": int(rows.max()),
            "maxCol": int(cols.max()),
        }
        centroid = {"row": float(rows.mean()), "col": float(cols.mean())}
        border_distances = np.minimum.reduce(
            [rows, cols, mask_array.shape[0] - 1 - rows, mask_array.shape[1] - 1 - cols]
        )
        min_distance = int(border_distances.min())
        touches = min_distance == 0
        component_areas = _connected_component_areas(raw0)
    largest = max(component_areas, default=0)
    return Raw0SliceAudit(
        split=split,
        patientId=str(patient_id),
        studyId=str(study_id),
        sliceId=str(slice_id),
        raw0Present=bool(count),
        raw0PixelCount=count,
        raw0AreaRatio=float(count / area) if area else 0.0,
        boundingBox=bbox,
        centroid=centroid,
        minDistanceToBorder=min_distance,
        touchesBorder=touches,
        connectedComponents=len(component_areas),
        largestComponentArea=int(largest),
        largestComponentRatio=float(largest / count) if count else None,
        imageHeight=int(image_height),
        imageWidth=int(image_width),
        maskHeight=int(mask_array.shape[0]),
        maskWidth=int(mask_array.shape[1]),
        otherClassesPresent=other,
    )


def summarize_raw0_by_patient(rows: Iterable[Raw0SliceAudit]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Raw0SliceAudit]] = {}
    for row in rows:
        grouped.setdefault(row.patientId, []).append(row)
    summary: list[dict[str, Any]] = []
    for patient_id, patient_rows in sorted(grouped.items()):
        positive_areas = [row.raw0PixelCount for row in patient_rows if row.raw0Present]
        components = [row.connectedComponents for row in patient_rows]
        summary.append(
            {
                "patientId": patient_id,
                "sliceCount": len(patient_rows),
                "raw0SliceCount": len(positive_areas),
                "raw0SliceRatio": len(positive_areas) / len(patient_rows) if patient_rows else 0.0,
                "raw0AreaMean": float(np.mean(positive_areas)) if positive_areas else 0.0,
                "raw0AreaMedian": float(np.median(positive_areas)) if positive_areas else 0.0,
                "raw0AreaMin": int(min(positive_areas)) if positive_areas else 0,
                "raw0AreaMax": int(max(positive_areas)) if positive_areas else 0,
                "componentCountMean": float(np.mean(components)) if components else 0.0,
                "raw0AreaStd": float(np.std(positive_areas)) if positive_areas else 0.0,
            }
        )
    return summary


def slice_audit_frame(rows: Iterable[Raw0SliceAudit]) -> pd.DataFrame:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        payload = asdict(row)
        bbox = payload.pop("boundingBox")
        centroid = payload.pop("centroid")
        for key, value in bbox.items():
            payload[f"bbox_{key}"] = value
        for key, value in centroid.items():
            payload[f"centroid_{key}"] = value
        payload["otherClassesPresent"] = ";".join(str(value) for value in payload["otherClassesPresent"])
        flattened.append(payload)
    return pd.DataFrame(flattened)


def write_audit_outputs(rows: Iterable[Raw0SliceAudit], output_dir: Path) -> dict[str, Any]:
    row_list = list(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_df = slice_audit_frame(row_list)
    patient_summary = summarize_raw0_by_patient(row_list)
    patient_df = pd.DataFrame(patient_summary)
    slice_csv = output_dir / "raw0_slice_audit.csv"
    patient_csv = output_dir / "raw0_patient_audit.csv"
    summary_json = output_dir / "raw0_audit_summary.json"
    slice_df.to_csv(slice_csv, index=False)
    patient_df.to_csv(patient_csv, index=False)
    summary = {
        "schemaVersion": "axial-v3-raw0-audit-v1",
        "sliceCount": len(row_list),
        "patientCount": len(patient_summary),
        "raw0PositiveSlices": int(sum(row.raw0Present for row in row_list)),
        "raw0BorderContactSlices": int(sum(row.touchesBorder for row in row_list)),
        "outputs": {
            "sliceCsv": str(slice_csv),
            "patientCsv": str(patient_csv),
        },
        "humanReviewPending": True,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
