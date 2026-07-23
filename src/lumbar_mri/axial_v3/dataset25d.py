"""Safe 2.5D scaffolding for axial v3 Iteration C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


ALLOWED_ORDER_SOURCES = {
    "InstanceNumber",
    "ImagePositionPatient",
    "SliceLocation",
    "curated_index",
    "validated_filename_index",
}
BLOCKED_ORDER_SOURCES = {"lexicographic_filename", "inferred", "unknown", None}


@dataclass(frozen=True)
class SliceRecord25D:
    image_path: str
    mask_path: str
    split: str
    patient_id: str
    study_id: str
    slice_id: str
    order_index: int | None = None
    order_source: str | None = None
    order_value_original: str | int | float | None = None


def validate_slice_order_source(record: SliceRecord25D, *, validated_filename_report: bool = False) -> None:
    if record.order_source in BLOCKED_ORDER_SOURCES:
        raise ValueError(f"Iteration C blocked: unreliable order source {record.order_source!r}")
    if record.order_source not in ALLOWED_ORDER_SOURCES:
        raise ValueError(f"Iteration C blocked: unsupported order source {record.order_source!r}")
    if record.order_source == "validated_filename_index" and not validated_filename_report:
        raise ValueError("Iteration C blocked: validated_filename_index requires prior validation report")


def require_reliable_slice_order(records: Sequence[SliceRecord25D], *, validated_filename_report: bool = False) -> None:
    for record in records:
        validate_slice_order_source(record, validated_filename_report=validated_filename_report)
    missing = [record.slice_id for record in records if record.order_index is None]
    if missing:
        raise ValueError(f"Iteration C blocked: missing reliable slice order for {len(missing)} slices")
    split_by_study: dict[tuple[str, str], set[str]] = {}
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for record in records:
        split_by_study.setdefault((record.patient_id, record.study_id), set()).add(record.split)
        grouped.setdefault((record.patient_id, record.study_id, record.split), []).append(int(record.order_index))
    for key, splits in split_by_study.items():
        if len(splits) > 1:
            raise ValueError(f"Iteration C blocked: patient/study crosses splits {key}")
    for key, values in grouped.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Iteration C blocked: duplicated order index for patient/study {key}")
        ordered = sorted(values)
        if any(b <= a for a, b in zip(ordered, ordered[1:])):
            raise ValueError(f"Iteration C blocked: order is not strictly increasing for {key}")


def neighbor_indices(records: Sequence[SliceRecord25D], center: int, *, validated_filename_report: bool = False) -> tuple[int, int, int]:
    require_reliable_slice_order(records, validated_filename_report=validated_filename_report)
    center_record = records[center]
    same_study = [
        (index, record)
        for index, record in enumerate(records)
        if record.patient_id == center_record.patient_id and record.study_id == center_record.study_id
        and record.split == center_record.split
    ]
    same_study.sort(key=lambda item: int(item[1].order_index))
    positions = [index for index, _ in same_study]
    position = positions.index(center)
    prev_index = positions[max(0, position - 1)]
    next_index = positions[min(len(positions) - 1, position + 1)]
    return prev_index, center, next_index


class AxialSegmentationDataset25D:
    """Minimal framework-neutral 2.5D dataset.

    The loader callback keeps real medical IO in notebooks or existing helpers.
    It must return a 2D image array for a record.
    """

    def __init__(
        self,
        records: Sequence[SliceRecord25D],
        image_loader: Callable[[SliceRecord25D], np.ndarray],
        mask_loader: Callable[[SliceRecord25D], np.ndarray],
        *,
        validated_filename_report: bool = False,
    ) -> None:
        require_reliable_slice_order(records, validated_filename_report=validated_filename_report)
        self.records = list(records)
        self.image_loader = image_loader
        self.mask_loader = mask_loader
        self.validated_filename_report = validated_filename_report

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
        prev_index, center_index, next_index = neighbor_indices(self.records, index, validated_filename_report=self.validated_filename_report)
        neighbors = [self.records[prev_index], self.records[center_index], self.records[next_index]]
        images = [np.asarray(self.image_loader(record), dtype=np.float32) for record in neighbors]
        if len({image.shape for image in images}) != 1:
            raise ValueError("2.5D neighbors must have matching shapes")
        center = self.records[center_index]
        return {
            "image": np.stack(images, axis=0),
            "mask": np.asarray(self.mask_loader(center)),
            "patientId": center.patient_id,
            "studyId": center.study_id,
            "sliceId": center.slice_id,
        }
