"""Safe 2.5D scaffolding for axial v3 Iteration C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class SliceRecord25D:
    image_path: str
    mask_path: str
    split: str
    patient_id: str
    study_id: str
    slice_id: str
    order_index: int | None = None


def require_reliable_slice_order(records: Sequence[SliceRecord25D]) -> None:
    missing = [record.slice_id for record in records if record.order_index is None]
    if missing:
        raise ValueError(f"Iteration C blocked: missing reliable slice order for {len(missing)} slices")
    grouped: dict[tuple[str, str], list[int]] = {}
    for record in records:
        grouped.setdefault((record.patient_id, record.study_id), []).append(int(record.order_index))
    for key, values in grouped.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Iteration C blocked: duplicated order index for patient/study {key}")


def neighbor_indices(records: Sequence[SliceRecord25D], center: int) -> tuple[int, int, int]:
    require_reliable_slice_order(records)
    center_record = records[center]
    same_study = [
        (index, record)
        for index, record in enumerate(records)
        if record.patient_id == center_record.patient_id and record.study_id == center_record.study_id
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
    ) -> None:
        require_reliable_slice_order(records)
        self.records = list(records)
        self.image_loader = image_loader
        self.mask_loader = mask_loader

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
        prev_index, center_index, next_index = neighbor_indices(self.records, index)
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
