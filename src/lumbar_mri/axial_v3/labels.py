"""Explicit axial raw-label mapping for axial v3."""

from __future__ import annotations

from typing import Literal

import numpy as np


RAW_ALLOWED_LABELS = {0, 50, 100, 150, 200, 250}
RAW_TO_CLASS_INDEX = {
    250: 0,
    0: 1,
    50: 2,
    100: 3,
    150: 4,
    200: 5,
}
CLASS_INDEX_TO_NAME = {
    0: "background_250",
    1: "raw_0",
    2: "raw_50",
    3: "raw_100",
    4: "raw_150",
    5: "raw_200",
}
INDEX_ALLOWED_LABELS = set(CLASS_INDEX_TO_NAME)

MaskLabelMode = Literal["raw", "indexed"]


def _unexpected(mask: np.ndarray, allowed: set[int]) -> set[int]:
    return {int(value) for value in np.unique(mask)} - allowed


def validate_raw_mask_labels(mask: np.ndarray) -> None:
    unexpected = _unexpected(np.asarray(mask), RAW_ALLOWED_LABELS)
    if unexpected:
        raise ValueError(f"unexpected raw axial labels: {sorted(unexpected)}")


def validate_indexed_mask_labels(mask: np.ndarray) -> None:
    unexpected = _unexpected(np.asarray(mask), INDEX_ALLOWED_LABELS)
    if unexpected:
        raise ValueError(f"unexpected indexed axial labels: {sorted(unexpected)}")


def map_raw_mask_to_class_indices(mask: np.ndarray) -> np.ndarray:
    raw = np.asarray(mask)
    validate_raw_mask_labels(raw)
    mapped = np.zeros(raw.shape, dtype=np.int64)
    for raw_value, index_value in RAW_TO_CLASS_INDEX.items():
        mapped[raw == raw_value] = index_value
    return mapped


def mask_to_class_indices(mask: np.ndarray, *, mode: MaskLabelMode) -> np.ndarray:
    """Convert a mask to class indices with an explicit declared input mode."""

    if mode == "raw":
        return map_raw_mask_to_class_indices(mask)
    if mode == "indexed":
        indexed = np.asarray(mask, dtype=np.int64)
        validate_indexed_mask_labels(indexed)
        return indexed
    raise ValueError(f"unknown mask label mode: {mode!r}")
