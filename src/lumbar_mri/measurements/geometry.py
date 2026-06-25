"""Mediciones geométricas simples desde máscaras segmentadas.

Estas mediciones son de apoyo y no deben interpretarse como diagnóstico clínico.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox2D:
    """Bounding box 2D en coordenadas de píxel."""

    min_row: int
    min_col: int
    max_row: int
    max_col: int

    @property
    def height_px(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def width_px(self) -> int:
        return self.max_col - self.min_col + 1


def binary_mask_for_class(mask: np.ndarray, class_id: int) -> np.ndarray:
    """Devuelve una máscara binaria para una clase."""

    return np.asarray(mask) == class_id


def projected_area(mask: np.ndarray, class_id: int, spacing: tuple[float, float] | None = None) -> float:
    """Calcula área proyectada de una clase.

    Si `spacing` está definido, se interpreta como `(row_spacing_mm, col_spacing_mm)`
    y el resultado queda en mm². Si no, el resultado queda en píxeles².
    """

    binary = binary_mask_for_class(mask, class_id)
    area_px = float(binary.sum())
    if spacing is None:
        return area_px
    return area_px * float(spacing[0]) * float(spacing[1])


def bounding_box(mask: np.ndarray, class_id: int) -> BoundingBox2D | None:
    """Calcula el bounding box de una clase. Devuelve None si la clase no existe."""

    binary = binary_mask_for_class(mask, class_id)
    coords = np.argwhere(binary)
    if coords.size == 0:
        return None
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    return BoundingBox2D(
        min_row=int(min_row),
        min_col=int(min_col),
        max_row=int(max_row),
        max_col=int(max_col),
    )


def centroid(mask: np.ndarray, class_id: int) -> tuple[float, float] | None:
    """Calcula centroide `(row, col)` para una clase."""

    binary = binary_mask_for_class(mask, class_id)
    coords = np.argwhere(binary)
    if coords.size == 0:
        return None
    center = coords.mean(axis=0)
    return float(center[0]), float(center[1])


def summarize_class_geometry(
    mask: np.ndarray,
    class_id: int,
    spacing: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Resume mediciones geométricas básicas de una clase."""

    box = bounding_box(mask, class_id)
    center = centroid(mask, class_id)
    return {
        "class_id": class_id,
        "area": projected_area(mask, class_id, spacing=spacing),
        "area_unit": "mm2" if spacing is not None else "px2",
        "bounding_box": None if box is None else box.__dict__,
        "centroid": center,
        "source": "mask",
        "clinical_interpretation": "not_provided",
        "review_status": "pending",
        "professional_comment": "",
    }
