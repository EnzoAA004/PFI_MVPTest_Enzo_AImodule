from __future__ import annotations

from typing import Any, List, Optional
import ast
import math


def parse_list_like(value: Any) -> List[Any]:
    """Parsea valores tipo list guardados en CSV: [], ['a'], [1, 2, 3]."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if isinstance(value, float) and math.isnan(value):
            return []
    except Exception:
        pass

    text = str(value).strip()
    if text in ("", "nan", "None", "[]"):
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
        return [parsed]
    except Exception:
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]
        return [text]


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        x = float(value)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default


def compute_quality_flags_from_values(
    *,
    foreground_ratio: Optional[float],
    n_components: Optional[int],
    present_classes: Optional[List[int]],
    mean_confidence: Optional[float],
    mean_fg_confidence: Optional[float],
    plane: str,
) -> List[str]:
    flags: List[str] = []
    fg = foreground_ratio if foreground_ratio is not None else 0.0
    comps = int(n_components or 0)
    classes = present_classes or []

    if fg < 0.002:
        flags.append("foreground_muy_bajo")
    if plane == "axial" and fg > 0.25:
        flags.append("foreground_muy_alto")
    if plane == "sagittal" and fg > 0.45:
        flags.append("foreground_muy_alto")
    if not classes:
        flags.append("sin_clases_no_fondo")

    if plane == "axial" and comps > 10:
        flags.append("muchos_componentes")
    if plane == "sagittal" and comps > 10:
        flags.append("muchos_componentes")

    if mean_confidence is not None and mean_confidence < 0.70:
        flags.append("baja_confianza_media")
    if mean_fg_confidence is not None and mean_fg_confidence < 0.75:
        flags.append("baja_confianza_foreground")

    return flags


def useful_classes_for_plane(plane: str) -> List[int]:
    """Clases útiles para métricas del MVP.

    Axial excluye raw_0 porque E11 lo documentó como minoritario/problemático.
    Sagital usa clases 1,2,3.
    """
    if plane == "axial":
        return [2, 3, 4, 5]
    if plane == "sagittal":
        return [1, 2, 3]
    return []
