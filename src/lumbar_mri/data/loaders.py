"""Carga de archivos de imagen médica.

Este módulo contiene utilidades mínimas. La integración específica con SPIDER se agregará
cuando se defina la estructura exacta del dataset en Drive/Colab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_numpy_array(path: str | Path) -> np.ndarray:
    """Lee un array NumPy desde `.npy` o `.npz`."""

    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data: Any = np.load(path)
        first_key = list(data.keys())[0]
        return data[first_key]
    raise ValueError(f"Formato no soportado para NumPy: {path}")


def validate_image_mask_pair(image: np.ndarray, mask: np.ndarray) -> None:
    """Valida compatibilidad dimensional entre imagen y máscara."""

    if image.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            "La imagen y la máscara deben compartir las dimensiones espaciales finales. "
            f"image.shape={image.shape}, mask.shape={mask.shape}"
        )
