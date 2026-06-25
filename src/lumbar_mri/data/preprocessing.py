"""Funciones de preprocesamiento para imágenes médicas."""

from __future__ import annotations

import numpy as np


def normalize_zscore(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normaliza una imagen con z-score.

    Parameters
    ----------
    image:
        Imagen de entrada.
    eps:
        Valor pequeño para evitar división por cero.

    Returns
    -------
    np.ndarray
        Imagen normalizada en `float32`.
    """

    image = np.asarray(image, dtype=np.float32)
    mean = float(np.mean(image))
    std = float(np.std(image))
    return ((image - mean) / (std + eps)).astype(np.float32)


def normalize_minmax(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Escala una imagen al rango [0, 1]."""

    image = np.asarray(image, dtype=np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    return ((image - min_value) / (max_value - min_value + eps)).astype(np.float32)


def ensure_channel_first(image: np.ndarray) -> np.ndarray:
    """Asegura formato `[C, H, W]` para una imagen 2D."""

    image = np.asarray(image)
    if image.ndim == 2:
        return image[None, ...]
    if image.ndim == 3:
        return image
    raise ValueError(f"Se esperaba una imagen 2D o 3D, se recibió shape={image.shape}")
