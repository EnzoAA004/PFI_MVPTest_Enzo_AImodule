"""Versioned 2D medical image I/O for axial MRI experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pydicom
except Exception:  # pragma: no cover - exercised through the explicit runtime error
    pydicom = None


STANDARD_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}
DICOM_SUFFIXES = {".dcm", ".ima"}


def open_2d_array(path: Path) -> np.ndarray:
    """Open a supported file as a two-dimensional NumPy array.

    Supported inputs are ``.npy``, conventional raster images, and DICOM
    files with ``.dcm`` or ``.ima`` suffixes. DICOM decoding is delegated to
    pydicom so the same Alkafri source files used by axial-final-v2 can be
    consumed without PIL fallbacks.
    """

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        array = np.asarray(np.load(path))
    elif suffix in STANDARD_IMAGE_SUFFIXES:
        array = np.asarray(Image.open(path))
    elif suffix in DICOM_SUFFIXES:
        if pydicom is None:
            raise RuntimeError(
                "pydicom is required to read .dcm/.ima files; install the project dependencies"
            )
        array = np.asarray(pydicom.dcmread(str(path)).pixel_array)
    else:
        raise ValueError(f"unsupported medical image format: {suffix} ({path})")

    array = np.asarray(array)
    if array.ndim == 2:
        return array

    if array.ndim == 3 and 1 in array.shape:
        squeezed = np.squeeze(array)
        if squeezed.ndim == 2:
            return squeezed

    if suffix in STANDARD_IMAGE_SUFFIXES and array.ndim == 3 and array.shape[-1] in {3, 4}:
        return array[..., 0]

    raise ValueError(f"expected a 2D image/mask at {path}, got {array.shape}")
