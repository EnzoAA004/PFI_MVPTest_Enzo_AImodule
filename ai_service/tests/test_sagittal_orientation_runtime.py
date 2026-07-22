from __future__ import annotations

from pathlib import Path

import numpy as np

from pfi_ai_service.real_inference_runtime import (
    LoadedInput,
    canonicalize_loaded_input,
    canonicalize_sagittal_array,
    slice_axis_for,
)


def loaded(array: np.ndarray) -> LoadedInput:
    return LoadedInput(array=array, path=Path("synthetic.mha"), suffix=".mha", spacing_xyz=None, metadata={})


def test_spider_simpleitk_shape_moves_axis_zero_to_last() -> None:
    array = np.zeros((17, 512, 512), dtype=np.float32)
    canonical, transform = canonicalize_sagittal_array(array)

    assert canonical.shape == (512, 512, 17)
    assert transform == "move_axis_0_to_last"


def test_already_canonical_sagittal_shape_is_not_transformed() -> None:
    array = np.zeros((512, 512, 17), dtype=np.float32)
    canonical, transform = canonicalize_sagittal_array(array)

    assert canonical.shape == (512, 512, 17)
    assert transform == "none"


def test_axial_plane_does_not_transform_spider_like_volume() -> None:
    array = np.zeros((17, 512, 512), dtype=np.float32)
    result = canonicalize_loaded_input(loaded(array), "axial", {})

    assert result.array.shape == (17, 512, 512)
    assert result.metadata["inputOrientationTransform"] == "none"


def test_sagittal_selected_axis_slice_count_and_index_are_after_canonicalization() -> None:
    array = np.zeros((17, 512, 512), dtype=np.float32)
    result = canonicalize_loaded_input(loaded(array), "sagittal", {})
    axis = slice_axis_for(result, "sagittal", {"sagittal_axis": 2}, {})

    assert result.metadata["inputShapeNative"] == [17, 512, 512]
    assert result.metadata["inputShapeCanonical"] == [512, 512, 17]
    assert result.metadata["inputOrientationTransform"] == "move_axis_0_to_last"
    assert axis == 2
    assert result.array.shape[axis] == 17
    assert 0 <= 9 < result.array.shape[axis]
