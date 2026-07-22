from __future__ import annotations

from pathlib import Path

import numpy as np

from pfi_ai_service.real_inference_runtime import (
    LoadedInput,
    build_measurements,
    canonicalize_loaded_input,
    canonicalize_sagittal_array,
    in_plane_spacing,
    slice_axis_for,
)


def loaded(array: np.ndarray, spacing_xyz: tuple[float, ...] | None = None) -> LoadedInput:
    return LoadedInput(array=array, path=Path("synthetic.mha"), suffix=".mha", spacing_xyz=spacing_xyz, metadata={})


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


def test_sagittal_spacing_after_move_axis_0_to_last_uses_in_plane_sy_sx() -> None:
    result = canonicalize_loaded_input(loaded(np.zeros((17, 512, 512)), (0.7, 0.8, 4.0)), "sagittal", {})
    axis = slice_axis_for(result, "sagittal", {"sagittal_axis": 2}, {})
    spacing = in_plane_spacing(result, axis)

    assert result.metadata["spacingXyz"] == [0.7, 0.8, 4.0]
    assert result.metadata["arrayAxisSpacingNative"] == [4.0, 0.8, 0.7]
    assert result.metadata["arrayAxisSpacingCanonical"] == [0.8, 0.7, 4.0]
    assert axis == 2
    assert result.array.shape[axis] == 17
    assert spacing == (0.8, 0.7)


def test_sagittal_spacing_already_canonical_uses_remaining_axes() -> None:
    result = canonicalize_loaded_input(loaded(np.zeros((512, 512, 17)), (0.7, 0.8, 4.0)), "sagittal", {})
    spacing = in_plane_spacing(result, 2)

    assert result.metadata["inputOrientationTransform"] == "none"
    assert result.metadata["arrayAxisSpacingCanonical"] == [4.0, 0.8, 0.7]
    assert spacing == (4.0, 0.8)


def test_axial_spacing_without_transform_keeps_native_axis_mapping() -> None:
    result = canonicalize_loaded_input(loaded(np.zeros((17, 512, 512)), (0.7, 0.8, 4.0)), "axial", {})

    assert result.metadata["inputOrientationTransform"] == "none"
    assert result.metadata["arrayAxisSpacingCanonical"] == [4.0, 0.8, 0.7]
    assert in_plane_spacing(result, 0) == (0.8, 0.7)


def test_volume_without_spacing_reports_no_in_plane_spacing() -> None:
    result = canonicalize_loaded_input(loaded(np.zeros((17, 512, 512))), "sagittal", {})

    assert result.metadata["spacingXyz"] is None
    assert result.metadata["arrayAxisSpacingNative"] is None
    assert result.metadata["arrayAxisSpacingCanonical"] is None
    assert in_plane_spacing(result, 2) is None


def test_measurements_use_row_sy_and_column_sx_not_slice_spacing() -> None:
    prediction = np.zeros((4, 5), dtype=np.uint8)
    prediction[1:3, 1:4] = 1
    confidence = np.ones_like(prediction, dtype=np.float32)
    values = build_measurements("sagittal_spider", "sagittal", prediction, confidence, (0.8, 0.7))
    by_id = {item["id"]: item for item in values}

    assert by_id["sagittal-vertebra_group-area"]["value"] == 3.36
    assert by_id["sagittal-vertebra_group-width"]["value"] == 2.1
    assert by_id["sagittal-vertebra_group-height"]["value"] == 1.6
    assert by_id["sagittal-vertebra_group-area"]["unit"] == "mm2"
