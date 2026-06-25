import numpy as np

from lumbar_mri.measurements.geometry import bounding_box, centroid, projected_area


def test_projected_area_without_spacing():
    mask = np.array([[0, 1, 1], [0, 1, 0]])

    assert projected_area(mask, class_id=1) == 3.0


def test_projected_area_with_spacing():
    mask = np.array([[0, 1], [1, 1]])

    assert projected_area(mask, class_id=1, spacing=(2.0, 3.0)) == 18.0


def test_bounding_box():
    mask = np.array([[0, 0, 0], [0, 2, 2], [0, 0, 2]])
    box = bounding_box(mask, class_id=2)

    assert box is not None
    assert box.min_row == 1
    assert box.min_col == 1
    assert box.max_row == 2
    assert box.max_col == 2


def test_centroid():
    mask = np.array([[0, 0], [3, 3]])

    assert centroid(mask, class_id=3) == (1.0, 0.5)
