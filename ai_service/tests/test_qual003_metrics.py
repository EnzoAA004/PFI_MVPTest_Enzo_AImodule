from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path("scripts/evaluate_model.py")
spec = importlib.util.spec_from_file_location("evaluate_model", SCRIPT_PATH)
evaluate_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["evaluate_model"] = evaluate_model
spec.loader.exec_module(evaluate_model)


def test_dice_iou_perfect_overlap_is_one() -> None:
    ground_truth = [[0, 1], [1, 0]]
    prediction = [[0, 1], [1, 0]]

    dice, iou, note = evaluate_model.dice_iou_for_class(prediction, ground_truth, class_id=1)

    assert dice == pytest.approx(1.0)
    assert iou == pytest.approx(1.0)
    assert note is None


def test_dice_iou_disjoint_is_zero() -> None:
    ground_truth = [[1, 1], [0, 0]]
    prediction = [[0, 0], [1, 1]]

    dice, iou, note = evaluate_model.dice_iou_for_class(prediction, ground_truth, class_id=1)

    assert dice == pytest.approx(0.0)
    assert iou == pytest.approx(0.0)
    assert note is None


def test_dice_iou_known_partial_overlap() -> None:
    ground_truth = [[1, 1], [0, 0]]
    prediction = [[1, 0], [0, 0]]

    dice, iou, note = evaluate_model.dice_iou_for_class(prediction, ground_truth, class_id=1)

    assert dice == pytest.approx(2.0 / 3.0)
    assert iou == pytest.approx(0.5)
    assert note is None


def test_absent_class_is_undefined_and_excluded_from_macro() -> None:
    ground_truth = [[0, 0], [0, 0]]
    prediction = [[0, 0], [0, 0]]

    dice, iou, note = evaluate_model.dice_iou_for_class(prediction, ground_truth, class_id=2)

    assert dice is None
    assert iou is None
    assert note == "absent_in_gt_and_prediction"
    assert evaluate_model.macro_foreground({0: 1.0, 1: None, 2: 0.5}) == pytest.approx(0.5)
    assert evaluate_model.macro_foreground({0: 1.0, 1: None}) is None