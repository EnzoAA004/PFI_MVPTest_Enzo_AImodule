from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
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


def test_raw_gt_label_group_mapping_aligns_classes_for_expected_dice() -> None:
    raw_ground_truth = np.array([[0, 10], [20, 20]], dtype=np.int64)
    prediction = np.array([[0, 1], [2, 2]], dtype=np.int64)
    mapping = evaluate_model.normalize_label_mapping({"10": 1, "20": 2}, num_classes=3, source="test")

    mapped_ground_truth = evaluate_model.apply_label_mapping(raw_ground_truth, mapping, num_classes=3, case_id="case-map")

    assert mapped_ground_truth.tolist() == [[0, 1], [2, 2]]
    assert evaluate_model.dice_iou_for_class(prediction, mapped_ground_truth, 1)[:2] == pytest.approx((1.0, 1.0))
    assert evaluate_model.dice_iou_for_class(prediction, mapped_ground_truth, 2)[:2] == pytest.approx((1.0, 1.0))


def test_gt_out_of_range_without_label_map_fails_loudly() -> None:
    raw_ground_truth = np.array([[0, 10], [0, 0]], dtype=np.int64)

    with pytest.raises(ValueError, match="Proveer --label-map"):
        evaluate_model.apply_label_mapping(raw_ground_truth, mapping=None, num_classes=3, case_id="case-out-of-range")


def test_report_includes_present_case_counts_and_empty_gt_warning(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "image.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(image_path, np.zeros((2, 2), dtype=np.float32))
    np.save(mask_path, np.array([[0, 1], [0, 0]], dtype=np.int64))
    pair = evaluate_model.CasePair("case-warning", image_path, mask_path)
    cached = object()

    def fake_predict_case(_cached, _plane, _image_path, _target_size):
        return np.array([[0, 1], [2, 0]], dtype=np.int64), {"processedShape": [2, 2]}

    monkeypatch.setattr(evaluate_model, "predict_case", fake_predict_case)

    report = evaluate_model.evaluate_pairs(cached, [pair], "sagittal", num_classes=3, target_size=(2, 2))

    assert report["classStats"]["1"]["gt_present_cases"] == 1
    assert report["classStats"]["1"]["pred_present_cases"] == 1
    assert report["classStats"]["2"]["gt_present_cases"] == 0
    assert report["classStats"]["2"]["pred_present_cases"] == 1
    assert report["macroForegroundReliable"] is False
    assert any("foreground class 2" in warning for warning in report["warnings"])
