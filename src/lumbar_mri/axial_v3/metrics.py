"""Validation metrics with the same raw_0 conventions used by axial v2."""

from __future__ import annotations

from typing import Any

import numpy as np

from .labels import CLASS_INDEX_TO_NAME


FOREGROUND_CLASSES = (1, 2, 3, 4, 5)
FOREGROUND_EXCLUDING_RAW0 = (2, 3, 4, 5)


def _as_batch(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 2:
        return values[None, ...]
    if values.ndim == 3:
        return values
    raise ValueError(f"expected [H,W] or [N,H,W], got {values.shape}")


def optional_div(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def confusion_matrix(prediction: np.ndarray, target: np.ndarray, *, num_classes: int = 6) -> np.ndarray:
    pred = _as_batch(prediction).astype(np.int64)
    truth = _as_batch(target).astype(np.int64)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} != {truth.shape}")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    flat_truth = truth.reshape(-1)
    flat_pred = pred.reshape(-1)
    valid = (flat_truth >= 0) & (flat_truth < num_classes) & (flat_pred >= 0) & (flat_pred < num_classes)
    np.add.at(matrix, (flat_truth[valid], flat_pred[valid]), 1)
    return matrix


def metrics_from_predictions(prediction: np.ndarray, target: np.ndarray, *, num_classes: int = 6) -> dict[str, Any]:
    pred = _as_batch(prediction).astype(np.int64)
    truth = _as_batch(target).astype(np.int64)
    matrix = confusion_matrix(pred, truth, num_classes=num_classes)
    total = int(matrix.sum())
    per_class: dict[str, dict[str, Any]] = {}
    for class_index in range(num_classes):
        name = CLASS_INDEX_TO_NAME[class_index]
        tp = int(matrix[class_index, class_index])
        fp = int(matrix[:, class_index].sum() - tp)
        fn = int(matrix[class_index, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        pred_present_by_case = (pred == class_index).reshape(pred.shape[0], -1).any(axis=1)
        gt_present_by_case = (truth == class_index).reshape(truth.shape[0], -1).any(axis=1)
        per_class[name] = {
            "evaluableCases": int(pred.shape[0]),
            "gtPresentCases": int(gt_present_by_case.sum()),
            "gtAbsentCases": int((~gt_present_by_case).sum()),
            "predPresentCases": int(pred_present_by_case.sum()),
            "predictedInGtAbsentCases": int((pred_present_by_case & ~gt_present_by_case).sum()),
            "truePositivePixels": tp,
            "falsePositivePixels": fp,
            "falseNegativePixels": fn,
            "trueNegativePixels": tn,
            "dice": optional_div(2 * tp, 2 * tp + fp + fn),
            "iou": optional_div(tp, tp + fp + fn),
            "precision": optional_div(tp, tp + fp),
            "recall": optional_div(tp, tp + fn),
        }

    def macro(classes: tuple[int, ...], key: str) -> float | None:
        values = [per_class[CLASS_INDEX_TO_NAME[index]][key] for index in classes]
        valid_values = [float(value) for value in values if value is not None]
        return float(np.mean(valid_values)) if valid_values else None

    raw0 = per_class["raw_0"]
    return {
        "confusionMatrix": matrix.tolist(),
        "perClass": per_class,
        "dice_macro_foreground": macro(FOREGROUND_CLASSES, "dice"),
        "iou_macro_foreground": macro(FOREGROUND_CLASSES, "iou"),
        "precision_macro_foreground": macro(FOREGROUND_CLASSES, "precision"),
        "recall_macro_foreground": macro(FOREGROUND_CLASSES, "recall"),
        "dice_macro_excluding_raw0": macro(FOREGROUND_EXCLUDING_RAW0, "dice"),
        "iou_macro_excluding_raw0": macro(FOREGROUND_EXCLUDING_RAW0, "iou"),
        "precision_macro_excluding_raw0": macro(FOREGROUND_EXCLUDING_RAW0, "precision"),
        "recall_macro_excluding_raw0": macro(FOREGROUND_EXCLUDING_RAW0, "recall"),
        "raw0Dice": raw0["dice"],
        "raw0Iou": raw0["iou"],
        "raw0Precision": raw0["precision"],
        "raw0Recall": raw0["recall"],
        "raw0FalsePositivePixels": raw0["falsePositivePixels"],
        "raw0FalseNegativePixels": raw0["falseNegativePixels"],
        "raw0PredictedInGtAbsentCases": raw0["predictedInGtAbsentCases"],
        "raw0GtAbsentCases": raw0["gtAbsentCases"],
    }
