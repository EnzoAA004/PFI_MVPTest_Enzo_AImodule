"""Guards that keep axial v3 development away from the observed test split."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable


ALLOWED_DEVELOPMENT_SPLITS = {"train", "val", "validation"}
FORBIDDEN_TEST_TOKENS = (
    "AXIAL_FINAL_TEST_CONFIRMATION",
    "test_metrics.json",
    "test_case_metrics.csv",
    "test_metrics_per_class.csv",
    "test_confusion_matrix.csv",
    "test_predictions.png",
    "test_evaluated_once.json",
    "build_eval_loaders",
    'loaders["test"]',
    "evaluate_test_once",
)


def normalize_split(split: str) -> str:
    value = str(split).strip().lower()
    return "val" if value == "validation" else value


def require_train_val_only(splits: Iterable[str], *, context: str = "axial_v3") -> set[str]:
    """Return normalized splits or raise if anything outside train/validation is requested."""

    normalized = {normalize_split(split) for split in splits}
    forbidden = normalized - {"train", "val"}
    if forbidden:
        raise ValueError(f"{context}: only train/val are allowed, got {sorted(forbidden)}")
    return normalized


def reject_test_paths(paths: Iterable[str | Path], *, context: str = "axial_v3") -> None:
    """Reject explicit reads of final v2 test artifacts from development code."""

    for path in paths:
        text = str(path).replace("\\", "/").lower()
        if "/test_" in text or text.endswith("test_metrics.json") or "test_evaluated_once" in text:
            raise ValueError(f"{context}: test artifact access is forbidden: {path}")


def find_forbidden_test_references(text: str) -> list[str]:
    """Find static references that should not appear in axial v3 iteration notebooks."""

    hits: list[str] = []
    for token in FORBIDDEN_TEST_TOKENS:
        if token in text:
            hits.append(token)
    regexes = [
        r"\bbuild_[A-Za-z0-9_]*loader[s]?\([^)]*[\"']test[\"']",
        r"\bDataLoader\([^)]*test",
        r"\bsplit\s*==\s*[\"']test[\"']",
        r"\bsplit\s*=\s*[\"']test[\"']",
    ]
    for pattern in regexes:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            hits.append(pattern)
    return hits
