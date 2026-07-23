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
    "test_evaluation_in_progress.json",
)
PROTECTED_V2_ARTIFACTS = (
    "test_evaluated_once.json",
    "test_metrics.json",
    "test_case_metrics.csv",
    "test_metrics_per_class.csv",
    "test_confusion_matrix.csv",
    "test_predictions.png",
    "axial_t2_alkafri_final_v2_candidate.pt",
    "final_artifact_verification.json",
    "axial_t2_alkafri_v2.best_checkpoint.pt",
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


def assert_no_test_records(records: Iterable[object], *, context: str = "axial_v3") -> None:
    splits = []
    for record in records:
        split = getattr(record, "split", None)
        if split is None and isinstance(record, dict):
            split = record.get("split")
        splits.append(str(split))
    require_train_val_only(splits, context=context)


def reject_protected_v2_paths(paths: Iterable[str | Path], *, write: bool = False, context: str = "axial_v3") -> None:
    for path in paths:
        text = str(path).replace("\\", "/")
        name = Path(text).name
        if name in PROTECTED_V2_ARTIFACTS:
            action = "write" if write else "read"
            raise ValueError(f"{context}: protected v2 artifact {action} is forbidden: {path}")


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
        r"(test_evaluated_once|test_evaluation_in_progress|test_metrics|test_predictions|final_artifact_verification).*\.unlink\(",
        r"\bos\.remove\([^)]*(test_evaluated_once|test_evaluation_in_progress|test_metrics|test_predictions|final_artifact_verification)",
        r"\bshutil\.rmtree\([^)]*(axial_final_v2|axial-final-v2)",
        r"(test_evaluated_once|test_metrics|test_predictions|final_artifact_verification).*write_(text|bytes)\(",
    ]
    for pattern in regexes:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            hits.append(pattern)
    return hits
