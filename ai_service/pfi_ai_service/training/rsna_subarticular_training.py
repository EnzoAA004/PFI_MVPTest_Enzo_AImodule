"""RSNA subarticular Axial T2 training adapter for Notebook 63."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, recall_score

from . import rsna_foraminal_training as base

CLASS_NAMES = base.CLASS_NAMES
CLASS_TO_INDEX = base.CLASS_TO_INDEX
DISPLAY_CLASS_NAMES = base.DISPLAY_CLASS_NAMES
SIDES = base.SIDES
SIDE_TO_INDEX = base.SIDE_TO_INDEX
LEVELS = base.LEVELS
LEVEL_TO_INDEX = base.LEVEL_TO_INDEX
TrainConfig = base.TrainConfig
DataRootAudit = base.DataRootAudit
sha256_file = base.sha256_file
audit_data_root = base.audit_data_root
find_data_root = base.find_data_root
download_selected_series = base.download_selected_series
prepare_samples = base.prepare_samples
build_cache = base.build_cache
sampling_audit = base.sampling_audit

_REQUIRED = {
    "study_id", "side", "level", "severity", "severity_code",
    "coordinate_series_id", "coordinate_instance_number",
    "coordinate_x", "coordinate_y", "coordinate_series_description",
    "sequence_category", "split", "internal_test_sealed",
    "human_review_required", "not_clinical_diagnosis",
    "official_test_accessed",
}


def _bools(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _validate_summary(summary: dict[str, Any]) -> None:
    checks = {
        "approved": summary.get("approved") is True,
        "status": summary.get("status") == "APPROVED_FOR_NOTEBOOK_63",
        "nextNotebook": summary.get("nextNotebook") == 63,
        "internalTestSealed": (
            summary.get("governance", {}).get("internalTestSealed") is True
        ),
        "authorizedOpenNotebook": (
            summary.get("governance", {})
            .get("authorizedInternalTestOpenNotebook") == 64
        ),
        "officialTestNotAccessed": (
            summary.get("governance", {}).get("officialTestAccessed") is False
        ),
        "internalTestNotAccessed": (
            summary.get("governance", {}).get("internalTestAccessed") is False
        ),
        "noStudyLeakage": (
            summary.get("gateResults", {}).get("noStudyLeakage") is True
        ),
        "rowsConserved": (
            summary.get("gateResults", {}).get("rowsConserved") is True
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Notebook 62 no habilita Notebook 63: {checks}")


def _normalize_manifest(frame: pd.DataFrame, expected_split: str) -> pd.DataFrame:
    missing = sorted(_REQUIRED - set(frame.columns))
    if missing:
        raise RuntimeError(f"Faltan columnas en {expected_split}: {missing}")
    result = frame.copy()
    result["study_id"] = result.study_id.astype(str)
    result["coordinate_series_id"] = result.coordinate_series_id.astype(str)
    result["side"] = result.side.astype(str).str.strip().str.lower()
    result["level"] = result.level.astype(str).str.strip()
    result["severity"] = result.severity.astype(str).str.strip().str.lower()
    result["split"] = result.split.astype(str).str.strip()
    result["sequence_category"] = (
        result.sequence_category.astype(str).str.strip().str.lower()
    )
    numbers = [
        "coordinate_instance_number", "coordinate_x",
        "coordinate_y", "severity_code",
    ]
    for column in numbers:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[numbers].isna().any().any():
        raise RuntimeError(
            f"Valores numéricos inválidos en {expected_split}: "
            f"{result[numbers].isna().sum().to_dict()}"
        )
    result["coordinate_instance_number"] = (
        result.coordinate_instance_number.round().astype(int)
    )
    result["severity_code"] = result.severity_code.round().astype(int)
    checks = {
        "split": result.split.eq(expected_split).all(),
        "sides": result.side.isin(SIDES).all(),
        "levels": result.level.isin(LEVELS).all(),
        "classes": result.severity.isin(CLASS_NAMES).all(),
        "axialCategory": result.sequence_category.eq("axial_t2").all(),
        "axialDescription": (
            result.coordinate_series_description.astype(str)
            .str.strip().str.lower().eq("axial t2").all()
        ),
        "codes": (
            result.severity.map(CLASS_TO_INDEX).astype(int)
            .eq(result.severity_code).all()
        ),
        "uniqueKeys": not result[
            ["study_id", "side", "level"]
        ].duplicated().any(),
        "officialTestNotAccessed": not _bools(
            result.official_test_accessed
        ).any(),
        "humanReviewRequired": _bools(
            result.human_review_required
        ).all(),
        "notClinicalDiagnosis": _bools(
            result.not_clinical_diagnosis
        ).all(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Manifiesto {expected_split} inválido: {checks}")
    result["side_index"] = result.side.map(SIDE_TO_INDEX).astype(int)
    result["level_index"] = result.level.map(LEVEL_TO_INDEX).astype(int)
    result["sample_id"] = (
        result.study_id + "__" + result.side + "__"
        + result.level.str.replace("-", "_", regex=False)
    )
    result["stratum"] = (
        result.side + "__" + result.level + "__" + result.severity
    )
    return result.reset_index(drop=True)


def load_manifests(
    split_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, str]]:
    split_root = Path(split_root)
    summary_path = split_root / "split_summary.json"
    seal_path = split_root / "internal_test_seal.json"
    internal_path = split_root / "internal_test_manifest.csv"
    for path in (summary_path, seal_path, internal_path):
        if not path.is_file():
            raise RuntimeError(f"Falta artefacto del Notebook 62: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_summary(summary)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal_checks = {
        "opened": seal.get("opened") is False,
        "doNotUseForTraining": seal.get("doNotUseForTraining") is True,
        "authorizedOpenNotebook": seal.get("authorizedOpenNotebook") == 64,
        "manifestHash": (
            seal.get("manifest", {}).get("sha256") == sha256_file(internal_path)
        ),
    }
    if not all(seal_checks.values()):
        raise RuntimeError(f"Sello del internal test inválido: {seal_checks}")

    original_validate = base._validate_summary
    original_normalize = base._normalize_manifest
    try:
        base._validate_summary = _validate_summary
        base._normalize_manifest = _normalize_manifest
        train, validation, loaded_summary, hashes = base.load_manifests(split_root)
    finally:
        base._validate_summary = original_validate
        base._normalize_manifest = original_normalize

    hashes["internal_test_seal"] = sha256_file(seal_path)
    expected = summary.get("outputSha256", {}).get("internalTestSeal")
    if expected and hashes["internal_test_seal"] != expected:
        raise RuntimeError("Hash de internal_test_seal no coincide.")
    return train, validation, loaded_summary, hashes


def _metrics(
    targets: list[int],
    predictions: list[int],
    probabilities: np.ndarray,
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    recalls = recall_score(
        targets, predictions, labels=[0, 1, 2],
        average=None, zero_division=0,
    )
    result = {
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "normal_mild_recall": float(recalls[0]),
        "moderate_recall": float(recalls[1]),
        "severe_recall": float(recalls[2]),
        "weighted_log_loss": float(
            log_loss(targets, probabilities, labels=[0, 1, 2])
        ),
    }
    result["selection_score"] = float(
        0.40 * result["macro_f1"]
        + 0.25 * result["balanced_accuracy"]
        + 0.25 * result["severe_recall"]
        + 0.10 * result["moderate_recall"]
    )
    return result


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def train_model(
    train_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    cache_root: Path,
    checkpoint_root: Path,
    run_root: Path,
    manifest_hashes: dict[str, str],
    repo_ref: str,
    repo_sha: str,
    config: TrainConfig,
) -> dict[str, Any]:
    original_metrics = base._metrics
    try:
        base._metrics = _metrics
        source = base.train_model(
            train_samples=train_samples,
            validation_samples=validation_samples,
            cache_root=cache_root,
            checkpoint_root=checkpoint_root,
            run_root=run_root,
            manifest_hashes=manifest_hashes,
            repo_ref=repo_ref,
            repo_sha=repo_sha,
            config=config,
        )
    finally:
        base._metrics = original_metrics

    checkpoint_path = Path(source["checkpoint"]["path"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint.update({
        "schemaVersion": "pfi.rsna-subarticular-training-checkpoint.v1",
        "task": "subarticular_stenosis_left_right",
        "sequence": "Axial T2",
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "officialTestAccessed": False,
        "internalTestAccessed": False,
    })
    base._save_checkpoint(checkpoint_path, checkpoint)

    metrics = {
        key: float(value)
        for key, value in source["validationMetrics"].items()
    }
    process_gates = {
        "sourceNotebook62Approved": True,
        "trainValidationStudyIsolation": not bool(
            set(train_samples.study_id) & set(validation_samples.study_id)
        ),
        "trainContainsAllClasses": (
            set(train_samples.severity) == set(CLASS_NAMES)
        ),
        "validationContainsAllClasses": (
            set(validation_samples.severity) == set(CLASS_NAMES)
        ),
        "bestCheckpointExists": checkpoint_path.is_file(),
        "finiteValidationMetrics": all(
            math.isfinite(value) for value in metrics.values()
        ),
        "internalTestNotAccessed": True,
        "officialTestNotAccessed": True,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
    }
    metric_gates = {
        "macroF1": metrics["macro_f1"] >= config.minimum_macro_f1,
        "balancedAccuracy": (
            metrics["balanced_accuracy"] >= config.minimum_balanced_accuracy
        ),
        "severeRecall": (
            metrics["severe_recall"] >= config.minimum_severe_recall
        ),
        "moderateRecall": (
            metrics["moderate_recall"] >= config.minimum_moderate_recall
        ),
    }
    process_gates = {key: bool(value) for key, value in process_gates.items()}
    metric_gates = {key: bool(value) for key, value in metric_gates.items()}
    approved = all(process_gates.values()) and all(metric_gates.values())

    model_card_path = Path(run_root) / "model_card.md"
    model_card = f"""# RSNA subarticular Axial T2 2.5D — model card

- Estado: {'APPROVED_FOR_NOTEBOOK_64' if approved else 'TRAINING_REVIEW_REQUIRED'}
- Arquitectura: `{config.model_name}` con embeddings de lado y nivel.
- Entrada: crop 2.5D de tres cortes Axial T2 alrededor de la coordenada subarticular.
- Clases: Normal/Mild, Moderate, Severe.
- Mejor época: {source['bestEpoch']}
- Macro F1 de validación: {metrics['macro_f1']:.4f}
- Balanced accuracy de validación: {metrics['balanced_accuracy']:.4f}
- Recall Severe de validación: {metrics['severe_recall']:.4f}
- Recall Moderate de validación: {metrics['moderate_recall']:.4f}

## Uso previsto

Herramienta de investigación y apoyo con revisión humana obligatoria. No es diagnóstico clínico autónomo y no está aprobada para uso comercial.

## Separación de datos

Solo se usan train y validation del Notebook 62. El internal test permanece sellado para el Notebook 64.
"""
    _write_text(model_card_path, model_card)

    summary = {
        **source,
        "schemaVersion": "pfi.rsna-subarticular-training.v1",
        "notebook": 63,
        "sourceNotebook": 62,
        "status": (
            "APPROVED_FOR_NOTEBOOK_64"
            if approved else "TRAINING_REVIEW_REQUIRED"
        ),
        "approved": bool(approved),
        "nextNotebook": 64 if approved else None,
        "task": "subarticular_stenosis_left_right",
        "sequence": "Axial T2",
        "architecture": {
            **source["architecture"],
            "input": "2.5D_three_adjacent_axial_slices",
        },
        "processGates": process_gates,
        "metricGates": metric_gates,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "governance": {
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
            "autonomousDiagnosis": False,
            "commercialUse": False,
            "officialTestAccessed": False,
            "internalTestAccessed": False,
            "internalTestSealedUntilNotebook64": True,
        },
    }
    summary["outputSha256"]["modelCard"] = sha256_file(model_card_path)
    _write_json(Path(run_root) / "training_summary.json", summary)
    return summary


__all__ = [
    "CLASS_NAMES", "DISPLAY_CLASS_NAMES", "LEVELS", "SIDES",
    "TrainConfig", "DataRootAudit", "audit_data_root", "build_cache",
    "download_selected_series", "find_data_root", "load_manifests",
    "prepare_samples", "sampling_audit", "sha256_file", "train_model",
]
