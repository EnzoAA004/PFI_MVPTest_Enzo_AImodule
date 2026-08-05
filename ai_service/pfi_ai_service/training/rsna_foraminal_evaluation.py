"""Internal-test evaluation and final export for RSNA foraminal classifier."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .rsna_foraminal_training import (
    CLASS_NAMES,
    DISPLAY_CLASS_NAMES,
    LEVEL_TO_INDEX,
    SIDE_TO_INDEX,
    CachedForaminalDataset,
    ForaminalClassifier,
    TrainConfig,
    _normalize_manifest,
    build_cache,
    download_selected_series,
    find_data_root,
    prepare_samples,
    sha256_file,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _probabilities(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return values / values.sum(axis=1, keepdims=True)


def _metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    recalls = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "normal_mild_recall": float(recalls[0]),
        "moderate_recall": float(recalls[1]),
        "severe_recall": float(recalls[2]),
        "weighted_log_loss": float(log_loss(y, prob, labels=[0, 1, 2])),
    }


def _group_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kind, columns in (("side", ["side"]), ("level", ["level"]), ("side_level", ["side", "level"])):
        grouper: Any = columns[0] if len(columns) == 1 else columns
        for key, group in frame.groupby(grouper, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            prob = _probabilities(group[["prob_normal_mild", "prob_moderate", "prob_severe"]].to_numpy())
            rows.append({
                "group_type": kind,
                "group_key": "__".join(map(str, key)),
                "n": int(len(group)),
                "normal_mild_support": int((group.target == 0).sum()),
                "moderate_support": int((group.target == 1).sum()),
                "severe_support": int((group.target == 2).sum()),
                **_metrics(group.target.to_numpy(), group.prediction.to_numpy(), prob),
            })
    return pd.DataFrame(rows)


def _stratum_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (side, level, severity), group in frame.groupby(["side", "level", "severity"], sort=True):
        target = int(group.target.iloc[0])
        rows.append({
            "side": side,
            "level": level,
            "severity": severity,
            "n": int(len(group)),
            "target_recall": float((group.prediction == target).mean()),
            "pred_normal_mild": int((group.prediction == 0).sum()),
            "pred_moderate": int((group.prediction == 1).sum()),
            "pred_severe": int((group.prediction == 2).sum()),
            "mean_prob_normal_mild": float(group.prob_normal_mild.mean()),
            "mean_prob_moderate": float(group.prob_moderate.mean()),
            "mean_prob_severe": float(group.prob_severe.mean()),
        })
    return pd.DataFrame(rows)


def evaluate_foraminal_internal(
    *,
    split_root: Path,
    training_root: Path,
    evaluation_root: Path,
    model_root: Path,
    cache_root: Path,
    data_candidates: Iterable[Path],
    competition: str,
    local_download_root: Path,
    repo_ref: str,
    repo_sha: str,
    kaggle_token: str = "",
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Seleccioná una GPU T4 o superior en Colab.")
    device = torch.device("cuda")
    evaluation_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    split_summary_path = split_root / "split_summary.json"
    training_summary_path = training_root / "training_summary.json"
    internal_path = split_root / "internal_test_manifest.csv"
    checkpoint_path = model_root / "checkpoints" / "best_checkpoint.pt"
    final_model_path = model_root / "rsna_foraminal_sagittal_t1_2p5d.pt"
    receipt_path = evaluation_root / "internal_test_access_receipt.json"
    marker_path = evaluation_root / "evaluation_complete.marker"
    summary_path = evaluation_root / "evaluation_summary.json"
    required = [split_summary_path, training_summary_path, internal_path, checkpoint_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan entradas:\n- " + "\n- ".join(missing))
    if marker_path.exists():
        raise RuntimeError(f"Notebook 60 ya fue completado: {marker_path}")

    split_summary = _read_json(split_summary_path)
    training_summary = _read_json(training_summary_path)
    if split_summary.get("approved") is not True or split_summary.get("governance", {}).get("internalTestSealed") is not True:
        raise RuntimeError("Notebook 58 no está aprobado o el internal test no figura sellado.")
    if training_summary.get("status") != "APPROVED_FOR_NOTEBOOK_60" or training_summary.get("approved") is not True:
        raise RuntimeError("Notebook 59 no habilita Notebook 60.")
    if training_summary.get("governance", {}).get("internalTestAccessed") is not False:
        raise RuntimeError("Notebook 59 declara acceso previo al internal test.")

    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != training_summary.get("checkpoint", {}).get("sha256"):
        raise RuntimeError("Hash del checkpoint distinto al congelado por Notebook 59.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**checkpoint["config"])
    gates = {
        "macro_f1": config.minimum_macro_f1,
        "balanced_accuracy": config.minimum_balanced_accuracy,
        "severe_recall": config.minimum_severe_recall,
        "moderate_recall": config.minimum_moderate_recall,
    }

    internal_hash = sha256_file(internal_path)
    expected_hash = split_summary.get("outputSha256", {}).get("internalTestManifest")
    if expected_hash and internal_hash != expected_hash:
        raise RuntimeError("Hash del internal test distinto al Notebook 58.")
    internal = _normalize_manifest(pd.read_csv(internal_path, dtype={"study_id": str, "coordinate_series_id": str}), "internal_test")
    train = pd.read_csv(split_root / "train_manifest.csv", dtype={"study_id": str})
    validation = pd.read_csv(split_root / "validation_manifest.csv", dtype={"study_id": str})
    internal_ids = set(internal.study_id)
    overlaps = {"train": len(internal_ids & set(train.study_id)), "validation": len(internal_ids & set(validation.study_id))}
    if any(overlaps.values()):
        raise RuntimeError(f"Fuga hacia internal test: {overlaps}")

    receipt = {
        "schemaVersion": "pfi.rsna-foraminal-internal-access.v1",
        "openedAtUtc": datetime.now(timezone.utc).isoformat(),
        "internalManifestSha256": internal_hash,
        "checkpointSha256": checkpoint_sha,
        "rows": int(len(internal)),
        "studies": int(internal.study_id.nunique()),
        "repoSha": repo_sha,
        "officialTestAccessed": False,
    }
    if receipt_path.exists():
        previous = _read_json(receipt_path)
        if previous.get("internalManifestSha256") != internal_hash or previous.get("checkpointSha256") != checkpoint_sha:
            raise RuntimeError("Existe un recibo previo para otros artefactos.")
    else:
        _write_json(receipt_path, receipt)

    empty = internal.iloc[0:0].copy()
    data_root, audits = find_data_root(data_candidates, internal, empty)
    if data_root is None:
        data_root = download_selected_series(internal, empty, local_download_root, competition, kaggle_token)
    samples = prepare_samples(internal, data_root, "internal_test")
    cache_audit = build_cache(samples, cache_root, "internal_test", config)

    eval_config = dataclasses.replace(config, pretrained=False)
    model = ForaminalClassifier(eval_config).to(device)
    model.load_state_dict(checkpoint["modelStateDict"])
    model.eval()
    dataset = CachedForaminalDataset(samples, cache_root, "internal_test", augment=False)
    loader = DataLoader(dataset, batch_size=min(max(config.batch_size, 1), 64), shuffle=False,
                        num_workers=config.num_workers, pin_memory=True, persistent_workers=config.num_workers > 0)
    targets: list[int] = []
    probabilities: list[list[float]] = []
    indices: list[int] = []
    started = time.time()
    with torch.no_grad():
        for images, labels, sides, levels, row_indices in tqdm(loader, desc="internal test"):
            images, sides, levels = images.to(device, non_blocking=True), sides.to(device, non_blocking=True), levels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images, sides, levels)
            probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().tolist())
            targets.extend(labels.tolist())
            indices.extend(row_indices.tolist())

    prob = _probabilities(np.asarray(probabilities))
    y = np.asarray(targets, dtype=int)
    pred = prob.argmax(axis=1)
    metadata = samples[["sample_id", "study_id", "coordinate_series_id", "coordinate_instance_number", "side", "level", "severity", "severity_code"]].reset_index(drop=True)
    values = pd.DataFrame({"dataset_index": indices, "target": y, "prediction": pred,
                           "prob_normal_mild": prob[:, 0], "prob_moderate": prob[:, 1], "prob_severe": prob[:, 2]}).sort_values("dataset_index").reset_index(drop=True)
    predictions = metadata.join(values.drop(columns="dataset_index"))
    predictions["predicted_severity"] = predictions.prediction.map(dict(enumerate(CLASS_NAMES)))
    metrics = _metrics(y, pred, prob)
    metrics["runtime_seconds"] = float(time.time() - started)
    metrics["samples_per_second"] = float(len(y) / max(metrics["runtime_seconds"], 1e-9))
    groups = _group_metrics(predictions)
    strata = _stratum_metrics(predictions)
    matrix = pd.DataFrame(confusion_matrix(y, pred, labels=[0, 1, 2]),
                          index=[f"true_{name}" for name in CLASS_NAMES],
                          columns=[f"pred_{name}" for name in CLASS_NAMES]).reset_index(names="target")
    severe_fn = predictions[(predictions.target == 2) & (predictions.prediction != 2)].sort_values("prob_severe")

    row_error = float(np.max(np.abs(prob.sum(axis=1) - 1.0)))
    process_gates = {
        "sourceNotebook58Approved": True,
        "sourceNotebook59Approved": True,
        "checkpointHashVerified": True,
        "trainInternalIsolation": overlaps["train"] == 0,
        "validationInternalIsolation": overlaps["validation"] == 0,
        "allClassesPresent": set(y.tolist()) == {0, 1, 2},
        "probabilitiesNormalized": row_error <= 1e-10,
        "finiteMetrics": all(math.isfinite(float(value)) for value in metrics.values()),
        "internalTestAccessRecorded": receipt_path.is_file(),
        "officialTestNotAccessed": True,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
    }
    metric_gates = {
        "macroF1": metrics["macro_f1"] >= gates["macro_f1"],
        "balancedAccuracy": metrics["balanced_accuracy"] >= gates["balanced_accuracy"],
        "severeRecall": metrics["severe_recall"] >= gates["severe_recall"],
        "moderateRecall": metrics["moderate_recall"] >= gates["moderate_recall"],
    }
    approved = all(process_gates.values()) and all(metric_gates.values())
    status = "APPROVED_FOR_NOTEBOOK_61" if approved else "EVALUATION_REVIEW_REQUIRED"

    paths = {
        "predictions": evaluation_root / "internal_test_predictions.csv",
        "metricsByGroup": evaluation_root / "internal_test_metrics_by_group.csv",
        "metricsByStratum": evaluation_root / "internal_test_metrics_by_stratum.csv",
        "confusionMatrix": evaluation_root / "internal_test_confusion_matrix.csv",
        "severeFalseNegatives": evaluation_root / "severe_false_negatives.csv",
        "modelCard": evaluation_root / "model_card_final.md",
        "finalManifest": evaluation_root / "final_model_manifest.json",
    }
    predictions.to_csv(paths["predictions"], index=False)
    groups.to_csv(paths["metricsByGroup"], index=False)
    strata.to_csv(paths["metricsByStratum"], index=False)
    matrix.to_csv(paths["confusionMatrix"], index=False)
    severe_fn.to_csv(paths["severeFalseNegatives"], index=False)

    final_sha = None
    if approved:
        payload = {
            "schemaVersion": "pfi.rsna-foraminal-final.v1",
            "modelStateDict": checkpoint["modelStateDict"],
            "config": checkpoint["config"],
            "classNames": list(CLASS_NAMES),
            "displayClassNames": list(DISPLAY_CLASS_NAMES),
            "sideToIndex": SIDE_TO_INDEX,
            "levelToIndex": LEVEL_TO_INDEX,
            "task": "neural_foraminal_narrowing",
            "sequence": "Sagittal T1",
            "sourceCheckpointSha256": checkpoint_sha,
            "internalTestManifestSha256": internal_hash,
            "internalTestMetrics": metrics,
            "gates": gates,
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
            "officialTestAccessed": False,
            "internalTestAccessed": True,
        }
        tmp = final_model_path.with_suffix(".tmp.pt")
        torch.save(payload, tmp)
        os.replace(tmp, final_model_path)
        final_sha = sha256_file(final_model_path)

    _write_text(paths["modelCard"], f"""# RSNA foraminal Sagittal T1 2.5D — model card final

- Estado: `{status}`
- Internal test: {len(internal)} muestras / {internal.study_id.nunique()} estudios.
- Macro F1: {metrics['macro_f1']:.4f}
- Balanced accuracy: {metrics['balanced_accuracy']:.4f}
- Recall Moderate: {metrics['moderate_recall']:.4f}
- Recall Severe: {metrics['severe_recall']:.4f}
- Log loss: {metrics['weighted_log_loss']:.4f}

Uso asistivo de investigación con revisión profesional obligatoria. No constituye diagnóstico autónomo ni dispositivo médico validado.
""")
    summary = {
        "schemaVersion": "pfi.rsna-foraminal-evaluation.v1",
        "notebook": 60,
        "status": status,
        "approved": approved,
        "nextNotebook": 61 if approved else None,
        "completedAtUtc": datetime.now(timezone.utc).isoformat(),
        "data": {"rows": int(len(internal)), "studies": int(internal.study_id.nunique()), "overlaps": overlaps},
        "internalTestMetrics": metrics,
        "frozenGates": gates,
        "processGates": process_gates,
        "metricGates": metric_gates,
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha, "epoch": int(checkpoint["epoch"])},
        "finalModel": {"path": str(final_model_path) if approved else None, "sha256": final_sha, "exported": approved},
        "dataRoot": str(data_root),
        "dataAudits": [dataclasses.asdict(audit) for audit in audits],
        "cacheAudit": cache_audit,
        "repoRef": repo_ref,
        "repoSha": repo_sha,
        "governance": {"humanReviewRequired": True, "notClinicalDiagnosis": True, "officialTestAccessed": False,
                       "internalTestAccessed": True, "gatesFrozenBeforeTest": True},
    }
    manifest = {
        "schemaVersion": "pfi.rsna-foraminal-final-manifest.v1",
        "status": status,
        "artifact": summary["finalModel"],
        "sourceCheckpoint": summary["checkpoint"],
        "internalTestManifestSha256": internal_hash,
        "internalTestMetrics": metrics,
        "processGates": process_gates,
        "metricGates": metric_gates,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "officialTestAccessed": False,
    }
    _write_json(paths["finalManifest"], manifest)
    summary["outputSha256"] = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    _write_json(summary_path, summary)
    _write_text(marker_path, json.dumps({"status": status, "completedAtUtc": summary["completedAtUtc"],
                                         "evaluationSummarySha256": sha256_file(summary_path)}) + "\n")
    if not approved:
        raise RuntimeError("No superó los gates. No ajustar el checkpoint usando el internal test; documentar la limitación.")
    return summary


__all__ = ["evaluate_foraminal_internal"]
