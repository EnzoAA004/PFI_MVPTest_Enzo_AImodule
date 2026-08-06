"""Notebook 64 helpers for frozen RSNA subarticular internal-test evaluation."""
from __future__ import annotations

import json
import os
import shutil
import tarfile
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from torch.utils.data import DataLoader

from . import rsna_foraminal_training as base
from . import rsna_subarticular_training as sub

TrainConfig = sub.TrainConfig
sha256_file = sub.sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _bools(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def prepare_context(pfi_root: Path, evaluation_repo_sha: str) -> dict[str, Any]:
    pfi_root = Path(pfi_root)
    results = pfi_root / "results" / "P10_6_rsna_findings"
    split_root = results / "notebook62_subarticular_split"
    training_root = results / "notebook63_subarticular_training"
    evaluation_root = results / "notebook64_subarticular_internal_test"
    model_root = (
        pfi_root / "models" / "P10_6_rsna_findings"
        / "subarticular_axial_t2_2p5d"
    )
    final_root = model_root / "final_internal_test_evaluation"
    checkpoint = model_root / "checkpoints" / "best_checkpoint.pt"
    paths = {
        "pfi_root": pfi_root,
        "split_root": split_root,
        "training_root": training_root,
        "evaluation_root": evaluation_root,
        "model_root": model_root,
        "final_root": final_root,
        "checkpoint": checkpoint,
        "training_summary": training_root / "training_summary.json",
        "split_summary": split_root / "split_summary.json",
        "seal": split_root / "internal_test_seal.json",
        "internal_manifest": split_root / "internal_test_manifest.csv",
        "open_record": evaluation_root / "internal_test_open_record.json",
        "prepared_manifest": evaluation_root / "internal_test_prepared_manifest.csv",
        "evaluation_summary": evaluation_root / "internal_test_evaluation_summary.json",
    }
    evaluation_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    required = [
        paths["training_summary"],
        training_root / "training_history.csv",
        training_root / "validation_predictions.csv",
        training_root / "validation_metrics_by_group.csv",
        training_root / "sampling_audit.json",
        training_root / "model_card.md",
        checkpoint,
        paths["split_summary"],
        paths["seal"],
        paths["internal_manifest"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Faltan artefactos:\n- " + "\n- ".join(missing))

    training = json.loads(paths["training_summary"].read_text(encoding="utf-8"))
    split = json.loads(paths["split_summary"].read_text(encoding="utf-8"))
    seal = json.loads(paths["seal"].read_text(encoding="utf-8"))

    checks = {
        "approved": training.get("approved") is True,
        "status": training.get("status") == "APPROVED_FOR_NOTEBOOK_64",
        "nextNotebook": training.get("nextNotebook") == 64,
        "task": training.get("task") == "subarticular_stenosis_left_right",
        "sequence": training.get("sequence") == "Axial T2",
        "internalTestNotAccessed": training.get("governance", {}).get(
            "internalTestAccessed"
        ) is False,
        "officialTestNotAccessed": training.get("governance", {}).get(
            "officialTestAccessed"
        ) is False,
        "humanReviewRequired": training.get("governance", {}).get(
            "humanReviewRequired"
        ) is True,
        "notClinicalDiagnosis": training.get("governance", {}).get(
            "notClinicalDiagnosis"
        ) is True,
    }
    if not all(checks.values()):
        raise RuntimeError({"message": "Notebook 63 no habilita 64", "checks": checks})

    expected_checkpoint_sha = training.get("checkpoint", {}).get("sha256")
    actual_checkpoint_sha = sha256_file(checkpoint)
    if actual_checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError("Hash del checkpoint inconsistente.")

    output_paths = {
        "history": training_root / "training_history.csv",
        "validationPredictions": training_root / "validation_predictions.csv",
        "validationMetricsByGroup": training_root / "validation_metrics_by_group.csv",
        "samplingAudit": training_root / "sampling_audit.json",
        "modelCard": training_root / "model_card.md",
    }
    output_checks = {
        name: training.get("outputSha256", {}).get(name) == sha256_file(path)
        for name, path in output_paths.items()
    }
    if not all(output_checks.values()):
        raise RuntimeError({"message": "Artefactos de Notebook 63 modificados", "checks": output_checks})

    seal_checks = {
        "openedWasFalse": seal.get("opened") is False,
        "doNotUseForTraining": seal.get("doNotUseForTraining") is True,
        "authorizedOpenNotebook64": seal.get("authorizedOpenNotebook") == 64,
        "manifestHash": seal.get("manifest", {}).get("sha256")
        == sha256_file(paths["internal_manifest"]),
        "sealHash": training.get("manifestHashes", {}).get("internal_test_seal")
        == sha256_file(paths["seal"]),
    }
    if not all(seal_checks.values()):
        raise RuntimeError({"message": "Sello inválido", "checks": seal_checks})

    train, validation, _, hashes = sub.load_manifests(split_root)
    for key in ("train_manifest", "validation_manifest", "split_summary"):
        if hashes.get(key) != training.get("manifestHashes", {}).get(key):
            raise RuntimeError(f"Hash inconsistente: {key}")

    context = {
        **paths,
        "training": training,
        "split": split,
        "seal_payload": seal,
        "train_manifest": train,
        "validation_manifest": validation,
        "config": TrainConfig(**training["config"]),
        "expected_checkpoint_sha": expected_checkpoint_sha,
        "actual_checkpoint_sha": actual_checkpoint_sha,
        "evaluation_repo_sha": evaluation_repo_sha,
    }
    report = {
        "status": "READY_FOR_AUTHORIZED_INTERNAL_TEST_OPEN",
        "trainingRepoSha": training.get("repoSha"),
        "evaluationRepoSha": evaluation_repo_sha,
        "bestEpoch": training.get("bestEpoch"),
        "checkpointSha256": actual_checkpoint_sha,
        "internalManifestSha256": sha256_file(paths["internal_manifest"]),
        "internalSealSha256": sha256_file(paths["seal"]),
        "trainingChecks": checks,
        "trainingOutputChecks": output_checks,
        "sealChecks": seal_checks,
        "internalTestParsed": False,
        "officialTestAccessed": False,
    }
    _atomic_json(evaluation_root / "preflight_report.json", report)
    return context


def _validate_internal(context: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    normalize = getattr(sub, "_normalize_manifest", None)
    if normalize is None:
        raise RuntimeError("El pipeline no expone _normalize_manifest.")
    result = normalize(frame, "internal_test").reset_index(drop=True)
    checks = {
        "sealed": _bools(result["internal_test_sealed"]).all(),
        "classes": set(result["severity"]) == set(base.CLASS_NAMES),
        "noTrainOverlap": not (
            set(result["study_id"]) & set(context["train_manifest"]["study_id"])
        ),
        "noValidationOverlap": not (
            set(result["study_id"]) & set(context["validation_manifest"]["study_id"])
        ),
        "humanReviewRequired": _bools(result["human_review_required"]).all(),
        "notClinicalDiagnosis": _bools(result["not_clinical_diagnosis"]).all(),
        "officialTestNotAccessed": not _bools(result["official_test_accessed"]).any(),
    }
    declared = context["split"].get("splits", {}).get("internal_test", {})
    if int(declared.get("rows", -1)) >= 0:
        checks["declaredRows"] = int(declared["rows"]) == len(result)
    if int(declared.get("studies", -1)) >= 0:
        checks["declaredStudies"] = int(declared["studies"]) == result["study_id"].nunique()
    if not all(checks.values()):
        raise RuntimeError({"message": "Internal test inválido", "checks": checks})
    return result


def open_or_reuse_internal(context: dict[str, Any]) -> pd.DataFrame:
    record_path = context["open_record"]
    prepared_path = context["prepared_manifest"]
    expected_sha = context["seal_payload"]["manifest"]["sha256"]

    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "PREPARED_FOR_EVALUATION":
            raise RuntimeError({"message": "Apertura parcial; revisión manual requerida", "record": record})
        if not prepared_path.is_file() or sha256_file(prepared_path) != record.get(
            "preparedManifestSha256"
        ):
            raise RuntimeError("La copia preparada falta o fue modificada.")
        raw = pd.read_csv(
            prepared_path,
            dtype={"study_id": str, "coordinate_series_id": str},
        )
        return _validate_internal(context, raw)

    if prepared_path.exists():
        raise RuntimeError("Copia preparada huérfana sin registro.")
    if sha256_file(context["internal_manifest"]) != expected_sha:
        raise RuntimeError("El manifiesto cambió antes de abrirse.")

    record = {
        "schemaVersion": "pfi.internal-test-open-record.v1",
        "ticket": "P10.6-AI",
        "notebook": 64,
        "status": "OPENING_AUTHORIZED",
        "authorizedSourceReadCount": 0,
        "sourceManifest": str(context["internal_manifest"]),
        "sourceManifestSha256": expected_sha,
        "checkpointSha256": context["actual_checkpoint_sha"],
        "trainingRepoSha": context["training"].get("repoSha"),
        "evaluationRepoSha": context["evaluation_repo_sha"],
        "startedAtUtc": _utc_now(),
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "officialTestAccessed": False,
    }
    _atomic_json(record_path, record)

    raw = pd.read_csv(
        context["internal_manifest"],
        dtype={"study_id": str, "coordinate_series_id": str},
    )
    result = _validate_internal(context, raw)
    if sha256_file(context["internal_manifest"]) != expected_sha:
        raise RuntimeError("El manifiesto cambió durante la apertura.")

    _atomic_csv(prepared_path, result)
    record.update({
        "status": "PREPARED_FOR_EVALUATION",
        "authorizedSourceReadCount": 1,
        "openedAtUtc": _utc_now(),
        "preparedManifest": str(prepared_path),
        "preparedManifestSha256": sha256_file(prepared_path),
        "rows": len(result),
        "studies": result["study_id"].nunique(),
        "classCounts": result["severity"].value_counts().sort_index().to_dict(),
        "sourceSealPreservedUnmodified": True,
    })
    _atomic_json(record_path, record)
    return result


def _expected_cache_names(samples: pd.DataFrame) -> set[str]:
    return set(samples["cache_file"].astype(str))


def build_cache_archive(context: dict[str, Any], manifest: pd.DataFrame) -> dict[str, Any]:
    pfi_root = context["pfi_root"]
    local_data = Path("/content/RSNA_LUMBAR_DISC")
    drive_data = pfi_root / "data" / "RSNA_LUMBAR_DISC"
    cache_root = pfi_root / "cache" / "notebook64_subarticular_internal_test_cache"
    archive_path = pfi_root / "cache" / "notebook64_subarticular_internal_test_cache.tar"
    local_archive = Path("/content/notebook64_subarticular_internal_test_cache.tar")
    cache_root.mkdir(parents=True, exist_ok=True)

    data_root = None
    audits = []
    for candidate in (local_data, drive_data):
        audit = base.audit_data_root(candidate, manifest)
        audits.append(audit)
        if audit.complete:
            data_root = candidate
            break
    if data_root is None:
        raise RuntimeError("No se encontró el root DICOM completo del internal test.")

    samples = sub.prepare_samples(manifest, data_root, "internal_test")
    for tmp in cache_root.rglob("*.tmp.npy"):
        tmp.unlink(missing_ok=True)
    cache_audit = sub.build_cache(samples, cache_root, "internal_test", context["config"])

    expected = _expected_cache_names(samples)
    actual = {path.name for path in (cache_root / "internal_test").glob("*.npy")}
    if actual != expected:
        raise RuntimeError("Caché interno incompleto.")
    expected_archive = {f"internal_test/{name}" for name in expected}

    def matches(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            with tarfile.open(path, "r") as archive:
                names = {member.name for member in archive if member.isfile()}
            return names == expected_archive
        except Exception:
            return False

    if not matches(archive_path):
        local_archive.unlink(missing_ok=True)
        files = sorted((cache_root / "internal_test").glob("*.npy"))
        with tarfile.open(local_archive, "w") as archive:
            for index, source in enumerate(files, 1):
                archive.add(source, arcname=f"internal_test/{source.name}", recursive=False)
                if index % 500 == 0 or index == len(files):
                    print({"archived": index, "total": len(files)})
        if not matches(local_archive):
            raise RuntimeError("TAR local inválido.")
        partial = Path(str(archive_path) + ".partial")
        partial.unlink(missing_ok=True)
        with local_archive.open("rb") as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=64 * 1024**2)
        os.replace(partial, archive_path)
    if not matches(archive_path):
        raise RuntimeError("TAR persistente inválido.")

    report = {
        "schemaVersion": "pfi.rsna-subarticular-internal-cache.v1",
        "ticket": "P10.6-AI",
        "notebook": 64,
        "split": "internal_test",
        "cacheRoot": str(cache_root),
        "archive": str(archive_path),
        "archiveSha256": sha256_file(archive_path),
        "archiveGiB": round(archive_path.stat().st_size / 1024**3, 3),
        "expectedSamples": len(samples),
        "cacheFiles": len(actual),
        "cacheComplete": True,
        "cacheAudit": cache_audit,
        "dataRoot": str(data_root),
        "dataAudits": [
            {
                "root": audit.root,
                "complete": audit.complete,
                "requiredSeries": audit.required_series,
                "missingSeries": audit.missing_series,
            }
            for audit in audits
        ],
        "preparedManifestSha256": sha256_file(context["prepared_manifest"]),
        "checkpointSha256": context["actual_checkpoint_sha"],
        "officialTestAccessed": False,
    }
    _atomic_json(context["evaluation_root"] / "internal_test_cache_audit.json", report)
    return report


def localize_cache(context: dict[str, Any], manifest: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere GPU CUDA.")
    archive_path = context["pfi_root"] / "cache" / "notebook64_subarticular_internal_test_cache.tar"
    if not archive_path.is_file():
        raise RuntimeError("Falta el TAR; ejecutar la fase CPU.")
    samples = manifest.copy()
    if "cache_file" not in samples:
        samples["cache_file"] = samples.apply(
            lambda row: (
                f"{row['study_id']}__{row['coordinate_series_id']}__{row['side']}__"
                f"{str(row['level']).replace('-', '_')}__"
                f"{int(row['coordinate_instance_number'])}.npy"
            ),
            axis=1,
        )
    expected = {f"internal_test/{name}" for name in samples["cache_file"].astype(str)}
    local_archive = Path("/content/notebook64_subarticular_internal_test_cache.tar")
    local_root = Path("/content/notebook64_subarticular_internal_test_cache")
    local_archive.unlink(missing_ok=True)
    shutil.rmtree(local_root, ignore_errors=True)
    local_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive_path, local_archive)
    with tarfile.open(local_archive, "r") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        if names != expected:
            raise RuntimeError("El TAR no coincide con el manifiesto preparado.")
        archive.extractall(local_root, filter="data")
    actual = {path.name for path in (local_root / "internal_test").glob("*.npy")}
    if actual != set(samples["cache_file"].astype(str)):
        raise RuntimeError("Caché local incompleto.")
    return samples.reset_index(drop=True), local_root


def _metrics(targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    recalls = recall_score(
        targets, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    result = {
        "support": int(len(targets)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "normal_mild_recall": float(recalls[0]),
        "moderate_recall": float(recalls[1]),
        "severe_recall": float(recalls[2]),
        "weighted_log_loss": float(log_loss(targets, probabilities, labels=[0, 1, 2])),
    }
    result["selection_score"] = float(
        0.40 * result["macro_f1"]
        + 0.25 * result["balanced_accuracy"]
        + 0.25 * result["severe_recall"]
        + 0.10 * result["moderate_recall"]
    )
    return result


def evaluate_frozen(
    context: dict[str, Any],
    samples: pd.DataFrame,
    cache_root: Path,
) -> dict[str, Any]:
    summary_path = context["evaluation_summary"]
    if summary_path.exists():
        raise RuntimeError("La evaluación final ya existe; no se repetirá.")
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere GPU CUDA.")
    record = json.loads(context["open_record"].read_text(encoding="utf-8"))
    if record.get("status") != "PREPARED_FOR_EVALUATION" or record.get(
        "authorizedSourceReadCount"
    ) != 1:
        raise RuntimeError("Registro de apertura inválido.")
    if sha256_file(context["checkpoint"]) != context["expected_checkpoint_sha"]:
        raise RuntimeError("El checkpoint cambió.")

    checkpoint = torch.load(context["checkpoint"], map_location="cpu", weights_only=False)
    required = {"modelStateDict", "epoch", "config", "classNames", "sideToIndex", "levelToIndex"}
    if required - set(checkpoint):
        raise RuntimeError("Checkpoint incompleto.")
    if int(checkpoint["epoch"]) != int(context["training"]["bestEpoch"]):
        raise RuntimeError("Epoch de checkpoint inconsistente.")
    if checkpoint["config"] != context["training"]["config"]:
        raise RuntimeError("Configuración de checkpoint inconsistente.")

    config = replace(context["config"], pretrained=False)
    device = torch.device("cuda")
    model = base.ForaminalClassifier(config)
    model.load_state_dict(checkpoint["modelStateDict"], strict=True)
    model.to(device).eval()
    dataset = base.CachedForaminalDataset(samples, cache_root, "internal_test", augment=False)
    loader = DataLoader(
        dataset,
        batch_size=context["config"].batch_size,
        shuffle=False,
        num_workers=context["config"].num_workers,
        pin_memory=True,
        persistent_workers=context["config"].num_workers > 0,
    )

    targets_all: list[int] = []
    predictions_all: list[int] = []
    probabilities_all: list[np.ndarray] = []
    indices_all: list[int] = []
    total_loss = 0.0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            images, targets, sides, levels, indices = batch
            images = images.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            sides = sides.to(device, non_blocking=True)
            levels = levels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(images, sides, levels)
                loss = criterion(logits, targets_device)
            probabilities = torch.softmax(logits.float(), dim=1)
            predictions = probabilities.argmax(dim=1)
            batch_size = len(targets)
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            targets_all.extend(targets.numpy().astype(int).tolist())
            predictions_all.extend(predictions.cpu().numpy().astype(int).tolist())
            probabilities_all.append(probabilities.cpu().numpy())
            indices_all.extend(indices.numpy().astype(int).tolist())
            if batch_index % 50 == 0 or batch_index == len(loader):
                print({"batch": batch_index, "totalBatches": len(loader)})

    targets = np.asarray(targets_all, dtype=np.int64)
    predictions = np.asarray(predictions_all, dtype=np.int64)
    probabilities = np.concatenate(probabilities_all).astype(np.float64)
    indices = np.asarray(indices_all, dtype=np.int64)
    order = np.argsort(indices)
    targets, predictions, probabilities, indices = (
        targets[order],
        predictions[order],
        probabilities[order],
        indices[order],
    )
    if not np.array_equal(indices, np.arange(len(samples))):
        raise RuntimeError("Orden de predicciones inválido.")

    class_names = list(base.CLASS_NAMES)
    frame = samples.copy()
    frame["target_index"] = targets
    frame["target_class"] = [class_names[index] for index in targets]
    frame["predicted_index"] = predictions
    frame["predicted_class"] = [class_names[index] for index in predictions]
    frame["prob_normal_mild"] = probabilities[:, 0]
    frame["prob_moderate"] = probabilities[:, 1]
    frame["prob_severe"] = probabilities[:, 2]
    frame["correct"] = targets == predictions
    frame["checkpoint_sha256"] = context["actual_checkpoint_sha"]
    frame["human_review_required"] = True
    frame["not_clinical_diagnosis"] = True
    frame["official_test_accessed"] = False

    metrics = _metrics(targets, predictions, probabilities)
    metrics["loss"] = float(total_loss / max(total, 1))
    groups: list[dict[str, Any]] = []

    def add_group(group_type: str, group_value: str, group_indices: np.ndarray) -> None:
        value = _metrics(
            targets[group_indices],
            predictions[group_indices],
            probabilities[group_indices],
        )
        groups.append({
            "group_type": group_type,
            "group_value": group_value,
            **value,
            "target_normal_mild": int(np.sum(targets[group_indices] == 0)),
            "target_moderate": int(np.sum(targets[group_indices] == 1)),
            "target_severe": int(np.sum(targets[group_indices] == 2)),
        })

    add_group("overall", "all", np.arange(len(frame)))
    for side in base.SIDES:
        idx = np.flatnonzero(frame["side"].to_numpy() == side)
        add_group("side", side, idx)
    for level in base.LEVELS:
        idx = np.flatnonzero(frame["level"].to_numpy() == level)
        if len(idx):
            add_group("level", level, idx)
    for side in base.SIDES:
        for level in base.LEVELS:
            idx = np.flatnonzero(
                (frame["side"].to_numpy() == side)
                & (frame["level"].to_numpy() == level)
            )
            if len(idx):
                add_group("side_level", f"{side}__{level}", idx)

    confusion = confusion_matrix(targets, predictions, labels=[0, 1, 2])
    confusion_frame = pd.DataFrame(
        confusion,
        columns=[f"predicted_{name}" for name in class_names],
    )
    confusion_frame.insert(0, "target_class", class_names)

    evaluation_root = context["evaluation_root"]
    final_root = context["final_root"]
    paths = {
        "internalTestPredictions": evaluation_root / "internal_test_predictions.csv",
        "internalTestMetrics": evaluation_root / "internal_test_metrics.json",
        "internalTestMetricsByGroup": evaluation_root / "internal_test_metrics_by_group.csv",
        "internalTestConfusionMatrix": evaluation_root / "internal_test_confusion_matrix.csv",
        "finalCheckpoint": final_root / "frozen_subarticular_checkpoint.pt",
        "finalCheckpointManifest": final_root / "final_checkpoint_manifest.json",
        "finalModelCard": final_root / "final_model_card.md",
    }
    _atomic_csv(paths["internalTestPredictions"], frame)
    _atomic_json(paths["internalTestMetrics"], metrics)
    _atomic_csv(paths["internalTestMetricsByGroup"], pd.DataFrame(groups))
    _atomic_csv(paths["internalTestConfusionMatrix"], confusion_frame)

    tmp_checkpoint = Path(str(paths["finalCheckpoint"]) + ".tmp")
    tmp_checkpoint.unlink(missing_ok=True)
    shutil.copyfile(context["checkpoint"], tmp_checkpoint)
    os.replace(tmp_checkpoint, paths["finalCheckpoint"])
    final_sha = sha256_file(paths["finalCheckpoint"])
    if final_sha != context["actual_checkpoint_sha"]:
        raise RuntimeError("Hash del checkpoint final inconsistente.")

    checkpoint_manifest = {
        "schemaVersion": "pfi.frozen-subarticular-checkpoint.v1",
        "ticket": "P10.6-AI",
        "sourceNotebook": 63,
        "evaluationNotebook": 64,
        "task": "subarticular_stenosis_left_right",
        "sequence": "Axial T2",
        "sourceCheckpoint": str(context["checkpoint"]),
        "finalCheckpoint": str(paths["finalCheckpoint"]),
        "sha256": final_sha,
        "bestEpoch": int(context["training"]["bestEpoch"]),
        "trainingRepoSha": context["training"].get("repoSha"),
        "evaluationRepoSha": context["evaluation_repo_sha"],
        "config": context["training"]["config"],
        "classNames": class_names,
        "sideToIndex": checkpoint["sideToIndex"],
        "levelToIndex": checkpoint["levelToIndex"],
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "autonomousDiagnosis": False,
        "commercialUse": False,
        "officialTestAccessed": False,
    }
    _atomic_json(paths["finalCheckpointManifest"], checkpoint_manifest)

    model_card = f"""# Modelo final congelado — estenosis subarticular Axial T2

Modelo de investigación evaluado sobre el conjunto interno sellado en Notebook 64.

- Mejor epoch seleccionado únicamente con validation: `{context['training']['bestEpoch']}`
- Checkpoint SHA-256: `{final_sha}`
- Macro F1 internal test: `{metrics['macro_f1']:.6f}`
- Balanced accuracy: `{metrics['balanced_accuracy']:.6f}`
- Recall moderate: `{metrics['moderate_recall']:.6f}`
- Recall severe: `{metrics['severe_recall']:.6f}`

## Gobernanza y limitaciones

- `humanReviewRequired=true`
- `notClinicalDiagnosis=true`
- `autonomousDiagnosis=false`
- `commercialUse=false`
- `officialTestAccessed=false`
- El internal test no se utilizó para reentrenar, ajustar hiperparámetros, thresholds o checkpoint.
- El clasificador utiliza coordenadas anatómicas del dataset; el puente desde ROI generada por el sistema todavía requiere implementación y validación.
- La salida representa hallazgos degenerativos asociados a estenosis y requiere revisión profesional.
"""
    _atomic_text(paths["finalModelCard"], model_card)

    evaluation = {
        "schemaVersion": "pfi.rsna-subarticular-internal-test-evaluation.v1",
        "ticket": "P10.6-AI",
        "notebook": 64,
        "sourceNotebook": 63,
        "status": "INTERNAL_TEST_EVALUATED_FOR_RESEARCH_EXPORT",
        "evaluationCompleted": True,
        "createdAtUtc": _utc_now(),
        "task": "subarticular_stenosis_left_right",
        "sequence": "Axial T2",
        "data": {
            "rows": len(samples),
            "studies": samples["study_id"].nunique(),
            "classCounts": samples["severity"].value_counts().sort_index().to_dict(),
            "authorizedSourceReadCount": 1,
            "sourceManifestSha256": context["seal_payload"]["manifest"]["sha256"],
            "preparedManifestSha256": sha256_file(context["prepared_manifest"]),
        },
        "checkpoint": {
            "path": str(paths["finalCheckpoint"]),
            "sha256": final_sha,
            "bestEpoch": int(context["training"]["bestEpoch"]),
            "frozenBeforeInternalTest": True,
        },
        "metrics": metrics,
        "processGates": {
            "notebook63Approved": True,
            "checkpointHashVerified": True,
            "checkpointConfigVerified": True,
            "sourceSealVerifiedBeforeOpen": True,
            "authorizedOpenNotebook64": True,
            "singleAuthorizedSourceRead": True,
            "trainInternalStudyIsolation": True,
            "validationInternalStudyIsolation": True,
            "allClassesPresent": True,
            "finiteMetrics": bool(np.isfinite(list(metrics.values())).all()),
            "noTrainingPerformed": True,
            "noHyperparameterAdjustment": True,
            "noThresholdAdjustment": True,
            "noCheckpointReselection": True,
            "officialTestNotAccessed": True,
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
        },
        "metricGatesAppliedToInternalTest": False,
        "acceptanceDecisionBasedOnInternalTest": False,
        "trainingRepoSha": context["training"].get("repoSha"),
        "evaluationRepoSha": context["evaluation_repo_sha"],
        "governance": {
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
            "autonomousDiagnosis": False,
            "commercialUse": False,
            "officialTestAccessed": False,
            "internalTestAccessed": True,
            "internalTestUsedOnlyForFinalEvaluation": True,
            "sourceSealPreservedUnmodified": True,
        },
        "outputSha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    _atomic_json(summary_path, evaluation)
    return evaluation


def final_gate(context: dict[str, Any]) -> dict[str, Any]:
    summary_path = context["evaluation_summary"]
    if not summary_path.is_file():
        raise RuntimeError("Falta el resumen final.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record = json.loads(context["open_record"].read_text(encoding="utf-8"))
    checks = {
        "status": summary.get("status") == "INTERNAL_TEST_EVALUATED_FOR_RESEARCH_EXPORT",
        "completed": summary.get("evaluationCompleted") is True,
        "singleSourceRead": record.get("authorizedSourceReadCount") == 1,
        "checkpointFrozen": summary.get("checkpoint", {}).get("frozenBeforeInternalTest") is True,
        "checkpointHash": summary.get("checkpoint", {}).get("sha256")
        == context["expected_checkpoint_sha"],
        "metricGatesNotApplied": summary.get("metricGatesAppliedToInternalTest") is False,
        "noAcceptanceDecision": summary.get("acceptanceDecisionBasedOnInternalTest") is False,
        "noTraining": summary.get("processGates", {}).get("noTrainingPerformed") is True,
        "noHyperparameterAdjustment": summary.get("processGates", {}).get(
            "noHyperparameterAdjustment"
        ) is True,
        "noThresholdAdjustment": summary.get("processGates", {}).get(
            "noThresholdAdjustment"
        ) is True,
        "noCheckpointReselection": summary.get("processGates", {}).get(
            "noCheckpointReselection"
        ) is True,
        "humanReviewRequired": summary.get("governance", {}).get(
            "humanReviewRequired"
        ) is True,
        "notClinicalDiagnosis": summary.get("governance", {}).get(
            "notClinicalDiagnosis"
        ) is True,
        "officialTestNotAccessed": summary.get("governance", {}).get(
            "officialTestAccessed"
        ) is False,
    }
    if not all(checks.values()):
        raise RuntimeError({"message": "Cierre inválido", "checks": checks})
    return {"status": summary["status"], "metrics": summary["metrics"], "checks": checks}
