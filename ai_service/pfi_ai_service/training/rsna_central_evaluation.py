from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import timm

from pfi_ai_service.training.rsna_central_training import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    CachedDataset,
    TrainConfig,
    build_cache,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_internal_test_manifest(split_root: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = split_root / "internal_test_manifest.csv"
    summary_path = split_root / "split_summary.json"
    required = [manifest_path, summary_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Faltan artefactos del Notebook 54:\n- " + "\n- ".join(missing))
    manifest = pd.read_csv(manifest_path, dtype={"study_id": str, "series_id": str})
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("leakageDetected") is not False:
        raise RuntimeError("El split reporta fuga.")
    if summary.get("officialTestAccessed") is not False:
        raise RuntimeError("El split reporta acceso al test oficial.")
    if set(manifest["split"].astype(str)) != {"internal_test"}:
        raise RuntimeError("El manifest no contiene exclusivamente internal_test.")
    return manifest, summary


def attach_coordinates_safe(
    manifest: pd.DataFrame,
    local_root: Path,
    report_path: Path,
    max_missing_rate: float = 0.02,
) -> pd.DataFrame:
    coordinates_path = local_root / "train_label_coordinates.csv"
    coordinates = pd.read_csv(
        coordinates_path,
        dtype={"study_id": str, "series_id": str},
    )
    rows = coordinates.loc[
        coordinates["condition"].astype(str).str.strip().eq("Spinal Canal Stenosis")
    ].copy()
    for column in ("instance_number", "x", "y"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = (
        rows.sort_values(["study_id", "series_id", "level", "instance_number"])
        .drop_duplicates(["study_id", "series_id", "level"], keep="first")
    )
    frame = manifest.copy()
    frame["study_id"] = frame["study_id"].astype(str)
    frame["series_id"] = frame["series_id"].astype(str)
    samples = frame.merge(
        rows[["study_id", "series_id", "level", "instance_number", "x", "y"]],
        on=["study_id", "series_id", "level"],
        how="left",
        validate="one_to_one",
    )
    missing_mask = samples[["instance_number", "x", "y"]].isna().any(axis=1)
    missing = samples.loc[
        missing_mask,
        ["study_id", "series_id", "level", "severity", "split"],
    ].copy()
    missing["reason"] = "missing_rsna_label_coordinate"
    missing["excludedFromEvaluation"] = True
    report_path.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(report_path, index=False)
    missing_rate = float(missing_mask.mean())
    if missing_rate > max_missing_rate:
        raise RuntimeError(
            f"Se excluyó {missing_rate:.2%} por falta de coordenadas; "
            f"supera el límite {max_missing_rate:.2%}."
        )
    usable = samples.loc[~missing_mask].copy()
    usable["instance_number"] = usable["instance_number"].astype(int)
    usable["severity_code"] = usable["severity"].map(CLASS_TO_INDEX)
    if usable["severity_code"].isna().any():
        unknown = usable.loc[usable["severity_code"].isna(), "severity"].unique().tolist()
        raise RuntimeError(f"Severidades no reconocidas: {unknown}")
    usable["severity_code"] = usable["severity_code"].astype(int)
    usable["local_series_path"] = usable.apply(
        lambda row: str(local_root / "train_images" / row["study_id"] / row["series_id"]),
        axis=1,
    )
    absent = usable.loc[
        ~usable["local_series_path"].map(lambda value: Path(value).is_dir())
    ]
    if not absent.empty:
        raise RuntimeError(f"Faltan {len(absent)} series locales.")
    print({
        "manifestRows": int(len(samples)),
        "usableSamples": int(len(usable)),
        "missingCoordinates": int(len(missing)),
        "missingCoordinateRate": round(missing_rate, 4),
        "missingCoordinateReport": str(report_path),
    })
    return usable.reset_index(drop=True)


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = ["modelName", "modelStateDict", "classNames", "config", "governance"]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(f"Checkpoint incompleto: {missing}")
    if list(checkpoint["classNames"]) != CLASS_NAMES:
        raise RuntimeError("Las clases del checkpoint no coinciden.")
    if checkpoint["governance"].get("officialTestAccessed") is not False:
        raise RuntimeError("Checkpoint no apto: acceso al test oficial.")
    model = timm.create_model(
        checkpoint["modelName"],
        pretrained=False,
        in_chans=3,
        num_classes=len(CLASS_NAMES),
    )
    model.load_state_dict(checkpoint["modelStateDict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _specificity_per_class(cm: np.ndarray) -> list[float]:
    total = cm.sum()
    values = []
    for index in range(len(CLASS_NAMES)):
        tp = cm[index, index]
        fn = cm[index, :].sum() - tp
        fp = cm[:, index].sum() - tp
        tn = total - tp - fn - fp
        values.append(float(tn / (tn + fp)) if (tn + fp) else 0.0)
    return values


def _ece(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        value += float(mask.mean()) * abs(
            float(correctness[mask].mean()) - float(confidence[mask].mean())
        )
    return value


def evaluate(
    model,
    samples: pd.DataFrame,
    cache_root: Path,
    output_root: Path,
    cfg: TrainConfig,
    device: torch.device,
) -> dict:
    dataset = CachedDataset(samples, cache_root, "internal_test", False)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    logits_parts = []
    targets_list = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="internal test"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
            logits_parts.append(logits.float().cpu().numpy())
            targets_list.extend(labels.numpy().tolist())
    logits = np.concatenate(logits_parts, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    targets = np.asarray(targets_list, dtype=int)
    predictions = probabilities.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    cm = confusion_matrix(targets, predictions, labels=list(range(len(CLASS_NAMES))))
    specificity = _specificity_per_class(cm)
    try:
        auc = roc_auc_score(
            targets,
            probabilities,
            multi_class="ovr",
            average=None,
            labels=list(range(len(CLASS_NAMES))),
        )
        macro_auc = float(np.mean(auc))
    except ValueError:
        auc = np.full(len(CLASS_NAMES), np.nan)
        macro_auc = float("nan")

    metrics = {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "severe_recall": float(
            recall_score(targets, predictions, labels=[2], average="macro", zero_division=0)
        ),
        "weighted_log_loss": float(
            log_loss(
                targets,
                probabilities,
                labels=list(range(len(CLASS_NAMES))),
                sample_weight=np.where(targets == 0, 1.0, np.where(targets == 1, 2.0, 4.0)),
            )
        ),
        "macro_roc_auc_ovr": macro_auc,
        "expected_calibration_error": _ece(targets, probabilities),
        "n_samples": int(len(targets)),
        "n_studies": int(samples["study_id"].nunique()),
    }

    predictions_frame = samples[
        ["study_id", "series_id", "level", "severity", "severity_code"]
    ].copy()
    predictions_frame = predictions_frame.rename(
        columns={"severity": "true_label", "severity_code": "true_code"}
    )
    predictions_frame["predicted_code"] = predictions
    predictions_frame["predicted_label"] = [CLASS_NAMES[index] for index in predictions]
    predictions_frame["prob_normal_mild"] = probabilities[:, 0]
    predictions_frame["prob_moderate"] = probabilities[:, 1]
    predictions_frame["prob_severe"] = probabilities[:, 2]
    predictions_frame["correct"] = predictions_frame["true_code"].eq(
        predictions_frame["predicted_code"]
    )
    predictions_frame.to_csv(output_root / "internal_test_predictions.csv", index=False)

    pd.DataFrame({
        "class": CLASS_NAMES,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "support": support.astype(int),
        "roc_auc_ovr": auc,
    }).to_csv(output_root / "metrics_by_class.csv", index=False)

    pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    ).to_csv(output_root / "confusion_matrix.csv")

    level_rows = []
    for level, group in predictions_frame.groupby("level", sort=False):
        y_true = group["true_code"].to_numpy()
        y_pred = group["predicted_code"].to_numpy()
        level_rows.append({
            "level": level,
            "n": int(len(group)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "severe_recall": float(
                recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
            ),
        })
    pd.DataFrame(level_rows).to_csv(output_root / "metrics_by_level.csv", index=False)

    predictions_frame.loc[
        predictions_frame["true_code"].eq(2)
        & ~predictions_frame["predicted_code"].eq(2)
    ].sort_values("prob_severe").to_csv(
        output_root / "severe_false_negatives.csv",
        index=False,
    )

    calibration_rows = []
    for index, name in enumerate(CLASS_NAMES):
        observed, predicted = calibration_curve(
            (targets == index).astype(int),
            probabilities[:, index],
            n_bins=10,
            strategy="uniform",
        )
        for obs, pred in zip(observed, predicted):
            calibration_rows.append({
                "class": name,
                "mean_predicted_probability": float(pred),
                "observed_fraction": float(obs),
            })
    pd.DataFrame(calibration_rows).to_csv(
        output_root / "calibration_curve.csv",
        index=False,
    )

    with (output_root / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    return metrics


def export_final_artifact(
    checkpoint_path: Path,
    destination: Path,
    evaluation_metrics: dict,
    evaluation_hashes: dict,
    approved: bool,
) -> Path | None:
    if not approved:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    artifact = {
        **checkpoint,
        "schemaVersion": "pfi.rsna-central-stenosis-model.v1",
        "artifactType": "image_classifier",
        "findingType": "spinal_canal_stenosis",
        "sourcePlane": "sagittal",
        "sourceSequence": "Sagittal T2/STIR",
        "evaluation": {
            "split": "internal_test",
            "metrics": evaluation_metrics,
            "outputHashes": evaluation_hashes,
        },
        "governance": {
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
            "predictionStatus": "pending_review",
            "officialTestAccessed": False,
            "measurementsAreSecondaryEvidence": True,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    return destination
