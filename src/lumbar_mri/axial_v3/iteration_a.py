"""Executable train/validation-only runner for axial v3 Iteration A."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .audit import audit_raw0_slice, sha256_file, slice_audit_frame, summarize_raw0_by_patient, write_audit_outputs
from .calibration import apply_raw0_threshold
from .guards import assert_no_test_records, require_train_val_only
from .labels import mask_to_class_indices
from .metrics import metrics_from_predictions
from .training import AxialV3Record, open_2d_array, resize_bilinear, resize_nearest, robust_normalize


@dataclass(frozen=True)
class AxialV3AuditConfig:
    RUN_ID: str = os.getenv("PFI_RUN_ID", "axial-v3-iteration-a")
    SEED: int = int(os.getenv("PFI_SEED", "2026"))
    REPO_ROOT: Path = Path(os.getenv("PFI_REPO_ROOT", "."))
    EXTERNAL_ROOT: Path = Path(os.getenv("PFI_EXTERNAL_ROOT", "/content/drive/MyDrive/PFI_MVP"))
    DATASET_ROOT: Path = Path(os.getenv("PFI_DATASET_ROOT", "/content/drive/MyDrive/PFI_MVP"))
    IMAGES_ROOT: Path = Path(os.getenv("AXIAL_IMAGES_ROOT", ""))
    MASKS_ROOT: Path = Path(os.getenv("AXIAL_MASKS_ROOT", ""))
    SPLIT_MANIFEST_PATH: Path = Path(os.getenv("AXIAL_E9_CURATED_SPLIT_CSV", "manifest.csv"))
    V2_CHECKPOINT_PATH: Path = Path(os.getenv("AXIAL_V2_CHECKPOINT_PATH", ""))
    OUTPUT_ROOT: Path = Path(os.getenv("PFI_OUTPUT_ROOT", "outputs")) / "axial_v3" / "iteration_a"
    TARGET_SIZE: tuple[int, int] = (256, 256)
    NUM_CLASSES: int = 6
    BASE_CHANNELS: int = 16
    BATCH_SIZE: int = int(os.getenv("PFI_BATCH_SIZE", "4"))
    NUM_WORKERS: int = int(os.getenv("PFI_NUM_WORKERS", "0"))
    PFI_USE_GOOGLE_DRIVE: bool = os.getenv("PFI_USE_GOOGLE_DRIVE", "0") == "1"
    PFI_DRIVE_ROOT: Path = Path(os.getenv("PFI_DRIVE_ROOT", "/content/drive/MyDrive/PFI_MVP"))
    MAX_PREVIEW_CASES: int = int(os.getenv("PFI_MAX_PREVIEW_CASES", "12"))
    PROBABILITY_HIST_BINS: int = int(os.getenv("PFI_PROBABILITY_HIST_BINS", "30"))
    THRESHOLD_GRID: tuple[float, ...] = tuple(float(v) for v in os.getenv("PFI_THRESHOLD_GRID", "0.3,0.4,0.5,0.6").split(","))
    MARGIN_GRID: tuple[float, ...] = tuple(float(v) for v in os.getenv("PFI_MARGIN_GRID", "0.0,0.05,0.1,0.2").split(","))
    MASK_LABEL_MODE: str = os.getenv("PFI_MASK_LABEL_MODE", "raw")


def validate_storage(config: AxialV3AuditConfig) -> None:
    if config.PFI_USE_GOOGLE_DRIVE and not config.PFI_DRIVE_ROOT.exists():
        raise FileNotFoundError(f"Google Drive root not available: {config.PFI_DRIVE_ROOT}")
    if not config.SPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"split manifest not found: {config.SPLIT_MANIFEST_PATH}")
    if not config.DATASET_ROOT.exists():
        raise FileNotFoundError(f"dataset root not found: {config.DATASET_ROOT}")
    if str(config.V2_CHECKPOINT_PATH) and not config.V2_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"v2 checkpoint not found: {config.V2_CHECKPOINT_PATH}")


def build_records(config: AxialV3AuditConfig) -> list[AxialV3Record]:
    df = pd.read_csv(config.SPLIT_MANIFEST_PATH)
    required = ["image_file_path", "final_label_file_path", "case_id_norm", "split"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    records: list[AxialV3Record] = []
    for _, row in df.iterrows():
        split = str(row["split"]).strip().lower()
        if split == "validation":
            split = "val"
        if split not in {"train", "val"}:
            continue
        image_path = Path(str(row["image_file_path"]))
        mask_path = Path(str(row["final_label_file_path"]))
        if not image_path.is_absolute():
            image_path = config.DATASET_ROOT / image_path
        if not mask_path.is_absolute():
            mask_path = config.DATASET_ROOT / mask_path
        records.append(
            AxialV3Record(
                str(image_path),
                str(mask_path),
                str(row["case_id_norm"]),
                str(row.get("study_id", row["case_id_norm"])),
                image_path.stem,
                split,
                image_path.name,
                mask_path.name,
            )
        )
    assert_no_test_records(records, context="iteration_a")
    return records


def run_structural_audit(config: AxialV3AuditConfig, records: list[AxialV3Record]) -> dict[str, Any]:
    rows = []
    for record in records:
        require_train_val_only([record.split], context="iteration_a.structural")
        image = open_2d_array(Path(record.image_path))
        raw_mask = open_2d_array(Path(record.mask_path))
        indexed_mask = mask_to_class_indices(raw_mask, mode=config.MASK_LABEL_MODE)
        resized_mask = resize_nearest(indexed_mask, config.TARGET_SIZE)
        rows.append(
            audit_raw0_slice(
                mask=resized_mask,
                split=record.split,
                patient_id=record.patientId,
                study_id=record.studyId,
                slice_id=record.sliceId,
                image_shape=config.TARGET_SIZE,
                original_image_shape=image.shape,
                original_mask_shape=raw_mask.shape,
                raw0_value=1,
                background_value=0,
                source_image=record.sourceImage,
                source_mask=record.sourceMask,
            )
        )
    output_dir = config.OUTPUT_ROOT / config.RUN_ID
    summary = write_audit_outputs(rows, output_dir)
    return {"rows": rows, "summary": summary}


def _save_basic_figures(config: AxialV3AuditConfig, rows: list[Any]) -> list[Path]:
    figures = config.OUTPUT_ROOT / config.RUN_ID / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    df = slice_audit_frame(rows)
    created: list[Path] = []
    specs = [
        ("raw0_area_histogram.png", "raw0PixelCount"),
        ("raw0_area_ratio_histogram.png", "raw0AreaRatio"),
        ("raw0_components_histogram.png", "connectedComponents"),
        ("raw0_distance_to_border_histogram.png", "minDistanceToBorder"),
    ]
    for filename, column in specs:
        path = figures / filename
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        plt.figure()
        values.hist(bins=config.PROBABILITY_HIST_BINS)
        plt.title(column)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        created.append(path)
    return created


def run_validation_probability_audit(config: AxialV3AuditConfig, records: list[AxialV3Record]) -> dict[str, Any]:
    if not str(config.V2_CHECKPOINT_PATH):
        return {"status": "skipped", "reason": "AXIAL_V2_CHECKPOINT_PATH not configured"}
    from .architectures import ArchitectureConfig, build_axial_v3_model

    val_records = [record for record in records if record.split == "val"]
    require_train_val_only(["val"], context="iteration_a.probability")
    checkpoint = torch.load(config.V2_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("runId") not in {None, "axial-final-v2"}:
        raise ValueError("checkpoint runId is not compatible with axial-final-v2")
    if checkpoint.get("smokeOnly") is True:
        raise ValueError("smoke checkpoint cannot be used for probability audit")
    model = build_axial_v3_model(ArchitectureConfig())
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    rows: list[dict[str, Any]] = []
    predictions = []
    targets = []
    with torch.inference_mode():
        for record in val_records:
            image = resize_bilinear(robust_normalize(open_2d_array(Path(record.image_path))), config.TARGET_SIZE)
            raw_mask = open_2d_array(Path(record.mask_path))
            target = resize_nearest(mask_to_class_indices(raw_mask, mode=config.MASK_LABEL_MODE), config.TARGET_SIZE)
            probs = torch.softmax(model(torch.from_numpy(image[None, None].astype(np.float32))), dim=1).numpy()[0]
            pred = np.argmax(probs, axis=0)
            raw0_prob = probs[1]
            metrics = metrics_from_predictions(pred, target)["perClass"]["raw_0"]
            rows.append(
                {
                    "split": record.split,
                    "patientId": record.patientId,
                    "studyId": record.studyId,
                    "sliceId": record.sliceId,
                    "raw0GtPresent": bool((target == 1).any()),
                    "raw0PredPresentArgmax": bool((pred == 1).any()),
                    "raw0MaxProbability": float(raw0_prob.max()),
                    "raw0MeanProbability": float(raw0_prob.mean()),
                    "raw0PredictedAreaPixels": int((pred == 1).sum()),
                    "raw0PredictedAreaRatio": float((pred == 1).mean()),
                    "raw0TrueAreaPixels": int((target == 1).sum()),
                    "raw0FalsePositivePixels": metrics["falsePositivePixels"],
                    "raw0FalseNegativePixels": metrics["falseNegativePixels"],
                    "raw0TruePositivePixels": metrics["truePositivePixels"],
                    "raw0Dice": metrics["dice"],
                    "raw0Precision": metrics["precision"],
                    "raw0Recall": metrics["recall"],
                    "raw0PredictedInGtAbsent": metrics["predictedInGtAbsentCases"],
                    "maxProbabilityInsideGt": float(raw0_prob[target == 1].max()) if (target == 1).any() else None,
                    "maxProbabilityOutsideGt": float(raw0_prob[target != 1].max()) if (target != 1).any() else None,
                    "meanProbabilityInsideGt": float(raw0_prob[target == 1].mean()) if (target == 1).any() else None,
                    "meanProbabilityOutsideGt": float(raw0_prob[target != 1].mean()) if (target != 1).any() else None,
                }
            )
            predictions.append(pred)
            targets.append(target)
    output_dir = config.OUTPUT_ROOT / config.RUN_ID
    tables = output_dir / "tables"
    metrics_dir = output_dir / "metrics"
    tables.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    probability_csv = tables / "raw0_validation_probability_audit.csv"
    pd.DataFrame(rows).to_csv(probability_csv, index=False)
    grid_rows = []
    if predictions:
        stacked_targets = np.stack(targets)
        for threshold in config.THRESHOLD_GRID:
            for margin in config.MARGIN_GRID:
                gated = []
                for record in val_records:
                    image = resize_bilinear(robust_normalize(open_2d_array(Path(record.image_path))), config.TARGET_SIZE)
                    probs = torch.softmax(model(torch.from_numpy(image[None, None].astype(np.float32))), dim=1).numpy()[0]
                    gated.append(apply_raw0_threshold(probs, min_probability=threshold, min_margin=margin))
                metric = metrics_from_predictions(np.stack(gated), stacked_targets)
                grid_rows.append({"minProbability": threshold, "minMargin": margin, **{k: v for k, v in metric.items() if k != "perClass"}})
    grid_csv = tables / "raw0_threshold_margin_grid.csv"
    pd.DataFrame(grid_rows).to_csv(grid_csv, index=False)
    summary = {
        "status": "completed",
        "validationSlices": len(val_records),
        "presenceScore": "raw0MaxProbability",
        "checkpointSha256": sha256_file(config.V2_CHECKPOINT_PATH),
        "artifacts": {
            "probabilityCsv": str(probability_csv),
            "thresholdGridCsv": str(grid_csv),
            "probabilityCsvSha256": sha256_file(probability_csv),
            "thresholdGridCsvSha256": sha256_file(grid_csv),
        },
    }
    summary_path = metrics_dir / "raw0_validation_probability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_iteration_a(config: AxialV3AuditConfig | None = None) -> dict[str, Any]:
    cfg = config or AxialV3AuditConfig()
    validate_storage(cfg)
    require_train_val_only(["train", "val"], context="iteration_a")
    records = build_records(cfg)
    structural = run_structural_audit(cfg, records)
    _save_basic_figures(cfg, structural["rows"])
    probability = run_validation_probability_audit(cfg, records)
    report_dir = cfg.OUTPUT_ROOT / cfg.RUN_ID / "reports"
    manifests_dir = cfg.OUTPUT_ROOT / cfg.RUN_ID / "manifests"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    config_path = manifests_dir / "iteration_a_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2, default=str, sort_keys=True), encoding="utf-8")
    result = {
        "status": "ready_outputs_created",
        "runId": cfg.RUN_ID,
        "recordCount": len(records),
        "structuralSummary": structural["summary"],
        "probabilitySummary": probability,
        "configSha256": sha256_file(config_path),
    }
    (report_dir / "iteration_a_report.json").write_text(json.dumps(result, indent=2, default=str, sort_keys=True), encoding="utf-8")
    (report_dir / "iteration_a_report.md").write_text("# Iteration A report\n\nGenerated from train/validation only.\n", encoding="utf-8")
    return result
