"""Train/validation-only runner primitives for axial v3 Iteration B."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .architectures import ArchitectureConfig, build_axial_v3_model
from .guards import assert_no_test_records, require_train_val_only
from .labels import CLASS_INDEX_TO_NAME, mask_to_class_indices
from .losses import combined_segmentation_loss, v2_baseline_segmentation_loss
from .low_cost import apply_raw0_effective_weight, cap_class_weight_ratio
from .metrics import metrics_from_predictions


@dataclass(frozen=True)
class AxialV3Record:
    image_path: str
    mask_path: str
    patientId: str
    studyId: str
    sliceId: str
    split: str
    sourceImage: str
    sourceMask: str


@dataclass(frozen=True)
class AxialV3TrainConfig:
    RUN_ID: str = "axial-v3-B0"
    SEED: int = 2026
    SPLIT_MANIFEST_PATH: Path = Path("manifest.csv")
    DATASET_ROOT: Path = Path(".")
    OUTPUT_ROOT: Path = Path("outputs") / "axial_v3" / "iteration_b"
    TARGET_SIZE: tuple[int, int] = (256, 256)
    NUM_CLASSES: int = 6
    BASE_CHANNELS: int = 16
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 0
    LEARNING_RATE: float = 0.0008
    WEIGHT_DECAY: float = 0.0001
    MAX_EPOCHS: int = 80
    EARLY_STOPPING_PATIENCE: int = 12
    RAW0_WEIGHT_BOOST: float = 1.0
    MAX_CLASS_WEIGHT_RATIO: float | None = None
    MONITOR_METRIC: str = "dice_macro_foreground"
    AMP: bool = True
    GRAD_CLIP_NORM: float = 1.0
    MASK_LABEL_MODE: str = "raw"
    IMAGE_COL: str = "image_file_path"
    MASK_COL: str = "final_label_file_path"
    PATIENT_COL: str = "case_id_norm"
    STUDY_COL: str = "case_id_norm"
    SPLIT_COL: str = "split"
    SLICE_COL: str | None = None
    PRESENCE_HEAD_ENABLED: bool = False
    LAMBDA_PRESENCE: float = 0.0
    RAW0_BALANCED_SAMPLER_ENABLED: bool = False
    POSITIVE_FRACTION: float = 0.5
    SMOKE_MAX_RECORDS_PER_SPLIT: int = 8


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def slice_id_from_paths(image_path: Path, mask_path: Path) -> str:
    image_stem = image_path.stem.replace("_mask", "")
    mask_stem = mask_path.stem.replace("_mask", "")
    return image_stem if image_stem == mask_stem else f"{image_stem}|{mask_stem}"


def build_train_val_records(config: AxialV3TrainConfig) -> list[AxialV3Record]:
    df = pd.read_csv(config.SPLIT_MANIFEST_PATH)
    required = [config.IMAGE_COL, config.MASK_COL, config.PATIENT_COL, config.SPLIT_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    records: list[AxialV3Record] = []
    for _, row in df.iterrows():
        split = str(row[config.SPLIT_COL]).strip().lower()
        if split == "validation":
            split = "val"
        if split not in {"train", "val"}:
            continue
        image_path = resolve_path(str(row[config.IMAGE_COL]), config.DATASET_ROOT)
        mask_path = resolve_path(str(row[config.MASK_COL]), config.DATASET_ROOT)
        study = str(row[config.STUDY_COL]) if config.STUDY_COL in df.columns else str(row[config.PATIENT_COL])
        slice_id = str(row[config.SLICE_COL]) if config.SLICE_COL and config.SLICE_COL in df.columns else slice_id_from_paths(image_path, mask_path)
        records.append(
            AxialV3Record(
                image_path=str(image_path),
                mask_path=str(mask_path),
                patientId=str(row[config.PATIENT_COL]),
                studyId=study,
                sliceId=slice_id,
                split=split,
                sourceImage=Path(str(row[config.IMAGE_COL])).name,
                sourceMask=Path(str(row[config.MASK_COL])).name,
            )
        )
    assert_no_test_records(records, context="build_train_val_records")
    validate_split_integrity(records)
    return records


def validate_split_integrity(records: Iterable[AxialV3Record]) -> dict[str, bool]:
    by_patient: dict[str, set[str]] = {}
    by_study: dict[str, set[str]] = {}
    for record in records:
        by_patient.setdefault(record.patientId, set()).add(record.split)
        by_study.setdefault(record.studyId, set()).add(record.split)
    patient_ok = all(len(splits) == 1 for splits in by_patient.values())
    study_ok = all(len(splits) == 1 for splits in by_study.values())
    if not patient_ok:
        raise ValueError("patientId leakage across train/val")
    if not study_ok:
        raise ValueError("studyId leakage across train/val")
    return {"patientHeldout": True, "studyHeldout": True}


def open_2d_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        array = np.asarray(Image.open(path))
    if array.ndim > 2:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"expected 2D image/mask at {path}, got {array.shape}")
    return array


def resize_nearest(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array)
    return np.asarray(image.resize((size[1], size[0]), resample=Image.Resampling.NEAREST))


def resize_bilinear(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array.astype(np.float32))
    return np.asarray(image.resize((size[1], size[0]), resample=Image.Resampling.BILINEAR), dtype=np.float32)


def robust_normalize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0, 1).astype(np.float32)


class AxialV3SegmentationDataset(Dataset):
    def __init__(self, records: list[AxialV3Record], config: AxialV3TrainConfig, split: str) -> None:
        require_train_val_only([split], context="AxialV3SegmentationDataset")
        self.records = [record for record in records if record.split == split]
        self.config = config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = resize_bilinear(robust_normalize(open_2d_array(Path(record.image_path))), self.config.TARGET_SIZE)
        raw_mask = resize_nearest(open_2d_array(Path(record.mask_path)), self.config.TARGET_SIZE)
        mask = mask_to_class_indices(raw_mask, mode=self.config.MASK_LABEL_MODE)  # explicit mode, never inferred
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "patientId": record.patientId,
            "studyId": record.studyId,
            "sliceId": record.sliceId,
        }


def _limit_smoke(records: list[AxialV3Record], config: AxialV3TrainConfig) -> list[AxialV3Record]:
    limited: list[AxialV3Record] = []
    for split in ("train", "val"):
        limited.extend([record for record in records if record.split == split][: config.SMOKE_MAX_RECORDS_PER_SPLIT])
    return limited


def build_raw0_balanced_sampler(dataset: AxialV3SegmentationDataset, *, positive_fraction: float, seed: int) -> WeightedRandomSampler:
    if not 0 < positive_fraction < 1:
        raise ValueError("positive_fraction must be between 0 and 1")
    positives: list[bool] = []
    for record in dataset.records:
        mask = mask_to_class_indices(open_2d_array(Path(record.mask_path)), mode=dataset.config.MASK_LABEL_MODE)
        positives.append(bool((mask == 1).any()))
    pos_count = sum(positives)
    neg_count = len(positives) - pos_count
    if pos_count == 0 or neg_count == 0:
        raise ValueError("balanced sampler requires both raw_0 positive and negative train slices")
    weights = [positive_fraction / pos_count if item else (1 - positive_fraction) / neg_count for item in positives]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def build_train_val_loaders(config: AxialV3TrainConfig, *, smoke: bool = False) -> tuple[dict[str, DataLoader], list[AxialV3Record]]:
    records = build_train_val_records(config)
    if smoke:
        records = _limit_smoke(records, config)
    train_ds = AxialV3SegmentationDataset(records, config, "train")
    val_ds = AxialV3SegmentationDataset(records, config, "val")
    sampler = None
    shuffle = True
    if config.RAW0_BALANCED_SAMPLER_ENABLED:
        sampler = build_raw0_balanced_sampler(train_ds, positive_fraction=config.POSITIVE_FRACTION, seed=config.SEED)
        shuffle = False
    return {
        "train": DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=shuffle, sampler=sampler, num_workers=config.NUM_WORKERS),
        "val": DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS),
    }, records


def compute_class_weights(records: list[AxialV3Record], config: AxialV3TrainConfig) -> tuple[torch.Tensor, dict[str, Any]]:
    train_records = [record for record in records if record.split == "train"]
    counts = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    for record in train_records:
        mask = mask_to_class_indices(open_2d_array(Path(record.mask_path)), mode=config.MASK_LABEL_MODE)
        values, value_counts = np.unique(mask, return_counts=True)
        for value, count in zip(values, value_counts):
            counts[int(value)] += int(count)
    safe_counts = np.maximum(counts, 1)
    raw_weights = safe_counts.sum() / (config.NUM_CLASSES * safe_counts)
    adjusted = apply_raw0_effective_weight(raw_weights, multiplier=config.RAW0_WEIGHT_BOOST)
    capped = cap_class_weight_ratio(adjusted, max_ratio=config.MAX_CLASS_WEIGHT_RATIO)
    normalized = capped / capped.mean()
    report = {
        "pixelCounts": {CLASS_INDEX_TO_NAME[i]: int(counts[i]) for i in range(config.NUM_CLASSES)},
        "rawWeights": raw_weights.tolist(),
        "adjustedWeights": adjusted.tolist(),
        "normalizedWeights": normalized.tolist(),
        "raw0Multiplier": config.RAW0_WEIGHT_BOOST,
        "maxClassWeightRatio": config.MAX_CLASS_WEIGHT_RATIO,
    }
    return torch.tensor(normalized, dtype=torch.float32), report


def build_optimizer(model: nn.Module, config: AxialV3TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)


def _extract_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    return output["segmentation_logits"] if isinstance(output, dict) else output


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, class_weights: torch.Tensor, config: AxialV3TrainConfig, device: torch.device) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        logits = _extract_logits(output)
        loss, _, _ = v2_baseline_segmentation_loss(logits, masks, class_weights=class_weights.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(losses)) if losses else math.inf}


def evaluate_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            logits = _extract_logits(model(batch["image"].to(device)))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(batch["mask"].cpu().numpy())
    if not predictions:
        raise ValueError("validation loader is empty")
    return metrics_from_predictions(np.concatenate(predictions), np.concatenate(targets))


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)
    return sha256_file(path)


def runtime_shape_smoke(model: nn.Module, config: AxialV3TrainConfig, device: torch.device) -> dict[str, Any]:
    model.eval()
    channels = 1
    with torch.inference_mode():
        output = _extract_logits(model(torch.randn(1, channels, *config.TARGET_SIZE, device=device)))
    return {"shape": list(output.shape), "finite": bool(torch.isfinite(output).all().item())}


def run_training(config: AxialV3TrainConfig, *, smoke: bool = False) -> dict[str, Any]:
    set_deterministic_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, records = build_train_val_loaders(config, smoke=smoke)
    class_weights, weight_report = compute_class_weights(records, config)
    model = build_axial_v3_model(ArchitectureConfig(base_channels=config.BASE_CHANNELS, presence_head=config.PRESENCE_HEAD_ENABLED)).to(device)
    optimizer = build_optimizer(model, config)
    max_epochs = min(config.MAX_EPOCHS, 2) if smoke else config.MAX_EPOCHS
    best_metric = -math.inf
    selected_epoch = 0
    patience_left = config.EARLY_STOPPING_PATIENCE
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, class_weights, config, device)
        val_metrics = evaluate_validation(model, loaders["val"], device)
        monitor = val_metrics.get(config.MONITOR_METRIC)
        improved = monitor is not None and float(monitor) > best_metric
        if improved:
            best_metric = float(monitor)
            selected_epoch = epoch
            patience_left = config.EARLY_STOPPING_PATIENCE
            checkpoint_path = config.OUTPUT_ROOT / config.RUN_ID / "checkpoints" / "best_checkpoint.pt"
            checkpoint_sha = save_checkpoint_atomic(
                checkpoint_path,
                {
                    "model_state_dict": model.state_dict(),
                    "runId": config.RUN_ID,
                    "smokeOnly": smoke,
                    "epoch": epoch,
                    "bestValidationMetric": best_metric,
                    "config": asdict(config),
                },
            )
        else:
            patience_left -= 1
        history.append({"epoch": epoch, "trainLoss": train_metrics["loss"], **{f"val_{k}": v for k, v in val_metrics.items() if k != "perClass"}})
        if patience_left <= 0:
            break
    duration = time.time() - started
    final_metrics = evaluate_validation(model, loaders["val"], device)
    return {
        "runId": config.RUN_ID,
        "smokeOnly": smoke,
        "selectedEpoch": selected_epoch,
        "monitorMetric": config.MONITOR_METRIC,
        "bestValidationMetric": best_metric,
        "validationMetrics": final_metrics,
        "classWeightReport": weight_report,
        "history": history,
        "durationSeconds": duration,
        "checkpointPath": str(locals().get("checkpoint_path", "")),
        "checkpointSha256": locals().get("checkpoint_sha", ""),
        "runtimeSmoke": runtime_shape_smoke(model, config, device),
    }
