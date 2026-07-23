"""Train/validation-only runner primitives for axial v3 Iteration B."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .architectures import ArchitectureConfig, build_axial_v3_model
from .calibration import apply_raw0_slice_presence_gate, apply_raw0_threshold
from .guards import assert_no_test_records, require_train_val_only
from .labels import CLASS_INDEX_TO_NAME, mask_to_class_indices
from .losses import build_segmentation_loss
from .low_cost import apply_raw0_effective_weight, cap_class_weight_ratio
from .metrics import metrics_from_predictions
from .presence import presence_targets, total_loss_with_presence
from .registry import ExperimentRegistryRow, upsert_registry_row


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
    RUN_MODE: str = "preflight"
    RUN_ID: str = "axial-v3-B0"
    EXPERIMENT_ID: str = "B0"
    EXPERIMENT_TYPE: str = "B0"
    SEED: int = 2026
    REPO_ROOT: Path = Path(".")
    SPLIT_MANIFEST_PATH: Path = Path("manifest.csv")
    DATASET_ROOT: Path = Path(".")
    OUTPUT_ROOT: Path = Path("outputs") / "axial_v3" / "iteration_b"
    RESUME_ROOT: Path = Path("outputs") / "axial_v3" / "iteration_b" / "resume"
    REGISTRY_PATH: Path = Path("outputs") / "axial_v3" / "iteration_b" / "experiment_registry.csv"
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
    LOSS_NAME: str = "baseline_v2"
    TVERSKY_ALPHA: float = 0.7
    TVERSKY_BETA: float = 0.3
    FOCAL_GAMMA: float = 1.333
    RAW0_TVERSKY_WEIGHT: float = 0.0
    RAW0_FP_PENALTY_WEIGHT: float = 0.0
    MIN_PROBABILITY: float | None = None
    MIN_MARGIN: float | None = None
    PRESENCE_HEAD_ENABLED: bool = False
    LAMBDA_PRESENCE: float = 0.0
    PRESENCE_THRESHOLD: float = 0.5
    RAW0_BALANCED_SAMPLER_ENABLED: bool = False
    POSITIVE_FRACTION: float = 0.5
    PARENT_EXPERIMENT_ID: str | None = None
    PARENT_RUN_ID: str | None = None
    SMOKE_MAX_RECORDS_PER_SPLIT: int = 8
    MAX_WEIGHT_RECORDS: int = 256
    RESUME_MODE: str = "auto"
    SAVE_VALIDATION_PROBABILITIES: bool = False
    ENABLE_TRAIN_AUGMENTATION: bool = True
    ENABLE_SPATIAL_AUGMENTATION: bool = False
    ENABLE_HORIZONTAL_FLIP: bool = False
    INTENSITY_JITTER_PROBABILITY: float = 0.25
    INTENSITY_SCALE_MIN: float = 0.95
    INTENSITY_SCALE_MAX: float = 1.05
    INTENSITY_SHIFT_MIN: float = -0.03
    INTENSITY_SHIFT_MAX: float = 0.03

    @classmethod
    def from_env(cls) -> "AxialV3TrainConfig":
        def env_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "y"}

        return cls(
            RUN_MODE=os.getenv("RUN_MODE", "preflight"),
            RUN_ID=os.getenv("PFI_RUN_ID", "axial-v3-B0"),
            EXPERIMENT_ID=os.getenv("PFI_AXIAL_V3_EXPERIMENT_ID", "B0"),
            EXPERIMENT_TYPE=os.getenv("PFI_AXIAL_V3_EXPERIMENT_TYPE", "B0"),
            SEED=int(os.getenv("PFI_SEED", "2026")),
            REPO_ROOT=Path(os.getenv("PFI_REPO_ROOT", ".")),
            DATASET_ROOT=Path(os.getenv("PFI_DATASET_ROOT", ".")),
            SPLIT_MANIFEST_PATH=Path(os.getenv("AXIAL_E9_CURATED_SPLIT_CSV", "manifest.csv")),
            OUTPUT_ROOT=Path(os.getenv("PFI_OUTPUT_ROOT", "outputs")) / "axial_v3" / "iteration_b",
            RESUME_ROOT=Path(os.getenv("PFI_RESUME_ROOT", "outputs")) / "axial_v3" / "iteration_b" / "resume",
            REGISTRY_PATH=Path(os.getenv("PFI_REGISTRY_PATH", str(Path("outputs") / "axial_v3" / "iteration_b" / "experiment_registry.csv"))),
            BATCH_SIZE=int(os.getenv("PFI_BATCH_SIZE", "8")),
            NUM_WORKERS=int(os.getenv("PFI_NUM_WORKERS", "0")),
            LEARNING_RATE=float(os.getenv("PFI_LR", "0.0008")),
            WEIGHT_DECAY=float(os.getenv("PFI_WEIGHT_DECAY", "0.0001")),
            MAX_EPOCHS=int(os.getenv("PFI_MAX_EPOCHS", "80")),
            EARLY_STOPPING_PATIENCE=int(os.getenv("PFI_EARLY_STOP_PATIENCE", "12")),
            RAW0_WEIGHT_BOOST=float(os.getenv("AXIAL_RAW0_WEIGHT_BOOST", "1.0")),
            MAX_CLASS_WEIGHT_RATIO=float(os.getenv("AXIAL_MAX_CLASS_WEIGHT_RATIO")) if os.getenv("AXIAL_MAX_CLASS_WEIGHT_RATIO") else None,
            MONITOR_METRIC=os.getenv("AXIAL_MONITOR_METRIC", "dice_macro_foreground"),
            AMP=env_bool("PFI_USE_AMP", True),
            GRAD_CLIP_NORM=float(os.getenv("PFI_GRADIENT_CLIP_NORM", "1.0")),
            MASK_LABEL_MODE=os.getenv("PFI_MASK_LABEL_MODE", "raw"),
            IMAGE_COL=os.getenv("AXIAL_IMAGE_COL", "image_file_path"),
            MASK_COL=os.getenv("AXIAL_MASK_COL", "final_label_file_path"),
            PATIENT_COL=os.getenv("AXIAL_PATIENT_COL", "case_id_norm"),
            STUDY_COL=os.getenv("AXIAL_STUDY_COL", "case_id_norm"),
            SPLIT_COL=os.getenv("AXIAL_SPLIT_COL", "split"),
            SLICE_COL=os.getenv("AXIAL_SLICE_COL") or None,
            RESUME_MODE=os.getenv("RESUME_MODE", "auto"),
            PARENT_EXPERIMENT_ID=os.getenv("PFI_PARENT_EXPERIMENT_ID") or None,
            PARENT_RUN_ID=os.getenv("PFI_PARENT_RUN_ID") or None,
        )


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
    if path.is_absolute():
        return path
    for candidate in (root / path, root / path.name):
        if candidate.exists():
            return candidate
    return root / path


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
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"missing axial image/mask: {image_path} / {mask_path}")
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
    source = np.asarray(array)
    image = Image.fromarray(source.astype(np.uint8) if source.dtype == np.int64 else source)
    return np.asarray(image.resize((size[1], size[0]), resample=Image.Resampling.NEAREST)).astype(source.dtype, copy=False)


def resize_bilinear(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array.astype(np.float32))
    return np.asarray(image.resize((size[1], size[0]), resample=Image.Resampling.BILINEAR), dtype=np.float32)


def robust_normalize(array: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0, 1).astype(np.float32)


def maybe_augment(image: np.ndarray, mask: np.ndarray, rng: random.Random, config: AxialV3TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    if not config.ENABLE_TRAIN_AUGMENTATION:
        return image, mask
    if config.ENABLE_SPATIAL_AUGMENTATION and config.ENABLE_HORIZONTAL_FLIP and rng.random() < 0.5:
        image, mask = np.flip(image, 1).copy(), np.flip(mask, 1).copy()
    if rng.random() < config.INTENSITY_JITTER_PROBABILITY:
        image = np.clip(
            image * rng.uniform(config.INTENSITY_SCALE_MIN, config.INTENSITY_SCALE_MAX)
            + rng.uniform(config.INTENSITY_SHIFT_MIN, config.INTENSITY_SHIFT_MAX),
            0,
            1,
        ).astype(np.float32)
    return image, mask


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
        if self.config.ENABLE_TRAIN_AUGMENTATION and record.split == "train":
            seed_material = f"{self.config.SEED}|{index}|{record.sliceId}"
            seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
            image, mask = maybe_augment(image, mask, random.Random(seed), self.config)
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "patientId": record.patientId,
            "studyId": record.studyId,
            "sliceId": record.sliceId,
        }


def _limit_smoke(records: list[AxialV3Record], config: AxialV3TrainConfig) -> list[AxialV3Record]:
    if config.RAW0_BALANCED_SAMPLER_ENABLED:
        selected: list[AxialV3Record] = []
        for split in ("train", "val"):
            split_records = [record for record in records if record.split == split]
            positive = []
            negative = []
            for record in split_records:
                mask = mask_to_class_indices(open_2d_array(Path(record.mask_path)), mode=config.MASK_LABEL_MODE)
                (positive if bool((mask == 1).any()) else negative).append(record)
            selected.extend(positive[:1] + negative[:1])
            selected.extend(split_records[: max(0, config.SMOKE_MAX_RECORDS_PER_SPLIT - 2)])
        return list(dict.fromkeys(selected))
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
    train_records = [record for record in records if record.split == "train"][: config.MAX_WEIGHT_RECORDS]
    counts = np.ones(config.NUM_CLASSES, dtype=np.float64)
    image_presence = np.zeros(config.NUM_CLASSES, dtype=np.int64)
    for record in train_records:
        original = open_2d_array(Path(record.mask_path))
        resized = resize_nearest(original, config.TARGET_SIZE)
        mask = mask_to_class_indices(resized, mode=config.MASK_LABEL_MODE)
        values, value_counts = np.unique(mask, return_counts=True)
        present = np.zeros(config.NUM_CLASSES, dtype=np.int64)
        for value, count in zip(values, value_counts):
            counts[int(value)] += int(count)
            present[int(value)] = 1
        image_presence += present
    freq = counts / counts.sum()
    raw_weights = 1.0 / np.sqrt(freq)
    raw_weights = raw_weights / raw_weights.mean()
    adjusted = apply_raw0_effective_weight(raw_weights, multiplier=config.RAW0_WEIGHT_BOOST)
    capped = cap_class_weight_ratio(adjusted, max_ratio=config.MAX_CLASS_WEIGHT_RATIO)
    normalized = capped / capped.mean()
    report = {
        "trainSamplesUsed": len(train_records),
        "pixelCounts": {CLASS_INDEX_TO_NAME[i]: int(counts[i]) for i in range(config.NUM_CLASSES)},
        "imagePresence": {CLASS_INDEX_TO_NAME[i]: int(image_presence[i]) for i in range(config.NUM_CLASSES)},
        "baseWeights": {CLASS_INDEX_TO_NAME[i]: float(raw_weights[i]) for i in range(config.NUM_CLASSES)},
        "adjustedWeights": {CLASS_INDEX_TO_NAME[i]: float(adjusted[i]) for i in range(config.NUM_CLASSES)},
        "normalizedWeights": {CLASS_INDEX_TO_NAME[i]: float(normalized[i]) for i in range(config.NUM_CLASSES)},
        "raw0Multiplier": config.RAW0_WEIGHT_BOOST,
        "maxClassWeightRatio": config.MAX_CLASS_WEIGHT_RATIO,
        "maxMinRatio": float(normalized.max() / max(normalized.min(), 1e-8)),
    }
    return torch.tensor(normalized, dtype=torch.float32), report


def build_optimizer(model: nn.Module, config: AxialV3TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)


def _extract_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    return output["segmentation_logits"] if isinstance(output, dict) else output


def _presence_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | None:
    return output.get("raw0_presence_logits") if isinstance(output, dict) else None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    config: AxialV3TrainConfig,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    amp_enabled = config.AMP and device.type == "cuda"
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    loss_fn = build_segmentation_loss(config, class_weights.to(device))
    totals: dict[str, list[float]] = {"totalLoss": [], "segmentationLoss": [], "presenceLoss": []}
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(images)
            logits = _extract_logits(output)
            seg_loss, components = loss_fn(logits, masks)
            presence_logits = _presence_logits(output)
            presence_loss = torch.tensor(0.0, device=device)
            if config.PRESENCE_HEAD_ENABLED and presence_logits is not None and config.LAMBDA_PRESENCE:
                target = presence_targets(masks).to(device)
                presence_loss = torch.nn.functional.binary_cross_entropy_with_logits(presence_logits, target)
            loss = seg_loss + config.LAMBDA_PRESENCE * presence_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
        totals["totalLoss"].append(float(loss.detach().cpu()))
        totals["segmentationLoss"].append(float(seg_loss.detach().cpu()))
        totals["presenceLoss"].append(float(presence_loss.detach().cpu()))
    return {key: float(np.mean(values)) if values else math.inf for key, values in totals.items()}


def _binary_auroc(scores: np.ndarray, truth: np.ndarray) -> float | None:
    positives = scores[truth]
    negatives = scores[~truth]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    wins = 0.0
    for value in positives:
        wins += float((value > negatives).sum())
        wins += 0.5 * float((value == negatives).sum())
    return wins / float(len(positives) * len(negatives))


def evaluate_validation(model: nn.Module, loader: DataLoader, device: torch.device, config: AxialV3TrainConfig | None = None) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    presence_scores: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(device))
            logits = _extract_logits(output)
            if _presence_logits(output) is not None:
                presence_scores.append(torch.sigmoid(_presence_logits(output)).cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(batch["mask"].cpu().numpy())
    if not predictions:
        raise ValueError("validation loader is empty")
    metrics = metrics_from_predictions(np.concatenate(predictions), np.concatenate(targets))
    if presence_scores:
        scores = np.concatenate(presence_scores)
        target_stack = np.concatenate(targets)
        truth = (target_stack == 1).reshape(len(scores), -1).any(axis=1)
        threshold = config.PRESENCE_THRESHOLD if config is not None else 0.5
        pred_present = scores >= threshold
        tp = int(np.logical_and(pred_present, truth).sum())
        fp = int(np.logical_and(pred_present, ~truth).sum())
        tn = int(np.logical_and(~pred_present, ~truth).sum())
        fn = int(np.logical_and(~pred_present, truth).sum())
        metrics["presenceHead"] = {
            "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "truePositive": tp,
            "falsePositive": fp,
            "trueNegative": tn,
            "falseNegative": fn,
            "gtPresentCases": int(truth.sum()),
            "gtAbsentCases": int((~truth).sum()),
            "threshold": threshold,
            "auroc": _binary_auroc(scores, truth),
        }
        if config is not None and config.PRESENCE_HEAD_ENABLED:
            gated = apply_raw0_slice_presence_gate(np.concatenate(probabilities), scores, threshold)
            metrics["presenceGatedSegmentation"] = metrics_from_predictions(gated, target_stack)
    return metrics


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)
    return sha256_file(path)


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: AxialV3TrainConfig,
    *,
    smoke: bool,
    epoch: int,
    selected_epoch: int,
    best_metric: float,
    history: list[dict[str, Any]],
    split_sha256: str,
    patience_left: int,
    config_sha256: str,
    git_commit: str,
    ai_service_commit: str,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, Any]:
    return {
        "schemaVersion": "axial-v3-train-validation-v1",
        "experimentId": config.EXPERIMENT_ID,
        "experimentType": config.EXPERIMENT_TYPE,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "runId": config.RUN_ID,
        "smokeOnly": smoke,
        "epoch": epoch,
        "selectedEpoch": selected_epoch,
        "bestValidationMetric": best_metric,
        "config": asdict(config),
        "splitSha256": split_sha256,
        "configSha256": config_sha256,
        "gitCommit": git_commit,
        "aiServiceCommit": ai_service_commit,
        "patienceLeft": patience_left,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": history,
    }


def _load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    config: AxialV3TrainConfig,
    *,
    smoke: bool,
    split_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected = {
        "runId": config.RUN_ID,
        "experimentId": config.EXPERIMENT_ID,
        "smokeOnly": smoke,
        "splitSha256": split_sha256,
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"resume checkpoint mismatch for {field}: {checkpoint.get(field)!r} != {value!r}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def runtime_shape_smoke(model: nn.Module, config: AxialV3TrainConfig, device: torch.device) -> dict[str, Any]:
    model.eval()
    channels = 1
    with torch.inference_mode():
        raw_output = model(torch.randn(1, channels, *config.TARGET_SIZE, device=device))
        output = _extract_logits(raw_output)
        result = {"shape": list(output.shape), "finite": bool(torch.isfinite(output).all().item())}
        if _presence_logits(raw_output) is not None:
            result["presenceShape"] = list(_presence_logits(raw_output).shape)
    return result


def _run_dirs(config: AxialV3TrainConfig, smoke: bool) -> dict[str, Path]:
    base = config.OUTPUT_ROOT / ("smoke" if smoke else "") / config.RUN_ID
    return {
        "base": base,
        "config": base / "config",
        "metrics": base / "metrics",
        "checkpoints": base / "checkpoints",
        "reports": base / "reports",
        "manifests": base / "manifests",
        "cache": base / "cache",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def _upsert_registry_status(
    config: AxialV3TrainConfig,
    *,
    dirs: dict[str, Path],
    status: str,
    smoke: bool,
    split_hash: str,
    config_hash: str,
    git_commit: str,
    ai_service_commit: str,
    started: float,
    notes: str,
) -> None:
    upsert_registry_row(
        config.REGISTRY_PATH,
        ExperimentRegistryRow(
            experimentId=config.EXPERIMENT_ID,
            iteration="B",
            experimentType=config.EXPERIMENT_TYPE,
            runId=config.RUN_ID,
            createdAtUtc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            updatedAtUtc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            gitCommit=git_commit,
            aiServiceCommit=ai_service_commit,
            seed=config.SEED,
            configPath=str(dirs["config"] / "run_config.json"),
            configSha256=config_hash,
            splitSha256=split_hash,
            trainingStatus=status,
            smokeOnly=smoke,
            selectedEpoch=None,
            monitorMetric=config.MONITOR_METRIC,
            durationSeconds=time.time() - started,
            notes=notes,
        ),
    )


def run_preflight(config: AxialV3TrainConfig, experiment: object | None = None) -> dict[str, Any]:
    records = build_train_val_records(config)
    class_weights, class_weight_report = compute_class_weights(records, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_axial_v3_model(ArchitectureConfig(base_channels=config.BASE_CHANNELS, presence_head=config.PRESENCE_HEAD_ENABLED)).to(device)
    report = {
        "status": "preflight_passed",
        "experimentId": config.EXPERIMENT_ID,
        "runId": config.RUN_ID,
        "trainSlices": sum(record.split == "train" for record in records),
        "valSlices": sum(record.split == "val" for record in records),
        "trainPatients": len({record.patientId for record in records if record.split == "train"}),
        "valPatients": len({record.patientId for record in records if record.split == "val"}),
        "raw0PresenceTrain": None,
        "classWeights": class_weight_report,
        "splitSha256": sha256_file(config.SPLIT_MANIFEST_PATH),
        "device": str(device),
        "ampRequested": config.AMP,
        "ampEnabled": config.AMP and device.type == "cuda",
        "runtimeShape": runtime_shape_smoke(model, config, device),
        "registryPath": str(config.REGISTRY_PATH),
    }
    dirs = _run_dirs(config, smoke=True)
    _write_json(dirs["reports"] / "preflight_report.json", report)
    return report


def run_calibration(config: AxialV3TrainConfig, *, parent_checkpoint_path: Path) -> dict[str, Any]:
    started = time.time()
    if not parent_checkpoint_path.is_file():
        raise FileNotFoundError(parent_checkpoint_path)
    if config.EXPERIMENT_TYPE != "B4" and not config.EXPERIMENT_ID.startswith("B4"):
        raise ValueError("run_calibration is only valid for B4 experiments")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, _ = build_train_val_loaders(config, smoke=False)
    model = build_axial_v3_model(ArchitectureConfig(base_channels=config.BASE_CHANNELS)).to(device)
    checkpoint = torch.load(parent_checkpoint_path, map_location=device, weights_only=False)
    parent_sha = sha256_file(parent_checkpoint_path)
    if config.PARENT_EXPERIMENT_ID and checkpoint.get("experimentId") != config.PARENT_EXPERIMENT_ID:
        raise ValueError(f"parent experiment mismatch: {checkpoint.get('experimentId')!r} != {config.PARENT_EXPERIMENT_ID!r}")
    if config.PARENT_RUN_ID and checkpoint.get("runId") != config.PARENT_RUN_ID:
        raise ValueError(f"parent run mismatch: {checkpoint.get('runId')!r} != {config.PARENT_RUN_ID!r}")
    split_hash = sha256_file(config.SPLIT_MANIFEST_PATH)
    if checkpoint.get("splitSha256") != split_hash:
        raise ValueError("parent checkpoint splitSha256 does not match current validation split")
    parent_config = checkpoint.get("config", {})
    if parent_config and int(parent_config.get("BASE_CHANNELS", config.BASE_CHANNELS)) != config.BASE_CHANNELS:
        raise ValueError("parent checkpoint architecture does not match BASE_CHANNELS")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    cache_dir = config.OUTPUT_ROOT / "calibration_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{parent_sha}_validation_probabilities.npz"
    if cache_path.exists():
        cache = np.load(cache_path)
        probs = cache["probabilities"]
        truth = cache["targets"]
    else:
        probabilities: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loaders["val"]:
                logits = _extract_logits(model(batch["image"].to(device)))
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
                targets.append(batch["mask"].cpu().numpy())
        probs = np.concatenate(probabilities)
        truth = np.concatenate(targets)
        np.savez_compressed(cache_path, probabilities=probs, targets=truth, parentCheckpointSha256=parent_sha)
    if config.MIN_PROBABILITY is None or config.MIN_MARGIN is None:
        raise ValueError("B4 calibrate requires MIN_PROBABILITY and MIN_MARGIN")
    pred = apply_raw0_threshold(probs, min_probability=config.MIN_PROBABILITY, min_margin=config.MIN_MARGIN)
    metrics = metrics_from_predictions(pred, truth)
    dirs = _run_dirs(config, smoke=False)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    _write_json(dirs["config"] / "run_config.json", asdict(config))
    config_hash = sha256_file(dirs["config"] / "run_config.json")
    git_commit = _git_commit(config.REPO_ROOT)
    ai_service_commit = os.getenv("PFI_AI_SERVICE_COMMIT", git_commit)
    _upsert_registry_status(
        config,
        dirs=dirs,
        status="running",
        smoke=smoke,
        split_hash=split_hash,
        config_hash=config_hash,
        git_commit=git_commit,
        ai_service_commit=ai_service_commit,
        started=started,
        notes="training started",
    )
    report_payload = {
        "status": "calibration_completed",
        "experimentId": config.EXPERIMENT_ID,
        "runId": config.RUN_ID,
        "parentExperimentId": config.PARENT_EXPERIMENT_ID,
        "parentRunId": config.PARENT_RUN_ID,
        "parentCheckpointPath": str(parent_checkpoint_path),
        "parentCheckpointSha256": parent_sha,
        "parentSplitSha256": checkpoint.get("splitSha256"),
        "splitSha256": split_hash,
        "probabilityCachePath": str(cache_path),
        "probabilityCacheSha256": sha256_file(cache_path),
        "minProbability": config.MIN_PROBABILITY,
        "minMargin": config.MIN_MARGIN,
        "validationMetrics": metrics,
    }
    report_path = dirs["reports"] / "calibration_report.json"
    _write_json(report_path, report_payload)
    upsert_registry_row(
        config.REGISTRY_PATH,
        ExperimentRegistryRow(
            experimentId=config.EXPERIMENT_ID,
            iteration="B",
            experimentType=config.EXPERIMENT_TYPE,
            runId=config.RUN_ID,
            createdAtUtc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            updatedAtUtc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            gitCommit=git_commit,
            aiServiceCommit=ai_service_commit,
            seed=config.SEED,
            configPath=str(dirs["config"] / "run_config.json"),
            configSha256=config_hash,
            splitSha256=split_hash,
            trainingStatus="completed",
            smokeOnly=False,
            selectedEpoch=None,
            monitorMetric=config.MONITOR_METRIC,
            validationDiceMacroForeground=metrics.get("dice_macro_foreground"),
            validationRaw0Dice=metrics.get("raw0Dice"),
            validationRaw0Precision=metrics.get("raw0Precision"),
            validationRaw0Recall=metrics.get("raw0Recall"),
            validationRaw0FalsePositivePixels=metrics.get("raw0FalsePositivePixels"),
            validationRaw0PredictedInGtAbsentCases=metrics.get("raw0PredictedInGtAbsentCases"),
            validationDiceMacroExcludingRaw0=metrics.get("dice_macro_excluding_raw0"),
            artifactPath=str(report_path),
            artifactSha256=sha256_file(report_path),
            checkpointPath=str(parent_checkpoint_path),
            checkpointSha256=parent_sha,
            durationSeconds=time.time() - started,
            notes=f"calibration parent={config.PARENT_EXPERIMENT_ID}/{config.PARENT_RUN_ID}",
        ),
    )
    return report_payload


def summarize_validation_runs(config: AxialV3TrainConfig) -> dict[str, Any]:
    from .low_cost import SelectionGuardrail, evaluate_other_class_guardrail, validation_ranking_key
    from .registry import read_registry

    rows = read_registry(config.REGISTRY_PATH)
    report_by_key = {row.get("experimentId"): _load_run_report_per_class(row) for row in rows}
    baseline_per_class = report_by_key.get("B0")
    guardrail = SelectionGuardrail(protected_classes=("raw_50", "raw_100", "raw_150", "raw_200"))
    accepted = []
    discarded = []
    for row in rows:
        smoke = str(row.get("smokeOnly", "")).lower() == "true"
        status = row.get("trainingStatus")
        reasons = []
        if smoke:
            reasons.append("smoke")
            if status not in {"completed", "validation_candidate"}:
                reasons.append("not_completed")
        metrics = {
            "dice_macro_foreground": _parse_float(row.get("validationDiceMacroForeground")),
            "raw0PredictedInGtAbsentCases": _parse_float(row.get("validationRaw0PredictedInGtAbsentCases")),
            "raw0Precision": _parse_float(row.get("validationRaw0Precision")),
            "dice_macro_excluding_raw0": _parse_float(row.get("validationDiceMacroExcludingRaw0")),
        }
        if any(value is None or not np.isfinite(value) for value in metrics.values()):
            reasons.append("missing_or_nonfinite_metrics")
        guardrail_report = {"passed": False, "reasons": ["baseline_per_class_missing"], "missingMetrics": []}
        if baseline_per_class:
            guardrail_report = evaluate_other_class_guardrail(report_by_key.get(row.get("experimentId"), {}), baseline_per_class, guardrail)
            if not guardrail_report["passed"]:
                reasons.extend(str(reason) for reason in guardrail_report["reasons"])
        else:
            reasons.append("baseline_per_class_missing")
        payload = {**row, **metrics, "guardrail": guardrail_report, "discardReasons": ";".join(reasons)}
        if reasons:
            discarded.append(payload)
        else:
            accepted.append(payload)
    accepted.sort(key=validation_ranking_key, reverse=True)
    if accepted:
        accepted[0]["selectionLabel"] = "validation_candidate"
        selected_payload = next((row for row in rows if row.get("experimentId") == accepted[0].get("experimentId") and row.get("runId") == accepted[0].get("runId")), None)
        if selected_payload:
            selected_payload = {**selected_payload, "trainingStatus": "validation_candidate", "guardrailPassed": True, "notes": "selected by validation-only summarize"}
            upsert_registry_row(config.REGISTRY_PATH, ExperimentRegistryRow(**{column: selected_payload.get(column) for column in ExperimentRegistryRow.__dataclass_fields__}))
    output = config.OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accepted + discarded).to_csv(output / "validation_ranking.csv", index=False)
    _write_json(output / "validation_ranking.json", {"accepted": accepted, "discarded": discarded})
    (output / "selection_report.md").write_text("# Axial v3 validation ranking\n\nValidation-only ranking. No test data read.\n", encoding="utf-8")
    return {"accepted": accepted, "discarded": discarded}


def _load_run_report_per_class(row: dict[str, Any]) -> dict[str, Any]:
    artifact_path = row.get("artifactPath")
    if artifact_path and Path(str(artifact_path)).exists():
        report_path = Path(str(artifact_path))
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = payload.get("finalValidationMetrics") or payload.get("validationMetrics") or {}
        per_class = metrics.get("perClass")
        return per_class if isinstance(per_class, dict) else {}
    checkpoint_path = row.get("checkpointPath")
    if not checkpoint_path:
        return {}
    report_path = Path(str(checkpoint_path)).parent.parent / "reports" / "run_report.json"
    if not report_path.exists():
        report_path = Path(str(checkpoint_path)).parent.parent / "reports" / "calibration_report.json"
    if not report_path.exists():
        return {}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = payload.get("finalValidationMetrics") or payload.get("validationMetrics") or {}
    per_class = metrics.get("perClass")
    return per_class if isinstance(per_class, dict) else {}


def _parse_float(value: object) -> float | None:
    if value in {None, "", "None"}:
        return None
    return float(value)


def run_training(config: AxialV3TrainConfig, *, smoke: bool = False) -> dict[str, Any]:
    if config.EXPERIMENT_TYPE == "B4" or config.EXPERIMENT_ID.startswith("B4"):
        raise ValueError("B4 calibration experiments must use RUN_MODE=calibrate, not train")
    set_deterministic_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, records = build_train_val_loaders(config, smoke=smoke)
    class_weights, weight_report = compute_class_weights(records, config)
    model = build_axial_v3_model(ArchitectureConfig(base_channels=config.BASE_CHANNELS, presence_head=config.PRESENCE_HEAD_ENABLED)).to(device)
    optimizer = build_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP and device.type == "cuda")
    max_epochs = min(config.MAX_EPOCHS, 2) if smoke else config.MAX_EPOCHS
    split_hash = sha256_file(config.SPLIT_MANIFEST_PATH)
    best_metric = -math.inf
    selected_epoch = 0
    patience_left = config.EARLY_STOPPING_PATIENCE
    history: list[dict[str, Any]] = []
    started = time.time()
    dirs = _run_dirs(config, smoke)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    _write_json(dirs["config"] / "run_config.json", asdict(config))
    config_hash = sha256_file(dirs["config"] / "run_config.json")
    git_commit = _git_commit(config.REPO_ROOT)
    ai_service_commit = os.getenv("PFI_AI_SERVICE_COMMIT", git_commit)
    resume_mode = config.RESUME_MODE.strip().lower()
    if resume_mode == "off":
        resume_mode = "never"
    if resume_mode not in {"auto", "required", "never"}:
        raise ValueError("RESUME_MODE must be auto, required, never or off")
    last_checkpoint_path = dirs["checkpoints"] / "last_checkpoint.pt"
    resume_status: dict[str, Any] = {"mode": resume_mode, "loaded": False, "path": str(last_checkpoint_path)}
    start_epoch = 1
    if resume_mode in {"auto", "required"}:
        if last_checkpoint_path.exists():
            checkpoint = _load_resume_checkpoint(last_checkpoint_path, model, optimizer, scaler, config, smoke=smoke, split_sha256=split_hash, device=device)
            history = list(checkpoint.get("history", []))
            best_metric = float(checkpoint.get("bestValidationMetric", -math.inf))
            selected_epoch = int(checkpoint.get("selectedEpoch", 0))
            patience_left = int(checkpoint.get("patienceLeft", config.EARLY_STOPPING_PATIENCE))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            resume_status.update({"loaded": True, "startEpoch": start_epoch, "checkpointSha256": sha256_file(last_checkpoint_path)})
        elif resume_mode == "required":
            raise FileNotFoundError(f"required resume checkpoint not found: {last_checkpoint_path}")
    checkpoint_path = dirs["checkpoints"] / "best_checkpoint.pt"
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.exists() else ""
    for epoch in range(start_epoch, max_epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, class_weights, config, device, scaler)
        val_metrics = evaluate_validation(model, loaders["val"], device, config)
        monitor = val_metrics.get(config.MONITOR_METRIC)
        improved = monitor is not None and float(monitor) > best_metric
        if improved:
            best_metric = float(monitor)
            selected_epoch = epoch
            patience_left = config.EARLY_STOPPING_PATIENCE
            checkpoint_sha = save_checkpoint_atomic(
                checkpoint_path,
                _checkpoint_payload(
                    model,
                    optimizer,
                    config,
                    smoke=smoke,
                    epoch=epoch,
                    selected_epoch=epoch,
                    best_metric=best_metric,
                    history=history,
                    split_sha256=split_hash,
                    patience_left=patience_left,
                    config_sha256=config_hash,
                    git_commit=git_commit,
                    ai_service_commit=ai_service_commit,
                    scaler=scaler,
                ),
            )
        else:
            patience_left -= 1
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items() if k not in {"perClass", "confusionMatrix"}}})
        save_checkpoint_atomic(
            last_checkpoint_path,
            _checkpoint_payload(
                model,
                optimizer,
                config,
                smoke=smoke,
                epoch=epoch,
                selected_epoch=selected_epoch,
                best_metric=best_metric,
                history=history,
                split_sha256=split_hash,
                patience_left=patience_left,
                config_sha256=config_hash,
                git_commit=git_commit,
                ai_service_commit=ai_service_commit,
                scaler=scaler,
            ),
        )
        if patience_left <= 0:
            break
    duration = time.time() - started
    if checkpoint_sha and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    final_metrics = evaluate_validation(model, loaders["val"], device, config)
    pd.DataFrame(history).to_csv(dirs["metrics"] / "history.csv", index=False)
    _write_json(dirs["metrics"] / "history.json", history)
    _write_json(dirs["metrics"] / "best_validation_metrics.json", final_metrics)
    _write_json(dirs["metrics"] / "class_weights.json", weight_report)
    result_payload = {
        "runId": config.RUN_ID,
        "experimentId": config.EXPERIMENT_ID,
        "smokeOnly": smoke,
        "selectedEpoch": selected_epoch,
        "monitorMetric": config.MONITOR_METRIC,
        "bestValidationMetric": best_metric,
        "finalValidationMetrics": final_metrics,
        "classWeightReport": weight_report,
        "history": history,
        "durationSeconds": duration,
        "checkpointPath": str(checkpoint_path) if checkpoint_sha else "",
        "checkpointSha256": checkpoint_sha,
        "lastCheckpointPath": str(last_checkpoint_path) if last_checkpoint_path.exists() else "",
        "lastCheckpointSha256": sha256_file(last_checkpoint_path) if last_checkpoint_path.exists() else "",
        "resumeStatus": resume_status,
        "runtimeSmoke": runtime_shape_smoke(model, config, device),
    }
    _write_json(dirs["reports"] / "run_report.json", result_payload)
    (dirs["reports"] / "run_report.md").write_text(f"# Axial v3 run {config.RUN_ID}\n\nTrain/validation only. Smoke: {smoke}.\n", encoding="utf-8")
    upsert_registry_row(
        config.REGISTRY_PATH,
        ExperimentRegistryRow(
            experimentId=config.EXPERIMENT_ID,
            iteration="B",
            experimentType=config.EXPERIMENT_TYPE,
            runId=config.RUN_ID,
            createdAtUtc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            updatedAtUtc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            gitCommit=git_commit,
            aiServiceCommit=ai_service_commit,
            seed=config.SEED,
            configPath=str(dirs["config"] / "run_config.json"),
            configSha256=config_hash,
            splitSha256=split_hash,
            trainingStatus="smoke_completed" if smoke else "completed",
            smokeOnly=smoke,
            selectedEpoch=selected_epoch,
            monitorMetric=config.MONITOR_METRIC,
            validationDiceMacroForeground=final_metrics.get("dice_macro_foreground"),
            validationRaw0Dice=final_metrics.get("raw0Dice"),
            validationRaw0Precision=final_metrics.get("raw0Precision"),
            validationRaw0Recall=final_metrics.get("raw0Recall"),
            validationRaw0FalsePositivePixels=final_metrics.get("raw0FalsePositivePixels"),
            validationRaw0PredictedInGtAbsentCases=final_metrics.get("raw0PredictedInGtAbsentCases"),
            validationDiceMacroExcludingRaw0=final_metrics.get("dice_macro_excluding_raw0"),
            checkpointPath=str(checkpoint_path) if checkpoint_sha else "",
            checkpointSha256=checkpoint_sha,
            durationSeconds=duration,
            notes="train/validation only",
        ),
    )
    return result_payload
