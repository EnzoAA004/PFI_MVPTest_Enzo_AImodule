"""Executable train/validation-only runner for axial v3 Iteration A."""

from __future__ import annotations

import json
import os
import hashlib
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .audit import audit_raw0_slice, sha256_file, slice_audit_frame, summarize_raw0_by_patient, write_audit_outputs
from .calibration import apply_raw0_threshold
from .guards import assert_no_test_records, require_train_val_only
from .labels import mask_to_class_indices
from .low_cost import metric_or_default
from .metrics import metrics_from_predictions
from .training import AxialV3Record, open_2d_array, resize_bilinear, resize_nearest, robust_normalize


def optional_env_path(name: str) -> Path | None:
    raw = os.getenv(name, "").strip()
    return Path(raw) if raw else None


@dataclass(frozen=True)
class AxialV3AuditConfig:
    RUN_ID: str = os.getenv("PFI_RUN_ID", "axial-v3-iteration-a")
    SEED: int = int(os.getenv("PFI_SEED", "2026"))
    REPO_ROOT: Path = Path(os.getenv("PFI_REPO_ROOT", "."))
    REPO_REF: str = os.getenv("PFI_REPO_REF", "")
    EXTERNAL_ROOT: Path = Path(os.getenv("PFI_EXTERNAL_ROOT", "/content/drive/MyDrive/PFI_MVP"))
    DATASET_ROOT: Path = Path(os.getenv("PFI_DATASET_ROOT", "/content/drive/MyDrive/PFI_MVP"))
    IMAGES_ROOT: Path = Path(os.getenv("AXIAL_IMAGES_ROOT", ""))
    MASKS_ROOT: Path = Path(os.getenv("AXIAL_MASKS_ROOT", ""))
    SPLIT_MANIFEST_PATH: Path = Path(os.getenv("AXIAL_E9_CURATED_SPLIT_CSV", "manifest.csv"))
    V2_CHECKPOINT_PATH: Path | None = optional_env_path("AXIAL_V2_CHECKPOINT_PATH")
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
    EXPECTED_V2_RUN_ID: str = os.getenv("AXIAL_EXPECTED_V2_RUN_ID", "axial-final-v2")
    EXPECTED_AI_SERVICE_COMMIT: str | None = os.getenv("AXIAL_EXPECTED_AI_SERVICE_COMMIT") or None
    EXPECTED_SPLIT_SHA256: str | None = os.getenv("AXIAL_EXPECTED_SPLIT_SHA256") or None
    EXPECTED_CHECKPOINT_SHA256: str | None = os.getenv("AXIAL_EXPECTED_CHECKPOINT_SHA256") or None
    PFI_ALLOW_INCOMPLETE_V2_CHECKPOINT_METADATA: bool = os.getenv("PFI_ALLOW_INCOMPLETE_V2_CHECKPOINT_METADATA", "0") == "1"

    def validate(self) -> None:
        if self.TARGET_SIZE != (256, 256):
            raise ValueError("TARGET_SIZE must be 256x256")
        if self.NUM_CLASSES != 6:
            raise ValueError("NUM_CLASSES must be 6")
        if self.MASK_LABEL_MODE not in {"raw", "indexed"}:
            raise ValueError("MASK_LABEL_MODE must be raw or indexed")
        if any(value < 0 or value > 1 for value in self.THRESHOLD_GRID):
            raise ValueError("thresholds must be between 0 and 1")
        if any(value < 0 for value in self.MARGIN_GRID):
            raise ValueError("margins must be non-negative")


def validate_storage(config: AxialV3AuditConfig) -> None:
    config.validate()
    if config.PFI_USE_GOOGLE_DRIVE and not config.PFI_DRIVE_ROOT.exists():
        raise FileNotFoundError(f"Google Drive root not available: {config.PFI_DRIVE_ROOT}")
    if not config.SPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"split manifest not found: {config.SPLIT_MANIFEST_PATH}")
    if not config.DATASET_ROOT.exists():
        raise FileNotFoundError(f"dataset root not found: {config.DATASET_ROOT}")
    if config.V2_CHECKPOINT_PATH is not None and not config.V2_CHECKPOINT_PATH.is_file():
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
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"missing axial image/mask: {image_path} / {mask_path}")
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
    _validate_record_integrity(records)
    return records


def _validate_record_integrity(records: list[AxialV3Record]) -> None:
    seen_pairs: set[tuple[str, str]] = set()
    by_patient: dict[str, set[str]] = {}
    by_study: dict[str, set[str]] = {}
    image_split: dict[str, set[str]] = {}
    mask_split: dict[str, set[str]] = {}
    for record in records:
        pair = (record.image_path, record.mask_path)
        if pair in seen_pairs:
            raise ValueError(f"duplicate image/mask pair: {pair}")
        seen_pairs.add(pair)
        by_patient.setdefault(record.patientId, set()).add(record.split)
        by_study.setdefault(record.studyId, set()).add(record.split)
        image_split.setdefault(record.image_path, set()).add(record.split)
        mask_split.setdefault(record.mask_path, set()).add(record.split)
    if any(len(splits) > 1 for splits in by_patient.values()):
        raise ValueError("patientId leakage across train/val")
    if any(len(splits) > 1 for splits in by_study.values()):
        raise ValueError("studyId leakage across train/val")
    if any(len(splits) > 1 for splits in image_split.values()):
        raise ValueError("image path repeated across train/val")
    if any(len(splits) > 1 for splits in mask_split.values()):
        raise ValueError("mask path repeated across train/val")


def run_structural_audit(config: AxialV3AuditConfig, records: list[AxialV3Record]) -> dict[str, Any]:
    rows = []
    for record in records:
        require_train_val_only([record.split], context="iteration_a.structural")
        image = open_2d_array(Path(record.image_path))
        raw_mask = open_2d_array(Path(record.mask_path))
        indexed_mask = mask_to_class_indices(raw_mask, mode=config.MASK_LABEL_MODE)
        resized_mask = resize_nearest(indexed_mask, config.TARGET_SIZE)
        original = audit_raw0_slice(
                mask=indexed_mask,
                split=record.split,
                patient_id=record.patientId,
                study_id=record.studyId,
                slice_id=record.sliceId,
                image_shape=indexed_mask.shape,
                original_image_shape=image.shape,
                original_mask_shape=raw_mask.shape,
                raw0_value=1,
                background_value=0,
                source_image=record.sourceImage,
                source_mask=record.sourceMask,
            )
        resized = audit_raw0_slice(
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
        row = original
        object.__setattr__(row, "resizedMetrics", resized)
        rows.append(row)
    output_dir = config.OUTPUT_ROOT / config.RUN_ID
    summary = write_audit_outputs(rows, output_dir)
    return {"rows": rows, "summary": summary}


def _save_structural_figures(config: AxialV3AuditConfig, rows: list[Any]) -> list[Path]:
    figures = config.OUTPUT_ROOT / config.RUN_ID / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    df = slice_audit_frame(rows)
    created: list[Path] = []
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None
    specs = [
        ("raw0_area_histogram.png", "raw0PixelCount"),
        ("raw0_area_ratio_histogram.png", "raw0AreaRatio"),
        ("raw0_components_histogram.png", "connectedComponents"),
        ("raw0_distance_to_border_histogram.png", "minDistanceToBorder"),
    ]
    for filename, column in specs:
        path = figures / filename
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if plt is None:
            _write_text_png(path, [column, f"count={len(values)}", f"median={float(values.median()) if len(values) else 'n/a'}"])
        else:
            plt.figure()
            values.hist(bins=config.PROBABILITY_HIST_BINS)
            plt.title(column)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        created.append(path)
    for filename in [
        "raw0_border_contact_by_split.png",
        "raw0_presence_by_patient.png",
        "raw0_area_boxplot_by_split.png",
    ]:
        path = figures / filename
        if plt is None:
            _write_text_png(path, [filename.replace(".png", ""), f"slices={len(df)}"])
        else:
            plt.figure()
            if filename == "raw0_border_contact_by_split.png":
                df.groupby("split")["touchesBorder"].mean().plot(kind="bar")
            elif filename == "raw0_presence_by_patient.png":
                df.groupby("patientId")["raw0Present"].mean().head(config.MAX_PREVIEW_CASES).plot(kind="bar")
            else:
                pd.to_numeric(df["raw0PixelCount"], errors="coerce").dropna().plot(kind="box")
            plt.title(filename.replace(".png", ""))
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        created.append(path)
    return created


def _save_probability_figures(config: AxialV3AuditConfig) -> list[Path]:
    figures = config.OUTPUT_ROOT / config.RUN_ID / "figures"
    tables = config.OUTPUT_ROOT / config.RUN_ID / "tables"
    probability_csv = tables / "raw0_validation_probability_audit.csv"
    grid_csv = tables / "raw0_threshold_margin_grid.csv"
    created: list[Path] = []
    for filename in [
        "raw0_false_positive_pixels_histogram.png",
        "raw0_predicted_area_histogram.png",
        "raw0_probability_positive_vs_negative.png",
        "raw0_precision_recall_presence.png",
    ]:
        path = figures / filename
        if not probability_csv.exists():
            _write_text_png(path, [filename.replace(".png", ""), "Probability audit skipped."])
            created.append(path)
            continue
        df = pd.read_csv(probability_csv)
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            plt = None
        if filename == "raw0_false_positive_pixels_histogram.png":
            values = pd.to_numeric(df["raw0FalsePositivePixels"], errors="coerce").dropna()
            if plt is None:
                _write_text_png(path, ["raw0 false positive pixels", f"count={len(values)}", f"max={int(values.max()) if len(values) else 'n/a'}"])
            else:
                plt.figure()
                values.hist(bins=config.PROBABILITY_HIST_BINS)
                plt.title("raw0FalsePositivePixels")
                plt.tight_layout()
                plt.savefig(path)
                plt.close()
        elif filename == "raw0_predicted_area_histogram.png":
            values = pd.to_numeric(df["raw0PredictedAreaPixels"], errors="coerce").dropna()
            if plt is None:
                _write_text_png(path, ["raw0 predicted area pixels", f"count={len(values)}", f"max={int(values.max()) if len(values) else 'n/a'}"])
            else:
                plt.figure()
                values.hist(bins=config.PROBABILITY_HIST_BINS)
                plt.title("raw0PredictedAreaPixels")
                plt.tight_layout()
                plt.savefig(path)
                plt.close()
        elif filename == "raw0_probability_positive_vs_negative.png":
            if plt is None:
                positives = df[df["raw0GtPresent"] == True]["raw0MaxProbability"]  # noqa: E712
                negatives = df[df["raw0GtPresent"] == False]["raw0MaxProbability"]  # noqa: E712
                _write_text_png(path, ["raw0 max probability", f"gt+ median={float(positives.median()) if len(positives) else 'n/a'}", f"gt- median={float(negatives.median()) if len(negatives) else 'n/a'}"])
            else:
                plt.figure()
                df.boxplot(column="raw0MaxProbability", by="raw0GtPresent")
                plt.suptitle("")
                plt.title("raw0MaxProbability by GT presence")
                plt.tight_layout()
                plt.savefig(path)
                plt.close()
        else:
            if not grid_csv.exists():
                _write_text_png(path, ["raw0 precision/recall", "Threshold grid missing."])
            else:
                grid = pd.read_csv(grid_csv)
                if plt is None or grid.empty:
                    _write_text_png(path, ["raw0 precision/recall", f"grid rows={len(grid)}"])
                else:
                    plt.figure()
                    plt.scatter(grid["raw0Recall"], grid["raw0Precision"], c=grid["dice_macro_foreground"])
                    plt.xlabel("raw0Recall")
                    plt.ylabel("raw0Precision")
                    plt.title("raw0 threshold precision/recall")
                    plt.tight_layout()
                    plt.savefig(path)
                    plt.close()
        created.append(path)
    return created


def _record_raw0_audit(record: AxialV3Record, config: AxialV3AuditConfig) -> Any:
    image = open_2d_array(Path(record.image_path))
    raw_mask = open_2d_array(Path(record.mask_path))
    indexed_mask = mask_to_class_indices(raw_mask, mode=config.MASK_LABEL_MODE)
    return audit_raw0_slice(
        mask=indexed_mask,
        split=record.split,
        patient_id=record.patientId,
        study_id=record.studyId,
        slice_id=record.sliceId,
        image_shape=indexed_mask.shape,
        original_image_shape=image.shape,
        original_mask_shape=raw_mask.shape,
        raw0_value=1,
        background_value=0,
        source_image=record.sourceImage,
        source_mask=record.sourceMask,
    )


def _preview_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = resize_bilinear(robust_normalize(image), (256, 256))
    mask = resize_nearest(mask, (256, 256))
    rgb = np.repeat(normalized[..., None], 3, axis=2)
    raw0 = mask == 1
    rgb[raw0, 0] = 1.0
    rgb[raw0, 1] *= 0.2
    rgb[raw0, 2] *= 0.2
    return rgb


def _overlay_raw0_on_normalized(normalized: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.repeat(normalized[..., None], 3, axis=2)
    raw0 = mask == 1
    rgb[raw0, 0] = 1.0
    rgb[raw0, 1] *= 0.2
    rgb[raw0, 2] *= 0.2
    return rgb


def _write_text_png(path: Path, lines: list[str], *, size: tuple[int, int] = (900, 480)) -> None:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    y = 24
    for line in lines:
        draw.text((24, y), str(line), fill=(20, 20, 20))
        y += 28
    canvas.save(path)


def _render_preview_pil(path: Path, records: list[AxialV3Record], config: AxialV3AuditConfig, title: str) -> None:
    from PIL import Image, ImageDraw

    selected = records[: max(1, min(config.MAX_PREVIEW_CASES, len(records)))]
    if not selected:
        _write_text_png(path, [title, "No matching train/validation slices."])
        return
    tile = 180
    label_h = 44
    cols = min(4, len(selected))
    rows = int(np.ceil(len(selected) / cols))
    canvas = Image.new("RGB", (cols * tile, rows * (tile + label_h) + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(20, 20, 20))
    for index, record in enumerate(selected):
        image = open_2d_array(Path(record.image_path))
        raw_mask = open_2d_array(Path(record.mask_path))
        mask = mask_to_class_indices(raw_mask, mode=config.MASK_LABEL_MODE)
        overlay = (_preview_overlay(image, mask) * 255).astype(np.uint8)
        tile_image = Image.fromarray(overlay).resize((tile, tile), resample=Image.Resampling.NEAREST)
        row = index // cols
        col = index % cols
        x = col * tile
        y = 28 + row * (tile + label_h)
        canvas.paste(tile_image, (x, y))
        draw.text((x + 4, y + tile + 4), f"{record.split} | {record.patientId}", fill=(20, 20, 20))
        draw.text((x + 4, y + tile + 22), record.sliceId[:28], fill=(20, 20, 20))
    canvas.save(path)


def _render_preview(path: Path, records: list[AxialV3Record], config: AxialV3AuditConfig, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _render_preview_pil(path, records, config, title)
        return

    selected = records[: max(1, min(config.MAX_PREVIEW_CASES, len(records)))]
    cols = min(4, max(1, len(selected)))
    rows = max(1, int(np.ceil(len(selected) / cols)))
    plt.figure(figsize=(3 * cols, 3 * rows))
    if not selected:
        plt.axis("off")
        plt.title(title)
        plt.text(0.01, 0.5, "No matching train/validation slices.")
    for index, record in enumerate(selected, start=1):
        image = open_2d_array(Path(record.image_path))
        raw_mask = open_2d_array(Path(record.mask_path))
        mask = mask_to_class_indices(raw_mask, mode=config.MASK_LABEL_MODE)
        plt.subplot(rows, cols, index)
        plt.imshow(_preview_overlay(image, mask))
        plt.axis("off")
        plt.title(f"{record.split} | {record.patientId}\n{record.sliceId}", fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _save_previews(config: AxialV3AuditConfig, records: list[AxialV3Record]) -> list[Path]:
    previews = config.OUTPUT_ROOT / config.RUN_ID / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    audited = [(record, _record_raw0_audit(record, config)) for record in records]
    positives = [record for record, row in audited if row.raw0Present]
    negatives = [record for record, row in audited if not row.raw0Present]
    border = [record for record, row in audited if row.touchesBorder]
    positive_sorted = sorted(
        [(record, row) for record, row in audited if row.raw0Present],
        key=lambda item: item[1].raw0PixelCount,
    )
    preview_specs = [
        ("raw0_positive_examples.png", positives, "raw_0 positive examples"),
        ("raw0_negative_examples.png", negatives, "raw_0 negative examples"),
        ("raw0_border_contact_examples.png", border, "raw_0 border contact examples"),
        ("raw0_smallest_annotations.png", [record for record, _ in positive_sorted], "smallest raw_0 annotations"),
        ("raw0_largest_annotations.png", [record for record, _ in reversed(positive_sorted)], "largest raw_0 annotations"),
    ]
    created: list[Path] = []
    for filename, selected, title in preview_specs:
        path = previews / filename
        _render_preview(path, selected, config, title)
        created.append(path)
    return created


def _save_probability_discordant_preview(config: AxialV3AuditConfig, records: list[AxialV3Record]) -> Path:
    from PIL import Image, ImageDraw

    output_dir = config.OUTPUT_ROOT / config.RUN_ID
    path = output_dir / "previews" / "raw0_probability_discordant_examples.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    probability_csv = output_dir / "tables" / "raw0_validation_probability_audit.csv"
    cache_path = output_dir / "cache" / "validation_probabilities.npz"
    if not probability_csv.exists() or not cache_path.exists():
        _write_text_png(path, ["raw0 probability discordant examples", "Probability audit skipped."])
        return path

    df = pd.read_csv(probability_csv).reset_index().rename(columns={"index": "probabilityIndex"})
    df["areaDisagreement"] = (pd.to_numeric(df["raw0PredictedAreaPixels"], errors="coerce") - pd.to_numeric(df["raw0TrueAreaPixels"], errors="coerce")).abs()
    candidates = pd.concat(
        [
            df[df["raw0GtPresent"] == False].sort_values("raw0MaxProbability", ascending=False).head(config.MAX_PREVIEW_CASES),  # noqa: E712
            df[df["raw0GtPresent"] == True].sort_values("raw0MaxProbability", ascending=True).head(config.MAX_PREVIEW_CASES),  # noqa: E712
            df.sort_values("raw0FalsePositivePixels", ascending=False).head(config.MAX_PREVIEW_CASES),
            df.sort_values("areaDisagreement", ascending=False).head(config.MAX_PREVIEW_CASES),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["probabilityIndex"]).head(config.MAX_PREVIEW_CASES)
    val_records = [record for record in records if record.split == "val"]
    cache = np.load(cache_path)
    probabilities = cache["probabilities"]
    tile = 160
    label_h = 54
    cols = 2
    rows = max(1, len(candidates))
    canvas = Image.new("RGB", (cols * tile, rows * (tile + label_h) + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "raw0 probability discordant examples: GT left, prediction right", fill=(20, 20, 20))
    for row_idx, item in enumerate(candidates.itertuples(index=False)):
        prob_index = int(item.probabilityIndex)
        if prob_index >= len(val_records) or prob_index >= len(probabilities):
            continue
        record = val_records[prob_index]
        image = resize_bilinear(robust_normalize(open_2d_array(Path(record.image_path))), config.TARGET_SIZE)
        target = resize_nearest(mask_to_class_indices(open_2d_array(Path(record.mask_path)), mode=config.MASK_LABEL_MODE), config.TARGET_SIZE)
        pred = np.argmax(probabilities[prob_index], axis=0)
        gt_tile = Image.fromarray((_overlay_raw0_on_normalized(image, target) * 255).astype(np.uint8)).resize((tile, tile), resample=Image.Resampling.NEAREST)
        pred_tile = Image.fromarray((_overlay_raw0_on_normalized(image, pred) * 255).astype(np.uint8)).resize((tile, tile), resample=Image.Resampling.NEAREST)
        y = 28 + row_idx * (tile + label_h)
        canvas.paste(gt_tile, (0, y))
        canvas.paste(pred_tile, (tile, y))
        label = f"{record.patientId} | maxP={float(item.raw0MaxProbability):.3f} | fpPx={int(item.raw0FalsePositivePixels)}"
        draw.text((4, y + tile + 4), label[:58], fill=(20, 20, 20))
        draw.text((4, y + tile + 24), record.sliceId[:58], fill=(20, 20, 20))
    canvas.save(path)
    return path


def _write_human_review_candidates(config: AxialV3AuditConfig, rows: list[Any]) -> Path:
    tables = config.OUTPUT_ROOT / config.RUN_ID / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df = slice_audit_frame(rows)
    candidates: list[pd.DataFrame] = []
    positive = df[df["raw0Present"] == True].copy()  # noqa: E712
    negative = df[df["raw0Present"] == False].copy()  # noqa: E712
    for reason, subset in [
        ("raw0_present", positive.head(config.MAX_PREVIEW_CASES)),
        ("raw0_absent", negative.head(max(1, config.MAX_PREVIEW_CASES // 2))),
        ("touches_border", df[df["touchesBorder"] == True].head(config.MAX_PREVIEW_CASES)),  # noqa: E712
        ("smallest_raw0", positive.sort_values("raw0PixelCount", ascending=True).head(config.MAX_PREVIEW_CASES)),
        ("largest_raw0", positive.sort_values("raw0PixelCount", ascending=False).head(config.MAX_PREVIEW_CASES)),
    ]:
        if subset.empty:
            continue
        item = subset.copy()
        item["reviewReason"] = reason
        candidates.append(item)
    output = tables / "raw0_human_review_candidates.csv"
    if candidates:
        merged = pd.concat(candidates, ignore_index=True)
        merged.drop_duplicates(subset=["split", "patientId", "studyId", "sliceId", "reviewReason"]).to_csv(output, index=False)
    else:
        pd.DataFrame(columns=[*df.columns, "reviewReason"]).to_csv(output, index=False)
    return output


def run_validation_probability_audit(config: AxialV3AuditConfig, records: list[AxialV3Record]) -> dict[str, Any]:
    if config.V2_CHECKPOINT_PATH is None:
        metrics_dir = config.OUTPUT_ROOT / config.RUN_ID / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        summary = {"status": "skipped", "reason": "AXIAL_V2_CHECKPOINT_PATH not configured", "validationSlices": sum(record.split == "val" for record in records)}
        (metrics_dir / "raw0_validation_probability_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary
    from .architectures import ArchitectureConfig, build_axial_v3_model

    val_records = [record for record in records if record.split == "val"]
    require_train_val_only(["val"], context="iteration_a.probability")
    checkpoint = torch.load(config.V2_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    validation = validate_v2_checkpoint_metadata(checkpoint, config)
    model = build_axial_v3_model(ArchitectureConfig())
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    rows: list[dict[str, Any]] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for record in val_records:
            image = resize_bilinear(robust_normalize(open_2d_array(Path(record.image_path))), config.TARGET_SIZE)
            raw_mask = open_2d_array(Path(record.mask_path))
            target = resize_nearest(mask_to_class_indices(raw_mask, mode=config.MASK_LABEL_MODE), config.TARGET_SIZE)
            probs = torch.softmax(model(torch.from_numpy(image[None, None].astype(np.float32))), dim=1).numpy()[0]
            probabilities.append(probs)
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
                    "raw0P99Probability": float(np.percentile(raw0_prob, 99)),
                    "raw0TopKMeanProbability": float(np.sort(raw0_prob.reshape(-1))[-min(100, raw0_prob.size):].mean()),
                    "raw0PredictedAreaPixels": int((pred == 1).sum()),
                    "raw0PredictedAreaRatio": float((pred == 1).mean()),
                    "raw0TrueAreaPixels": int((target == 1).sum()),
                    "raw0FalsePositivePixels": metrics["falsePositivePixels"],
                    "raw0FalseNegativePixels": metrics["falseNegativePixels"],
                    "raw0TruePositivePixels": metrics["truePositivePixels"],
                    "raw0Dice": metrics["dice"],
                    "raw0IoU": metrics["iou"],
                    "raw0Precision": metrics["precision"],
                    "raw0Recall": metrics["recall"],
                    "raw0PredictedInGtAbsent": metrics["predictedInGtAbsentCases"],
                    "maxProbabilityInsideGt": float(raw0_prob[target == 1].max()) if (target == 1).any() else None,
                    "maxProbabilityOutsideGt": float(raw0_prob[target != 1].max()) if (target != 1).any() else None,
                    "meanProbabilityInsideGt": float(raw0_prob[target == 1].mean()) if (target == 1).any() else None,
                    "meanProbabilityOutsideGt": float(raw0_prob[target != 1].mean()) if (target != 1).any() else None,
                }
            )
            targets.append(target)
    output_dir = config.OUTPUT_ROOT / config.RUN_ID
    tables = output_dir / "tables"
    metrics_dir = output_dir / "metrics"
    tables.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    probability_csv = tables / "raw0_validation_probability_audit.csv"
    pd.DataFrame(rows).to_csv(probability_csv, index=False)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "validation_probabilities.npz"
    if probabilities:
        np.savez_compressed(cache_path, probabilities=np.stack(probabilities), targets=np.stack(targets))
    grid_rows = []
    if probabilities:
        stacked_targets = np.stack(targets)
        stacked_probabilities = np.stack(probabilities)
        for threshold in config.THRESHOLD_GRID:
            for margin in config.MARGIN_GRID:
                gated = apply_raw0_threshold(stacked_probabilities, min_probability=threshold, min_margin=margin)
                metric = metrics_from_predictions(gated, stacked_targets)
                grid_rows.append({"minProbability": threshold, "minMargin": margin, **{k: v for k, v in metric.items() if k != "perClass"}})
        grid_rows.sort(
            key=lambda row: (
                metric_or_default(row, "dice_macro_foreground", -1.0),
                -metric_or_default(row, "raw0PredictedInGtAbsentCases", 10**9),
                metric_or_default(row, "raw0Precision", -1.0),
                metric_or_default(row, "dice_macro_excluding_raw0", -1.0),
            ),
            reverse=True,
        )
    grid_csv = tables / "raw0_threshold_margin_grid.csv"
    pd.DataFrame(grid_rows).to_csv(grid_csv, index=False)
    summary = {
        "status": "completed",
        "validationSlices": len(val_records),
        "presenceScore": "raw0MaxProbability",
        "checkpointSha256": sha256_file(config.V2_CHECKPOINT_PATH),
        "checkpointMetadataValidation": validation,
        "artifacts": {
            "probabilityCsv": str(probability_csv),
            "thresholdGridCsv": str(grid_csv),
            "probabilityCsvSha256": sha256_file(probability_csv),
            "thresholdGridCsvSha256": sha256_file(grid_csv),
            "probabilityCache": str(cache_path) if probabilities else None,
            "probabilityCacheSha256": sha256_file(cache_path) if probabilities else None,
        },
    }
    summary_path = metrics_dir / "raw0_validation_probability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def validate_v2_checkpoint_metadata(checkpoint: Any, config: AxialV3AuditConfig) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("v2 checkpoint must be a mapping")
    if "model_state_dict" not in checkpoint and not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise ValueError("v2 checkpoint does not contain a usable model_state_dict")
    missing: list[str] = []
    validated: list[str] = []

    def check_optional(field: str, expected: Any) -> None:
        if field not in checkpoint:
            missing.append(field)
            return
        value = checkpoint[field]
        if field == "target_size":
            ok = tuple(value) == tuple(expected)
        else:
            ok = value == expected
        if not ok:
            raise ValueError(f"v2 checkpoint metadata mismatch for {field}: {value!r} != {expected!r}")
        validated.append(field)

    check_optional("runId", config.EXPECTED_V2_RUN_ID)
    check_optional("smokeOnly", False)
    check_optional("num_classes", config.NUM_CLASSES)
    check_optional("base_channels", config.BASE_CHANNELS)
    check_optional("target_size", config.TARGET_SIZE)
    check_optional("monitorMetric", "dice_macro_foreground")
    check_optional("raw0Boost", 1.0)
    if config.EXPECTED_AI_SERVICE_COMMIT:
        check_optional("aiServiceCommit", config.EXPECTED_AI_SERVICE_COMMIT)
    if config.EXPECTED_SPLIT_SHA256:
        check_optional("splitSha256", config.EXPECTED_SPLIT_SHA256)
    if config.EXPECTED_CHECKPOINT_SHA256 and config.V2_CHECKPOINT_PATH is not None:
        actual = sha256_file(config.V2_CHECKPOINT_PATH)
        if actual != config.EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("v2 checkpoint SHA-256 mismatch")
        validated.append("checkpointSha256")
    if missing and not config.PFI_ALLOW_INCOMPLETE_V2_CHECKPOINT_METADATA:
        raise ValueError(f"v2 checkpoint metadata is incomplete: {missing}")
    return {"missingMetadataFields": missing, "validatedMetadataFields": validated}


def _artifact_manifest(output_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    manifest_path = output_dir / "manifests" / "iteration_a_artifacts.json"
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        stat = path.stat()
        artifacts.append(
            {
                "relativePath": path.relative_to(output_dir).as_posix(),
                "sizeBytes": stat.st_size,
                "sha256": sha256_file(path),
                "createdAtUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                "role": path.parent.name,
            }
        )
    return artifacts


def run_iteration_a(config: AxialV3AuditConfig | None = None) -> dict[str, Any]:
    cfg = config or AxialV3AuditConfig()
    validate_storage(cfg)
    require_train_val_only(["train", "val"], context="iteration_a")
    records = build_records(cfg)
    structural = run_structural_audit(cfg, records)
    _save_structural_figures(cfg, structural["rows"])
    _save_previews(cfg, records)
    human_review_path = _write_human_review_candidates(cfg, structural["rows"])
    probability = run_validation_probability_audit(cfg, records)
    _save_probability_figures(cfg)
    _save_probability_discordant_preview(cfg, records)
    report_dir = cfg.OUTPUT_ROOT / cfg.RUN_ID / "reports"
    manifests_dir = cfg.OUTPUT_ROOT / cfg.RUN_ID / "manifests"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    config_path = manifests_dir / "iteration_a_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2, default=str, sort_keys=True), encoding="utf-8")
    environment_path = manifests_dir / "iteration_a_environment.json"
    environment_path.write_text(
        json.dumps(
            {"python": platform.python_version(), "platform": platform.platform(), "timeUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = {
        "status": "ready_outputs_created",
        "runId": cfg.RUN_ID,
        "recordCount": len(records),
        "structuralSummary": structural["summary"],
        "probabilitySummary": probability,
        "humanReviewCandidatesCsv": str(human_review_path),
        "configSha256": sha256_file(config_path),
    }
    (report_dir / "iteration_a_report.json").write_text(json.dumps(result, indent=2, default=str, sort_keys=True), encoding="utf-8")
    (report_dir / "iteration_a_report.md").write_text(render_iteration_a_markdown(result, cfg, records), encoding="utf-8")
    artifacts_path = manifests_dir / "iteration_a_artifacts.json"
    artifacts_path.write_text(json.dumps(_artifact_manifest(cfg.OUTPUT_ROOT / cfg.RUN_ID), indent=2, sort_keys=True), encoding="utf-8")
    return result


def render_iteration_a_markdown(result: dict[str, Any], config: AxialV3AuditConfig, records: list[AxialV3Record]) -> str:
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    output_dir = config.OUTPUT_ROOT / config.RUN_ID
    audit_csv = output_dir / "tables" / "raw0_slice_audit.csv"
    grid_csv = output_dir / "tables" / "raw0_threshold_margin_grid.csv"
    split_sha = sha256_file(config.SPLIT_MANIFEST_PATH)
    checkpoint_line = "- checkpoint v2: not configured"
    if config.V2_CHECKPOINT_PATH is not None:
        checkpoint_line = f"- checkpoint v2 sha256: {sha256_file(config.V2_CHECKPOINT_PATH)}"
    structural_lines = ["- raw_0 slices: unavailable", "- border-contact slices: unavailable"]
    if audit_csv.exists():
        df = pd.read_csv(audit_csv)
        positives = int(df["raw0Present"].sum()) if "raw0Present" in df else 0
        border = int(df["touchesBorder"].sum()) if "touchesBorder" in df else 0
        pixel_counts = pd.to_numeric(df.get("raw0PixelCount", pd.Series(dtype=float)), errors="coerce")
        structural_lines = [
            f"- raw_0 positive slices: {positives}/{len(df)}",
            f"- raw_0 border-contact slices: {border}",
            f"- raw_0 pixel count median: {float(pixel_counts.median()) if not pixel_counts.dropna().empty else 'n/a'}",
            f"- raw_0 pixel count max: {int(pixel_counts.max()) if not pixel_counts.dropna().empty else 'n/a'}",
        ]
    grid_lines = ["- probability audit: skipped or no threshold rows"]
    if grid_csv.exists():
        grid = pd.read_csv(grid_csv)
        if not grid.empty:
            top = grid.head(5)
            grid_lines = [
                f"- p>={row.minProbability}, margin>={row.minMargin}: dice_fg={row.dice_macro_foreground:.4f}, raw0_precision={row.raw0Precision:.4f}, raw0_absent_fp_cases={row.raw0PredictedInGtAbsentCases}"
                for row in top.itertuples(index=False)
            ]
    return "\n".join(
        [
            "# Iteration A report",
            "",
            "Generated from train/validation only. Test was not accessed.",
            "",
            "## Configuration",
            f"- runId: {config.RUN_ID}",
            f"- targetSize: {config.TARGET_SIZE}",
            f"- maskLabelMode: {config.MASK_LABEL_MODE}",
            f"- repoRef: {config.REPO_REF or 'not configured'}",
            checkpoint_line,
            "",
            "## Split",
            f"- train slices: {len(train)}",
            f"- validation slices: {len(val)}",
            f"- train patients: {len({r.patientId for r in train})}",
            f"- validation patients: {len({r.patientId for r in val})}",
            f"- split manifest sha256: {split_sha}",
            "",
            "## Structural raw_0 audit",
            *structural_lines,
            "",
            "## Artifacts",
            json.dumps(result.get("structuralSummary", {}).get("outputs", {}), indent=2, default=str),
            f"- human review candidates: {result.get('humanReviewCandidatesCsv')}",
            "",
            "## Probability audit summary",
            json.dumps(result.get("probabilitySummary", {}), indent=2, default=str),
            "",
            "## Top threshold candidates",
            *grid_lines,
            "",
            "## Limitations",
            "- No train/validation conclusions are claimed until the notebook is run on the real dataset.",
            "- raw_0 keeps unresolved anatomical semantics.",
        ]
    )
