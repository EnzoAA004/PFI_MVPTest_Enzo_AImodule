from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for relative in ("ai_service", "src"):
    candidate = REPO_ROOT / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pfi_ai_service.model_architectures import build_checkpoint_model  # noqa: E402
from pfi_ai_service.real_inference_runtime import (  # noqa: E402
    CachedModel,
    LoadedInput,
    load_input,
    resize_image,
    runtime_device,
    select_slice,
)

IMAGE_EXTENSIONS = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".mha", ".mhd", ".dcm"}
MASK_EXTENSIONS = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CasePair:
    case_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class ClassAccumulator:
    class_id: int
    intersection: int = 0
    pred_pixels: int = 0
    gt_pixels: int = 0
    union: int = 0
    absent_both_cases: int = 0

    def add(self, prediction: np.ndarray, ground_truth: np.ndarray) -> "ClassAccumulator":
        pred_mask = np.asarray(prediction) == self.class_id
        gt_mask = np.asarray(ground_truth) == self.class_id
        intersection = int(np.logical_and(pred_mask, gt_mask).sum())
        union = int(np.logical_or(pred_mask, gt_mask).sum())
        pred_pixels = int(pred_mask.sum())
        gt_pixels = int(gt_mask.sum())
        absent_both = int(pred_pixels == 0 and gt_pixels == 0)
        return ClassAccumulator(
            class_id=self.class_id,
            intersection=self.intersection + intersection,
            pred_pixels=self.pred_pixels + pred_pixels,
            gt_pixels=self.gt_pixels + gt_pixels,
            union=self.union + union,
            absent_both_cases=self.absent_both_cases + absent_both,
        )

    def dice(self) -> float | None:
        denominator = self.pred_pixels + self.gt_pixels
        if denominator == 0:
            return None
        return float((2.0 * self.intersection) / denominator)

    def iou(self) -> float | None:
        if self.union == 0:
            return None
        return float(self.intersection / self.union)


def dice_iou_for_class(prediction: np.ndarray, ground_truth: np.ndarray, class_id: int) -> tuple[float | None, float | None, str | None]:
    accumulator = ClassAccumulator(class_id).add(prediction, ground_truth)
    if accumulator.pred_pixels == 0 and accumulator.gt_pixels == 0:
        return None, None, "absent_in_gt_and_prediction"
    return accumulator.dice(), accumulator.iou(), None


def macro_foreground(values: dict[int, float | None]) -> float | None:
    defined = [value for class_id, value in values.items() if class_id != 0 and value is not None]
    if not defined:
        return None
    return float(np.mean(defined))


def load_mask(mask_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    suffix = mask_path.suffix.lower()
    if suffix == ".npy":
        mask = np.load(mask_path)
    elif suffix in MASK_EXTENSIONS:
        mask = np.asarray(Image.open(mask_path).convert("L"))
    else:
        raise ValueError(f"Formato de mascara no soportado: {suffix}")
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] == 1 else mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Se esperaba mascara 2D; shape={mask.shape}")
    if tuple(mask.shape) != target_size:
        image = Image.fromarray(mask.astype(np.int32), mode="I")
        image = image.resize((target_size[1], target_size[0]), resample=Image.Resampling.NEAREST)
        mask = np.asarray(image)
    return mask.astype(np.int64)


def find_case_pairs(test_dir: Path) -> list[CasePair]:
    images_dir = test_dir / "images"
    masks_dir = test_dir / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError("test-dir debe contener subdirectorios images/ y masks/")

    mask_by_stem: dict[str, Path] = {}
    for mask_path in sorted(path for path in masks_dir.rglob("*") if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS):
        relative_stem = str(mask_path.relative_to(masks_dir).with_suffix(""))
        mask_by_stem[relative_stem] = mask_path
        if relative_stem.endswith("_mask"):
            mask_by_stem.setdefault(relative_stem[:-5], mask_path)
        else:
            mask_by_stem.setdefault(f"{relative_stem}_mask", mask_path)

    pairs: list[CasePair] = []
    for image_path in sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
        relative_stem = str(image_path.relative_to(images_dir).with_suffix(""))
        mask_path = mask_by_stem.get(relative_stem) or mask_by_stem.get(f"{relative_stem}_mask")
        if mask_path is None:
            raise FileNotFoundError(f"No se encontro mascara GT para caseId={relative_stem}")
        pairs.append(CasePair(case_id=relative_stem.replace("\\", "/"), image_path=image_path, mask_path=mask_path))
    if not pairs:
        raise FileNotFoundError("No se encontraron imagenes evaluables en images/")
    return pairs


def model_key_for_plane(plane: str) -> str:
    if plane == "sagittal":
        return "sagittal_spider"
    if plane == "axial":
        return "axial_t2_alkafri"
    raise ValueError(f"plane invalido: {plane}")


def load_checkpoint_model(checkpoint_path: Path, plane: str) -> CachedModel:
    device = runtime_device()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model, runtime_metadata = build_checkpoint_model(model_key_for_plane(plane), checkpoint)
    model.to(device)
    model.eval()
    return CachedModel(
        model_key=model_key_for_plane(plane),
        path=checkpoint_path,
        mtime_ns=checkpoint_path.stat().st_mtime_ns,
        device=str(device),
        model=model,
        checkpoint=checkpoint,
        runtime_metadata=runtime_metadata,
    )


def predict_case(cached: CachedModel, plane: str, image_path: Path, target_size: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    loaded = load_input(str(image_path), plane)
    image, selected_slice, slice_count, selected_axis = select_slice(loaded, plane, cached, target_size, {})
    tensor = torch.from_numpy(image[None, None]).float().to(cached.device)
    with torch.inference_mode():
        logits = cached.model(tensor)
        prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1)[0]
    return prediction.detach().cpu().numpy().astype(np.int64), {
        "sourceShape": [int(value) for value in loaded.array.shape],
        "processedShape": [int(value) for value in image.shape],
        "selectedSlice": int(selected_slice),
        "sliceCount": int(slice_count),
        "selectedAxis": int(selected_axis),
    }


def evaluate_pairs(cached: CachedModel, pairs: Iterable[CasePair], plane: str, num_classes: int, target_size: tuple[int, int]) -> dict[str, Any]:
    accumulators = {class_id: ClassAccumulator(class_id) for class_id in range(num_classes)}
    cases: list[dict[str, Any]] = []
    for pair in pairs:
        prediction, metadata = predict_case(cached, plane, pair.image_path, target_size)
        ground_truth = load_mask(pair.mask_path, target_size)
        if prediction.shape != ground_truth.shape:
            raise ValueError(f"Shape mismatch luego de resize para caseId={pair.case_id}: pred={prediction.shape}, gt={ground_truth.shape}")
        for class_id in range(num_classes):
            accumulators[class_id] = accumulators[class_id].add(prediction, ground_truth)
        cases.append({"caseId": pair.case_id, **metadata})

    dice_by_class = {class_id: accumulator.dice() for class_id, accumulator in accumulators.items()}
    iou_by_class = {class_id: accumulator.iou() for class_id, accumulator in accumulators.items()}
    absent_notes = {
        class_id: "excluded_absent_in_gt_and_prediction"
        for class_id, accumulator in accumulators.items()
        if accumulator.pred_pixels + accumulator.gt_pixels == 0
    }
    return {
        "plane": plane,
        "numClasses": num_classes,
        "targetSize": list(target_size),
        "nCases": len(cases),
        "diceByClass": {str(key): value for key, value in dice_by_class.items()},
        "iouByClass": {str(key): value for key, value in iou_by_class.items()},
        "diceMacroForeground": macro_foreground(dice_by_class),
        "iouMacroForeground": macro_foreground(iou_by_class),
        "absentClassNotes": {str(key): value for key, value in absent_notes.items()},
        "cases": cases,
        "deidentified": True,
        "diagnosisGenerated": False,
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def parse_target_size(raw: str) -> tuple[int, int]:
    parts = [part.strip() for part in raw.replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return size, size
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise argparse.ArgumentTypeError("--target-size debe ser N o H,W")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evalua Dice/IoU por clase contra un held-out local deidentificado.")
    parser.add_argument("--plane", choices=["sagittal", "axial"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True, help="Directorio con images/ y masks/.")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--target-size", type=parse_target_size, default=(256, 256))
    parser.add_argument("--output", type=Path, default=Path("docs/qual-003-eval-report.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint no encontrado: {args.checkpoint}")
    pairs = find_case_pairs(args.test_dir)
    cached = load_checkpoint_model(args.checkpoint, args.plane)
    report = evaluate_pairs(cached, pairs, args.plane, args.num_classes, args.target_size)
    report["checkpointFile"] = args.checkpoint.name
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Reporte JSON escrito en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())