from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS, build_agent_decision
from .asset_registry import registered_assets_for_run, register_run_assets
from .model_architectures import build_checkpoint_model
from .model_artifacts import model_artifact_path, model_status
from .settings import MODEL_REGISTRY, get_settings

SUPPORTED_EXTENSIONS = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".mha", ".mhd", ".dcm"}
PALETTE = {
    1: (230, 25, 75),
    2: (60, 180, 75),
    3: (0, 130, 200),
    4: (245, 130, 48),
    5: (145, 30, 180),
    6: (70, 240, 240),
}


@dataclass
class CachedModel:
    model_key: str
    path: Path
    mtime_ns: int
    device: str
    model: torch.nn.Module
    checkpoint: Any
    runtime_metadata: Dict[str, Any]


@dataclass
class LoadedInput:
    array: np.ndarray
    path: Path
    suffix: str
    spacing_xyz: tuple[float, ...] | None
    metadata: Dict[str, Any]


_MODEL_CACHE: dict[str, CachedModel] = {}


def runtime_device() -> torch.device:
    configured = os.getenv("PFI_INFERENCE_DEVICE", "auto").strip().lower()
    if configured == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if configured not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"PFI_INFERENCE_DEVICE invalido: {configured}")
    return torch.device("cuda" if configured == "auto" and torch.cuda.is_available() else "cpu")


def runtime_status() -> Dict[str, Any]:
    device = runtime_device()
    return {
        "status": "pytorch_runtime_ready",
        "torchVersion": torch.__version__,
        "cudaAvailable": torch.cuda.is_available(),
        "device": str(device),
        "loadedModels": sorted(_MODEL_CACHE.keys()),
        "supportedExtensions": sorted(SUPPORTED_EXTENSIONS),
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_key: str) -> CachedModel:
    path = model_artifact_path(model_key)
    if path is None or not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Artifact no disponible para {model_key}: {path}")
    device = runtime_device()
    mtime_ns = path.stat().st_mtime_ns
    cached = _MODEL_CACHE.get(model_key)
    if cached and cached.path == path and cached.mtime_ns == mtime_ns and cached.device == str(device):
        return cached

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model, runtime_metadata = build_checkpoint_model(model_key, checkpoint)
    model.to(device)
    model.eval()
    cached = CachedModel(
        model_key=model_key,
        path=path,
        mtime_ns=mtime_ns,
        device=str(device),
        model=model,
        checkpoint=checkpoint,
        runtime_metadata=runtime_metadata,
    )
    _MODEL_CACHE[model_key] = cached
    return cached


def resolve_input_path(input_path: str, plane: str) -> Path:
    path = Path(input_path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Input real no encontrado: {input_path}")
    if not path.is_dir():
        raise ValueError(f"Input no soportado: {input_path}")
    files = sorted(file for file in path.rglob("*") if file.is_file() and normalized_suffix(file) in SUPPORTED_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No hay archivos soportados dentro de {input_path}")
    if plane == "sagittal":
        volume = next((file for file in files if normalized_suffix(file) in {".mha", ".mhd", ".npy"}), None)
        return volume or files[len(files) // 2]
    dicoms = [file for file in files if normalized_suffix(file) == ".dcm"]
    selected = dicoms if dicoms else files
    return selected[len(selected) // 2]


def normalized_suffix(path: Path) -> str:
    return path.suffix.lower()


def load_input(input_path: str, plane: str) -> LoadedInput:
    path = resolve_input_path(input_path, plane)
    suffix = normalized_suffix(path)
    spacing: tuple[float, ...] | None = None
    metadata: Dict[str, Any] = {}

    if suffix == ".npy":
        array = np.load(path)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        array = np.asarray(Image.open(path).convert("L"))
    elif suffix in {".mha", ".mhd"}:
        try:
            import SimpleITK as sitk
        except Exception as exc:
            raise ImportError("SimpleITK es requerido para archivos MHA/MHD") from exc
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        spacing = tuple(float(value) for value in image.GetSpacing())
        metadata["origin"] = tuple(float(value) for value in image.GetOrigin())
        metadata["direction"] = tuple(float(value) for value in image.GetDirection())
    elif suffix == ".dcm":
        try:
            import pydicom
        except Exception as exc:
            raise ImportError("pydicom es requerido para archivos DICOM") from exc
        dataset = pydicom.dcmread(str(path), force=True)
        array = dataset.pixel_array
        pixel_spacing = getattr(dataset, "PixelSpacing", None)
        if pixel_spacing is not None and len(pixel_spacing) >= 2:
            spacing = (float(pixel_spacing[1]), float(pixel_spacing[0]))
        metadata["seriesInstanceUid"] = str(getattr(dataset, "SeriesInstanceUID", ""))
        metadata["sopInstanceUid"] = str(getattr(dataset, "SOPInstanceUID", ""))
    else:
        raise ValueError(f"Formato de input no soportado: {suffix}")

    array = np.asarray(array)
    if array.ndim not in {2, 3}:
        raise ValueError(f"Se esperaba imagen 2D o volumen 3D; shape={array.shape}")
    return LoadedInput(array=array, path=path, suffix=suffix, spacing_xyz=spacing, metadata=metadata)


def robust_percentile_normalize(array: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(value)
    if not finite.any():
        return np.zeros_like(value, dtype=np.float32)
    low, high = np.percentile(value[finite], [p_low, p_high])
    if float(high) <= float(low):
        return np.zeros_like(value, dtype=np.float32)
    clipped = np.clip(value, low, high)
    return ((clipped - low) / (float(high) - float(low) + 1e-8)).astype(np.float32)


def resize_image(array: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    normalized = robust_percentile_normalize(array)
    image = Image.fromarray(np.clip(normalized * 255.0, 0, 255).astype(np.uint8))
    resized = image.resize((target_size[1], target_size[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def slice_axis_for(loaded: LoadedInput, plane: str, checkpoint: Any, metadata: Dict[str, Any]) -> int:
    if loaded.array.ndim == 2:
        return 0
    if metadata.get("sliceAxis") is not None:
        axis = int(metadata["sliceAxis"])
    elif plane == "sagittal" and isinstance(checkpoint, dict) and checkpoint.get("sagittal_axis") is not None:
        axis = int(checkpoint["sagittal_axis"])
    elif plane == "sagittal":
        axis = int(np.argmin(loaded.array.shape))
    else:
        axis = 0
    if axis < 0 or axis >= loaded.array.ndim:
        raise ValueError(f"sliceAxis fuera de rango: {axis} para shape={loaded.array.shape}")
    return axis


def candidate_indices(length: int, radius: int = 3) -> list[int]:
    center = length // 2
    return list(range(max(0, center - radius), min(length - 1, center + radius) + 1))


def select_slice(
    loaded: LoadedInput,
    plane: str,
    cached: CachedModel,
    target_size: tuple[int, int],
    metadata: Dict[str, Any],
) -> tuple[np.ndarray, int, int, int]:
    if loaded.array.ndim == 2:
        return resize_image(loaded.array, target_size), 0, 1, 0

    axis = slice_axis_for(loaded, plane, cached.checkpoint, metadata)
    count = int(loaded.array.shape[axis])
    if metadata.get("sliceIndex") is not None:
        selected = max(0, min(int(metadata["sliceIndex"]), count - 1))
        raw_slice = np.take(loaded.array, selected, axis=axis)
        return resize_image(raw_slice, target_size), selected, count, axis

    normalized_volume = robust_percentile_normalize(loaded.array)
    if plane == "sagittal":
        indices = candidate_indices(count, int(metadata.get("sliceWindowRadius", 3)))
        prepared = [resize_image(np.take(normalized_volume, index, axis=axis), target_size) for index in indices]
        tensor = torch.from_numpy(np.stack(prepared)[:, None]).float().to(cached.device)
        with torch.inference_mode():
            probabilities = torch.softmax(cached.model(tensor), dim=1)
            foreground_scores = (1.0 - probabilities[:, 0]).mean(dim=(1, 2))
        best = int(torch.argmax(foreground_scores).detach().cpu().item())
        return prepared[best], int(indices[best]), count, axis

    selected = count // 2
    raw_slice = np.take(normalized_volume, selected, axis=axis)
    return resize_image(raw_slice, target_size), selected, count, axis


def in_plane_spacing(loaded: LoadedInput, selected_axis: int) -> tuple[float, float] | None:
    spacing = loaded.spacing_xyz
    if spacing is None:
        return None
    if loaded.array.ndim == 2 and len(spacing) >= 2:
        return float(spacing[1]), float(spacing[0])
    if loaded.array.ndim == 3 and len(spacing) >= 3:
        array_axis_spacing = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
        remaining = [value for index, value in enumerate(array_axis_spacing) if index != selected_axis]
        if len(remaining) == 2:
            return float(remaining[0]), float(remaining[1])
    return None


def boundary_polygon(binary: np.ndarray, max_points: int = 96) -> list[Dict[str, float]]:
    mask = np.asarray(binary, dtype=bool)
    if not mask.any():
        return []
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = center & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    coords = np.argwhere(mask & ~interior)
    if len(coords) < 3:
        coords = np.argwhere(mask)
    cy, cx = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 0] - cy, coords[:, 1] - cx)
    ordered = coords[np.argsort(angles)]
    if len(ordered) > max_points:
        sample_indices = np.linspace(0, len(ordered) - 1, max_points, dtype=int)
        ordered = ordered[sample_indices]
    return [{"x": round(float(x), 1), "y": round(float(y), 1)} for y, x in ordered]


def class_name(model_key: str, class_id: int) -> str:
    names = MODEL_REGISTRY.get(model_key, {}).get("class_names", {})
    return str(names.get(class_id, f"class_{class_id}"))


def class_color(class_id: int) -> str:
    red, green, blue = PALETTE.get(class_id, (255, 255, 0))
    return f"#{red:02x}{green:02x}{blue:02x}"


def build_masks(model_key: str, plane: str, prediction: np.ndarray, confidence: np.ndarray, series_id: str, slice_index: int) -> list[Dict[str, Any]]:
    masks: list[Dict[str, Any]] = []
    for class_id in sorted(int(value) for value in np.unique(prediction) if int(value) != 0):
        binary = prediction == class_id
        points = boundary_polygon(binary)
        if not points:
            continue
        class_confidence = float(confidence[binary].mean()) if binary.any() else 0.0
        masks.append({
            "id": f"mask-{plane}-{class_name(model_key, class_id).replace('_', '-')}",
            "label": class_name(model_key, class_id),
            "className": class_name(model_key, class_id),
            "classId": class_id,
            "color": class_color(class_id),
            "confidence": round(class_confidence, 4),
            "editable": True,
            "enabled": True,
            "contours": [{"seriesId": series_id, "sliceIndex": slice_index, "points": points}],
        })
    return masks


def build_landmarks(masks: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    landmarks: list[Dict[str, Any]] = []
    for mask in masks:
        contour = (mask.get("contours") or [{}])[0]
        points = contour.get("points") or []
        if not points:
            continue
        x = sum(float(point["x"]) for point in points) / len(points)
        y = sum(float(point["y"]) for point in points) / len(points)
        landmarks.append({
            "id": f"lm-{mask['id']}-centroid",
            "label": f"{mask['label']} centroid",
            "seriesId": contour.get("seriesId"),
            "sliceIndex": contour.get("sliceIndex", 0),
            "x": round(x, 1),
            "y": round(y, 1),
            "editable": True,
            "linkedMaskId": mask.get("id"),
        })
    return landmarks


def build_measurements(
    model_key: str,
    plane: str,
    prediction: np.ndarray,
    confidence: np.ndarray,
    spacing: tuple[float, float] | None,
) -> list[Dict[str, Any]]:
    values: list[Dict[str, Any]] = []
    row_spacing, col_spacing = spacing if spacing else (1.0, 1.0)
    physical = spacing is not None
    for class_id in sorted(int(value) for value in np.unique(prediction) if int(value) != 0):
        binary = prediction == class_id
        ys, xs = np.where(binary)
        if len(xs) == 0:
            continue
        label = class_name(model_key, class_id)
        class_confidence = float(confidence[binary].mean())
        width = float(xs.max() - xs.min() + 1) * col_spacing
        height = float(ys.max() - ys.min() + 1) * row_spacing
        area = float(len(xs)) * row_spacing * col_spacing
        dimension_unit = "mm" if physical else "px"
        area_unit = "mm2" if physical else "px2"
        common = {
            "level": plane,
            "source": "AI",
            "confidence": round(class_confidence, 4),
            "status": "pendiente",
            "outlier": False,
            "linkedLandmarks": [f"lm-mask-{plane}-{label.replace('_', '-')}-centroid"],
        }
        values.extend([
            {"id": f"{plane}-{label}-area", "label": f"{label} area", "value": round(area, 2), "aiValue": round(area, 2), "reviewerValue": None, "unit": area_unit, **common},
            {"id": f"{plane}-{label}-width", "label": f"{label} width", "value": round(width, 2), "aiValue": round(width, 2), "reviewerValue": None, "unit": dimension_unit, **common},
            {"id": f"{plane}-{label}-height", "label": f"{label} height", "value": round(height, 2), "aiValue": round(height, 2), "reviewerValue": None, "unit": dimension_unit, **common},
        ])
    return values


def save_outputs(run_id: str, plane: str, image: np.ndarray, prediction: np.ndarray, confidence: np.ndarray) -> Dict[str, str]:
    output_dir = get_settings().output_dir / "real_inference" / run_id / plane
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "input.png"
    mask_path = output_dir / "mask.npy"
    confidence_path = output_dir / "confidence.npy"
    overlay_path = output_dir / "overlay.png"

    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(image_path)
    np.save(mask_path, prediction.astype(np.uint8))
    np.save(confidence_path, confidence.astype(np.float32))

    gray = np.stack([image, image, image], axis=-1)
    overlay = gray.copy()
    alpha = 0.42
    for class_id in sorted(int(value) for value in np.unique(prediction) if int(value) != 0):
        color = np.asarray(PALETTE.get(class_id, (255, 255, 0)), dtype=np.float32) / 255.0
        selected = prediction == class_id
        overlay[selected] = (1.0 - alpha) * overlay[selected] + alpha * color
    Image.fromarray(np.clip(overlay * 255.0, 0, 255).astype(np.uint8)).save(overlay_path)
    outputs = {
        "imagePath": str(image_path),
        "maskPath": str(mask_path),
        "confidencePath": str(confidence_path),
        "overlayPath": str(overlay_path),
    }
    register_run_assets(run_id, plane, outputs)
    return outputs


def run_real_inference(request: Any, run_id: str) -> Dict[str, Any]:
    cached = load_model(request.model_key)
    artifact = model_status(request.model_key, dict(MODEL_REGISTRY.get(request.model_key, {})))
    if not artifact.get("availableForRealInference", False):
        raise RuntimeError(f"Modelo no habilitado para real_baseline: {request.model_key}")

    loaded = load_input(request.input_path, request.plane)
    target_size = tuple(cached.runtime_metadata.get("targetSize", (256, 256)))
    image, selected_slice, slice_count, selected_axis = select_slice(
        loaded,
        request.plane,
        cached,
        (int(target_size[0]), int(target_size[1])),
        dict(request.metadata or {}),
    )
    tensor = torch.from_numpy(image[None, None]).float().to(cached.device)
    with torch.inference_mode():
        logits = cached.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    prediction = torch.argmax(probabilities, dim=0).detach().cpu().numpy().astype(np.uint8)
    confidence = torch.max(probabilities, dim=0).values.detach().cpu().numpy().astype(np.float32)
    foreground = prediction > 0
    mean_confidence = float(confidence.mean())
    mean_foreground_confidence = float(confidence[foreground].mean()) if foreground.any() else 0.0
    foreground_ratio = float(foreground.mean())
    present_classes = sorted(int(value) for value in np.unique(prediction) if int(value) != 0)

    series_id = "series-sag-t2" if request.plane == "sagittal" else "series-ax-t2"
    outputs = save_outputs(run_id, request.plane, image, prediction, confidence)
    assets = registered_assets_for_run(run_id, request.plane)
    spacing = in_plane_spacing(loaded, selected_axis)
    masks = build_masks(request.model_key, request.plane, prediction, confidence, series_id, selected_slice)
    landmarks = build_landmarks(masks)
    measurement_values = build_measurements(request.model_key, request.plane, prediction, confidence, spacing)

    flags = ["real_baseline_inference_completed"]
    if not present_classes:
        flags.append("real_inference_empty_foreground")
    if mean_foreground_confidence < 0.70:
        flags.append("real_inference_low_foreground_confidence")
    agent_decision = build_agent_decision(plane=request.plane, model_key=request.model_key, flags=flags)
    trace_id = request.metadata.get("traceId") or request.metadata.get("correlationId") or request.metadata.get("backendTraceId")
    quality = {
        "maskCount": len(masks),
        "landmarkCount": len(landmarks),
        "measurementCount": len(measurement_values),
        "meanConfidence": round(mean_confidence, 4),
        "meanForegroundConfidence": round(mean_foreground_confidence, 4),
        "foregroundRatio": round(foreground_ratio, 6),
        "presentClasses": present_classes,
        "pixelSpacing": list(spacing) if spacing else None,
        "measurementsDerivedFromPredictionMask": True,
    }
    requested_mode = str(request.metadata.get("inferenceMode", "real_baseline"))
    return {
        "run_id": run_id,
        "runId": run_id,
        "traceId": trace_id,
        "case_id": request.case_id,
        "caseId": request.case_id,
        "studyId": f"STUDY-{request.case_id.replace('CASE-', '')}",
        "patientId": request.metadata.get("patientId", "PAT-DEIDENTIFIED"),
        "studyDate": request.metadata.get("studyDate"),
        "modality": "MRI",
        "bodyRegion": "Lumbar Spine",
        "reviewStatus": "pendiente",
        "plane": request.plane,
        "model_key": request.model_key,
        "modelKey": request.model_key,
        "modelVersion": artifact.get("version"),
        "input_path": request.input_path,
        "inputPath": request.input_path,
        "series": [{
            "id": series_id,
            "name": "Sagittal T2" if request.plane == "sagittal" else "Axial T2",
            "plane": request.plane,
            "sequence": "T2",
            "sliceCount": slice_count,
            "selectedSlice": selected_slice,
            "imageUrl": None,
            "overlayUrl": None,
            "imagePath": outputs["imagePath"],
            "overlayPath": outputs["overlayPath"],
            "overlayOpacity": 0.74,
            "status": "real_baseline_ready",
        }],
        "masks": masks,
        "landmarks": landmarks,
        "measurements": {
            "status": "real_baseline_ready",
            "values": measurement_values,
            "source": "pytorch_real_baseline",
            "description": "Mediciones descriptivas derivadas de la mascara predicha; requieren revision profesional.",
        },
        "measurementValues": measurement_values,
        "overlay_path": outputs["overlayPath"],
        "overlayPath": outputs["overlayPath"],
        "aiOutput": {
            "status": "real_baseline_ready",
            "label": "Inferencia real baseline",
            "description": "Salida generada por el checkpoint PyTorch real del modelo seleccionado.",
            "inferenceMode": "real_baseline",
            "requestedInferenceMode": requested_mode,
            "realInferenceAvailable": True,
            "modelReadiness": artifact.get("readiness"),
            "runtime": "pytorch",
            "device": cached.device,
            "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
            "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
            "agentDecision": agent_decision,
        },
        "agent_decision": agent_decision,
        "agentDecision": agent_decision,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "not_clinical_diagnosis": NOT_CLINICAL_DIAGNOSIS,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
        "quality": quality,
        "assets": assets,
        "modelArtifact": artifact,
        "metadata": {
            **request.metadata,
            "traceId": trace_id,
            "inferenceMode": "real_baseline",
            "requestedInferenceMode": requested_mode,
            "modelReadiness": artifact.get("readiness"),
            "artifactHash": artifact.get("artifactHash"),
            "runtime": "pytorch",
            "device": cached.device,
            "selectedSlice": selected_slice,
            "selectedAxis": selected_axis,
            "sliceCount": slice_count,
            "sourceShape": [int(value) for value in loaded.array.shape],
            "processedShape": [int(value) for value in prediction.shape],
            "inputFormat": loaded.suffix,
            "sourcePath": str(loaded.path),
            "outputFiles": outputs,
            "assets": assets,
            "quality": quality,
            "deidentified": True,
            "diagnosisGenerated": False,
        },
    }
