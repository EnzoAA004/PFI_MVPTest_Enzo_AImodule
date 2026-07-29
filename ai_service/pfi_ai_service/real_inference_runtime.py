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

SUPPORTED_EXTENSIONS = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".mha", ".mhd", ".dcm", ".ima"}
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


def metadata_bool(metadata: Dict[str, Any], key: str, default: bool) -> bool:
    value = metadata.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", "off"}


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
    # A pre-assembled 3D volume file wins for any plane.
    volume = next(
        (file for file in sorted(path.rglob("*")) if file.is_file() and normalized_suffix(file) in {".mha", ".mhd", ".npy"}),
        None,
    )
    if volume is not None:
        return volume
    # A DICOM series (2+ slices) reads as a 3D volume via read_dicom_series. Detection
    # is by content (GDCM), so it works for .dcm, Siemens .ima and extension-less PACS
    # exports alike — not just files literally named *.dcm.
    if count_dicom_slices(path) >= 2:
        return path
    # Legacy fallback: a single slice or a folder of flat 2D images stays 2D.
    files = sorted(file for file in path.rglob("*") if file.is_file() and normalized_suffix(file) in SUPPORTED_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No hay archivos soportados dentro de {input_path}")
    return files[len(files) // 2]


def count_dicom_slices(directory: Path) -> int:
    """Count DICOM slices under a directory by content (GDCM), ignoring file extension."""
    try:
        import SimpleITK as sitk
    except Exception:  # pragma: no cover - dependency guard
        return 0
    reader = sitk.ImageSeriesReader()
    total = 0
    for dirpath, _dirs, filenames in os.walk(directory):
        if not filenames:
            continue
        for series_id in reader.GetGDCMSeriesIDs(dirpath):
            total += len(reader.GetGDCMSeriesFileNames(dirpath, series_id))
    return total


def read_dicom_series(directory: Path) -> tuple[np.ndarray, tuple[float, ...] | None, Dict[str, Any]]:
    """Assemble a folder of DICOM slices into a 3D volume using SimpleITK/GDCM.

    Picks the series with the most slices when several are present, orders slices by
    geometry, and returns the array as [slices, rows, cols] with its physical spacing.
    """
    try:
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("SimpleITK es requerido para series DICOM") from exc
    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(str(directory)))
    best_id = ""
    if series_ids:
        best_files: list[str] = []
        for sid in series_ids:
            names = list(reader.GetGDCMSeriesFileNames(str(directory), sid))
            if len(names) > len(best_files):
                best_id, best_files = sid, names
        file_names = best_files
    else:
        # No series metadata at the top level: search recursively for a single series.
        file_names = list(reader.GetGDCMSeriesFileNames(str(directory), "", False, True))
    if len(file_names) < 2:
        raise ValueError(f"La serie DICOM debe tener al menos 2 cortes; encontrados={len(file_names)}")
    reader.SetFileNames(file_names)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)
    spacing = tuple(float(value) for value in image.GetSpacing())
    metadata: Dict[str, Any] = {
        "seriesInstanceUid": best_id,
        "origin": tuple(float(value) for value in image.GetOrigin()),
        "direction": tuple(float(value) for value in image.GetDirection()),
        "sliceCount": int(array.shape[0]),
    }
    return array, spacing, metadata


def normalized_suffix(path: Path) -> str:
    return path.suffix.lower()


def load_input(input_path: str, plane: str) -> LoadedInput:
    path = resolve_input_path(input_path, plane)
    spacing: tuple[float, ...] | None = None
    metadata: Dict[str, Any] = {}

    if path.is_dir():
        # A directory reaching here is a DICOM series (resolve_input_path only
        # returns the folder when it holds 2+ .dcm slices) -> stack into a 3D volume.
        array, spacing, metadata = read_dicom_series(path)
        suffix = ".dcm"
    elif (suffix := normalized_suffix(path)) == ".npy":
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
    metadata["inputShapeNative"] = [int(value) for value in array.shape]
    return LoadedInput(array=array, path=path, suffix=suffix, spacing_xyz=spacing, metadata=metadata)


def sagittal_orientation_transform(array: np.ndarray) -> str:
    if array.ndim == 3 and array.shape[0] <= 64 and array.shape[-1] > 64:
        return "move_axis_0_to_last"
    return "none"


def canonicalize_sagittal_array(array: np.ndarray, transform: str | None = None) -> tuple[np.ndarray, str]:
    selected = transform or sagittal_orientation_transform(array)
    if selected == "none":
        return array, "none"
    if selected == "move_axis_0_to_last":
        if array.ndim != 3:
            raise ValueError(f"Transformacion sagital requiere volumen 3D; shape={array.shape}")
        return np.moveaxis(array, 0, -1), selected
    raise ValueError(f"Transformacion de orientacion no soportada: {selected}")


def array_axis_spacing_native(spacing_xyz: tuple[float, ...] | None, ndim: int) -> list[float] | None:
    if spacing_xyz is None:
        return None
    if ndim == 2 and len(spacing_xyz) >= 2:
        sx, sy = float(spacing_xyz[0]), float(spacing_xyz[1])
        return [sy, sx]
    if ndim == 3 and len(spacing_xyz) >= 3:
        sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
        return [sz, sy, sx]
    return None


def canonical_axis_spacing(native_spacing: list[float] | None, transform: str) -> list[float] | None:
    if native_spacing is None:
        return None
    if transform == "none":
        return list(native_spacing)
    if transform == "move_axis_0_to_last":
        if len(native_spacing) != 3:
            raise ValueError("move_axis_0_to_last requiere spacing de 3 ejes")
        return [native_spacing[1], native_spacing[2], native_spacing[0]]
    raise ValueError(f"Transformacion de spacing no soportada: {transform}")


def canonicalize_loaded_input(loaded: LoadedInput, plane: str, metadata: Dict[str, Any]) -> LoadedInput:
    native_shape = [int(value) for value in loaded.array.shape]
    transform = "none"
    array = loaded.array
    if plane == "sagittal":
        override = metadata.get("inputOrientationTransform") or os.getenv("PFI_SAGITTAL_ORIENTATION_TRANSFORM")
        array, transform = canonicalize_sagittal_array(array, str(override) if override else None)
    native_spacing = array_axis_spacing_native(loaded.spacing_xyz, loaded.array.ndim)
    canonical_spacing = canonical_axis_spacing(native_spacing, transform)
    canonical_metadata = {
        **loaded.metadata,
        "spacingXyz": [float(value) for value in loaded.spacing_xyz] if loaded.spacing_xyz is not None else None,
        "arrayAxisSpacingNative": native_spacing,
        "arrayAxisSpacingCanonical": canonical_spacing,
        "inputShapeNative": native_shape,
        "inputShapeCanonical": [int(value) for value in array.shape],
        "inputOrientationTransform": transform,
    }
    return LoadedInput(
        array=np.asarray(array),
        path=loaded.path,
        suffix=loaded.suffix,
        spacing_xyz=loaded.spacing_xyz,
        metadata=canonical_metadata,
    )


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
    axis_spacing = loaded.metadata.get("arrayAxisSpacingCanonical")
    if axis_spacing is None:
        return None
    if loaded.array.ndim == 2 and len(axis_spacing) >= 2:
        return float(axis_spacing[0]), float(axis_spacing[1])
    if loaded.array.ndim == 3 and len(axis_spacing) >= 3:
        remaining = [float(value) for index, value in enumerate(axis_spacing) if index != selected_axis]
        if len(remaining) == 2:
            return float(remaining[0]), float(remaining[1])
    return None


def slice_asset_name(index: int) -> str:
    if index < 0 or index > 999:
        raise ValueError("slice_index_out_of_catalog_range")
    return f"slice-{index:03d}.png"


def slice_overlay_asset_name(index: int) -> str:
    if index < 0 or index > 999:
        raise ValueError("slice_index_out_of_catalog_range")
    return f"slice-{index:03d}-overlay.png"


def slice_count_for_catalog(loaded: LoadedInput, selected_axis: int) -> int:
    if loaded.array.ndim == 2:
        return 1
    return int(loaded.array.shape[selected_axis])


def slice_image_for_catalog(loaded: LoadedInput, index: int, selected_axis: int, target_size: tuple[int, int]) -> np.ndarray:
    if loaded.array.ndim == 2:
        if index != 0:
            raise ValueError("slice_index_out_of_range")
        source = loaded.array
    else:
        source = np.take(loaded.array, index, axis=selected_axis)
    return resize_image(source, target_size)


def save_slice_catalog_assets(
    run_id: str,
    plane: str,
    loaded: LoadedInput,
    selected_axis: int,
    selected_slice: int,
    target_size: tuple[int, int],
    overlay_path: str,
) -> dict[str, str]:
    output_dir = get_settings().output_dir / "real_inference" / run_id / plane
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    count = slice_count_for_catalog(loaded, selected_axis)
    for index in range(count):
        asset_name = slice_asset_name(index)
        preview_path = output_dir / asset_name
        image = slice_image_for_catalog(loaded, index, selected_axis, target_size)
        Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(preview_path)
        outputs[f"slicePreview{index:03d}Path"] = str(preview_path)

    overlay_source = Path(overlay_path)
    if overlay_source.exists() and overlay_source.is_file():
        overlay_asset = slice_overlay_asset_name(selected_slice)
        overlay_target = output_dir / overlay_asset
        overlay_target.write_bytes(overlay_source.read_bytes())
        outputs[f"sliceOverlay{selected_slice:03d}Path"] = str(overlay_target)
    register_run_assets(run_id, plane, outputs)
    return outputs


def input_geometry_metadata(loaded: LoadedInput) -> dict[str, Any]:
    origin = float_list_or_none(loaded.metadata.get("origin"))
    direction = float_list_or_none(loaded.metadata.get("direction"))
    spacing = float_list_or_none(loaded.metadata.get("spacingXyz"))
    canonical_spacing = float_list_or_none(loaded.metadata.get("arrayAxisSpacingCanonical"))
    origin_mm = origin if origin and any(abs(value) > 1e-9 for value in origin) else None
    direction_matrix = direction if direction and len(direction) in {4, 9} else None
    geometry_complete = bool(spacing and canonical_spacing and origin_mm and direction_matrix)
    return {
        "originMm": origin_mm,
        "directionMatrix": direction_matrix,
        "geometryComplete": geometry_complete,
        "geometryMetadataSource": "image_header" if geometry_complete else "incomplete_image_header",
    }


def float_list_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def strip_dot(value: str) -> str:
    return str(value).lstrip(".")


def build_volume_slice_catalog(
    *,
    run_id: str,
    plane: str,
    slice_count: int,
    selected_slice: int,
    measurement_values: list[Dict[str, Any]],
    landmarks: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    catalog: list[Dict[str, Any]] = []
    measurement_ids = [str(item["id"]) for item in measurement_values if isinstance(item, dict) and item.get("id")]
    landmark_ids = [str(item["id"]) for item in landmarks if isinstance(item, dict) and item.get("id")]
    for index in range(slice_count):
        preview_name = slice_asset_name(index)
        has_results = index == selected_slice
        entry: Dict[str, Any] = {
            "index": index,
            "displayIndex": index + 1,
            "previewAsset": {
                "assetName": preview_name,
                "role": "slice-preview",
                "contentType": "image/png",
                "generated": True,
                "url": f"/assets/{run_id}/{plane}/{preview_name}",
            },
            "hasResults": has_results,
            "overlayAsset": None,
            "measurementIds": [],
            "landmarkIds": [],
        }
        if has_results:
            overlay_name = slice_overlay_asset_name(index)
            entry["overlayAsset"] = {
                "assetName": overlay_name,
                "role": "slice-overlay",
                "contentType": "image/png",
                "generated": True,
                "url": f"/assets/{run_id}/{plane}/{overlay_name}",
            }
            entry["measurementIds"] = measurement_ids
            entry["landmarkIds"] = landmark_ids
        catalog.append(entry)
    return catalog


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
    mask_preview_path = output_dir / "mask-preview.png"

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

    preview = np.zeros((*prediction.shape, 3), dtype=np.float32)
    for class_id in sorted(int(value) for value in np.unique(prediction) if int(value) != 0):
        preview[prediction == class_id] = np.asarray(PALETTE.get(class_id, (255, 255, 0)), dtype=np.float32) / 255.0
    Image.fromarray(np.clip(preview * 255.0, 0, 255).astype(np.uint8)).save(mask_preview_path)
    outputs = {
        "imagePath": str(image_path),
        "maskPath": str(mask_path),
        "confidencePath": str(confidence_path),
        "overlayPath": str(overlay_path),
        "maskPreviewPath": str(mask_preview_path),
    }
    register_run_assets(run_id, plane, outputs)
    return outputs


def run_real_inference(request: Any, run_id: str) -> Dict[str, Any]:
    cached = load_model(request.model_key)
    artifact = model_status(request.model_key, dict(MODEL_REGISTRY.get(request.model_key, {})))
    if not artifact.get("availableForRealInference", False):
        raise RuntimeError(f"Modelo no habilitado para real_baseline: {request.model_key}")

    request_metadata = dict(request.metadata or {})
    loaded = canonicalize_loaded_input(load_input(request.input_path, request.plane), request.plane, request_metadata)
    target_size = tuple(cached.runtime_metadata.get("targetSize", (256, 256)))
    image, selected_slice, slice_count, selected_axis = select_slice(
        loaded,
        request.plane,
        cached,
        (int(target_size[0]), int(target_size[1])),
        request_metadata,
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
    spacing = in_plane_spacing(loaded, selected_axis)
    spacing_unit = "mm" if spacing else None
    masks = build_masks(request.model_key, request.plane, prediction, confidence, series_id, selected_slice)
    landmarks = build_landmarks(masks)
    measurement_values = build_measurements(request.model_key, request.plane, prediction, confidence, spacing)
    catalog_outputs = save_slice_catalog_assets(
        run_id,
        request.plane,
        loaded,
        selected_axis,
        selected_slice,
        (int(target_size[0]), int(target_size[1])),
        outputs["overlayPath"],
    )
    outputs = {**outputs, **catalog_outputs}
    assets = registered_assets_for_run(run_id, request.plane)
    slice_catalog = build_volume_slice_catalog(
        run_id=run_id,
        plane=request.plane,
        slice_count=slice_count,
        selected_slice=selected_slice,
        measurement_values=measurement_values,
        landmarks=landmarks,
    )
    geometry = input_geometry_metadata(loaded)

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
        "inPlaneSpacing": list(spacing) if spacing else None,
        "inPlaneSpacingUnit": spacing_unit,
        "measurementsDerivedFromPredictionMask": True,
    }
    requested_mode = str(request.metadata.get("inferenceMode", "real_baseline"))
    allow_contract_fallback = metadata_bool(request.metadata, "allowContractFallback", True)
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
        "artifactHash": artifact.get("artifactHash"),
        "inferenceMode": "real_baseline",
        "requestedInferenceMode": requested_mode,
        "allowContractFallback": allow_contract_fallback,
        "synthetic": False,
        "fallbackReason": None,
        "input_path": request.input_path,
        "inputPath": request.input_path,
        "series": [{
            "id": series_id,
            "name": "Sagittal T2" if request.plane == "sagittal" else "Axial T2",
            "plane": request.plane,
            "sequence": "T2",
            "sliceCount": slice_count,
            "slices": slice_catalog,
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
            "artifactHash": artifact.get("artifactHash"),
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
            "sagittalAxis": selected_axis if request.plane == "sagittal" else None,
            "seriesId": request.input_id or series_id,
            "sourceFormat": strip_dot(loaded.suffix),
            "inputShapeNative": loaded.metadata.get("inputShapeNative"),
            "inputShapeCanonical": loaded.metadata.get("inputShapeCanonical"),
            "inputOrientationTransform": loaded.metadata.get("inputOrientationTransform"),
            "spacingXyz": loaded.metadata.get("spacingXyz"),
            "arrayAxisSpacingNative": loaded.metadata.get("arrayAxisSpacingNative"),
            "arrayAxisSpacingCanonical": loaded.metadata.get("arrayAxisSpacingCanonical"),
            "originMm": geometry["originMm"],
            "directionMatrix": geometry["directionMatrix"],
            "geometryComplete": geometry["geometryComplete"],
            "geometryMetadataSource": geometry["geometryMetadataSource"],
            "inPlaneSpacing": list(spacing) if spacing else None,
            "inPlaneSpacingUnit": spacing_unit,
            "slices": slice_catalog,
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
