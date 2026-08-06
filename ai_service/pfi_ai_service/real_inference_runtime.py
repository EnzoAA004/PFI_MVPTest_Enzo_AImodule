from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, NamedTuple

import numpy as np
import torch
from PIL import Image

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS, build_agent_decision
from .asset_registry import is_slice_asset_name, registered_assets_for_run, register_run_assets, slice_asset_name, slice_pixels_asset_name
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
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOff()
    image = reader.Execute()
    slice_positions = dicom_slice_positions(reader, len(file_names))
    array = sitk.GetArrayFromImage(image)
    spacing = tuple(float(value) for value in image.GetSpacing())
    metadata: Dict[str, Any] = {
        "seriesInstanceUid": best_id,
        "origin": tuple(float(value) for value in image.GetOrigin()),
        "direction": tuple(float(value) for value in image.GetDirection()),
        # El marco de referencia es lo que decide si dos series se pueden comparar en
        # el espacio. Dos estudios distintos tienen origenes y direcciones bien
        # formados y completamente incomparables entre si: sin este identificador, una
        # linea de referencia entre planos podria trazarse con geometria valida y
        # apuntar a un lugar que no existe en la otra imagen.
        "frameOfReferenceUid": reader.GetMetaData(0, "0020|0052").strip()
        if reader.HasMetaDataKey(0, "0020|0052") else None,
        "slicePositions": slice_positions,
        "sliceOrientations": dicom_slice_orientations(reader, len(file_names)),
        "sliceSpacingUniform": spacing_is_uniform(slice_positions),
        "sliceCount": int(array.shape[0]),
    }
    return array, spacing, metadata


def dicom_slice_positions(reader: Any, count: int) -> list[list[float]] | None:
    """Posicion declarada de cada corte de la serie, en coordenadas del paciente.

    Es el dato exacto, y evita el supuesto de que los cortes estan a pasos parejos.
    Sin esto habria que ubicar el corte N como `origen + N x espaciado`, que es
    correcto hasta el primer salto de la serie y falso despues: en este dataset las
    series axiales tienen huecos de decenas de milimetros, asi que la cuenta ubicaria
    medio estudio en la altura equivocada.

    Devuelve None si alguna posicion no se puede leer: es preferible caer al modelo
    del espaciado unico -y declararlo- que mezclar posiciones reales con inventadas.
    """
    positions: list[list[float]] = []
    for index in range(count):
        try:
            raw = reader.GetMetaData(index, "0020|0032")
        except Exception:
            return None
        parts = [value.strip() for value in str(raw).split("\\")]
        if len(parts) != 3:
            return None
        try:
            positions.append([float(value) for value in parts])
        except ValueError:
            return None
    return positions


def dicom_slice_orientations(reader: Any, count: int) -> list[list[float]] | None:
    """Orientacion declarada de cada corte, como [fila(3), columna(3)].

    Una serie axial lumbar **no es un unico plano repetido**: se adquiere en bloques
    angulados, uno por disco. En el estudio de referencia los cortes 1-5 estan a 3.5
    grados, los 6-10 a 5.9 y los 11-15 a 23, con huecos de decenas de milimetros entre
    bloques. Describir el volumen con una sola direccion -la que devuelve ITK- toma la
    de uno de los bloques y la aplica a todos: acertaba en 5 cortes de 15 y en los
    otros 10 dibujaba la linea de referencia a 23 grados donde la real es casi
    horizontal, y con el signo antero-posterior invertido.

    Se lee de 0020|0037, que son los dos vectores unitarios de fila y columna.
    Devuelve None si alguno no se puede leer: es preferible caer a la direccion unica
    del volumen -y que el visor lo sepa- que mezclar orientaciones reales con
    heredadas.
    """
    orientations: list[list[float]] = []
    for index in range(count):
        try:
            raw = reader.GetMetaData(index, "0020|0037")
        except Exception:
            return None
        parts = [value.strip() for value in str(raw).split("\\")]
        if len(parts) != 6:
            return None
        try:
            orientations.append([float(value) for value in parts])
        except ValueError:
            return None
    return orientations


def spacing_is_uniform(positions: list[list[float]] | None, tolerance: float = 0.2) -> bool:
    """Si los cortes de la serie estan a pasos parejos.

    Importa para ubicar un corte en el espacio: la geometria que publica el modulo
    describe la serie con un unico espaciado, y con esa cifra el corte N esta en
    `origen + N x espaciado`. Si la serie tiene un salto -una region no adquirida, un
    corte perdido- esa cuenta es correcta hasta el salto y falsa despues, y una linea
    de referencia trazada con ella apuntaria a la altura equivocada sin avisar.

    Se mide sobre las posiciones declaradas por cada archivo y no sobre el espaciado
    que resume ITK, que ya asume uniformidad. Ante la duda -si las posiciones no se
    pueden leer- se responde que no es uniforme: es la respuesta que hace que el
    consumidor no extrapole.
    """
    if not positions:
        return False
    if len(positions) < 3:
        return True
    steps = [
        sum((a - b) ** 2 for a, b in zip(first, second)) ** 0.5
        for first, second in zip(positions, positions[1:])
    ]
    reference = sorted(steps)[len(steps) // 2]
    if reference <= 0:
        return False
    return all(abs(step - reference) <= tolerance * reference for step in steps)


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


def native_slice(loaded: LoadedInput, axis: int, index: int) -> np.ndarray:
    """Corte en su resolucion original, sin el resize a la grilla del modelo.

    El resize a targetSize existe solo porque es la entrada que el checkpoint
    espera. Guardar esa version como input.png hacia que el visor mostrara una
    imagen de 256x256: el frente dimensiona el marco con naturalWidth/Height, asi
    que el medico veia el estudio escalado a 256 px y con la relacion de aspecto
    deformada. Los PNG se generan desde el corte nativo; la mascara y la
    confianza siguen viviendo en la grilla del modelo, que es donde se midieron.
    """
    raw = loaded.array if loaded.array.ndim == 2 else np.take(loaded.array, index, axis=axis)
    return robust_percentile_normalize(raw)


def upsample_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Lleva un mapa de clases a `shape` sin inventar clases intermedias."""
    if labels.shape == shape:
        return labels
    image = Image.fromarray(labels.astype(np.uint8))
    resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


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


def is_compact(binary: np.ndarray, max_elongation: float = 4.0) -> bool:
    """Si la forma es lo bastante compacta como para describirla con un poligono
    ordenado por angulo alrededor de su centroide.

    Se mide la relacion entre el lado largo y el corto de su caja: un disco o un
    cuerpo vertebral quedan por debajo de 4, el canal de un estudio lumbar completo
    la supera holgadamente. No es una constante arbitraria sino el limite donde el
    ordenamiento angular deja de describir el borde real.
    """
    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return False
    height = float(rows.max() - rows.min() + 1)
    width = float(cols.max() - cols.min() + 1)
    short = min(height, width)
    return short > 0 and max(height, width) / short <= max_elongation


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


#: Paleta cualitativa para distinguir instancias entre si.
#:
#: No codifica clase ni severidad: su unico trabajo es que dos estructuras vecinas
#: no compartan color, para que el medico vea de un vistazo si la IA separo los
#: discos donde correspondia o fusiono dos. Los colores por clase (PALETTE) siguen
#: usandose para el overlay compuesto y para las capas por clase.
INSTANCE_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#800000", "#808000",
]


def instance_color(index: int) -> str:
    return INSTANCE_COLORS[index % len(INSTANCE_COLORS)]


#: Clases que son fondo y no se pintan ni se miden.
#:
#: El modelo axial tiene dos: `background_250`, que quedo como id 0 en el remapeo
#: del entrenamiento, y `raw_0`, que es el fondo del dataset Al-Kafri. Pintar la segunda dibujaba un
#: marco rojo alrededor del estudio, porque es justamente lo que rodea al paciente.
BACKGROUND_CLASSES = {"background", "background_250", "raw_0"}


def is_background_class(name: str) -> bool:
    return name in BACKGROUND_CLASSES


def rle_encode(labels: np.ndarray) -> list[int]:
    """Codifica un mapa de etiquetas como pares (valor, repeticiones).

    Es el transporte de la segmentacion: exacto -no hay perdida ni interpolacion-
    y mas chico que el PNG equivalente, porque una segmentacion son pocas regiones
    grandes. Sobre un corte real de 256x256 da unos 1200 pares, ~9 KB en JSON.

    Se manda el mapa y no una imagen ya pintada porque el color es una decision de
    presentacion: quien dibuja elige que instancia resalta, cual oculta y con que
    opacidad, sin volver a pedirle nada al backend.
    """
    flat = np.asarray(labels, dtype=np.int32).ravel()
    if flat.size == 0:
        return []
    changes = np.flatnonzero(np.diff(flat))
    starts = np.concatenate(([0], changes + 1))
    lengths = np.diff(np.concatenate((starts, [flat.size])))
    encoded: list[int] = []
    for value, length in zip(flat[starts], lengths):
        encoded.append(int(value))
        encoded.append(int(length))
    return encoded


def build_segmentation(
    model_key: str,
    plane: str,
    prediction: np.ndarray,
) -> Dict[str, Any]:
    """Segmentacion como mapa de instancias, no como imagen.

    Cada pixel lleva el indice de la instancia a la que pertenece (0 = ninguna), y
    aparte viaja la lista de instancias con su clase y su nivel. Asi el visor pinta
    cada vertebra y cada disco de un color propio, puede ocultar una sin tocar las
    demas, y tiene el dato exacto para corregirlo -sin reconstruir el borde con un
    poligono, que sobre formas no compactas une puntos que no son.
    """
    labels = np.zeros(prediction.shape, dtype=np.int32)
    instances: list[Dict[str, Any]] = []
    class_ids = {
        name: class_id
        for class_id in sorted(int(value) for value in np.unique(prediction) if int(value) != 0)
        if not is_background_class(name := class_name(model_key, class_id))
    }

    def register(identifier: str, label: str, class_key: str, binary: np.ndarray, level: str | None) -> None:
        index = len(instances) + 1
        labels[binary] = index
        instances.append({
            "index": index,
            "id": identifier,
            "label": label,
            "classKey": class_key,
            "level": level,
        })

    if "disc_group" in class_ids:
        components = connected_instances(prediction == class_ids["disc_group"])
        for position, (component, level) in enumerate(zip(components, lumbar_disc_levels(len(components))), start=1):
            slug = level.lower() if level else f"d{position}"
            register(f"{plane}-disc-{slug}", "disc", "disc_group", component, level)

    if "vertebra_group" in class_ids:
        discs = connected_instances(prediction == class_ids["disc_group"]) if "disc_group" in class_ids else []
        bodies, posterior = split_vertebral_bodies(connected_instances(prediction == class_ids["vertebra_group"]), discs)
        body_names = name_vertebral_bodies(bodies, discs, lumbar_disc_levels(len(discs)))
        # El id es posicional y el nivel viaja aparte. Derivarlo del nivel producia
        # ids repetidos: cuando la segmentacion parte el arco de una vertebra en dos
        # componentes, ambos son legitimamente "L4" y colisionaban en un mismo id,
        # que es justo lo que el front usa como clave de lista.
        for index, (component, name) in enumerate(zip(bodies, body_names), start=1):
            register(f"{plane}-vertebra-b{index}", "vertebra", "vertebra_group", component, name)
        for position, (component, name) in enumerate(zip(posterior, name_posterior_elements(posterior, bodies, body_names)), start=1):
            register(f"{plane}-posterior-p{position}", "posterior_element", "vertebra_group", component, name)

    for label, class_id in class_ids.items():
        if label in {"disc_group", "vertebra_group"}:
            continue
        register(f"{plane}-{label}", label, label, prediction == class_id, None)

    return {
        "encoding": "rle-v1",
        "width": int(prediction.shape[1]),
        "height": int(prediction.shape[0]),
        "data": rle_encode(labels),
        "instances": instances,
    }


def split_vertebral_bodies(
    vertebrae: list[np.ndarray],
    discs: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Separa cuerpos vertebrales de elementos posteriores.

    La clase `vertebra_group` no distingue uno de otro: sobre un estudio real da 14
    componentes para 7 vertebras, porque cada una aporta su cuerpo y su arco
    posterior. En un corte sagital el eje horizontal es el anteroposterior, y el
    cuerpo esta delante mientras el arco esta detras.

    El corte se ancla en los discos y no en un umbral inventado: los discos estan
    alineados con los cuerpos por definicion -son lo que hay entre dos cuerpos-, asi
    que su borde posterior marca donde termina la columna anterior. Sin discos no
    hay ancla y se devuelven todos como cuerpos, que es lo que se veia antes.
    """
    if not vertebrae:
        return [], []
    if not discs:
        return list(vertebrae), []
    disc_backs = [float(np.where(mask.any(axis=0))[0].max()) for mask in discs]
    boundary = float(np.median(disc_backs))
    bodies: list[np.ndarray] = []
    posterior: list[np.ndarray] = []
    for mask in vertebrae:
        columns = np.where(mask.any(axis=0))[0]
        (bodies if float(columns.mean()) <= boundary else posterior).append(mask)
    return bodies, posterior


def name_posterior_elements(
    posterior: list[np.ndarray],
    bodies: list[np.ndarray],
    body_names: list[str | None],
) -> list[str | None]:
    """Nombra cada arco posterior por el cuerpo que tiene enfrente.

    El arco posterior de L4 esta a la misma altura que el cuerpo de L4: comparten el
    rango de filas porque son la misma vertebra vista de perfil. Asi que se empareja
    por solapamiento vertical en vez de contar de nuevo desde abajo, que volveria a
    introducir el supuesto de donde empieza la lumbar.

    Un arco que no solapa con ningun cuerpo nombrado queda sin nivel. Se sigue viendo
    como elemento posterior -eso lo dice la estructura, no el nombre- pero no se le
    inventa una vertebra que el encuadre no muestra.
    """
    names: list[str | None] = [None] * len(posterior)
    if not posterior or not bodies:
        return names
    spans = [(float(np.where(mask.any(axis=1))[0].min()), float(np.where(mask.any(axis=1))[0].max())) for mask in bodies]
    for index, mask in enumerate(posterior):
        rows = np.where(mask.any(axis=1))[0]
        top, bottom = float(rows.min()), float(rows.max())
        overlaps = [
            (min(bottom, span_bottom) - max(top, span_top), position)
            for position, (span_top, span_bottom) in enumerate(spans)
        ]
        overlap, position = max(overlaps)
        if overlap > 0 and body_names[position]:
            names[index] = body_names[position]
    return names


def name_vertebral_bodies(
    bodies: list[np.ndarray],
    discs: list[np.ndarray],
    disc_levels: list[str | None],
) -> list[str | None]:
    """Nombra cada cuerpo a partir de los discos ya identificados.

    No se vuelve a contar desde cero: el cuerpo inmediatamente superior al disco
    L4-L5 es L4, y el inferior al ultimo disco lumbar es S1. Anclar en los discos
    evita introducir un segundo supuesto sobre donde empieza la lumbar que podria
    contradecir al primero.

    Un cuerpo que no queda entre dos discos identificados se deja sin nombre: es el
    que el encuadre corto por arriba o por abajo.
    """
    names: list[str | None] = [None] * len(bodies)
    if not bodies or not discs:
        return names
    tops = [float(np.where(mask.any(axis=1))[0].min()) for mask in bodies]
    for disc, level in zip(discs, disc_levels):
        if not level or "-" not in level:
            continue
        upper, lower = level.split("-", 1)
        disc_top = float(np.where(disc.any(axis=1))[0].min())
        disc_bottom = float(np.where(disc.any(axis=1))[0].max())
        above = [index for index, top in enumerate(tops) if top < disc_top]
        if above:
            names[max(above, key=lambda index: tops[index])] = upper
        below = [index for index, top in enumerate(tops) if top > disc_bottom]
        if below:
            candidate = min(below, key=lambda index: tops[index])
            if names[candidate] is None:
                names[candidate] = lower
    return names


def build_masks(
    model_key: str,
    plane: str,
    prediction: np.ndarray,
    confidence: np.ndarray,
    series_id: str,
    slice_index: int,
) -> list[Dict[str, Any]]:
    """Una mascara por instancia, con su contorno propio.

    Antes se emitia una mascara por clase, con un unico poligono calculado sobre
    toda la clase: el contorno de `disc_group` envolvia los seis discos juntos, un
    trazo que no corresponde a ninguna estructura y que el revisor no puede
    corregir. Por instancia, cada disco tiene su borde, su color y -cuando se pudo
    determinar- su nivel.

    Los cuerpos vertebrales se emiten por instancia aunque no se les pueda asignar
    nivel: verlos separados es justamente lo que deja ver que la clase
    `vertebra_group` no contiene solo cuerpos, y esa evidencia le sirve al medico
    mas que un unico contorno que la esconde.
    """
    masks: list[Dict[str, Any]] = []
    class_ids = {
        name: class_id
        for class_id in sorted(int(value) for value in np.unique(prediction) if int(value) != 0)
        if not is_background_class(name := class_name(model_key, class_id))
    }

    def add(identifier: str, label: str, class_id: int, binary: np.ndarray, level: str | None) -> None:
        # El contorno se ordena por angulo alrededor del centroide, lo que solo
        # describe una figura estrellada respecto de ese centro. Sirve para discos y
        # cuerpos vertebrales, que son compactos; sobre el canal -un tubo largo y
        # curvo- el poligono se cruza a si mismo y dibuja una estrella que no es la
        # estructura. Para esas formas no se emite geometria: el PNG por clase la
        # muestra bien, y un contorno falso seria peor que ninguno.
        points = boundary_polygon(binary) if is_compact(binary) else []
        masks.append({
            "id": identifier,
            "label": label,
            "className": class_name(model_key, class_id),
            "classId": class_id,
            "level": level,
            "color": instance_color(len(masks)),
            "confidence": round(float(confidence[binary].mean()), 4) if binary.any() else 0.0,
            "editable": True,
            "enabled": True,
            "contours": [{"seriesId": series_id, "sliceIndex": slice_index, "points": points}] if points else [],
        })

    if "disc_group" in class_ids:
        class_id = class_ids["disc_group"]
        instances = connected_instances(prediction == class_id)
        levels = lumbar_disc_levels(len(instances))
        for position, (component, level) in enumerate(zip(instances, levels), start=1):
            slug = level.lower() if level else f"d{position}"
            add(f"mask-{plane}-disc-{slug}", "disc", class_id, component, level)

    if "vertebra_group" in class_ids:
        class_id = class_ids["vertebra_group"]
        discs = connected_instances(prediction == class_ids["disc_group"]) if "disc_group" in class_ids else []
        bodies, posterior = split_vertebral_bodies(connected_instances(prediction == class_id), discs)
        for component, name in zip(bodies, name_vertebral_bodies(bodies, discs, lumbar_disc_levels(len(discs)))):
            slug = name.lower() if name else f"b{bodies.index(component) + 1}"
            add(f"mask-{plane}-vertebra-{slug}", "vertebra", class_id, component, name)
        for position, component in enumerate(posterior, start=1):
            add(f"mask-{plane}-posterior-p{position}", "posterior_element", class_id, component, None)

    for label, class_id in class_ids.items():
        if label in {"disc_group", "vertebra_group"}:
            continue
        add(f"mask-{plane}-{label.replace('_', '-')}", label, class_id, prediction == class_id, None)

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


LUMBAR_DISC_LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")


def connected_instances(binary: np.ndarray, min_pixels: int = 20) -> list[np.ndarray]:
    """Separa una mascara de clase en sus componentes conexas, de superior a inferior.

    El modelo sagital segmenta `disc_group` como una unica clase, no un disco por
    instancia. Las componentes conexas de esa mascara son los espacios discales
    individuales, que es lo unico que permite hablar de un nivel concreto.

    En el arreglo sagital canonico la fila 0 es superior, asi que ordenar por
    centroide de fila da los discos de arriba hacia abajo.
    """
    try:
        import SimpleITK as sitk
    except Exception:  # pragma: no cover - dependency guard
        return []
    labelled = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(binary.astype(np.uint8))))
    instances = [
        component
        for value in sorted(int(item) for item in np.unique(labelled) if int(item) != 0)
        if int((component := labelled == value).sum()) >= min_pixels
    ]
    instances.sort(key=lambda mask: float(np.where(mask)[0].mean()))
    return instances


#: Espacios discales por encima de L1-L2, hacia arriba. Un encuadre lumbar suele
#: incluir uno o dos: nombrarlos evita que caigan en "sin nivel asignado" cuando en
#: realidad se sabe cuales son.
THORACIC_DISC_LEVELS = ("T12-L1", "T11-T12", "T10-T11", "T9-T10", "T8-T9")


def lumbar_disc_levels(count: int) -> list[str | None]:
    """Nivel anatomico de cada espacio discal detectado, de superior a inferior.

    Se cuenta desde abajo, que es como se numera una lumbar en la practica: el
    espacio discal mas inferior del estudio es L5-S1 y desde ahi se sube L4-L5,
    L3-L4, L2-L3, L1-L2. Los espacios por encima del quinto quedan sin nivel: son
    T12-L1 o mas altos, fuera de la nomenclatura lumbar.

    El supuesto es que el encuadre llega a la union lumbosacra, que es lo que
    define a un protocolo de RM lumbar y el revisor puede verificar de un vistazo
    en la imagen. Si el estudio muestra menos de cinco espacios discales el
    encuadre no es una lumbar completa: no se sabe desde donde empezar a contar y
    todas las mediciones quedan sin nivel, agrupadas aparte, antes que
    desplazar la numeracion entera en un nivel.
    """
    if count < len(LUMBAR_DISC_LEVELS):
        return [None] * count
    extra = count - len(LUMBAR_DISC_LEVELS)
    # Los que sobran por encima de L1-L2 son toracicos, y se nombran hacia arriba
    # desde T12-L1. Mas alla de lo que cubre la tabla quedan sin nivel antes que
    # seguir contando vertebras que el encuadre ya casi no muestra.
    above = [THORACIC_DISC_LEVELS[index] if index < len(THORACIC_DISC_LEVELS) else None for index in range(extra)]
    return list(reversed(above)) + list(LUMBAR_DISC_LEVELS)


#: Como se reordenan los ejes del arreglo al canonicalizar. El eje `i` de la lista es
#: el eje canonico; su valor es el eje nativo del que salio.
CANONICAL_TO_NATIVE_AXIS = {
    "none": (0, 1, 2),
    # `np.moveaxis(array, 0, -1)`: el eje nativo 0 pasa al final y los otros suben.
    "move_axis_0_to_last": (1, 2, 0),
}


def native_axis_directions(direction: Any) -> list[list[float]] | None:
    """Direccion en el paciente de cada eje del arreglo nativo.

    ITK entrega la matriz de direccion por filas y sus **columnas** son los cosenos de
    los ejes de indice i, j, k. El arreglo de numpy viene en orden (k, j, i), asi que
    el eje 0 del arreglo corresponde a la tercera columna y no a la primera. Es
    exactamente el tipo de inversion que, si se toma al reves, dibuja una linea de
    referencia perpendicular a donde tiene que ir.
    """
    if not direction or len(direction) != 9:
        return None
    d = [float(value) for value in direction]
    i_dir = [d[0], d[3], d[6]]
    j_dir = [d[1], d[4], d[7]]
    k_dir = [d[2], d[5], d[8]]
    return [k_dir, j_dir, i_dir]


def slice_plane_geometry(
    loaded: LoadedInput,
    selected_axis: int,
    slice_index: int,
) -> Dict[str, Any] | None:
    """Ubica el corte que se muestra en el espacio del paciente.

    Devuelve el origen del pixel (0,0) del corte y las direcciones de sus filas y
    columnas, que es lo que permite pasar de un punto de la imagen a una coordenada
    del paciente y al reves. Con eso, dos planos de un mismo estudio se pueden cruzar.

    El razonamiento de ejes se hace aca y no en el visor a proposito: aca estan el
    arreglo, su forma nativa y la transformacion que lo canonicalizo, y se puede
    probar contra geometria conocida. Mandar la matriz cruda y que el cliente la
    interprete es repartir el mismo razonamiento en dos lugares, donde uno de los dos
    va a equivocarse de eje.
    """
    directions = native_axis_directions(loaded.metadata.get("direction"))
    origin = loaded.metadata.get("origin")
    spacing = loaded.metadata.get("arrayAxisSpacingNative")
    transform = loaded.metadata.get("inputOrientationTransform") or "none"
    mapping = CANONICAL_TO_NATIVE_AXIS.get(transform)
    if not directions or not origin or not spacing or mapping is None or len(origin) != 3:
        return None
    if loaded.array.ndim != 3 or not 0 <= selected_axis < 3:
        return None

    native_slice_axis = mapping[selected_axis]
    plane_axes = [axis for axis in range(3) if axis != selected_axis]
    row_axis, col_axis = mapping[plane_axes[0]], mapping[plane_axes[1]]
    step = float(spacing[native_slice_axis])
    normal = directions[native_slice_axis]
    # La posicion declarada por el propio corte, cuando la serie la trae. Solo si no
    # esta se cae al modelo del espaciado unico, que supone la serie sin huecos.
    declared = loaded.metadata.get("slicePositions")
    exact = (
        [float(value) for value in declared[slice_index]]
        if isinstance(declared, list) and 0 <= slice_index < len(declared)
        else None
    )
    return {
        # Origen del pixel (0,0) del corte que se esta mostrando.
        "position": exact or [float(origin[k]) + slice_index * step * normal[k] for k in range(3)],
        "positionSource": "declared" if exact else "uniform_spacing",
        "rowDirection": directions[row_axis],
        "colDirection": directions[col_axis],
        "normal": normal,
        "rowSpacing": float(spacing[row_axis]),
        "colSpacing": float(spacing[col_axis]),
        "sliceSpacing": step,
        "rowCount": int(loaded.array.shape[plane_axes[0]]),
        "colCount": int(loaded.array.shape[plane_axes[1]]),
    }


def volume_bounds(loaded: LoadedInput, plane_geometry: Dict[str, Any] | None, slice_count: int) -> Dict[str, Any] | None:
    """Caja que ocupa el volumen en coordenadas del paciente.

    Es la evidencia que reemplaza al identificador cuando el identificador se perdio.
    El `frameOfReferenceUid` es la garantia declarada de que dos series comparten
    sistema de coordenadas, pero la anonimizacion de este dataset lo regenera por
    serie: las coordenadas siguen siendo coherentes y el identificador ya no lo dice.
    Con las cajas de los dos volumenes se puede verificar lo mismo por geometria -que
    se solapen y que cada uno contenga el centro del otro- y decir en pantalla cual de
    las dos evidencias sostiene la linea.
    """
    if plane_geometry is None:
        return None
    positions = loaded.metadata.get("slicePositions")
    if not isinstance(positions, list) or not positions:
        origin = plane_geometry["position"]
        normal = plane_geometry["normal"]
        step = plane_geometry["sliceSpacing"]
        positions = [[origin[k] + index * step * normal[k] for k in range(3)] for index in range(slice_count)]

    row = plane_geometry["rowDirection"]
    col = plane_geometry["colDirection"]
    height = (plane_geometry["rowCount"] - 1) * plane_geometry["rowSpacing"]
    width = (plane_geometry["colCount"] - 1) * plane_geometry["colSpacing"]
    corners = [
        [float(position[k]) + row[k] * r + col[k] * c for k in range(3)]
        for position in positions
        for r in (0.0, height)
        for c in (0.0, width)
    ]
    return {
        "min": [min(corner[k] for corner in corners) for k in range(3)],
        "max": [max(corner[k] for corner in corners) for k in range(3)],
    }


def volume_geometry(loaded: LoadedInput, selected_axis: int, slice_count: int, slice_index: int = 0) -> Dict[str, Any] | None:
    """Geometria del volumen en el espacio del paciente, para ubicar un corte.

    Es lo que falta para trazar una linea de referencia real entre el sagital y el
    axial: sin origen, direccion y spacing no hay forma de saber donde corta un plano
    al otro, y dibujar una linea sin eso seria inventar una coordenada.

    Se publica el `inputOrientationTransform` junto con la geometria y no por separado.
    El origen y la direccion describen el arreglo **nativo**, mientras que el analisis
    corre sobre el canonico; entregarlos sin decir que transformacion los separa
    invitaria a leerlos contra los ejes equivocados, que es la clase de error que la
    correccion de escala ya costo una vez.

    Devuelve None cuando el formato de entrada no trae geometria -un PNG o un .npy no
    la tienen- en vez de rellenar con una identidad que se leeria como un dato.
    """
    origin = loaded.metadata.get("origin")
    direction = loaded.metadata.get("direction")
    if not origin or not direction:
        return None
    origin_values = [float(value) for value in origin]
    direction_values = [float(value) for value in direction]
    spacing = loaded.metadata.get("spacingXyz")
    # Un origen todo en cero y una matriz de largo raro son cabeceras por defecto, no
    # geometria: el archivo trae los campos pero no dice donde esta el paciente.
    # Publicarlos igual y dejar que el consumidor lo deduzca es pedirle que repita esta
    # verificacion, y que la haga mal el dia que no la haga.
    complete = bool(
        spacing
        and any(abs(value) > 1e-9 for value in origin_values)
        and len(direction_values) in {4, 9}
    )
    return {
        "origin": origin_values,
        "direction": direction_values,
        "geometryComplete": complete,
        "spacingXyz": spacing,
        "arrayAxisSpacingNative": loaded.metadata.get("arrayAxisSpacingNative"),
        "inputShapeNative": loaded.metadata.get("inputShapeNative"),
        "inputOrientationTransform": loaded.metadata.get("inputOrientationTransform"),
        "frameOfReferenceUid": loaded.metadata.get("frameOfReferenceUid"),
        "sliceAxis": int(selected_axis),
        "sliceCount": int(slice_count),
        # Sin pasos parejos, el corte N no esta donde la cuenta dice. Se declara para
        # que el consumidor no ubique un corte que la serie no garantiza.
        "sliceSpacingUniform": bool(loaded.metadata.get("sliceSpacingUniform", False)),
        "slicePlane": (plane_geometry := slice_plane_geometry(loaded, selected_axis, slice_index)),
        # La posicion declarada de todos los cortes, no solo la del que se analizo.
        #
        # Es lo que permite que la linea de referencia se mueva mientras el medico
        # recorre la serie. Con un solo plano, el visor tendria que extrapolar los
        # demas como `origen + N x espaciado`, y en este dataset las series axiales
        # tienen huecos: el corte 4 quedaria a 22 mm de donde esta. Son quince ternas
        # de numeros, y evitan que el cliente rehaga -mal- una cuenta que aca es exacta.
        "slicePositions": loaded.metadata.get("slicePositions"),
        # Por corte, porque una serie axial lumbar son bloques angulados por nivel y no
        # un plano unico repetido. Ver dicom_slice_orientations.
        "sliceOrientations": loaded.metadata.get("sliceOrientations"),
        "boundsMm": volume_bounds(loaded, plane_geometry, slice_count),
    }


def prediction_grid_spacing(
    spacing: tuple[float, float] | None,
    array_shape: tuple[int, ...],
    selected_axis: int,
    prediction_shape: tuple[int, ...],
) -> tuple[float, float] | None:
    """Convierte el spacing nativo al de la grilla de la prediccion.

    La red recibe el corte reescalado a su tamano de entrada -384x384 pasa a
    256x256-, asi que un pixel de la prediccion cubre 1.5 pixeles nativos. Medir
    contando pixeles de la prediccion y multiplicar por el spacing nativo mezclaba
    dos grillas distintas y devolvia todo un 33% mas chico: un disco de 28 mm se
    informaba como 19 mm, y las areas quedaban a menos de la mitad porque el error
    entra al cuadrado.

    Lo que se ajusta es el spacing, no la medicion, porque el error no esta en el
    conteo de pixeles sino en cuanto mide cada pixel de esa grilla.
    """
    if spacing is None:
        return None
    native = [dim for index, dim in enumerate(array_shape) if index != selected_axis]
    if len(native) < 2 or len(prediction_shape) < 2:
        return spacing
    if prediction_shape[0] <= 0 or prediction_shape[1] <= 0:
        return spacing
    return (
        float(spacing[0]) * float(native[0]) / float(prediction_shape[0]),
        float(spacing[1]) * float(native[1]) / float(prediction_shape[1]),
    )


class AxisMeasure(NamedTuple):
    """Una magnitud y los dos puntos de los que salio.

    Van juntas a proposito. El visor dibuja el segmento sobre la imagen para que el
    medico vea de donde a donde se midio, y si el numero y la linea se calcularan por
    separado podrian discrepar: la pantalla mostraria una medicion que no es la que
    dice la tabla. Aca `length` es exactamente la distancia entre `start` y `end`.

    Los puntos van en la grilla de la prediccion, que es el mismo espacio en el que
    viajan mascaras y landmarks.
    """

    length: float
    start: tuple[float, float]
    end: tuple[float, float]


def oriented_extent(
    binary: np.ndarray,
    row_spacing: float,
    col_spacing: float,
) -> tuple[AxisMeasure, AxisMeasure]:
    """Ancho y alto de una estructura medidos sobre sus propios ejes.

    La caja alineada a la imagen no mide el disco: mide la caja que lo contiene. En
    L5-S1, que es el nivel mas angulado de la columna, eso daba 25 mm de alto para un
    disco que anda por 8-12. Cuanto mas inclinada la estructura, mas mide de mas.

    Los ejes propios salen de la nube de pixeles: los autovectores de su covarianza
    son las direcciones en las que la estructura se extiende. Se calcula sobre las
    coordenadas ya en milimetros y no en pixeles, para que un spacing anisotropico no
    incline los ejes por si solo -en mm el disco esta donde esta, en pixeles estaria
    estirado.

    El eje mas horizontal se informa como ancho y el otro como alto: el que rota es
    el instrumento, no el significado. En un sagital el ancho sigue siendo el
    anteroposterior y el alto el craneocaudal, que es como se lee un informe. Ordenar
    por eje mayor y menor los intercambiaria en cualquier vertebra mas alta que ancha,
    y llamaria "ancho" al largo del canal, que es una estructura vertical.

    A 45 grados exactos los dos ejes estan igual de cerca de la horizontal y el
    rotulo deja de significar algo. Las dos magnitudes siguen siendo correctas; cual
    es el ancho y cual el alto, en ese caso, no lo decide la geometria.

    Se suma la huella del pixel proyectada sobre cada eje, que es lo que hacia el
    `+1` de la caja: sin eso una estructura de un pixel de espesor mediria cero. A
    cero grados la formula da exactamente lo mismo que la caja, que es la unica forma
    de que este cambio no mueva las medidas que ya estaban bien. El segmento se
    extiende media huella por punta, asi que su largo es el valor informado: llega
    hasta el borde exterior del pixel, no hasta su centro.
    """
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return AxisMeasure(0.0, (0.0, 0.0), (0.0, 0.0)), AxisMeasure(0.0, (0.0, 0.0), (0.0, 0.0))
    points = np.stack((xs * col_spacing, ys * row_spacing), axis=1)
    center = points.mean(axis=0)
    grid = lambda point: (round(float(point[0]) / col_spacing, 2), round(float(point[1]) / row_spacing, 2))
    if len(xs) < 2:
        half = np.array([col_spacing / 2, row_spacing / 2])
        return (
            AxisMeasure(col_spacing, grid(center - [half[0], 0]), grid(center + [half[0], 0])),
            AxisMeasure(row_spacing, grid(center - [0, half[1]]), grid(center + [0, half[1]])),
        )
    centered = points - center
    # `eigh` en vez de `eig`: la covarianza es simetrica y garantiza autovectores
    # reales y ortonormales, sin la parte imaginaria residual que trae el caso general.
    _, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    horizontal, vertical = sorted(vectors.T, key=lambda vector: abs(float(vector[0])), reverse=True)

    def measure(axis: np.ndarray) -> AxisMeasure:
        projected = centered @ axis
        footprint = abs(float(axis[0])) * col_spacing + abs(float(axis[1])) * row_spacing
        low = center + axis * (float(projected.min()) - footprint / 2)
        high = center + axis * (float(projected.max()) + footprint / 2)
        return AxisMeasure(float(projected.max() - projected.min()) + footprint, grid(low), grid(high))

    return measure(horizontal), measure(vertical)


def segmental_angle(upper: AxisMeasure, lower: AxisMeasure) -> tuple[float, list[Dict[str, float]]]:
    """Angulo entre los platillos de dos cuerpos vertebrales consecutivos.

    Sale de los ejes propios de cada cuerpo, que ya se calcularon para medir su ancho:
    el eje mas horizontal de un cuerpo vertebral es la direccion de sus platillos. No
    se introduce ninguna geometria nueva, y la figura que se dibuja -los dos segmentos
    de ancho- es literalmente la que produjo el numero.

    Se informa el angulo agudo porque una recta no tiene sentido: si dependiera de en
    que orden quedaron los extremos, el mismo segmento daria 8 grados o 172.
    """
    first = math.atan2(upper.end[1] - upper.start[1], upper.end[0] - upper.start[0])
    second = math.atan2(lower.end[1] - lower.start[1], lower.end[0] - lower.start[0])
    degrees = abs(math.degrees(first - second)) % 180
    if degrees > 90:
        degrees = 180 - degrees
    return degrees, segment_points(upper) + segment_points(lower)


def vertebral_listhesis(
    upper: AxisMeasure,
    lower: AxisMeasure,
    col_spacing: float,
    row_spacing: float,
) -> tuple[float, str, list[Dict[str, float]]] | None:
    """Cuanto se corrio un cuerpo vertebral sobre el de abajo.

    En un corte sagital el eje horizontal es el anteroposterior, asi que de cada
    segmento de ancho el extremo de mayor columna es la esquina posterior. El
    deslizamiento es la proyeccion sobre el platillo inferior y no la distancia
    directa: lo que se informa es cuanto se corrio hacia adelante o atras, no cuanto se
    separo en altura.

    El grado de Meyerding sale de la proporcion contra la longitud anteroposterior del
    cuerpo inferior, que es la misma que se midio como su ancho.
    """
    def corners(segment: AxisMeasure) -> tuple[tuple[float, float], tuple[float, float]]:
        return (segment.start, segment.end) if segment.start[0] <= segment.end[0] else (segment.end, segment.start)

    lower_anterior, lower_posterior = corners(lower)
    _, upper_posterior = corners(upper)
    axis_x = (lower_posterior[0] - lower_anterior[0]) * col_spacing
    axis_y = (lower_posterior[1] - lower_anterior[1]) * row_spacing
    endplate = math.hypot(axis_x, axis_y)
    if endplate == 0:
        return None
    delta_x = (upper_posterior[0] - lower_posterior[0]) * col_spacing
    delta_y = (upper_posterior[1] - lower_posterior[1]) * row_spacing
    slip = abs((delta_x * axis_x + delta_y * axis_y) / endplate)
    grade = "I" if slip / endplate <= 0.25 else "II" if slip / endplate <= 0.5 else "III" if slip / endplate <= 0.75 else "IV" if slip / endplate <= 1 else "V"
    points = [
        {"x": lower_anterior[0], "y": lower_anterior[1]},
        {"x": lower_posterior[0], "y": lower_posterior[1]},
        {"x": upper_posterior[0], "y": upper_posterior[1]},
    ]
    return slip, f"grado {grade}", points


def segment_points(segment: AxisMeasure | None) -> list[Dict[str, float]]:
    """Los dos extremos de la medicion, en la forma que consume el visor.

    Vacio cuando la magnitud no es una distancia. Un area no tiene de donde a donde.
    """
    if segment is None:
        return []
    return [
        {"x": segment.start[0], "y": segment.start[1]},
        {"x": segment.end[0], "y": segment.end[1]},
    ]


def build_measurements(
    model_key: str,
    plane: str,
    prediction: np.ndarray,
    confidence: np.ndarray,
    spacing: tuple[float, float] | None,
    slice_index: int,
) -> list[Dict[str, Any]]:
    values: list[Dict[str, Any]] = []
    row_spacing, col_spacing = spacing if spacing else (1.0, 1.0)
    physical = spacing is not None
    dimension_unit = "mm" if physical else "px"
    area_unit = "mm2" if physical else "px2"

    def emit(identifier: str, label: str, binary: np.ndarray, level: str | None, linked: list[str]) -> None:
        ys, xs = np.where(binary)
        if len(xs) == 0:
            return
        width, height = oriented_extent(binary, row_spacing, col_spacing)
        # El area cuenta pixeles, asi que no depende de como este orientada la
        # estructura: es la unica de las tres magnitudes que la caja no distorsionaba.
        area = float(len(xs)) * row_spacing * col_spacing
        common = {
            "level": level,
            # Estas miden su propia mascara, no se derivan de otras mediciones.
            "experimental": False,
            # Distingue "no le pude asignar nivel" de "no corresponde a un nivel".
            # Sin esto las dos llegan como level=null y la pantalla las junta bajo el
            # mismo rotulo, que en el segundo caso acusa a la IA de un fallo que no
            # tuvo. Aca el nivel siempre corresponde, aunque a veces no se conozca.
            "levelScope": "level",
            # Todas las mediciones de una corrida salen del unico corte inferido.
            "sliceIndex": slice_index,
            "source": "AI",
            "confidence": round(float(confidence[binary].mean()), 4),
            "status": "pendiente",
            "outlier": False,
            "linkedLandmarks": linked,
        }
        for metric, magnitude, unit, segment in (
            # El area no lleva segmento: no tiene dos extremos, y la mascara pintada
            # ya la muestra. Dibujarle una linea seria decorar, no ubicar la medicion.
            ("area", area, area_unit, None),
            ("width", width.length, dimension_unit, width),
            ("height", height.length, dimension_unit, height),
        ):
            values.append({
                "id": f"{identifier}-{metric}",
                "label": f"{label} {metric}",
                "labelKey": f"{label} {metric}",
                "value": round(magnitude, 2),
                "aiValue": round(magnitude, 2),
                "reviewerValue": None,
                "unit": unit,
                "points": segment_points(segment),
                **common,
            })

    def emit_single(identifier, label, magnitude, unit, level, binary, level_scope="study", segment=None,
                    points=None, experimental=False, detail=None):
        """Una sola magnitud, para lo que no tiene sentido medir en tres ejes.

        Por defecto describe el estudio y no un nivel: es el caso del canal, una
        mascara continua que atraviesa toda la columna. Que no tenga nivel no es una
        falla de la IA, y el `levelScope` es lo que permite decirlo en la pantalla en
        vez de mostrarlo junto a lo que si quedo sin identificar.
        """
        values.append({
            "id": identifier,
            "label": label,
            "labelKey": label,
            "value": round(magnitude, 2),
            "aiValue": round(magnitude, 2),
            "reviewerValue": None,
            "unit": unit,
            "points": points if points is not None else segment_points(segment),
            "level": level,
            "levelScope": level_scope,
            "sliceIndex": slice_index,
            "source": "AI",
            # Una medicion derivada no se apoya en su propia mascara sino en la
            # geometria de otras dos, asi que se marca para que el visor pueda
            # mostrarla aparte y el medico decida si la quiere.
            "experimental": experimental,
            "detail": detail,
            "confidence": round(float(confidence[binary].mean()), 4) if binary.any() else 0.0,
            "status": "pendiente",
            "outlier": False,
            "linkedLandmarks": [],
        })

    by_class = {
        name: prediction == class_id
        for class_id in sorted(int(value) for value in np.unique(prediction) if int(value) != 0)
        if not is_background_class(name := class_name(model_key, class_id))
    }

    # --- Discos: una instancia por espacio discal ---------------------------
    #
    # "altura del disco L4-L5" es un hallazgo reportable; "altura del grupo de
    # discos" no significa nada clinico.
    disc_instances = connected_instances(by_class["disc_group"]) if "disc_group" in by_class else []
    disc_levels = lumbar_disc_levels(len(disc_instances))
    for position, (component, level) in enumerate(zip(disc_instances, disc_levels), start=1):
        slug = level.lower() if level else f"d{position}"
        # Sin nivel confirmado el landmark del grupo no identifica a este disco en
        # particular: se deja sin vinculo antes que apuntar a un punto ajeno.
        emit(f"{plane}-disc-{slug}", "disc", component, level, [])

    # --- Canal ---------------------------------------------------------------
    #
    # El diametro AP por nivel es la medicion con la que se describe una estenosis, y
    # se publica. Costo dos intentos llegar aca, los dos por medir mal antes que por
    # el modelo.
    #
    # La primera vez la evidencia estaba tomada con el spacing equivocado. La segunda,
    # el argumento para retenerla fue que la mascara ocupa 47 mm de ancho, demasiado
    # para un canal. Ese 47 era la caja envolvente: el mismo error que se acababa de
    # corregir en los discos. El canal recorre la lordosis, asi que su centro se
    # desplaza 35 mm de arriba abajo y la caja mide el recorrido, no el canal.
    #
    # Medido fila por fila, que es como corresponde, la mascara es un canal: una sola
    # componente conexa, continua a lo largo de 230 mm, de 10 a 15 mm de ancho con
    # mediana en 13.1. Por nivel da 13.1, 14.2, 13.1, 12.0, 9.8 y 12.0 mm de T11-T12
    # a L4-L5, valores compatibles con el saco tecal y con un hallazgo real en L3-L4.
    #
    # Lo que se sigue sin afirmar es si la clase delimita el canal oseo o su contenido
    # dural: el dataset la nombra "canal espinal" y nada en el repo lo precisa mas. La
    # medicion describe la mascara segmentada y el umbral de estenosis de las dos
    # lecturas -12 mm para el canal, 10 para el saco- cae del mismo lado en este
    # estudio, asi que la conclusion clinica no depende de cual sea.
    #
    # El area va aparte y sin nivel, porque describe la mascara entera y no un nivel.
    canal = by_class.get("canal")
    if canal is not None and canal.any():
        _, xs = np.where(canal)
        emit_single(
            f"{plane}-canal-area",
            "canal area",
            float(len(xs)) * row_spacing * col_spacing,
            area_unit,
            None,
            canal,
        )
        for position, (component, level) in enumerate(zip(disc_instances, disc_levels), start=1):
            diameter = canal_ap_diameter(canal, component, col_spacing)
            if diameter is None:
                continue
            slug = level.lower() if level else f"d{position}"
            rows = np.where(component.any(axis=1))[0]
            # La confianza es la de la porcion de canal que se midio, no la de toda la
            # mascara: un canal bien segmentado arriba no avala el numero de abajo.
            band = np.zeros_like(canal)
            band[int(rows.min()):int(rows.max()) + 1] = canal[int(rows.min()):int(rows.max()) + 1]
            emit_single(
                f"{plane}-canal-ap-{slug}",
                "canal ap",
                diameter.length,
                dimension_unit,
                level,
                band,
                level_scope="level",
                segment=diameter,
            )

    # --- Cuerpos vertebrales -------------------------------------------------
    #
    # La clase `vertebra_group` no contiene solo cuerpos: incluye los elementos
    # posteriores, que en un sagital aparecen como blobs aparte. Se separan con el
    # mismo criterio que usa la segmentacion -los discos como ancla- y se nombran
    # desde los discos ya identificados, asi que la altura de un cuerpo viaja con su
    # nivel. Esa altura es un hallazgo reportable: una perdida de altura del cuerpo
    # es como se describe un aplastamiento.
    #
    # No se mide el arco posterior. Se ve en la segmentacion y lleva su nivel, pero
    # el ancho de un arco no es una magnitud que un informe reporte.
    #
    # Un cuerpo que el encuadre corto se mide igual pero sin nivel: la medicion
    # describe la mascara, y es el nombre lo que no se puede afirmar.
    vertebra_group = by_class.get("vertebra_group")
    if vertebra_group is not None and vertebra_group.any():
        bodies, _ = split_vertebral_bodies(connected_instances(vertebra_group), disc_instances)
        body_levels = name_vertebral_bodies(bodies, disc_instances, disc_levels)
        # El id es posicional y el nivel viaja aparte, por lo mismo que en la
        # segmentacion: derivarlo del nombre produce colisiones cuando dos
        # componentes reciben el mismo nivel, y el id es la clave de la lista.
        body_segments: list[AxisMeasure] = []
        for position, (component, level) in enumerate(zip(bodies, body_levels), start=1):
            emit(f"{plane}-vertebra-v{position}", "vertebra", component, level, [])
            body_segments.append(oriented_extent(component, row_spacing, col_spacing)[0])

        # --- Derivadas de dos cuerpos: angulo segmentario y listesis ---------
        #
        # No salen de una mascara propia sino de la geometria de dos cuerpos vecinos,
        # que ya se calculo para medir su ancho. Se publican marcadas como
        # experimentales para que el visor las muestre en su propia capa: son las dos
        # mediciones que un informe de columna reporta y que el modelo no fue
        # entrenado para dar, asi que la decision de usarlas es del medico y no del
        # sistema.
        #
        # El nivel es el espacio discal entre los dos cuerpos, que es el segmento de
        # movimiento al que pertenece la medida. Si alguno de los dos no tiene nombre,
        # no se publica: sin saber entre que vertebras esta, el numero no ubica nada.
        for position in range(len(bodies) - 1):
            upper_name, lower_name = body_levels[position], body_levels[position + 1]
            if not upper_name or not lower_name:
                continue
            level = f"{upper_name}-{lower_name}"
            upper, lower = body_segments[position], body_segments[position + 1]
            degrees, angle_points = segmental_angle(upper, lower)
            emit_single(
                f"{plane}-segmental-angle-{level.lower()}",
                "segmental angle",
                degrees,
                "deg",
                level,
                bodies[position] | bodies[position + 1],
                level_scope="level",
                points=angle_points,
                experimental=True,
            )
            # La listesis no se publica, aunque el calculo esta implementado
            # (vertebral_listhesis) y es el segundo hallazgo estructural mas reportado
            # despues de la estenosis.
            #
            # Toda la medicion depende de un solo punto: la esquina posterior del
            # platillo. Y ese punto es justo donde la segmentacion es menos precisa,
            # porque es el borde del cuerpo que toca el disco. Sobre el estudio de
            # prueba, derivarlo del eje de ancho -que cruza el centroide- y derivarlo
            # del borde de la mascara dan resultados que difieren hasta 17 mm en el
            # mismo nivel: 7.30 contra 24.06 en L1-L2, 6.64 contra 20.78 en L2-L3.
            # Con esa dispersion, cualquiera de los dos numeros que se publique es
            # arbitrario.
            #
            # La diferencia con el angulo segmentario, que si se publica, no es de
            # metodo sino de verificacion: el angulo se pudo contrastar contra un
            # rango conocido -la lordosis lumbar anda entre 40 y 60 grados, y la suma
            # de L1 a L5 dio 36- mientras que para la listesis haria falta un estudio
            # con un deslizamiento medido por alguien. Sin eso, medirla mejor tampoco
            # alcanzaria: seguiria siendo un numero sin contra que validarse.
            #
            # Mientras tanto el revisor la mide a mano con su propia herramienta, que
            # es donde corresponde: es el quien reconoce las esquinas del platillo.

    # --- Clases que el modelo no separa en instancias ------------------------
    for label, binary in by_class.items():
        if label in {"disc_group", "canal", "vertebra_group"}:
            continue
        emit(f"{plane}-{label}", label, binary, None, [f"lm-mask-{plane}-{label.replace('_', '-')}-centroid"])

    return values


def canal_ap_diameter(canal, disc, col_spacing):
    """Diametro anteroposterior del canal a la altura de un disco.

    En un corte sagital el eje horizontal es el anteroposterior, asi que el ancho
    del canal medido en las filas que ocupa el disco es el diametro AP a ese nivel
    -la medicion con la que se describe una estenosis.

    Se toma el **minimo** de esas filas y no el promedio: una estenosis se define
    por el punto mas estrecho, y promediarlo lo diluiria justo donde importa.

    No se informa nada cuando la mascara del canal empieza o termina dentro de las
    filas del disco. En ese borde la mascara se adelgaza hasta desaparecer, y esa
    caida es indistinguible de un estrechamiento real: sobre el estudio de prueba el
    canal terminaba en la ultima fila de L5-S1 y el minimo daba 2.19 mm, que leido
    como diametro AP describe un bloqueo completo donde solo hay un encuadre que se
    acaba. Se exige que el canal se extienda mas alla del disco por arriba y por
    abajo, que es la unica forma de saber que lo que se mide es el canal y no su
    final.
    """
    rows = np.where(disc.any(axis=1))[0]
    canal_rows = np.where(canal.any(axis=1))[0]
    if rows.size == 0 or canal_rows.size == 0:
        return None
    top, bottom = int(rows.min()), int(rows.max())
    if int(canal_rows.min()) >= top or int(canal_rows.max()) <= bottom:
        return None
    widths = []
    for row in range(top, bottom + 1):
        columns = np.where(canal[row])[0]
        if columns.size == 0:
            continue
        widths.append((float(columns.max() - columns.min() + 1), row, float(columns.min()), float(columns.max())))
    if not widths:
        return None
    span, row, left, right = min(widths)
    # El segmento va de borde a borde del pixel, no de centro a centro, para que su
    # largo sea exactamente el valor informado.
    return AxisMeasure(span * col_spacing, (left - 0.5, float(row)), (right + 0.5, float(row)))


#: Techo del catalogo de previsualizaciones. Una serie mas larga que esto no se
#: recorta: se deja sin catalogo y el visor lo informa, antes que persistir un
#: subconjunto que haria que unos cortes tengan imagen y otros no sin explicacion.
MAX_SLICE_PREVIEWS = 512


def save_slice_pixels(run_id: str, plane: str, loaded: LoadedInput, axis: int) -> Dict[str, Any]:
    """Escribe cada corte como enteros de 16 bits, sin ventanear.

    El PNG que se servia ya venia a 8 bits y con una ventana aplicada: el W/L del
    visor era un filtro de brillo y contraste sobre esa imagen, no ventaneo. Con las
    intensidades originales el visor puede mapear centro y ancho de verdad, que es
    lo que hace un PACS y lo que permite ver una hernia contra el disco.

    Se manda int16 little-endian, el mismo tipo en el que viene una RM, y aparte el
    rango real de la serie para que el visor pueda proponer una ventana inicial sin
    tener que recorrer el volumen entero.
    """
    if loaded.array.ndim != 3:
        return {}
    count = int(loaded.array.shape[axis])
    if count <= 0 or count > MAX_SLICE_PREVIEWS:
        return {}
    output_dir = get_settings().output_dir / "real_inference" / run_id / plane
    output_dir.mkdir(parents=True, exist_ok=True)
    volume = np.asarray(loaded.array)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return {}
    minimum = float(finite.min())
    maximum = float(finite.max())
    outputs: Dict[str, str] = {}
    for index in range(count):
        frame = np.nan_to_num(np.take(volume, index, axis=axis), nan=minimum)
        path = output_dir / slice_pixels_asset_name(index)
        path.write_bytes(np.clip(frame, -32768, 32767).astype("<i2").tobytes())
        outputs[f"slicePixels{index}"] = str(path)
    register_run_assets(run_id, plane, outputs)
    height, width = np.take(volume, 0, axis=axis).shape
    return {
        "count": count,
        "width": int(width),
        "height": int(height),
        "dtype": "int16",
        "byteOrder": "little",
        "min": minimum,
        "max": maximum,
    }


def save_slice_previews(run_id: str, plane: str, loaded: LoadedInput, axis: int) -> Dict[str, str]:
    """Escribe un PNG por corte de la serie, en resolucion nativa.

    Sin esto solo el corte que la IA analizo tiene imagen persistida, y navegar la
    serie muestra el estudio completo en la barra de cortes pero un unico cuadro
    con contenido. Cada PNG es la imagen sola, sin superposicion: la segmentacion
    existe unicamente para el corte inferido y pintarla sobre los demas mostraria
    una mascara que no les corresponde.
    """
    if loaded.array.ndim != 3:
        return {}
    count = int(loaded.array.shape[axis])
    if count <= 1 or count > MAX_SLICE_PREVIEWS:
        return {}
    output_dir = get_settings().output_dir / "real_inference" / run_id / plane
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    for index in range(count):
        frame = robust_percentile_normalize(np.take(loaded.array, index, axis=axis))
        path = output_dir / slice_asset_name(index)
        Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8)).save(path)
        outputs[f"slice{index}"] = str(path)
    register_run_assets(run_id, plane, outputs)
    return outputs


def series_preview_dir(input_id: str) -> Path:
    return get_settings().output_dir / "series_preview" / input_id


def render_series_previews(input_id: str, input_path: str) -> int:
    """Escribe un PNG por corte de una serie registrada y devuelve cuantos hay.

    Es lo que permite mostrar una serie sobre la que no corrio la IA -la T1, el axial
    T1 que no tiene modelo, el localizer-. No hay inferencia: se apila el volumen, se
    normaliza cada corte igual que en `save_slice_previews` y se guarda la imagen sola,
    sin superposicion, porque no hay mascara que le corresponda.

    Se rinde una vez por serie y queda en disco. Armar el volumen desde los DICOM es lo
    caro, y hacerlo en cada pedido de corte significaria apilar la serie entera quince
    veces para recorrerla una.
    """
    output_dir = series_preview_dir(input_id)
    # El contador se escribe al final: si el render se corta a la mitad, el marcador no
    # esta y el proximo pedido rehace la serie en vez de servir un tramo incompleto.
    marker = output_dir / "count"
    if marker.exists():
        return int(marker.read_text().strip() or 0)

    # El plano no incide: `resolve_input_path` no lo usa, y aca no hay canonicalizacion
    # por plano porque no se va a inferir sobre esto, solo mostrarlo.
    loaded = load_input(input_path, "sagittal")
    array = loaded.array
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        return 0
    count = min(int(array.shape[0]), MAX_SLICE_PREVIEWS)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        frame = robust_percentile_normalize(array[index])
        Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8)).save(output_dir / slice_asset_name(index))
    marker.write_text(str(count))
    return count


def class_mask_asset_name(class_name_value: str) -> str:
    return f"mask-{class_name_value}.png"


def save_class_masks(
    output_dir: Path,
    model_key: str,
    render_mask: np.ndarray,
    present: list[int],
) -> Dict[str, str]:
    """Escribe una mascara por clase, en RGBA con fondo transparente.

    El overlay compuesto en un solo PNG es lo que obliga al visor a mostrar la
    segmentacion entera o nada: no hay forma de ocultar el canal y dejar los discos.
    Con una capa por clase el medico elige que mira, que es como se lee cuando una
    estructura tapa a la que interesa.

    El alfa es binario -0 o 255- y no lleva la opacidad de presentacion: esa la
    decide el visor con su propio control, y hornearla aca la volveria imposible de
    cambiar sin reprocesar el estudio.
    """
    outputs: Dict[str, str] = {}
    for class_id in present:
        name = class_name(model_key, class_id)
        if is_background_class(name):
            continue
        color = np.asarray(PALETTE.get(class_id, (255, 255, 0)), dtype=np.uint8)
        selected = render_mask == class_id
        rgba = np.zeros((*render_mask.shape, 4), dtype=np.uint8)
        rgba[selected, 0:3] = color
        rgba[selected, 3] = 255
        path = output_dir / class_mask_asset_name(name)
        Image.fromarray(rgba, mode="RGBA").save(path)
        outputs[f"classMask_{name}"] = str(path)
    return outputs


def save_outputs(
    run_id: str,
    plane: str,
    model_key: str,
    image: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
    render_image: np.ndarray | None = None,
) -> Dict[str, str]:
    """Persiste los artefactos de la corrida.

    `image`/`prediction`/`confidence` estan en la grilla del modelo y definen las
    mediciones. `render_image` es el mismo corte en resolucion nativa y es lo que
    se dibuja: los PNG salen de ahi, con la mascara remuestreada por vecino mas
    cercano para que superponga exactamente sobre esos pixeles.
    """
    output_dir = get_settings().output_dir / "real_inference" / run_id / plane
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "input.png"
    mask_path = output_dir / "mask.npy"
    confidence_path = output_dir / "confidence.npy"
    overlay_path = output_dir / "overlay.png"
    mask_preview_path = output_dir / "mask-preview.png"

    render = image if render_image is None else np.asarray(render_image, dtype=np.float32)
    render_mask = upsample_labels(prediction, (int(render.shape[0]), int(render.shape[1])))
    present = sorted(int(value) for value in np.unique(prediction) if int(value) != 0)

    Image.fromarray(np.clip(render * 255.0, 0, 255).astype(np.uint8)).save(image_path)
    np.save(mask_path, prediction.astype(np.uint8))
    np.save(confidence_path, confidence.astype(np.float32))

    overlay = np.stack([render, render, render], axis=-1)
    alpha = 0.42
    for class_id in present:
        color = np.asarray(PALETTE.get(class_id, (255, 255, 0)), dtype=np.float32) / 255.0
        selected = render_mask == class_id
        overlay[selected] = (1.0 - alpha) * overlay[selected] + alpha * color
    Image.fromarray(np.clip(overlay * 255.0, 0, 255).astype(np.uint8)).save(overlay_path)

    preview = np.zeros((*render_mask.shape, 3), dtype=np.float32)
    for class_id in present:
        preview[render_mask == class_id] = np.asarray(PALETTE.get(class_id, (255, 255, 0)), dtype=np.float32) / 255.0
    Image.fromarray(np.clip(preview * 255.0, 0, 255).astype(np.uint8)).save(mask_preview_path)
    outputs = {
        "imagePath": str(image_path),
        "maskPath": str(mask_path),
        "confidencePath": str(confidence_path),
        "overlayPath": str(overlay_path),
        "maskPreviewPath": str(mask_preview_path),
        # El overlay compuesto se conserva: es lo que ve un consumidor que no sabe
        # componer capas, y evita romper corridas ya persistidas.
        **save_class_masks(output_dir, model_key, render_mask, present),
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
    outputs = save_outputs(
        run_id,
        request.plane,
        request.model_key,
        image,
        prediction,
        confidence,
        native_slice(loaded, selected_axis, selected_slice),
    )
    slice_previews = save_slice_previews(run_id, request.plane, loaded, selected_axis)
    slice_pixels = save_slice_pixels(run_id, request.plane, loaded, selected_axis)
    # Los PNG por corte quedan registrados para poder servirlos, pero fuera de la
    # lista de assets del contrato: son cientos de entradas que solo repetirian el
    # patron `slice-NNN.png`. El consumidor necesita un numero, no el catalogo.
    assets = {
        name: metadata
        for name, metadata in registered_assets_for_run(run_id, request.plane).items()
        if not is_slice_asset_name(name)
    }
    spacing = in_plane_spacing(loaded, selected_axis)
    spacing_unit = "mm" if spacing else None
    masks = build_masks(request.model_key, request.plane, prediction, confidence, series_id, selected_slice)
    segmentation = build_segmentation(request.model_key, request.plane, prediction)
    landmarks = build_landmarks(masks)
    measurement_values = build_measurements(
        request.model_key,
        request.plane,
        prediction,
        confidence,
        # Las mediciones se cuentan en pixeles de la prediccion, no de la imagen
        # original, asi que necesitan el spacing de esa grilla. `spacing` sigue siendo
        # el nativo porque describe la imagen que se muestra, y es el que usa la
        # herramienta de medir del visor.
        prediction_grid_spacing(spacing, loaded.array.shape, selected_axis, prediction.shape),
        selected_slice,
    )

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
        "segmentation": segmentation,
        "volumeGeometry": volume_geometry(loaded, selected_axis, slice_count, selected_slice),
        "slicePreviewCount": len(slice_previews),
        "slicePixels": slice_pixels or None,
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
            "inputShapeNative": loaded.metadata.get("inputShapeNative"),
            "inputShapeCanonical": loaded.metadata.get("inputShapeCanonical"),
            "inputOrientationTransform": loaded.metadata.get("inputOrientationTransform"),
            "spacingXyz": loaded.metadata.get("spacingXyz"),
            "arrayAxisSpacingNative": loaded.metadata.get("arrayAxisSpacingNative"),
            "arrayAxisSpacingCanonical": loaded.metadata.get("arrayAxisSpacingCanonical"),
            "inPlaneSpacing": list(spacing) if spacing else None,
            "inPlaneSpacingUnit": spacing_unit,
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
