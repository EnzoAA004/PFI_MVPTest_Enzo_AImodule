from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .real_inference_runtime import SUPPORTED_EXTENSIONS

# A .zip is accepted only at upload time as a container for a DICOM series; it is
# never a direct inference format (SUPPORTED_EXTENSIONS drives inference dispatch).
ALLOWED_UPLOAD_EXTENSIONS = SUPPORTED_EXTENSIONS | {".zip"}

DEFAULT_MAX_UPLOAD_BYTES = 300 * 1024 * 1024
DEFAULT_MAX_SERIES_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_SERIES_FILES = 4096
UPLOAD_CHUNK_BYTES = 1024 * 1024


class InputRegistryError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class InputRegistrationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_by_name=True)

    case_id: str = Field(..., alias="caseId")
    plane: Literal["sagittal", "axial"]
    source_key: str = Field(..., alias="sourceKey")


@dataclass(frozen=True)
class InputRecord:
    input_id: str
    case_id: str
    plane: str
    path: Path
    format: str
    size: int
    source_key: str


SERVER_SIDE_SOURCES = {
    "fixture:sagittal_sample": {
        "plane": "sagittal",
        "path": Path("ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy"),
    },
    "fixture:axial_sample": {
        "plane": "axial",
        "path": Path("ai_service/tests/fixtures/real_baseline/axial_sample_input.npy"),
    },
}

_INPUT_REGISTRY: dict[str, InputRecord] = {}


def upload_root() -> Path:
    return Path(os.getenv("PFI_UPLOAD_DIR", "uploads/inputs"))


def max_upload_bytes() -> int:
    raw = os.getenv("PFI_MAX_UPLOAD_BYTES")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_UPLOAD_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise InputRegistryError("PFI_MAX_UPLOAD_BYTES invalido", status_code=500) from exc
    if value <= 0:
        raise InputRegistryError("PFI_MAX_UPLOAD_BYTES debe ser positivo", status_code=500)
    return value


def register_server_side_input(request: InputRegistrationRequest) -> dict[str, object]:
    source = SERVER_SIDE_SOURCES.get(request.source_key)
    if source is None:
        raise InputRegistryError("sourceKey no registrado", status_code=404)
    expected_plane = str(source["plane"])
    if request.plane != expected_plane:
        raise InputRegistryError(f"plane incompatible con sourceKey: expected={expected_plane}", status_code=400)

    path = Path(source["path"])
    if not path.exists() or not path.is_file():
        raise InputRegistryError("recurso server-side no disponible", status_code=404)
    suffix = validate_suffix(path.name)

    return register_existing_path(
        case_id=request.case_id,
        plane=request.plane,
        path=path,
        source_key=request.source_key,
        suffix=suffix,
    )


def register_uploaded_input(
    *,
    case_id: str,
    plane: str,
    client_filename: str | None,
    stream: BinaryIO,
) -> dict[str, object]:
    normalized_plane = validate_plane(plane)
    suffix = validate_suffix(client_filename or "")
    input_id = f"inp_{uuid4().hex}"
    if suffix == ".zip":
        destination, size, fmt = store_zip_series(input_id, normalized_plane, stream)
    else:
        destination = upload_destination(input_id, normalized_plane, suffix)
        size = write_limited_upload(stream, destination, max_upload_bytes())
        fmt = suffix.lstrip(".")
    record = InputRecord(
        input_id=input_id,
        case_id=case_id,
        plane=normalized_plane,
        path=destination,
        format=fmt,
        size=size,
        source_key="upload",
    )
    _INPUT_REGISTRY[input_id] = record
    return public_input_metadata(record)


def register_existing_path(
    *,
    case_id: str,
    plane: str,
    path: Path,
    source_key: str,
    suffix: str | None = None,
) -> dict[str, object]:
    normalized_plane = validate_plane(plane)
    clean_suffix = suffix or validate_suffix(path.name)
    input_id = f"inp_{uuid4().hex}"
    record = InputRecord(
        input_id=input_id,
        case_id=case_id,
        plane=normalized_plane,
        path=path,
        format=clean_suffix.lstrip("."),
        size=path.stat().st_size,
        source_key=source_key,
    )
    _INPUT_REGISTRY[input_id] = record
    return public_input_metadata(record)


def register_series_files(*, case_id: str, plane: str, file_paths: list) -> dict[str, object]:
    """Copy a classified series' files into a fresh per-plane input directory and register it.

    Used by study ingestion: the caller has already selected which series belongs to
    this plane; here we materialize it as an ordinary directory-backed input so the
    existing pipeline (read_dicom_series) can stack it into a 3D volume.
    """
    normalized_plane = validate_plane(plane)
    if not file_paths:
        raise InputRegistryError("serie sin archivos", status_code=400)
    input_id = f"inp_{uuid4().hex}"
    series_dir = upload_root() / normalized_plane / input_id
    series_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for index, source in enumerate(file_paths):
        # These are DICOM files (GDCM-classified); normalize to .dcm so downstream
        # extension-based checks work even when the source was .ima or extension-less.
        destination = series_dir / f"{index:05d}.dcm"
        shutil.copyfile(Path(source), destination)
        total += destination.stat().st_size
    record = InputRecord(
        input_id=input_id,
        case_id=case_id,
        plane=normalized_plane,
        path=series_dir,
        format="dicom_series",
        size=total,
        source_key="study-upload",
    )
    _INPUT_REGISTRY[input_id] = record
    return public_input_metadata(record)


def validate_plane(plane: str) -> str:
    normalized = str(plane).strip().lower()
    if normalized not in {"sagittal", "axial"}:
        raise InputRegistryError("plane invalido", status_code=400)
    return normalized


def validate_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise InputRegistryError(f"extension no permitida: {suffix or 'sin_extension'}", status_code=400)
    return suffix


def max_series_uncompressed_bytes() -> int:
    raw = os.getenv("PFI_MAX_SERIES_UNCOMPRESSED_BYTES")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_SERIES_UNCOMPRESSED_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise InputRegistryError("PFI_MAX_SERIES_UNCOMPRESSED_BYTES invalido", status_code=500) from exc
    if value <= 0:
        raise InputRegistryError("PFI_MAX_SERIES_UNCOMPRESSED_BYTES debe ser positivo", status_code=500)
    return value


def max_series_files() -> int:
    raw = os.getenv("PFI_MAX_SERIES_FILES")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_SERIES_FILES
    try:
        value = int(raw)
    except ValueError as exc:
        raise InputRegistryError("PFI_MAX_SERIES_FILES invalido", status_code=500) from exc
    if value <= 0:
        raise InputRegistryError("PFI_MAX_SERIES_FILES debe ser positivo", status_code=500)
    return value


def store_zip_series(input_id: str, plane: str, stream: BinaryIO) -> tuple[Path, int, str]:
    """Persist an uploaded .zip and extract it into a per-input directory.

    Guards against path traversal (zip-slip), decompression bombs (total-size and
    file-count caps) and empty archives. Returns the extraction directory, its total
    uncompressed size and the "dicom_series" format tag.
    """
    plane_dir = upload_root() / plane
    plane_dir.mkdir(parents=True, exist_ok=True)
    zip_path = plane_dir / f"{input_id}.zip"
    write_limited_upload(stream, zip_path, max_upload_bytes())

    extract_dir = plane_dir / input_id
    try:
        total, count = safe_extract_zip(zip_path, extract_dir)
    finally:
        if zip_path.exists():
            zip_path.unlink()
    if count == 0:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise InputRegistryError("zip vacio", status_code=400)
    return extract_dir, total, "dicom_series"


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> tuple[int, int]:
    """Extract a zip into extract_dir, guarding against zip-slip and zip-bombs.

    Returns (total_uncompressed_bytes, file_count). On any failure the partial
    extraction directory is removed and an InputRegistryError is raised.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    resolved_extract = extract_dir.resolve()
    max_bytes = max_series_uncompressed_bytes()
    max_files = max_series_files()
    total = 0
    count = 0
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                count += 1
                if count > max_files:
                    raise InputRegistryError("zip con demasiados archivos", status_code=413)
                total += member.file_size
                if total > max_bytes:
                    raise InputRegistryError("zip descomprimido excede el limite", status_code=413)
                member_path = Path(member.filename)
                if member_path.is_absolute() or member_path.drive:
                    raise InputRegistryError("zip con ruta invalida (path traversal)", status_code=400)
                target = (extract_dir / member.filename).resolve()
                if not str(target).startswith(str(resolved_extract)):
                    raise InputRegistryError("zip con ruta invalida (path traversal)", status_code=400)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle, UPLOAD_CHUNK_BYTES)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise InputRegistryError("archivo zip invalido o corrupto", status_code=400) from exc
    except Exception:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise
    return total, count


def upload_destination(input_id: str, plane: str, suffix: str) -> Path:
    root = upload_root()
    destination_dir = root / plane
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{input_id}{suffix}"
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if not str(resolved_destination).startswith(str(resolved_root)):
        raise InputRegistryError("ruta de upload invalida", status_code=400)
    return destination


def write_limited_upload(stream: BinaryIO, destination: Path, max_bytes: int) -> int:
    size = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = stream.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise InputRegistryError("archivo excede el limite de tama?o", status_code=413)
                handle.write(chunk)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    if size == 0:
        raise InputRegistryError("archivo vacio", status_code=400)
    return size


def public_input_metadata(record: InputRecord) -> dict[str, object]:
    return {
        "inputId": record.input_id,
        "caseId": record.case_id,
        "plane": record.plane,
        "format": record.format,
        "size": record.size,
    }


def resolve_input_id(input_id: str, *, case_id: str, plane: str) -> InputRecord:
    record = _INPUT_REGISTRY.get(input_id)
    if record is None:
        raise InputRegistryError("inputId no registrado", status_code=404)
    if record.case_id != case_id:
        raise InputRegistryError("inputId no pertenece al caseId solicitado", status_code=409)
    if record.plane != plane:
        raise InputRegistryError("inputId no pertenece al plano solicitado", status_code=409)
    # A record path is a file for single uploads, or a directory for an extracted
    # DICOM series (store_zip_series) — accept either as long as it still exists.
    if not record.path.exists() or not (record.path.is_file() or record.path.is_dir()):
        raise InputRegistryError("archivo asociado al inputId no disponible", status_code=404)
    return record
