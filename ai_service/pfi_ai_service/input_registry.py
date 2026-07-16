from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .real_inference_runtime import SUPPORTED_EXTENSIONS

DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
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
    destination = upload_destination(input_id, normalized_plane, suffix)
    size = write_limited_upload(stream, destination, max_upload_bytes())
    record = InputRecord(
        input_id=input_id,
        case_id=case_id,
        plane=normalized_plane,
        path=destination,
        format=suffix.lstrip("."),
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


def validate_plane(plane: str) -> str:
    normalized = str(plane).strip().lower()
    if normalized not in {"sagittal", "axial"}:
        raise InputRegistryError("plane invalido", status_code=400)
    return normalized


def validate_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise InputRegistryError(f"extension no permitida: {suffix or 'sin_extension'}", status_code=400)
    return suffix


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
    if not record.path.exists() or not record.path.is_file():
        raise InputRegistryError("archivo asociado al inputId no disponible", status_code=404)
    return record
