from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .real_inference_runtime import SUPPORTED_EXTENSIONS


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
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise InputRegistryError(f"formato no soportado: {suffix}", status_code=400)

    input_id = f"inp_{uuid4().hex}"
    record = InputRecord(
        input_id=input_id,
        case_id=request.case_id,
        plane=request.plane,
        path=path,
        format=suffix.lstrip("."),
        size=path.stat().st_size,
        source_key=request.source_key,
    )
    _INPUT_REGISTRY[input_id] = record
    return public_input_metadata(record)


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
