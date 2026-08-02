from __future__ import annotations

import re
from urllib.parse import unquote
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ALLOWED_ASSET_NAMES = frozenset({"input.png", "mask.npy", "confidence.npy", "overlay.png", "mask-preview.png", "lumbar-3d-mesh.json"})
PUBLIC_BROWSER_ASSET_NAMES = frozenset({"input.png", "overlay.png", "mask-preview.png", "lumbar-3d-mesh.json"})
INTERNAL_RAW_ASSET_NAMES = ALLOWED_ASSET_NAMES - PUBLIC_BROWSER_ASSET_NAMES

# Catalogo de previsualizaciones por corte: `slice-007.png`. Es el unico nombre de
# asset que no puede estar en una lista fija, porque la cantidad de cortes depende
# del estudio. El patron es igual de estricto que la lista: solo digitos, sin
# separadores de ruta ni extension arbitraria, de modo que no amplia la superficie
# de path traversal que la lista ya cerraba.
_SLICE_ASSET_PATTERN = re.compile(r"^slice-\d{3,5}\.(png|raw)$")

# Mascara de una clase de segmentacion: `mask-vertebra_group.png`. Tampoco puede ir
# en una lista fija, porque las clases dependen del modelo. El patron acepta solo
# minusculas, digitos y guion bajo -el formato de los class_names del registro- asi
# que no admite separadores de ruta ni extension arbitraria.
_CLASS_MASK_PATTERN = re.compile(r"^mask-[a-z][a-z0-9_]{0,31}\.png$")


def is_slice_asset_name(asset_name: str) -> bool:
    return bool(_SLICE_ASSET_PATTERN.fullmatch(asset_name))


def is_class_mask_asset_name(asset_name: str) -> bool:
    return bool(_CLASS_MASK_PATTERN.fullmatch(asset_name))


def slice_asset_name(index: int) -> str:
    return f"slice-{int(index):03d}.png"


def slice_pixels_asset_name(index: int) -> str:
    return f"slice-{int(index):03d}.raw"


_VALID_PLANES = {"sagittal", "axial", "workspace"}
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,96}$")


class AssetRegistryError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class AssetRecord:
    run_id: str
    plane: Literal["sagittal", "axial", "workspace"]
    asset_name: str
    path: Path
    size: int


_ASSET_REGISTRY: dict[tuple[str, str, str], AssetRecord] = {}


def clear_asset_registry() -> None:
    _ASSET_REGISTRY.clear()


def register_run_assets(run_id: str, plane: str, outputs: dict[str, str]) -> dict[str, dict[str, object]]:
    normalized_plane = validate_plane(plane)
    registered: dict[str, dict[str, object]] = {}
    for raw_path in outputs.values():
        path = Path(raw_path)
        asset_name = path.name
        if asset_name not in ALLOWED_ASSET_NAMES and not is_slice_asset_name(asset_name) and not is_class_mask_asset_name(asset_name):
            continue
        if not path.exists() or not path.is_file():
            continue
        record = AssetRecord(
            run_id=run_id,
            plane=normalized_plane,  # type: ignore[arg-type]
            asset_name=asset_name,
            path=path,
            size=path.stat().st_size,
        )
        _ASSET_REGISTRY[(run_id, normalized_plane, asset_name)] = record
        registered[asset_name] = public_asset_metadata(record)
    return registered


def register_workspace_asset(run_id: str, asset_name: str, path: Path) -> dict[str, object] | None:
    normalized_run_id = validate_run_id(run_id)
    normalized_asset = validate_asset_name(asset_name)
    if not path.exists() or not path.is_file():
        return None
    expected_path = workspace_asset_path(normalized_run_id, normalized_asset)
    if path.resolve() != expected_path:
        raise AssetRegistryError("ruta de asset workspace invalida", status_code=400)
    record = AssetRecord(
        run_id=normalized_run_id,
        plane="workspace",
        asset_name=normalized_asset,
        path=expected_path,
        size=path.stat().st_size,
    )
    _ASSET_REGISTRY[(normalized_run_id, "workspace", normalized_asset)] = record
    return public_asset_metadata(record)


def resolve_run_asset(run_id: str, plane: str, asset_name: str) -> AssetRecord:
    normalized_plane = validate_plane(plane)
    normalized_asset = validate_asset_name(asset_name)
    normalized_run_id = validate_run_id(run_id) if normalized_plane == "workspace" else run_id
    record = _ASSET_REGISTRY.get((normalized_run_id, normalized_plane, normalized_asset))
    if record is None and normalized_plane == "workspace" and normalized_asset == "lumbar-3d-mesh.json":
        record = rehydrate_workspace_asset(normalized_run_id, normalized_asset)
    if record is None:
        raise AssetRegistryError("asset no registrado", status_code=404)
    if not record.path.exists() or not record.path.is_file():
        raise AssetRegistryError("archivo de asset no disponible", status_code=404)
    return record


def rehydrate_workspace_asset(run_id: str, asset_name: str) -> AssetRecord | None:
    normalized_run_id = validate_run_id(run_id)
    path = workspace_asset_path(normalized_run_id, asset_name)
    if not path.exists() or not path.is_file():
        return None
    record = AssetRecord(
        run_id=normalized_run_id,
        plane="workspace",
        asset_name=asset_name,
        path=path,
        size=path.stat().st_size,
    )
    _ASSET_REGISTRY[(normalized_run_id, "workspace", asset_name)] = record
    return record


def validate_run_id(run_id: str) -> str:
    raw = str(run_id or "").strip()
    decoded = raw
    for _ in range(3):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    lowered_values = {raw.lower(), decoded.lower()}
    forbidden = ("..", "/", "\\", "%2e", "%2f", "%5c")
    if not raw or any(token in value for value in lowered_values for token in forbidden):
        raise AssetRegistryError("runId invalido", status_code=403)
    if raw != decoded or not _RUN_ID_PATTERN.fullmatch(decoded):
        raise AssetRegistryError("runId invalido", status_code=403)
    return decoded


def workspace_asset_root() -> Path:
    from .settings import get_settings

    return (get_settings().output_dir / "multiplanar_3d").resolve()


def workspace_asset_path(run_id: str, asset_name: str) -> Path:
    normalized_run_id = validate_run_id(run_id)
    normalized_asset = validate_asset_name(asset_name)
    root = workspace_asset_root()
    path = (root / normalized_run_id / normalized_asset).resolve()
    if path.parent.parent != root:
        raise AssetRegistryError("ruta de asset workspace invalida", status_code=403)
    return path


def public_asset_metadata(record: AssetRecord) -> dict[str, object]:
    return {
        "runId": record.run_id,
        "plane": record.plane,
        "assetName": record.asset_name,
        "size": record.size,
    }


def registered_assets_for_run(run_id: str, plane: str) -> dict[str, dict[str, object]]:
    normalized_plane = validate_plane(plane)
    return {
        asset_name: public_asset_metadata(record)
        for (stored_run_id, stored_plane, asset_name), record in sorted(_ASSET_REGISTRY.items())
        if stored_run_id == run_id and stored_plane == normalized_plane
    }


def validate_plane(plane: str) -> str:
    normalized = str(plane).strip().lower()
    if normalized not in _VALID_PLANES:
        raise AssetRegistryError("plane invalido", status_code=400)
    return normalized


def validate_asset_name(asset_name: str) -> str:
    normalized = str(asset_name).strip()
    if normalized != Path(normalized).name or "/" in normalized or "\\" in normalized or ".." in normalized:
        raise AssetRegistryError("assetName invalido", status_code=403)
    if normalized not in ALLOWED_ASSET_NAMES and not is_slice_asset_name(normalized) and not is_class_mask_asset_name(normalized):
        raise AssetRegistryError("assetName no permitido", status_code=403)
    return normalized


def is_public_browser_asset(asset_name: str) -> bool:
    normalized = validate_asset_name(asset_name)
    return normalized in PUBLIC_BROWSER_ASSET_NAMES or is_slice_asset_name(normalized) or is_class_mask_asset_name(normalized)
