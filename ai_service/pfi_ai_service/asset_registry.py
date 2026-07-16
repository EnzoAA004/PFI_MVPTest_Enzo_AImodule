from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ALLOWED_ASSET_NAMES = frozenset({"input.png", "mask.npy", "confidence.npy", "overlay.png", "mask-preview.png"})
PUBLIC_BROWSER_ASSET_NAMES = frozenset({"input.png", "overlay.png", "mask-preview.png"})
INTERNAL_RAW_ASSET_NAMES = ALLOWED_ASSET_NAMES - PUBLIC_BROWSER_ASSET_NAMES
_VALID_PLANES = {"sagittal", "axial"}


class AssetRegistryError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class AssetRecord:
    run_id: str
    plane: Literal["sagittal", "axial"]
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
        if asset_name not in ALLOWED_ASSET_NAMES:
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


def resolve_run_asset(run_id: str, plane: str, asset_name: str) -> AssetRecord:
    normalized_plane = validate_plane(plane)
    normalized_asset = validate_asset_name(asset_name)
    record = _ASSET_REGISTRY.get((run_id, normalized_plane, normalized_asset))
    if record is None:
        raise AssetRegistryError("asset no registrado", status_code=404)
    if not record.path.exists() or not record.path.is_file():
        raise AssetRegistryError("archivo de asset no disponible", status_code=404)
    return record


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
    if normalized not in ALLOWED_ASSET_NAMES:
        raise AssetRegistryError("assetName no permitido", status_code=403)
    return normalized


def is_public_browser_asset(asset_name: str) -> bool:
    return validate_asset_name(asset_name) in PUBLIC_BROWSER_ASSET_NAMES
