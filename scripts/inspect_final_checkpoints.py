from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from pfi_ai_service.model_artifacts import model_artifact_path
from pfi_ai_service.model_manifest import manifest_path_for_artifact, read_model_manifest


ARTIFACTS = {
    "sagittal_spider": "sagittal_spider_multiclass_final_best.pt",
    "axial_t2_alkafri": "axial_t2_alkafri_final_best.pt",
}

HASH_FIELDS = (
    ("sha256",),
    ("artifactSha256",),
    ("artifactSHA256",),
    ("artifactHash",),
    ("modelSha256",),
    ("checkpointSha256",),
    ("checksum", "sha256"),
    ("artifact", "sha256"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_top_keys(checkpoint: Any) -> list[str]:
    if isinstance(checkpoint, Mapping):
        return sorted(str(key) for key in checkpoint.keys())
    return [f"<{type(checkpoint).__name__}>"]


def normalize_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        clean_key = str(key)
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        normalized[clean_key] = value
    return normalized


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, Any] | None, str]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return normalize_state_dict(value), key
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return normalize_state_dict(checkpoint), "<checkpoint>"
    return None, "AUSENTE"


def compact_list(values: list[str], limit: int = 24) -> list[str]:
    if len(values) <= limit:
        return values
    remaining = len(values) - limit
    return values[:limit] + [f"... ({remaining} mas)"]


def nested_get(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def expected_sha_from_manifest(manifest: Mapping[str, Any]) -> str | None:
    content = manifest.get("content") if isinstance(manifest.get("content"), Mapping) else manifest
    if not isinstance(content, Mapping):
        return None
    for field_path in HASH_FIELDS:
        value = nested_get(content, field_path)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def manifest_status(path: Path, actual_sha: str) -> tuple[str, str]:
    manifest_path = manifest_path_for_artifact(path)
    manifest = read_model_manifest(path)
    if not manifest.get("exists"):
        return "SIN_MANIFEST", "manifest lateral ausente"
    expected_sha = expected_sha_from_manifest(manifest)
    if not expected_sha:
        return "SIN_MANIFEST", "sha256 esperado AUSENTE en manifest"
    if expected_sha == actual_sha.lower():
        return "MATCH", "sha256 esperado coincide"
    return "MISMATCH", f"sha256 esperado={expected_sha}"


def stored_value(checkpoint: Any, keys: tuple[str, ...]) -> tuple[Any, str] | None:
    if not isinstance(checkpoint, Mapping):
        return None
    for key in keys:
        if checkpoint.get(key) is not None:
            return checkpoint[key], key
    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        for key in keys:
            if config.get(key) is not None:
                return config[key], f"config.{key}"
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, Mapping):
        for key in keys:
            if metadata.get(key) is not None:
                return metadata[key], f"metadata.{key}"
    return None


def infer_num_classes(model_key: str, state_dict: Mapping[str, Any]) -> tuple[int | None, str]:
    candidates = ("out_conv.weight",) if model_key == "sagittal_spider" else ("out.weight",)
    for key in candidates:
        tensor = state_dict.get(key)
        if torch.is_tensor(tensor) and tensor.ndim >= 1:
            return int(tensor.shape[0]), key
    return None, "AUSENTE"


def infer_base_channels(model_key: str, state_dict: Mapping[str, Any]) -> tuple[int | None, str]:
    candidates = ("enc1.block.0.weight",) if model_key == "sagittal_spider" else ("e1.net.0.weight",)
    for key in candidates:
        tensor = state_dict.get(key)
        if torch.is_tensor(tensor) and tensor.ndim >= 1:
            return int(tensor.shape[0]), key
    return None, "AUSENTE"


def report_field(
    checkpoint: Any,
    state_dict: Mapping[str, Any],
    model_key: str,
    name: str,
) -> str:
    if name == "num_classes":
        stored = stored_value(checkpoint, ("num_classes", "numClasses", "n_classes", "classes"))
        if stored is not None:
            return f"{stored[0]} (STORED desde {stored[1]})"
        inferred, source = infer_num_classes(model_key, state_dict)
        return f"{inferred if inferred is not None else 'AUSENTE'} (INFERIDO desde {source})"
    if name == "base_channels":
        stored = stored_value(checkpoint, ("base_channels", "baseChannels"))
        if stored is not None:
            return f"{stored[0]} (STORED desde {stored[1]})"
        inferred, source = infer_base_channels(model_key, state_dict)
        return f"{inferred if inferred is not None else 'AUSENTE'} (INFERIDO desde {source})"
    if name == "target_size":
        stored = stored_value(checkpoint, ("target_size", "targetSize", "input_size", "inputSize"))
        if stored is not None:
            return f"{stored[0]} (STORED desde {stored[1]})"
        return "AUSENTE (INFERIDO no disponible sin cargar arquitectura ni inferencia)"
    raise KeyError(name)


def display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.name


def inspect_artifact(model_key: str, expected_file: str) -> dict[str, Any]:
    path = model_artifact_path(model_key)
    if path is None:
        return {
            "model_key": model_key,
            "name": expected_file,
            "path": expected_file,
            "exists": False,
            "blocker": "path no resuelto desde settings",
        }

    path = Path(path)
    result: dict[str, Any] = {
        "model_key": model_key,
        "name": expected_file,
        "path": display_path(path),
        "source": "PFI_MODEL_DIR/settings",
        "exists": path.exists() and path.is_file(),
    }
    if not result["exists"]:
        result["size_bytes"] = 0
        result["sha256"] = "AUSENTE"
        result["manifest_status"] = "SIN_MANIFEST"
        result["manifest_detail"] = "artifact no materializado localmente"
        result["blocker"] = "artifact no materializado localmente"
        return result

    actual_sha = sha256_file(path)
    status, detail = manifest_status(path, actual_sha)
    checkpoint = load_checkpoint(path)
    state_dict, state_source = extract_state_dict(checkpoint)
    state_keys = sorted(state_dict.keys()) if state_dict else []

    result.update(
        {
            "size_bytes": path.stat().st_size,
            "sha256": actual_sha,
            "manifest_status": status,
            "manifest_detail": detail,
            "top_level_keys": checkpoint_top_keys(checkpoint),
            "state_dict_source": state_source,
            "state_dict_keys": compact_list(state_keys),
            "state_dict_key_count": len(state_keys),
        }
    )
    if state_dict is None:
        result["num_classes"] = "AUSENTE (state_dict AUSENTE)"
        result["base_channels"] = "AUSENTE (state_dict AUSENTE)"
        result["target_size"] = "AUSENTE (state_dict AUSENTE)"
    else:
        result["num_classes"] = report_field(checkpoint, state_dict, model_key, "num_classes")
        result["base_channels"] = report_field(checkpoint, state_dict, model_key, "base_channels")
        result["target_size"] = report_field(checkpoint, state_dict, model_key, "target_size")
    return result


def print_result(result: Mapping[str, Any]) -> None:
    print(f"## {result['model_key']}")
    print(f"- Nombre / path relativo: {result['name']} / {result['path']}")
    print(f"- Fuente de path: {result.get('source', 'AUSENTE')}")
    print(f"- Existe: {'si' if result['exists'] else 'no'} | Tamaño (bytes): {result.get('size_bytes', 0)}")
    print(f"- SHA-256: {result.get('sha256', 'AUSENTE')} | vs manifest: {result.get('manifest_status', 'SIN_MANIFEST')}")
    print(f"- Detalle manifest: {result.get('manifest_detail', 'AUSENTE')}")
    if not result["exists"]:
        print(f"- Bloqueo: {result.get('blocker', 'artifact ausente')}")
        print()
        return
    print(f"- Keys top-level del checkpoint: {json.dumps(result.get('top_level_keys', []), ensure_ascii=True)}")
    print(f"- State_dict source: {result.get('state_dict_source', 'AUSENTE')}")
    print(
        "- Keys del state_dict (muestra): "
        f"{json.dumps(result.get('state_dict_keys', []), ensure_ascii=True)}"
        f" | total={result.get('state_dict_key_count', 0)}"
    )
    print(f"- num_classes: {result.get('num_classes', 'AUSENTE')}")
    print(f"- base_channels: {result.get('base_channels', 'AUSENTE')}")
    print(f"- target_size: {result.get('target_size', 'AUSENTE')}")
    print()


def main() -> int:
    print("# AI-001 checkpoint inspection")
    print()
    results = [inspect_artifact(model_key, file_name) for model_key, file_name in ARTIFACTS.items()]
    for result in results:
        print_result(result)

    blockers = [f"{result['name']}: {result.get('blocker')}" for result in results if result.get("blocker")]
    print("## Bloqueos")
    if blockers:
        for blocker in blockers:
            print(f"- {blocker}")
        return 2
    print("- ninguno")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
