"""Cloud-portable helpers for the final training notebook.

This module contains infrastructure helpers only. It intentionally avoids model
architecture, loss, optimizer, dataset, or metric logic so the notebook can keep
scientific parity with the validated v4 flow.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_SYNC_MODES = {"push-resume", "push-final"}


def _bash_executable() -> str:
    if os.name == "nt":
        candidates = [
            os.getenv("PFI_BASH_BIN", ""),
            r"C:\\Program Files\\Git\\bin\\bash.exe",
            r"C:\\Program Files\\Git\\usr\\bin\\bash.exe",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
    return "bash"



class PortablePathError(ValueError):
    """Raised when a dataset path cannot be interpreted portably."""


def detect_runtime() -> str:
    """Return colab, gce, or local without consulting cloud metadata."""
    if os.getenv("PFI_CLOUD_MODE") == "1":
        return "gce"
    try:
        __import__("google.colab")
    except Exception:
        return "local"
    return "colab"


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"{name} debe ser 0 o 1; obtenido {value!r}")


def validate_run_id(value: str) -> str:
    if not RUN_ID_RE.fullmatch(value or ""):
        raise ValueError("PFI_RUN_ID invalido; usar ^[a-z0-9][a-z0-9-]{0,62}$")
    return value


def _is_nan(value: object) -> bool:
    try:
        return bool(math.isnan(value))  # type: ignore[arg-type]
    except Exception:
        return False


def _candidate_under(root: Path, suffix: str) -> Path:
    clean = suffix.replace("\\", "/").lstrip("/")
    if not clean or clean == ".":
        raise PortablePathError("suffix axial vacio")
    if any(part in {"", ".", ".."} for part in clean.split("/")):
        raise PortablePathError(f"suffix axial inseguro: {suffix!r}")
    candidate = root / Path(*clean.split("/"))
    root_abs = root.absolute()
    cand_abs = candidate.absolute()
    try:
        cand_abs.relative_to(root_abs)
    except ValueError as exc:
        raise PortablePathError(f"path axial escapa primary_root: {candidate}") from exc
    return candidate


def resolve_portable_axial_path(
    raw_value: object,
    *,
    primary_root: Path,
    additional_roots: Sequence[Path],
    anchor: str = "AXIAL_ALKAFRI",
) -> tuple[Path, bool]:
    """Resolve E9 axial CSV paths without mutating the original CSV."""
    if raw_value is None or _is_nan(raw_value):
        raise PortablePathError("path axial nulo")
    text = str(raw_value).strip()
    if not text:
        raise PortablePathError("path axial vacio")
    normalized = text.replace("\\", "/")
    direct = Path(text)
    if direct.is_absolute() and direct.exists():
        return direct, False

    roots = [primary_root, *additional_roots]
    if not direct.is_absolute():
        for root in roots:
            candidate = root / direct
            if candidate.exists():
                return candidate, False

    parts = normalized.split("/")
    lower_anchor = anchor.lower()
    anchor_index = next((i for i, part in enumerate(parts) if part.lower() == lower_anchor), None)
    if anchor_index is not None:
        suffix_parts = parts[anchor_index + 1 :]
        suffix = "/".join(suffix_parts)
        return _candidate_under(primary_root, suffix), True

    if not direct.is_absolute():
        return primary_root / direct, False
    raise PortablePathError(f"no se pudo interpretar path axial: {raw_value!r}")


def _reject_symlink(destination: Path) -> None:
    if destination.exists() and destination.is_symlink():
        raise RuntimeError(f"destination es symlink: {destination}")


def _tmp_for(destination: Path) -> Path:
    if not destination.parent.exists():
        raise FileNotFoundError(f"directorio padre inexistente: {destination.parent}")
    _reject_symlink(destination)
    return destination.parent / f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"


def _fsync_file(handle) -> None:
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


def _replace_atomic(tmp: Path, destination: Path) -> None:
    os.replace(tmp, destination)


def atomic_write_text(text: str, destination: Path) -> None:
    tmp = _tmp_for(destination)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            _fsync_file(fh)
        _replace_atomic(tmp, destination)
    except Exception:
        if tmp.exists() and not tmp.is_symlink():
            tmp.unlink()
        raise


def atomic_write_json(data, destination: Path) -> None:
    atomic_write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", destination)


def atomic_write_dataframe_csv(dataframe, destination: Path, *, index: bool = False) -> None:
    tmp = _tmp_for(destination)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            dataframe.to_csv(fh, index=index)
            _fsync_file(fh)
        _replace_atomic(tmp, destination)
    except Exception:
        if tmp.exists() and not tmp.is_symlink():
            tmp.unlink()
        raise


def atomic_torch_save(payload, destination: Path) -> None:
    tmp = _tmp_for(destination)
    try:
        import torch

        torch.save(payload, tmp)
        with open(tmp, "ab") as fh:
            _fsync_file(fh)
        _replace_atomic(tmp, destination)
    except Exception:
        if tmp.exists() and not tmp.is_symlink():
            tmp.unlink()
        raise


def wait_for_minimum_file_age(
    paths: Sequence[Path],
    minimum_age_seconds: int,
    *,
    timeout_seconds: int | None = None,
) -> None:
    if minimum_age_seconds < 0:
        raise ValueError("minimum_age_seconds debe ser >= 0")
    deadline = time.time() + (timeout_seconds if timeout_seconds is not None else max(30, minimum_age_seconds + 30))
    unique = [Path(p) for p in paths]
    while True:
        now = time.time()
        pending: list[Path] = []
        for path in unique:
            if path.is_symlink():
                raise RuntimeError(f"archivo symlink no permitido: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"archivo requerido inexistente: {path}")
            if now - path.stat().st_mtime < minimum_age_seconds:
                pending.append(path)
        if not pending:
            return
        if now >= deadline:
            raise TimeoutError(f"archivos no alcanzaron edad minima: {[p.name for p in pending]}")
        time.sleep(min(0.25, max(0.01, minimum_age_seconds / 10 if minimum_age_seconds else 0.01)))


def build_sync_command(*, repo_root: Path, env_file: Path, mode: str, execute: bool) -> list[str]:
    if mode not in _SYNC_MODES:
        raise ValueError(f"modo sync invalido: {mode}")
    script = repo_root / "infra" / "gcp" / "sync-training-artifacts.sh"
    if not script.exists():
        raise FileNotFoundError(f"sync script inexistente: {script}")
    if not env_file.exists():
        raise FileNotFoundError(f"env file inexistente: {env_file}")
    return ["bash", str(script), "--mode", mode, "--execute" if execute else "--dry-run", "--env-file", str(env_file)]


def _redact_output(text: str) -> str:
    forbidden = ("access_token", "PRIVATE" + " KEY", "BEGIN" + " RSA", "BEGIN" + " OPENSSH")
    lines = []
    for line in text.splitlines():
        if any(item.lower() in line.lower() for item in forbidden):
            lines.append("[REDACTED]")
        else:
            lines.append(line)
    return "\n".join(lines)


def run_sync_command(
    *,
    repo_root: Path,
    env_file: Path,
    mode: str,
    execute: bool,
    failure_is_fatal: bool,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    cmd = build_sync_command(repo_root=repo_root, env_file=env_file, mode=mode, execute=execute)
    env = os.environ.copy()
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    run_cmd = [_bash_executable(), *cmd[1:]] if cmd and cmd[0] == "bash" else cmd
    result = subprocess.run(run_cmd, check=False, text=True, capture_output=True, env=env)
    print(f"sync command mode={mode} execute={execute} returncode={result.returncode}")
    if result.stdout:
        print(_redact_output(result.stdout[-4000:]))
    if result.stderr:
        print(_redact_output(result.stderr[-4000:]), file=sys.stderr)
    if result.returncode != 0 and failure_is_fatal:
        raise RuntimeError(f"sync {mode} fallo con returncode={result.returncode}")
    return result


def load_bash_env_contract(env_file: Path, keys: Sequence[str]) -> dict[str, str]:
    if not env_file.exists():
        raise FileNotFoundError(f"env file inexistente: {env_file}")
    if any(not re.fullmatch(r"[A-Z0-9_]+", key) for key in keys):
        raise ValueError("keys de env invalidas")
    script = (
        "set -Eeuo pipefail; "
        "set -a; source \"$1\"; set +a; "
        "python3 -c 'import json, os, sys; keys=sys.argv[1:]; print(json.dumps({k: os.environ.get(k, \"\") for k in keys}, sort_keys=True))' "
        + " ".join(f"'{key}'" for key in keys)
    )
    result = subprocess.run([_bash_executable(), "-c", script, "bash", str(env_file)], check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"no se pudo cargar env contract: {result.stderr[-1000:]}")
    return json.loads(result.stdout)