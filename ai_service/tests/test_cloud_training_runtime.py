from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pfi_ai_service.training.cloud_runtime import (
    PortablePathError,
    atomic_torch_save,
    atomic_write_dataframe_csv,
    atomic_write_json,
    atomic_write_text,
    build_sync_command,
    env_bool,
    load_bash_env_contract,
    resolve_portable_axial_path,
    run_sync_command,
    validate_run_id,
    wait_for_minimum_file_age,
)


def test_validate_run_id():
    assert validate_run_id("pfi-final-training-v1") == "pfi-final-training-v1"
    for bad in ["../otro", "a/b", "ABC", "", "a" * 64]:
        with pytest.raises(ValueError):
            validate_run_id(bad)


def test_env_bool(monkeypatch):
    monkeypatch.delenv("X_BOOL", raising=False)
    assert env_bool("X_BOOL", True) is True
    monkeypatch.setenv("X_BOOL", "0")
    assert env_bool("X_BOOL", True) is False
    monkeypatch.setenv("X_BOOL", "1")
    assert env_bool("X_BOOL", False) is True
    monkeypatch.setenv("X_BOOL", "true")
    with pytest.raises(ValueError):
        env_bool("X_BOOL", False)


def test_resolve_portable_axial_path_variants(tmp_path):
    root = tmp_path / "AXIAL_ALKAFRI"
    sample = root / "case" / "img.ima"
    sample.parent.mkdir(parents=True)
    sample.write_text("x")
    rel, rebased = resolve_portable_axial_path("case/img.ima", primary_root=root, additional_roots=[])
    assert rel == sample and rebased is False
    abs_path, rebased = resolve_portable_axial_path(sample, primary_root=root, additional_roots=[])
    assert abs_path == sample and rebased is False
    for raw in [
        f"/content/drive/MyDrive/PFI/data/AXIAL_ALKAFRI/case/img.ima",
        f"/content/drive/PFI/data/AXIAL_ALKAFRI/case/img.ima",
        "C:\\legacy\\AXIAL_ALKAFRI\\case\\img.ima",
        "C:/legacy/axial_alkafri/case/img.ima",
    ]:
        out, rebased = resolve_portable_axial_path(raw, primary_root=root, additional_roots=[])
        assert out == sample
        assert rebased is True
    missing, rebased = resolve_portable_axial_path("C:/legacy/AXIAL_ALKAFRI/case/missing.ima", primary_root=root, additional_roots=[])
    assert missing == root / "case" / "missing.ima"
    assert rebased is True
    with pytest.raises(PortablePathError):
        resolve_portable_axial_path("", primary_root=root, additional_roots=[])
    with pytest.raises(PortablePathError):
        resolve_portable_axial_path("C:/legacy/AXIAL_ALKAFRI/../evil.ima", primary_root=root, additional_roots=[])


def test_atomic_write_text_json_csv(tmp_path):
    text_path = tmp_path / "a.txt"
    atomic_write_text("uno", text_path)
    atomic_write_text("dos", text_path)
    assert text_path.read_text(encoding="utf-8") == "dos"
    json_path = tmp_path / "a.json"
    atomic_write_json({"ñ": 1}, json_path)
    assert '"ñ"' in json_path.read_text(encoding="utf-8")
    csv_path = tmp_path / "a.csv"
    atomic_write_dataframe_csv(pd.DataFrame([{"a": 1}]), csv_path)
    assert "a" in csv_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch no disponible")
def test_atomic_torch_save(tmp_path):
    import torch

    dest = tmp_path / "x.pt"
    atomic_torch_save({"x": torch.tensor([1])}, dest)
    loaded = torch.load(dest, map_location="cpu", weights_only=False)
    assert loaded["x"].item() == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_rejects_symlink_when_supported(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("x")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink no soportado")
    with pytest.raises(RuntimeError):
        atomic_write_text("z", link)


def test_wait_for_minimum_file_age(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    wait_for_minimum_file_age([f], 0)
    with pytest.raises(FileNotFoundError):
        wait_for_minimum_file_age([tmp_path / "missing"], 0)
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(f)
    except OSError:
        return
    with pytest.raises(RuntimeError):
        wait_for_minimum_file_age([link], 0)
    os.utime(f, None)
    with pytest.raises(TimeoutError):
        wait_for_minimum_file_age([f], 2, timeout_seconds=0)


def _repo_with_sync(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    script = repo / "infra" / "gcp" / "sync-training-artifacts.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\necho sync $@\nexit ${SYNC_RC:-0}\n", encoding="utf-8")
    env = repo / "infra" / "gcp" / "training-vm.env"
    env.write_text("PFI_RUN_ID=pfi-final-training-v1\nPFI_LOCAL_RESUME_DIR=/tmp/resume\n", encoding="utf-8")
    return repo, script, env


def test_build_sync_command(tmp_path):
    repo, _, env = _repo_with_sync(tmp_path)
    cmd = build_sync_command(repo_root=repo, env_file=env, mode="push-resume", execute=False)
    assert isinstance(cmd, list)
    assert "--dry-run" in cmd
    cmd2 = build_sync_command(repo_root=repo, env_file=env, mode="push-final", execute=True)
    assert "--execute" in cmd2
    with pytest.raises(ValueError):
        build_sync_command(repo_root=repo, env_file=env, mode="pull-resume", execute=False)


def test_run_sync_command(tmp_path, monkeypatch):
    repo, _, env = _repo_with_sync(tmp_path)
    ok = run_sync_command(repo_root=repo, env_file=env, mode="push-resume", execute=False, failure_is_fatal=True)
    assert ok.returncode == 0
    bad = run_sync_command(repo_root=repo, env_file=env, mode="push-resume", execute=False, failure_is_fatal=False, extra_env={"SYNC_RC": "1"})
    assert bad.returncode == 1
    with pytest.raises(RuntimeError):
        run_sync_command(repo_root=repo, env_file=env, mode="push-resume", execute=False, failure_is_fatal=True, extra_env={"SYNC_RC": "1"})


def test_load_bash_env_contract(tmp_path, monkeypatch):
    env = tmp_path / "env"
    env.write_text('A=1\nB="${A}2"\nSECRET=hidden\n', encoding="utf-8")
    stub = tmp_path / "bin"
    stub.mkdir()
    py = stub / ("python3" if os.name != "nt" else "python3")
    py.write_text(f"#!/usr/bin/env bash\nexec {sys.executable!r} \"$@\"\n", encoding="utf-8")
    try:
        py.chmod(0o755)
    except OSError:
        pass
    monkeypatch.setenv("PATH", str(stub) + os.pathsep + os.environ.get("PATH", ""))
    data = load_bash_env_contract(env, ["A", "B"])
    assert data == {"A": "1", "B": "12"}
    with pytest.raises(FileNotFoundError):
        load_bash_env_contract(tmp_path / "missing", ["A"])
    with pytest.raises(ValueError):
        load_bash_env_contract(env, ["BAD;echo"])