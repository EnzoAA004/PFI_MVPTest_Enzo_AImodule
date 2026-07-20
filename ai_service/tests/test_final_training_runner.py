from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "infra" / "gcp" / "run-final-training.sh"


def _bash() -> str:
    candidates = [os.environ.get("PFI_TEST_BASH"), r"C:\Program Files\Git\bin\bash.exe", "bash"]
    for candidate in candidates:
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return candidate
    pytest.skip("bash no disponible")


def _to_bash_path(path: Path) -> str:
    bash = _bash()
    proc = subprocess.run([bash, "-lc", f"cygpath -u {str(path)!r}"], text=True, capture_output=True)
    if proc.returncode == 0:
        return proc.stdout.strip()
    return path.as_posix()


def _make_stub_dir(tmp_path: Path) -> tuple[Path, Path]:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    calls = tmp_path / "calls.log"
    for name in ["preflight-training-vm.sh", "download-training-data.sh", "sync-training-artifacts.sh"]:
        script = stub_dir / name
        script.write_text(
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
            f"printf '%s %s\\n' {name!r} \"$*\" >> {str(calls)!r}\n"
            "case \"${PFI_STUB_FAIL:-}\" in\n"
            f"  {name}) exit 1;;\n"
            "esac\n"
            "exit 0\n",
            encoding="utf-8",
            newline="\n",
        )
    return stub_dir, calls


def _copy_training_package(repo_root: Path) -> None:
    target = repo_root / "ai_service" / "pfi_ai_service" / "training"
    target.mkdir(parents=True)
    (repo_root / "ai_service" / "pfi_ai_service" / "__init__.py").write_text("", encoding="utf-8")
    for name in ["cloud_runtime.py", "notebook_executor.py", "__init__.py"]:
        shutil.copy(REPO / "ai_service" / "pfi_ai_service" / "training" / name, target / name)


def _env_file(tmp_path: Path, *, overrides: dict[str, str] | None = None) -> tuple[Path, Path, Path]:
    vm_root = tmp_path / "vm"
    repo_root = vm_root / "repo"
    output_root = vm_root / "outputs" / "final_training"
    for d in [repo_root, output_root / "models", output_root / "resume", output_root / "manifests", output_root / "logs", vm_root / "venv"]:
        d.mkdir(parents=True, exist_ok=True)
    _copy_training_package(repo_root)
    env_path = repo_root / "infra" / "gcp" / "training-vm.env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "PFI_CLOUD_MODE": "1",
        "PFI_PREFLIGHT_ONLY": "1",
        "PFI_SYNC_DRY_RUN": "1",
        "PFI_SYNC_RESUME": "1",
        "PFI_SYNC_FINAL_ARTIFACTS": "1",
        "PFI_SYNC_FAILURE_IS_FATAL": "1",
        "PFI_DOWNLOAD_DATASETS": "1",
        "PFI_DOWNLOAD_RESUME": "1",
        "PFI_RUN_ID": "test-run",
        "PFI_GCP_PROJECT_ID": "pfi-test",
        "PFI_GCP_ZONE": "us-central1-a",
        "PFI_VM_NAME": "pfi-vm",
        "PFI_VM_SERVICE_ACCOUNT": "pfi-vm@pfi-test.iam.gserviceaccount.com",
        "PFI_VM_ROOT": _to_bash_path(vm_root),
        "PFI_REPO_ROOT": _to_bash_path(repo_root),
        "PFI_VENV_DIR": _to_bash_path(vm_root / "venv"),
        "PFI_TRAIN_OUTPUT_DIR": _to_bash_path(output_root),
        "PFI_LOCAL_MODELS_DIR": _to_bash_path(output_root / "models"),
        "PFI_LOCAL_RESUME_DIR": _to_bash_path(output_root / "resume"),
        "PFI_LOCAL_MANIFESTS_DIR": _to_bash_path(output_root / "manifests"),
        "PFI_LOCAL_LOGS_DIR": _to_bash_path(output_root / "logs"),
        "PFI_TRAINING_ENV_FILE": _to_bash_path(env_path),
        "PFI_GCS_BUCKET_URI": "gs://pfi-test-bucket",
        "PFI_GCS_RUN_MODELS_URI": "gs://pfi-test-bucket/models/test-run",
        "PFI_GCS_RUN_RESUME_URI": "gs://pfi-test-bucket/resume/test-run",
        "PFI_GCS_RUN_MANIFESTS_URI": "gs://pfi-test-bucket/manifests/test-run",
        "PFI_GCS_RUN_OUTPUTS_URI": "gs://pfi-test-bucket/outputs/test-run",
        "PFI_SYNC_MIN_FILE_AGE_SECONDS": "0",
        "RUN_SAGITTAL": "1",
        "RUN_AXIAL": "1",
        "REQUIRE_GPU": "1",
    }
    if overrides:
        values.update(overrides)
    env_path.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8", newline="\n")
    return env_path, repo_root, output_root


def _run_runner(tmp_path: Path, *args: str, overrides: dict[str, str] | None = None, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env_path, repo_root, output_root = _env_file(tmp_path, overrides=overrides)
    stub_dir, calls = _make_stub_dir(tmp_path)
    env = os.environ.copy()
    env["PFI_RUNNER_STUB_DIR"] = _to_bash_path(stub_dir)
    env["PFI_RUNNER_PYTHON_BIN"] = _to_bash_path(Path(sys.executable))
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run([_bash(), str(RUNNER), "--env-file", _to_bash_path(env_path), *args], text=True, capture_output=True, env=env)
    proc.calls_file = calls  # type: ignore[attr-defined]
    proc.output_root = output_root  # type: ignore[attr-defined]
    return proc


def test_help() -> None:
    proc = subprocess.run([_bash(), str(RUNNER), "--help"], text=True, capture_output=True)
    assert proc.returncode == 0
    assert "--execute" in proc.stdout


def test_unknown_argument_returns_two() -> None:
    proc = subprocess.run([_bash(), str(RUNNER), "--wat"], text=True, capture_output=True)
    assert proc.returncode == 2


def test_default_dry_run_uses_child_dry_run_and_does_not_execute_notebook(tmp_path: Path) -> None:
    proc = _run_runner(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = proc.calls_file.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    assert "preflight-training-vm.sh --mode static" in calls
    assert "download-training-data.sh --component datasets --dry-run" in calls
    assert "sync-training-artifacts.sh --mode pull-resume --dry-run" in calls
    assert "--input" not in calls
    assert not list(proc.output_root.rglob("*.pt"))  # type: ignore[attr-defined]


def test_execute_requires_cloud_mode(tmp_path: Path) -> None:
    proc = _run_runner(tmp_path, "--execute", overrides={"PFI_CLOUD_MODE": "0", "PFI_PREFLIGHT_ONLY": "0", "PFI_SYNC_DRY_RUN": "0"})
    assert proc.returncode == 2
    assert "guards --execute" in proc.stderr


def test_execute_requires_preflight_only_zero(tmp_path: Path) -> None:
    proc = _run_runner(tmp_path, "--execute", overrides={"PFI_SYNC_DRY_RUN": "0"})
    assert proc.returncode == 2
    assert "guards --execute" in proc.stderr


def test_execute_requires_sync_dry_run_zero(tmp_path: Path) -> None:
    proc = _run_runner(tmp_path, "--execute", overrides={"PFI_PREFLIGHT_ONLY": "0"})
    assert proc.returncode == 2
    assert "guards --execute" in proc.stderr


def test_require_resume_rejected_outside_execute(tmp_path: Path) -> None:
    proc = _run_runner(tmp_path, "--require-resume")
    assert proc.returncode == 2


def test_status_json_written_on_dry_run(tmp_path: Path) -> None:
    proc = _run_runner(tmp_path)
    assert proc.returncode == 0
    status = json.loads((proc.output_root / "logs" / "runner_status_test-run.json").read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    assert status["status"] == "success"
    assert status["run_id"] == "test-run"


def test_child_failure_prevents_later_steps(tmp_path: Path) -> None:
    proc = _run_runner(tmp_path, extra_env={"PFI_STUB_FAIL": "download-training-data.sh"})
    assert proc.returncode == 1
    calls = proc.calls_file.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    assert "download-training-data.sh" in calls
    assert "sync-training-artifacts.sh --mode pull-resume" not in calls


def test_commands_do_not_contain_forbidden_operations() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for bad in ["gcloud storage rm", "gcloud storage mv", "gsutil", "delete-unmatched-destination-objects", "auth print-" + "access-token", "git reset", "git clean", "git pull", "sudo", "pip install", "rm -rf"]:
        assert bad not in text
