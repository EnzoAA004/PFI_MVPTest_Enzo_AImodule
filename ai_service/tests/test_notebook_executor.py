from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pfi_ai_service.training.notebook_executor import main, redact_text

HAS_NB_DEPS = all(importlib.util.find_spec(m) for m in ["nbclient", "nbformat", "ipykernel", "jupyter_client"])


def _write_notebook(path: Path, sources: list[str]) -> None:
    cells = [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
        for src in sources
    ]
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")


@pytest.mark.skipif(not HAS_NB_DEPS, reason="nbclient/nbformat/ipykernel no instalados en este venv")
def test_successful_notebook_streams_stdout_and_saves_outputs(tmp_path: Path) -> None:
    src = tmp_path / "ok.ipynb"
    out = tmp_path / "ok.executed.ipynb"
    log = tmp_path / "ok.log"
    _write_notebook(src, ["print('hola runner')"])

    rc = main(["--input", str(src), "--output", str(out), "--log", str(log), "--kernel-python", sys.executable, "--timeout", "30"])

    assert rc == 0
    assert "hola runner" in log.read_text(encoding="utf-8")
    executed = json.loads(out.read_text(encoding="utf-8"))
    assert executed["cells"][0]["outputs"]
    assert json.loads(src.read_text(encoding="utf-8"))["cells"][0]["outputs"] == []


@pytest.mark.skipif(not HAS_NB_DEPS, reason="nbclient/nbformat/ipykernel no instalados en este venv")
def test_failed_notebook_returns_one_and_keeps_partial_output(tmp_path: Path) -> None:
    src = tmp_path / "fail.ipynb"
    out = tmp_path / "fail.executed.ipynb"
    log = tmp_path / "fail.log"
    _write_notebook(src, ["print('antes')", "raise RuntimeError('boom')"])

    rc = main(["--input", str(src), "--output", str(out), "--log", str(log), "--kernel-python", sys.executable, "--timeout", "30"])

    assert rc == 1
    assert out.exists()
    text = log.read_text(encoding="utf-8")
    assert "antes" in text
    assert "RuntimeError" in text


@pytest.mark.skipif(not HAS_NB_DEPS, reason="nbclient/nbformat/ipykernel no instalados en este venv")
def test_execute_result_and_display_data_text_plain(tmp_path: Path) -> None:
    src = tmp_path / "display.ipynb"
    out = tmp_path / "display.executed.ipynb"
    log = tmp_path / "display.log"
    _write_notebook(src, ["1 + 1", "from IPython.display import display; display({'text/plain':'plain display'}, raw=True)"])

    rc = main(["--input", str(src), "--output", str(out), "--log", str(log), "--kernel-python", sys.executable, "--timeout", "30"])

    assert rc == 0
    text = log.read_text(encoding="utf-8")
    assert "2" in text
    assert "plain display" in text


def test_missing_kernel_python_returns_configuration_error(tmp_path: Path) -> None:
    src = tmp_path / "ok.ipynb"
    _write_notebook(src, ["print('x')"])

    rc = main(["--input", str(src), "--output", str(tmp_path / "out.ipynb"), "--log", str(tmp_path / "log.txt"), "--kernel-python", str(tmp_path / "missing-python"), "--timeout", "0"])

    assert rc == 2


def test_timeout_argument_invalid_exits_two(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pfi_ai_service.training.notebook_executor", "--timeout", "-1"],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 2


def test_redaction_masks_secret_like_text() -> None:
    assert "abc" not in redact_text("access" + "_token=abc")
    assert "top" not in redact_text("client" + "_secret: top")
    assert "[REDACTED]" in redact_text("BEGIN " + "PRIVATE " + "KEY")
