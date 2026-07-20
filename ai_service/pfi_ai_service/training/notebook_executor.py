"""Notebook execution helper for the final training VM runner."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, TextIO

SECRET_PATTERNS = (
    re.compile("access" + r"_token\s*[:=]\s*[^\s]+", re.IGNORECASE),
    re.compile("private" + r"_key\s*[:=]\s*[^\n]+", re.IGNORECASE),
    re.compile("client" + r"_secret\s*[:=]\s*[^\s]+", re.IGNORECASE),
    re.compile(r"authorization:\s*[^\n]+", re.IGNORECASE),
    re.compile(r"BEGIN\s+PRIVATE\s+KEY", re.IGNORECASE),
    re.compile(r"BEGIN\s+RSA", re.IGNORECASE),
    re.compile(r"BEGIN\s+OPENSSH", re.IGNORECASE),
)


class NotebookExecutorError(RuntimeError):
    """Operational notebook executor error."""


def redact_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _write_line(log_handle: TextIO, text: str) -> None:
    safe = redact_text(text)
    print(safe, end="")
    sys.stdout.flush()
    log_handle.write(safe)
    log_handle.flush()


def _atomic_write_notebook(nbformat_module, notebook, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            nbformat_module.write(notebook, fh)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, destination)
    except Exception:
        if tmp.exists() and not tmp.is_symlink():
            tmp.unlink()
        raise


def _check_kernel_python(kernel_python: Path) -> None:
    if not kernel_python.is_file():
        raise NotebookExecutorError(f"kernel-python inexistente: {kernel_python}")
    result = subprocess.run(
        [str(kernel_python), "-c", "import ipykernel, sys; print(sys.executable)"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise NotebookExecutorError("kernel-python no puede importar ipykernel")
    resolved = Path(result.stdout.strip()).resolve()
    if resolved != kernel_python.resolve():
        raise NotebookExecutorError(f"kernel-python inesperado: {resolved}")


def _make_kernel_dir(kernel_python: Path) -> Path:
    kernel_dir = Path(tempfile.mkdtemp(prefix="pfi-nb-kernel-"))
    kernel_json = kernel_dir / "kernel.json"
    kernel_json.write_text(
        "{\n"
        '  "argv": ['
        f'"{str(kernel_python)}", "-m", "ipykernel_launcher", "-f", "{{connection_file}}"],\n'
        '  "display_name": "PFI temporary training kernel",\n'
        '  "language": "python"\n'
        "}\n",
        encoding="utf-8",
    )
    return kernel_dir


def _validate_timeout(value: str) -> int:
    try:
        timeout = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--timeout debe ser entero") from exc
    if timeout < 0:
        raise argparse.ArgumentTypeError("--timeout debe ser >= 0")
    return timeout


def _load_dependencies():
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError, CellTimeoutError
    except ModuleNotFoundError as exc:
        raise NotebookExecutorError(f"dependencia faltante: {exc.name}") from exc
    return nbformat, NotebookClient, CellExecutionError, CellTimeoutError


def _stream_payload(msg: dict) -> str | None:
    msg_type = msg.get("msg_type")
    content = msg.get("content", {})
    if msg_type == "stream":
        return str(content.get("text", ""))
    if msg_type == "error":
        traceback = content.get("traceback") or []
        if traceback:
            ansi = re.compile(r"\x1b\[[0-9;]*m")
            return "\n".join(ansi.sub("", str(line)) for line in traceback) + "\n"
        return f"{content.get('ename', 'Error')}: {content.get('evalue', '')}\n"
    if msg_type in {"execute_result", "display_data"}:
        data = content.get("data", {})
        if "text/plain" in data:
            payload = data["text/plain"]
            if isinstance(payload, list):
                payload = "".join(str(item) for item in payload)
            return str(payload).rstrip("\n") + "\n"
    return None


def execute_notebook(input_path: Path, output_path: Path, log_path: Path, kernel_python: Path, timeout: int) -> int:
    nbformat, NotebookClient, CellExecutionError, CellTimeoutError = _load_dependencies()
    _check_kernel_python(kernel_python)
    if not input_path.is_file():
        raise NotebookExecutorError(f"notebook input inexistente: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise NotebookExecutorError("output no puede ser igual al input")

    notebook = nbformat.read(str(input_path), as_version=4)
    nbformat.validate(notebook)
    kernel_dir = _make_kernel_dir(kernel_python)
    kernel_name = kernel_dir.name
    kernel_path_parent = str(kernel_dir.parent)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    class StreamingNotebookClient(NotebookClient):  # type: ignore[misc, valid-type]
        def process_message(self, msg, cell, cell_index):  # noqa: ANN001
            payload = _stream_payload(msg)
            if payload:
                _write_line(log_handle, payload)
            return super().process_message(msg, cell, cell_index)

    old_kernel_path = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = kernel_path_parent if not old_kernel_path else kernel_path_parent + os.pathsep + old_kernel_path
    code = 0
    try:
        with open(log_path, "a", encoding="utf-8", newline="") as log_handle:
            _write_line(log_handle, f"[INFO] executing notebook: {input_path.name}\n")
            client = StreamingNotebookClient(
                notebook,
                timeout=None if timeout == 0 else timeout,
                kernel_name=kernel_name,
                allow_errors=False,
                force_raise_errors=True,
                store_widget_state=False,
            )
            try:
                client.execute()
            except KeyboardInterrupt:
                code = 130
                _write_line(log_handle, "[ERROR] notebook interrupted by KeyboardInterrupt\n")
            except CellTimeoutError as exc:
                code = 1
                _write_line(log_handle, f"[ERROR] notebook timeout: {exc}\n")
            except CellExecutionError as exc:
                code = 1
                _write_line(log_handle, f"[ERROR] notebook cell failed: {exc}\n")
            finally:
                _atomic_write_notebook(nbformat, notebook, output_path)
                _write_line(log_handle, f"[INFO] executed notebook saved: {output_path.name}\n")
    finally:
        if old_kernel_path is None:
            os.environ.pop("JUPYTER_PATH", None)
        else:
            os.environ["JUPYTER_PATH"] = old_kernel_path
        shutil.rmtree(kernel_dir, ignore_errors=True)
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute a notebook with a temporary kernelspec.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--kernel-python", required=True, type=Path)
    parser.add_argument("--timeout", default=0, type=_validate_timeout)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        return execute_notebook(args.input, args.output, args.log, args.kernel_python, args.timeout)
    except NotebookExecutorError as exc:
        print(f"[ERROR] {redact_text(str(exc))}", file=sys.stderr)
        return 2
    except SystemExit:
        raise
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # defensive CLI boundary
        print(f"[ERROR] {redact_text(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
