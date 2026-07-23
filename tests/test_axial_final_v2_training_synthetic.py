from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_49 = ROOT / "notebooks" / "49_axial_final_v2_train_validation.ipynb"
NOTEBOOK_50 = ROOT / "notebooks" / "50_axial_final_v2_test_once.ipynb"


def _write_case(root: Path, name: str, split: str, patient: str, labels: list[int], relative: bool) -> dict[str, str]:
    image = (np.arange(16 * 16, dtype=np.float32).reshape(16, 16) + sum(ord(ch) for ch in name)) / 255.0
    mask = np.full((16, 16), 250, dtype=np.int16)
    for index, label in enumerate(labels):
        row = (index * 3) % 14
        col = (index * 2) % 14
        mask[row : row + 2, col : col + 2] = label
    image_path = root / "images" / f"{name}.npy"
    mask_path = root / "masks" / f"{name}_mask.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_path, image)
    from PIL import Image

    Image.fromarray(mask.astype(np.int16)).save(mask_path)
    image_value = str(image_path.relative_to(root)) if relative else str(image_path)
    mask_value = str(mask_path.relative_to(root)) if relative else str(mask_path)
    return {
        "image_file_path": image_value,
        "final_label_file_path": mask_value,
        "case_id_norm": patient,
        "split": split,
    }


def _synthetic_manifest(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    rows = [
        _write_case(data_root, "train_a", "train", "patient_train_a", [0, 50, 100], True),
        _write_case(data_root, "train_b", "train", "patient_train_b", [150, 200, 0], False),
        _write_case(data_root, "train_c", "train", "patient_train_c", [50, 100, 150, 200], True),
        _write_case(data_root, "val_a", "val", "patient_val_a", [0, 50, 100], True),
        _write_case(data_root, "val_b", "val", "patient_val_b", [150, 200], False),
        _write_case(data_root, "test_a", "test", "patient_test_a", [0, 50, 100], True),
        _write_case(data_root, "test_b", "test", "patient_test_b", [150, 200], False),
    ]
    manifest = data_root / "split.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _prepare_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, notebook: str = "49") -> Path:
    manifest = _synthetic_manifest(tmp_path)
    monkeypatch.setenv("PFI_INSTALL_NOTEBOOK_DEPS", "0")
    monkeypatch.setenv("PFI_USE_GOOGLE_DRIVE", "0")
    monkeypatch.setenv("PFI_REPO_ROOT", str(ROOT))
    monkeypatch.setenv("PFI_DATASET_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AXIAL_IMAGES_DIR", str(tmp_path / "data" / "images"))
    monkeypatch.setenv("AXIAL_MASKS_DIR", str(tmp_path / "data" / "masks"))
    monkeypatch.setenv("AXIAL_E9_CURATED_SPLIT_CSV", str(manifest))
    monkeypatch.setenv("PFI_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("PFI_RESUME_ROOT", str(tmp_path / "resume"))
    monkeypatch.setenv("PFI_RUN_ID", "axial-final-v2")
    monkeypatch.setenv("PFI_NUM_WORKERS", "0")
    monkeypatch.setenv("PFI_BATCH_SIZE", "2")
    monkeypatch.setenv("PFI_MAX_EPOCHS", "2")
    monkeypatch.setenv("PFI_EARLY_STOP_PATIENCE", "2")
    monkeypatch.setenv("PFI_WEIGHT_MAX_RECORDS", "8")
    monkeypatch.setenv("SMOKE_MAX_RECORDS_PER_SPLIT", "4")
    monkeypatch.setenv("AXIAL_RAW0_WEIGHT_BOOST", "1.0")
    monkeypatch.setenv("AXIAL_MONITOR_METRIC", "dice_macro_foreground")
    monkeypatch.setenv("RUN_MODE", "smoke" if notebook == "49" else "evaluate")
    return manifest


def _exec_notebook(path: Path, *, skip_run: bool = True) -> dict[str, object]:
    if "matplotlib" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")

        class _Axis:
            def imshow(self, *args, **kwargs):
                return None

            def set_title(self, *args, **kwargs):
                return None

            def axis(self, *args, **kwargs):
                return None

        class _Fig:
            def tight_layout(self):
                return None

            def savefig(self, path, *args, **kwargs):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"synthetic-figure")

            def suptitle(self, *args, **kwargs):
                return None

        def subplots(*args, **kwargs):
            nrows = int(args[0]) if args else 1
            ncols = int(args[1]) if len(args) > 1 else 1
            axes = np.array([[_Axis() for _ in range(ncols)] for _ in range(nrows)])
            if nrows == 1 and ncols == 1:
                axes = axes[0, 0]
            elif nrows == 1:
                axes = axes[0]
            return _Fig(), axes

        pyplot.subplots = subplots
        pyplot.close = lambda *args, **kwargs: None
        matplotlib.pyplot = pyplot
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = pyplot
    notebook = nbformat.read(path, as_version=4)
    module_name = f"_notebook_{path.stem}_{os.urandom(4).hex()}"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    ns = module.__dict__
    ns["__name__"] = module_name
    ns["AI_SERVICE_COMMIT_SHA"] = "285159982832abb604a176b4302ac83a837ff1c9"
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if cell.get("id") in {"deps", "drive-mount", "repo-prep"}:
            continue
        if cell.get("id") == "run-block":
            if path == NOTEBOOK_50:
                source = cell.source.split("TEST_ONCE_RESULT = test_once_pipeline()", 1)[0]
                exec(compile(source, f"{path.name}:{cell.get('id')}", "exec"), ns)
            continue
        exec(compile(cell.source, f"{path.name}:{cell.get('id')}", "exec"), ns)
    return ns


def test_notebook_49_real_train_validation_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare_env(monkeypatch, tmp_path, notebook="49")
    ns = _exec_notebook(NOTEBOOK_49)

    records = ns["build_records_from_split_manifest"]()
    assert len(records) == 7
    assert {r.split for r in records} == {"train", "val", "test"}
    assert Path(records[0].image_path).exists()

    integrity = ns["validate_split_integrity"](records)
    assert integrity["patientHeldout"] is True
    label_report = ns["scan_labels_and_shapes"](records)
    assert label_report["casePresenceBySplit"]["train"]["raw_0"] >= 1

    ns["mkdirs_for_run"]()
    duplicate_df = ns["compute_duplicate_report"](records, full_hash=True)
    duplicate_report = ns["validate_no_duplicate_leakage"](duplicate_df)
    assert duplicate_report["duplicateLeakage"] is False

    loaders = ns["build_train_val_loaders"](records, smoke=True)
    assert set(loaders) == {"train", "val"}
    batch = next(iter(loaders["train"]))
    assert list(batch["image"].shape[1:]) == [1, 256, 256]
    assert list(batch["mask"].shape[1:]) == [256, 256]
    assert int(batch["mask"].max()) <= 5

    class_report = ns["class_weight_report"](records)
    assert class_report["trainSamplesUsed"] >= 1
    weights_boost_1 = np.array(list(class_report["finalWeights"].values()))
    monkeypatch.setenv("AXIAL_RAW0_WEIGHT_BOOST", "3.0")
    ns_boost_3 = _exec_notebook(NOTEBOOK_49)
    weights_boost_3 = np.array(list(ns_boost_3["class_weight_report"](records)["finalWeights"].values()))
    assert weights_boost_1[1] < weights_boost_3[1]

    model = ns["build_model"]().to(ns["DEVICE"])
    ce = torch.nn.CrossEntropyLoss(weight=ns["estimate_class_weights"](records).to(ns["DEVICE"]))
    before = next(model.parameters()).detach().clone()
    metrics_train = ns["run_epoch"](model, loaders["train"], ce, torch.optim.AdamW(model.parameters(), lr=0.001), None)
    after = next(model.parameters()).detach().clone()
    assert metrics_train["loss"] > 0
    assert not torch.equal(before, after)
    with torch.inference_mode():
        metrics_val = ns["run_epoch"](model, loaders["val"], ce, None, None)
    assert metrics_val["dice_macro_foreground"] is not None

    report = ns["train_model"](records, smoke_only=True)
    assert report["history"]
    assert report["testEvaluated"] is False
    assert (ns["RESUME_DIR"] / "axial_t2_alkafri_v2.best_checkpoint.pt").exists()
    assert (ns["RESUME_DIR"] / "axial_t2_alkafri_v2.last_checkpoint.pt").exists()


def test_notebook_49_resume_incompatibilities(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare_env(monkeypatch, tmp_path, notebook="49")
    monkeypatch.setenv("RESUME_MODE", "required")
    ns = _exec_notebook(NOTEBOOK_49)
    records = ns["build_records_from_split_manifest"]()
    ns["mkdirs_for_run"]()
    model = ns["build_model"]()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")
    payload = ns["checkpoint_payload"](model, opt, scheduler, None, 1, 0.1, 2, [], ns["estimate_class_weights"](records), smoke_only=True)
    payload["runId"] = "wrong-run"
    ns["save_checkpoint"](ns["RESUME_DIR"] / "axial_t2_alkafri_v2.last_checkpoint.pt", payload)
    with pytest.raises(ValueError, match="Resume incompatible"):
        ns["load_resume_if_allowed"](model, opt, scheduler, None, smoke_only=True)
    payload["runId"] = "axial-final-v2"
    payload["smokeOnly"] = False
    ns["save_checkpoint"](ns["RESUME_DIR"] / "axial_t2_alkafri_v2.last_checkpoint.pt", payload)
    with pytest.raises(ValueError, match="Resume incompatible"):
        ns["load_resume_if_allowed"](model, opt, scheduler, None, smoke_only=True)


def test_notebook_50_test_once_and_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare_env(monkeypatch, tmp_path, notebook="49")
    ns49 = _exec_notebook(NOTEBOOK_49)
    records = ns49["build_records_from_split_manifest"]()
    ns49["train_model"](records, smoke_only=True)

    monkeypatch.setenv("RUN_MODE", "evaluate")
    ns50 = _exec_notebook(NOTEBOOK_50)
    with pytest.raises(RuntimeError, match="AXIAL_FINAL_TEST_CONFIRMATION"):
        ns50["evaluate_test_once"](records)
    monkeypatch.setenv("AXIAL_FINAL_TEST_CONFIRMATION", "axial-final-v2")
    result = ns50["test_once_pipeline"]()
    assert result["testMetrics"]["dice_macro_foreground"] is not None
    assert "raw0Precision" in result["testMetrics"]
    assert "raw0Recall" in result["testMetrics"]
    assert result["verification"]["runtimeShape"] == [1, 6, 256, 256]
    assert result["verification"]["runtimeFinite"] is True
    artifact_name = result["artifact"]
    assert artifact_name in {"axial_t2_alkafri_final_v2_best.pt", "axial_t2_alkafri_final_v2_candidate.pt"}
    manifest_path = ns50["manifest_path_for"](ns50["select_artifact_path"](result["qualityGate"]["qualityGatePassed"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifactFile"] == artifact_name
    assert manifest["artifactSha256"] == ns50["sha256_stream"](ns50["select_artifact_path"](result["qualityGate"]["qualityGatePassed"]))

    with pytest.raises(RuntimeError, match="evaluado"):
        ns50["evaluate_test_once"](records)
    in_progress = ns50["OUTPUT_DIRS"]["metrics"] / "test_evaluation_in_progress.json"
    marker = ns50["OUTPUT_DIRS"]["metrics"] / "test_evaluated_once.json"
    marker.unlink()
    ns50["TEST_EVALUATED_IN_MEMORY"] = False
    in_progress.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="in_progress"):
        ns50["evaluate_test_once"](records)


def test_static_separation_and_no_placeholders() -> None:
    source49 = "\n".join(cell.source for cell in nbformat.read(NOTEBOOK_49, as_version=4).cells if cell.cell_type == "code")
    source50 = "\n".join(cell.source for cell in nbformat.read(NOTEBOOK_50, as_version=4).cells if cell.cell_type == "code")
    assert "PFI_EXECUTE_NOTEBOOK" not in source49 + source50
    assert "placeholder_metrics" not in source49 + source50
    assert 'loaders["test"]' not in source49
    assert "evaluate_test_once" not in source49
    assert "def train_model" not in source50
    assert ".backward(" not in source50
