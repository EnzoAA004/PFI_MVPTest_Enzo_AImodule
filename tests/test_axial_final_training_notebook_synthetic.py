from __future__ import annotations

import dataclasses
import os
import sys
import types
import warnings
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "48_axial_final_training_patient_split.ipynb"


def _write_pair(root: Path, name: str, labels: list[int], background_only: bool = False) -> tuple[Path, Path]:
    image_offset = sum(ord(ch) for ch in name)
    image = np.arange(16 * 16, dtype=np.float32).reshape(16, 16) + image_offset
    mask = np.full((16, 16), 250, dtype=np.int16)
    if not background_only:
        for index, label in enumerate(labels):
            row = (index * 3) % 16
            mask[row : row + 2, index : index + 2] = label
    image_path = root / "images" / f"{name}.npy"
    mask_path = root / "masks" / f"{name}.npy"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_path, image)
    np.save(mask_path, mask)
    return image_path, mask_path


def _write_shape_pair(root: Path, name: str, image_shape: tuple[int, int], mask_shape: tuple[int, int]) -> tuple[Path, Path]:
    image = np.arange(np.prod(image_shape), dtype=np.float32).reshape(image_shape)
    mask = np.full(mask_shape, 250, dtype=np.int16)
    image_path = root / "images" / f"{name}_image.npy"
    mask_path = root / "masks" / f"{name}_mask.npy"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_path, image)
    np.save(mask_path, mask)
    return image_path, mask_path


def _synthetic_split(tmp_path: Path) -> Path:
    rows = []
    specs = [
        ("train_a", "train", "patient_a", [0, 50]),
        ("train_b", "train", "patient_b", [100, 150, 200]),
        ("val_a", "val", "patient_c", [50, 100]),
        ("val_b", "val", "patient_d", [150, 200]),
        ("test_a", "test", "patient_e", [0, 150]),
        ("test_b", "test", "patient_f", [50, 200]),
        ("test_bg_a", "test", "patient_g", []),
        ("test_bg_b", "test", "patient_h", []),
    ]
    for name, split, patient, labels in specs:
        image_path, mask_path = _write_pair(tmp_path, name, labels, background_only=name.startswith("test_bg_"))
        rows.append(
            {
                "image_file_path": str(image_path),
                "final_label_file_path": str(mask_path),
                "case_id_norm": patient,
                "split": split,
            }
        )
    csv_path = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _load_notebook_symbols(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    split_csv = _synthetic_split(tmp_path / "data")
    out_root = tmp_path / "out"
    resume_root = tmp_path / "resume_root"
    monkeypatch.setenv("PFI_INSTALL_NOTEBOOK_DEPS", "0")
    monkeypatch.setenv("PFI_USE_GOOGLE_DRIVE", "0")
    monkeypatch.setenv("RUN_MODE", "preflight")
    monkeypatch.setenv("PFI_REPO_ROOT", str(ROOT))
    monkeypatch.setenv("PFI_DATASET_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AXIAL_IMAGES_DIR", str(tmp_path / "data" / "images"))
    monkeypatch.setenv("AXIAL_MASKS_DIR", str(tmp_path / "data" / "masks"))
    monkeypatch.setenv("AXIAL_E9_CURATED_SPLIT_CSV", str(split_csv))
    monkeypatch.setenv("PFI_OUTPUT_ROOT", str(out_root))
    monkeypatch.setenv("PFI_RESUME_ROOT", str(resume_root))
    monkeypatch.setenv("PFI_RUN_ID", "synthetic-run")
    monkeypatch.setenv("PFI_NUM_WORKERS", "0")
    monkeypatch.setenv("PFI_BATCH_SIZE", "2")
    monkeypatch.setenv("PFI_WEIGHT_MAX_RECORDS", "6")
    monkeypatch.setenv("SMOKE_MAX_RECORDS_PER_SPLIT", "4")

    if "matplotlib" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")
        pyplot.subplots = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pyplot no se usa en este test"))
        pyplot.close = lambda *args, **kwargs: None
        matplotlib.pyplot = pyplot
        monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    nb = nbformat.read(NOTEBOOK, as_version=4)
    symbols: dict[str, object] = {
        "AI_SERVICE_COMMIT_SHA": "285159982832abb604a176b4302ac83a837ff1c9",
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source.lstrip()
        if source.startswith("import importlib.util"):
            continue
        if source.startswith("# Montaje idempotente"):
            continue
        if source.startswith("# Commit validado"):
            continue
        if source.startswith("def select_preflight_examples"):
            continue
        exec(compile(cell.source, "<notebook-cell>", "exec"), symbols)
    symbols["mkdirs_for_run"]()
    return symbols


def test_axial_final_notebook_synthetic_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ns = _load_notebook_symbols(monkeypatch, tmp_path)
    records = ns["build_records_from_split_manifest"]()

    assert len(records) == 8
    integrity = ns["validate_split_integrity"](records)
    assert integrity["patientHeldout"] is True

    duplicate_df = ns["compute_duplicate_report"](records, full_hash=False)
    assert int(duplicate_df["fileSize"].duplicated().sum()) > 0
    assert int(duplicate_df["sha256"].notna().sum()) == len(duplicate_df)
    with pytest.warns(UserWarning, match="grupos de mascaras"):
        duplicate_report = ns["validate_no_duplicate_leakage"](duplicate_df)
    assert duplicate_report["duplicateLeakage"] is False
    assert duplicate_report["duplicateImages"] == []
    assert duplicate_report["duplicatePairs"] == []
    assert duplicate_report["repeatedMasks"]
    assert duplicate_report["maskOnlyWarnings"]

    label_report = ns["scan_labels_and_shapes"](records)
    assert label_report["casePresenceBySplit"]["train"]["raw_0"] == 1

    loaders = ns["build_loaders"](records, smoke=True)
    train_patients = {item for batch in loaders["train"] for item in batch["patientId"]}
    assert len(train_patients) >= 2

    model = ns["build_model"]()
    batch = next(iter(loaders["train"]))
    logits = model(batch["image"])
    ce = torch.nn.CrossEntropyLoss(weight=ns["estimate_class_weights"](records))
    loss, ce_loss, dice_loss = ns["multiclass_loss"](logits, batch["mask"], ce)
    assert torch.isfinite(loss)
    assert torch.isfinite(ce_loss)
    assert torch.isfinite(dice_loss)

    metrics = ns["metrics_from_logits"](logits, batch["mask"])
    assert "confusionMatrix" in metrics
    absent_metrics = ns["metrics_from_predictions"](
        np.zeros((1, 8, 8), dtype=np.int64),
        np.zeros((1, 8, 8), dtype=np.int64),
    )
    assert absent_metrics["perClass"]["raw_0"]["dice"] is None
    assert absent_metrics["perClass"]["raw_0"]["gt_present_cases"] == 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    class_weights = ns["estimate_class_weights"](records)
    history = [{"epoch": 1, "val_dice_macro_excluding_raw0": 0.25}]
    payload = ns["checkpoint_payload"](
        model,
        optimizer,
        scheduler,
        None,
        1,
        0.25,
        7,
        history,
        class_weights,
        True,
    )
    resume_path = ns["RESUME_DIR"] / "axial_t2_alkafri_final.last_checkpoint.pt"
    assert ns["RESUME_DIR"].parent == tmp_path / "resume_root"
    assert not str(ns["RESUME_DIR"]).startswith(str(ns["CFG"].OUTPUT_ROOT))
    ns["save_checkpoint"](resume_path, payload)
    start_epoch, best_metric, patience_left, restored_history = ns["load_resume_if_allowed"](
        model,
        optimizer,
        scheduler,
        None,
        smoke_only=True,
    )
    assert (start_epoch, best_metric, patience_left, restored_history) == (1, 0.25, 7, history)

    incompatible = dict(payload)
    incompatible["smokeOnly"] = False
    incompatible["smoke_only"] = False
    ns["save_checkpoint"](resume_path, incompatible)
    with pytest.warns(UserWarning, match="Resume incompatible"):
        assert ns["load_resume_if_allowed"](model, optimizer, scheduler, None, smoke_only=True)[0] == 0

    original_cfg = ns["CFG"]
    ns["CFG"] = dataclasses.replace(original_cfg, RESUME_MODE="required")
    with pytest.raises(ValueError, match="Resume incompatible"):
        ns["load_resume_if_allowed"](model, optimizer, scheduler, None, smoke_only=True)
    ns["CFG"] = original_cfg

    final_path = ns["OUTPUT_DIRS"]["models"] / "axial_t2_alkafri_final_best.pt"
    candidate_path = ns["OUTPUT_DIRS"]["models"] / "axial_t2_alkafri_final_candidate.pt"
    ns["save_checkpoint"](
        final_path,
        {
            "model_state_dict": model.state_dict(),
            "num_classes": ns["CFG"].NUM_CLASSES,
            "base_channels": ns["CFG"].BASE_CHANNELS,
            "target_size": ns["CFG"].TARGET_SIZE,
        },
    )
    approved_gate = {"qualityGatePassed": True}
    manifest = ns["generate_manifest_and_model_card"](final_path, metrics, approved_gate)
    assert isinstance(manifest, dict)
    assert manifest["artifactFile"] == "axial_t2_alkafri_final_best.pt"
    assert manifest["metrics"]["dicePerClass"]
    assert manifest["metrics"]["iouPerClass"]
    assert (ns["OUTPUT_DIRS"]["manifests"] / "axial_t2_alkafri_final_best.pt.manifest.json").exists()
    assert (ns["OUTPUT_DIRS"]["manifests"] / "axial_t2_alkafri_final_best.pt.modelcard.md").exists()

    ns["save_checkpoint"](
        candidate_path,
        {
            "model_state_dict": model.state_dict(),
            "num_classes": ns["CFG"].NUM_CLASSES,
            "base_channels": ns["CFG"].BASE_CHANNELS,
            "target_size": ns["CFG"].TARGET_SIZE,
        },
    )
    candidate_manifest = ns["generate_manifest_and_model_card"](candidate_path, metrics, {"qualityGatePassed": False})
    assert candidate_manifest["artifactFile"] == "axial_t2_alkafri_final_candidate.pt"
    assert (ns["OUTPUT_DIRS"]["manifests"] / "axial_t2_alkafri_final_candidate.pt.manifest.json").exists()
    assert (ns["OUTPUT_DIRS"]["manifests"] / "axial_t2_alkafri_final_candidate.pt.modelcard.md").exists()

    shape = ns["round_trip_model_from_manifest"](
        final_path,
        manifest,
        torch.randn(1, 1, *ns["CFG"].TARGET_SIZE),
    )
    assert shape == [1, 6, 256, 256]


def test_axial_final_notebook_structure_is_reproducible() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(cell.source for cell in nb.cells)
    code_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

    cfg_index = code_text.index("CFG = TrainConfig()")
    first_cfg_use = code_text.index("CFG.")
    assert cfg_index < first_cfg_use
    assert text.count("def scan_labels_and_shapes") == 1
    assert text.count("def build_model") == 1
    assert text.count("def preflight") == 1
    assert "filter_known_corrupt_records" not in text
    assert "KNOWN_CORRUPT_SOURCE_PAIRS" not in text
    assert 'OUTPUT_DIRS["resume"]' not in text
    assert "shape_mismatch_df" not in text
    assert "all_border_df" not in text
    assert "target_slice_id" not in text
    assert "print_medical_geometry" not in text
    assert "run_records = build_records_from_split_manifest()" in text
    assert "records = build_records_from_split_manifest()" in text
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None
            compile(cell.source, "notebook_cell", "exec")


def test_known_patient_56_shape_mismatch_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ns = _load_notebook_symbols(monkeypatch, tmp_path)
    records = ns["build_records_from_split_manifest"]()
    record_cls = ns["AxialRecord"]

    image_path, mask_path = _write_shape_pair(tmp_path / "known", "patient56", (384, 384), (320, 320))
    known_record = record_cls(
        str(image_path),
        str(mask_path),
        "56",
        "study_56",
        "known_patient_56",
        "train",
        image_path.name,
        mask_path.name,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        report = ns["scan_labels_and_shapes"]([*records, known_record])
    assert report["shapeMismatchCount"] == 1
    assert any("patron conocido del paciente 56" in str(item.message) for item in captured)

    other_image, other_mask = _write_shape_pair(tmp_path / "bad_patient", "patient57", (384, 384), (320, 320))
    bad_patient_record = record_cls(
        str(other_image),
        str(other_mask),
        "57",
        "study_57",
        "bad_patient",
        "train",
        other_image.name,
        other_mask.name,
    )
    with pytest.raises(ValueError, match="mismatch image/mask no autorizado"):
        ns["scan_labels_and_shapes"]([*records, bad_patient_record])

    ratio_image, ratio_mask = _write_shape_pair(tmp_path / "bad_ratio", "patient56_ratio", (384, 320), (320, 320))
    bad_ratio_record = record_cls(
        str(ratio_image),
        str(ratio_mask),
        "56",
        "study_56_ratio",
        "bad_ratio",
        "train",
        ratio_image.name,
        ratio_mask.name,
    )
    with pytest.raises(ValueError, match="mismatch image/mask no autorizado"):
        ns["scan_labels_and_shapes"]([*records, bad_ratio_record])
