from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import nbformat
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai_service"))

from pfi_ai_service.model_architectures import AxialUNet2D


NOTEBOOK_49 = ROOT / "notebooks" / "49_axial_final_v2_train_validation.ipynb"
NOTEBOOK_50 = ROOT / "notebooks" / "50_axial_final_v2_test_once.ipynb"


def notebook_source(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def exec_notebook_defs(path: Path) -> dict:
    notebook = nbformat.read(path, as_version=4)
    module_name = f"_notebook_{path.stem}"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    namespace: dict = module.__dict__
    namespace["__name__"] = module_name
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        if "if os.getenv(\"PFI_EXECUTE_NOTEBOOK\"" in source:
            continue
        exec(compile(source, f"{path.name}:cell", "exec"), namespace)
    return namespace


def test_axial_model_round_trip_backward_is_finite() -> None:
    model = AxialUNet2D(num_classes=6, base_channels=16)
    model.train()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    assert list(y.shape) == [1, 6, 256, 256]
    assert torch.isfinite(y).all()
    loss = y.mean()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_metrics_count_raw0_false_positives_and_macros() -> None:
    ns = exec_notebook_defs(NOTEBOOK_49)
    metrics_from_predictions = ns["metrics_from_predictions"]
    classes = [ns["CLASS_INDEX_TO_NAME"][index] for index in range(6)]
    pred = torch.tensor([[1, 1, 2], [0, 3, 3]])
    target = torch.tensor([[1, 0, 2], [0, 0, 3]])

    metrics = metrics_from_predictions(pred.numpy(), target.numpy())

    raw0 = metrics["perClass"]["raw_0"]
    assert raw0["falsePositivePixels"] == pytest.approx(1.0)
    assert raw0["precision"] == pytest.approx(0.5)
    assert raw0["recall"] == pytest.approx(1.0)
    assert raw0["dice"] is not None
    assert raw0["dice"] < 1.0
    assert metrics["dice_macro_foreground"] is not None
    assert metrics["dice_macro_excluding_raw0"] is not None


def test_raw0_boost_one_is_lower_than_boost_three() -> None:
    ns = exec_notebook_defs(NOTEBOOK_49)
    compute_class_weights = ns["class_weights_from_pixel_counts"]
    counts = {0: 1000, 1: 100, 2: 250, 3: 250, 4: 250, 5: 250}

    boost_1 = compute_class_weights(counts, raw0_boost=1.0)
    boost_3 = compute_class_weights(counts, raw0_boost=3.0)

    assert boost_1["finalWeights"]["raw_0"] < boost_3["finalWeights"]["raw_0"]
    assert boost_1["maxMinRatio"] < boost_3["maxMinRatio"]


def test_checkpoint_resume_uses_external_resume_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resume_root = tmp_path / "resume-external"
    output_root = tmp_path / "output-root"
    monkeypatch.setenv("PFI_RESUME_ROOT", str(resume_root))
    monkeypatch.setenv("PFI_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("PFI_RUN_ID", "synthetic-run")

    ns = exec_notebook_defs(NOTEBOOK_49)
    resume_dir = ns["RESUME_DIR"]
    assert resume_dir == resume_root / "synthetic-run"
    assert str(resume_dir).startswith(str(resume_root))
    assert not str(resume_dir).startswith(str(output_root / "synthetic-run" / "resume"))

    model = AxialUNet2D(num_classes=6, base_channels=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008)
    payload = ns["checkpoint_payload"](
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max"),
        scaler=None,
        epoch=3,
        best_monitor_value=0.4,
        patience_counter=2,
        history=[{"dice_macro_foreground": 0.4}],
        class_weight_report={"finalWeights": {}},
        smoke_only=False,
    )
    checkpoint_path = resume_dir / "axial_t2_alkafri_v2.last_checkpoint.pt"
    ns["save_checkpoint"](checkpoint_path, payload)
    assert checkpoint_path.exists()

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 3
    assert loaded["runId"] == "synthetic-run"


def test_resume_rejects_incompatible_mode_and_monitor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFI_RESUME_ROOT", str(tmp_path))
    monkeypatch.setenv("PFI_RUN_ID", "resume-check")
    monkeypatch.setenv("RESUME_MODE", "required")
    ns = exec_notebook_defs(NOTEBOOK_49)
    checkpoint = tmp_path / "resume-check" / "axial_t2_alkafri_v2.last_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "run_id": "resume-check",
            "run_mode": "smoke",
            "monitor_metric": "dice_macro_excluding_raw0",
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "epoch": 1,
            "best_metric": 0.0,
        },
        checkpoint,
    )

    model = AxialUNet2D(num_classes=6, base_channels=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    with pytest.raises(ValueError, match="Resume incompatible"):
        ns["load_resume_if_allowed"](checkpoint, model, optimizer, scheduler, None, smoke_only=False)


def test_notebook_49_has_no_test_evaluation_and_50_has_no_training() -> None:
    source49 = notebook_source(NOTEBOOK_49)
    source50 = notebook_source(NOTEBOOK_50)

    for forbidden in [
        "evaluate_test_once(",
        'loaders["test"]',
        "test_metrics",
        "quality_gate(",
        "AXIAL_FINAL_TEST_CONFIRMATION",
    ]:
        assert forbidden not in source49

    for forbidden in ["optimizer", ".backward(", "def train_model"]:
        assert forbidden not in source50


def test_best_selection_uses_configured_foreground_monitor() -> None:
    source = notebook_source(NOTEBOOK_49)
    assert 'AXIAL_MONITOR_METRIC", "dice_macro_foreground"' in source
    assert "def monitor_value(metrics: dict[str, Any]) -> float:" in source
    assert "value = metrics.get(CFG.AXIAL_MONITOR_METRIC)" in source
    assert "current = monitor_value(" in source
    assert "scheduler.step(current)" in source
    assert "improved = current > best_monitor" in source


def test_test_once_confirmation_marker_and_dynamic_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFI_OUTPUT_ROOT", str(tmp_path))
    ns = exec_notebook_defs(NOTEBOOK_50)

    with pytest.raises(RuntimeError, match="AXIAL_FINAL_TEST_CONFIRMATION"):
        ns["require_test_confirmation"]()

    monkeypatch.setenv("AXIAL_FINAL_TEST_CONFIRMATION", "axial-final-v2")
    ns["require_test_confirmation"]()

    marker = tmp_path / "run" / "metrics" / "test_evaluated_once.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"already": True}), encoding="utf-8")
    ns["marker_path"] = lambda: marker
    with pytest.raises(RuntimeError, match="evaluado"):
        ns["assert_not_already_evaluated"]()

    best = ns["select_artifact_path"](True)
    candidate = ns["select_artifact_path"](False)
    assert best.name == "axial_t2_alkafri_final_v2_best.pt"
    assert candidate.name == "axial_t2_alkafri_final_v2_candidate.pt"
    assert ns["manifest_path_for"](best).name == "axial_t2_alkafri_final_v2_best.pt.manifest.json"
    assert ns["model_card_path_for"](candidate).name == "axial_t2_alkafri_final_v2_candidate.pt.modelcard.md"
