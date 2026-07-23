from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_49 = ROOT / "notebooks" / "49_axial_final_v2_train_validation.ipynb"
NOTEBOOK_50 = ROOT / "notebooks" / "50_axial_final_v2_test_once.ipynb"


def load_notebook(path: Path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    ids: set[str] = set()
    for index, cell in enumerate(notebook.cells):
        cell_id = cell.get("id")
        if not cell_id:
            raise AssertionError(f"{path.name}: cell {index} sin id")
        if cell_id in ids:
            raise AssertionError(f"{path.name}: id duplicado {cell_id}")
        ids.add(cell_id)
        if cell.cell_type == "code":
            if cell.get("execution_count") is not None:
                raise AssertionError(f"{path.name}: cell {index} execution_count no nulo")
            if cell.get("outputs") != []:
                raise AssertionError(f"{path.name}: cell {index} tiene outputs")
            compile(cell.source, f"{path.name}:cell_{index}", "exec")
        elif cell.cell_type == "markdown":
            if "outputs" in cell or "execution_count" in cell:
                raise AssertionError(f"{path.name}: markdown cell {index} contiene metadata de ejecucion")
    return notebook


def source(notebook) -> str:
    return "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def assert_contains(source_text: str, needles: list[str], label: str) -> None:
    missing = [needle for needle in needles if needle not in source_text]
    if missing:
        raise AssertionError(f"{label}: faltan {missing}")


def assert_absent(source_text: str, needles: list[str], label: str) -> None:
    present = [needle for needle in needles if needle in source_text]
    if present:
        raise AssertionError(f"{label}: prohibidos presentes {present}")


def function_counts(source_text: str) -> dict[str, int]:
    tree = ast.parse(source_text)
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            counts[node.name] = counts.get(node.name, 0) + 1
    return counts


def assert_cfg_before_use(source_text: str) -> None:
    match = re.search(r"\bCFG\s*=\s*TrainConfig\(\)", source_text)
    if not match:
        raise AssertionError("CFG no se inicializa")
    first_use = source_text.find("CFG.")
    if first_use != -1 and first_use < match.start():
        raise AssertionError("CFG se usa antes de inicializarse")


def validate_49(source_text: str) -> None:
    assert_cfg_before_use(source_text)
    counts = function_counts(source_text)
    for name in ["preflight", "train_model", "run_epoch", "metrics_from_predictions", "build_train_val_loaders"]:
        if counts.get(name) != 1:
            raise AssertionError(f"notebook 49 debe definir {name} una vez, tiene {counts.get(name)}")
    if counts.get("evaluate_test_once", 0) != 0:
        raise AssertionError("notebook 49 no debe definir evaluate_test_once")
    assert_contains(
        source_text,
        [
            'RUN_MODE: Literal["preflight", "smoke", "train"]',
            'RUN_MODE", "preflight"',
            'AXIAL_RAW0_WEIGHT_BOOST", "1.0"',
            'AXIAL_MONITOR_METRIC", "dice_macro_foreground"',
            "class AxialSegmentationDataset",
            "def build_train_val_loaders",
            "def soft_dice_loss",
            "def multiclass_loss",
            "def run_epoch",
            "optimizer.zero_grad",
            ".backward()",
            "scaler.step",
            "clip_grad_norm_",
            "model.train",
            "model.eval",
            "torch.inference_mode",
            "nn.CrossEntropyLoss",
            "DataLoader",
            "validation_metrics_best.json",
            "validation_case_metrics.csv",
            "validation_predictions_best.png",
            "DEVICE",
            'if self.RUN_MODE == "train" and not torch.cuda.is_available()',
            "class_weight_report(records)",
            "PFI_WEIGHT_MAX_RECORDS",
            "AXIAL_ALLOW_EXTREME_CLASS_WEIGHTS",
            "RESUME_DIR = CFG.RESUME_ROOT / CFG.RUN_ID",
            "axial_t2_alkafri_v2.last_checkpoint.pt",
            "axial_t2_alkafri_v2.best_checkpoint.pt",
            "PREFLIGHT_REPORT = preflight()",
            'if CFG.RUN_MODE in {"smoke", "train"}',
            "AI_SERVICE_COMMIT_SHA",
        ],
        "notebook 49",
    )
    assert_absent(
        source_text,
        [
            "placeholder_metrics",
            "{0: 100, 1: 20",
            "PFI_EXECUTE_NOTEBOOK",
            "evaluate_test_once(",
            'loaders["test"]',
            "test_metrics",
            "quality_gate(",
            "AXIAL_FINAL_TEST_CONFIRMATION",
            'RUN_MODE", "full"',
            "runtime_shape = [1, 6, 256, 256]",
        ],
        "notebook 49",
    )


def validate_50(source_text: str) -> None:
    assert_cfg_before_use(source_text)
    counts = function_counts(source_text)
    for name in ["evaluate_test_once", "test_once_pipeline", "quality_gate", "round_trip_model_from_manifest"]:
        if counts.get(name) != 1:
            raise AssertionError(f"notebook 50 debe definir {name} una vez, tiene {counts.get(name)}")
    assert_contains(
        source_text,
        [
            "AxialSegmentationDataset",
            "DataLoader",
            "torch.inference_mode",
            "logits",
            "torch.argmax",
            "metrics_from_predictions",
            "test_metrics.json",
            "test_case_metrics.csv",
            "test_confusion_matrix.csv",
            "test_evaluation_in_progress.json",
            "test_evaluated_once.json",
            "torch.isfinite",
            "output.shape",
            "AXIAL_FINAL_TEST_CONFIRMATION",
            "axial-final-v2",
            "axial_t2_alkafri_v2.best_checkpoint.pt",
            "axial_t2_alkafri_final_v2_best.pt",
            "axial_t2_alkafri_final_v2_candidate.pt",
            "artifact_path.name",
            "final_artifact_verification.json",
            "The held-out test partition was previously evaluated for the axial-full-v1 baseline.",
        ],
        "notebook 50",
    )
    assert_absent(
        source_text,
        [
            "placeholder_metrics",
            "optimizer",
            ".backward(",
            "def train_model",
            "scheduler",
            "runtime_shape = [1, 6, 256, 256]",
            "PFI_EXECUTE_NOTEBOOK",
        ],
        "notebook 50",
    )


def main() -> int:
    nb49 = load_notebook(NOTEBOOK_49)
    nb50 = load_notebook(NOTEBOOK_50)
    validate_49(source(nb49))
    validate_50(source(nb50))
    print(
        "validate_axial_v2_notebooks: OK "
        f"notebook49_cells={len(nb49.cells)} notebook50_cells={len(nb50.cells)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
