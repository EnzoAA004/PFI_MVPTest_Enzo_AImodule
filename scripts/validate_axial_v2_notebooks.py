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
    seen_ids: set[str] = set()
    for index, cell in enumerate(notebook.cells):
        cell_id = cell.get("id")
        if not cell_id:
            raise AssertionError(f"{path.name}: cell {index} no tiene id")
        if cell_id in seen_ids:
            raise AssertionError(f"{path.name}: id duplicado {cell_id}")
        seen_ids.add(cell_id)
        if cell.cell_type == "code":
            if cell.get("execution_count") is not None:
                raise AssertionError(f"{path.name}: cell {index} tiene execution_count")
            if cell.get("outputs") != []:
                raise AssertionError(f"{path.name}: cell {index} tiene outputs")
            compile(cell.source, f"{path.name}:cell_{index}", "exec")
        elif cell.cell_type == "markdown":
            if "execution_count" in cell:
                raise AssertionError(f"{path.name}: markdown cell {index} tiene execution_count")
            if "outputs" in cell:
                raise AssertionError(f"{path.name}: markdown cell {index} tiene outputs")
    return notebook


def code_source(notebook) -> str:
    return "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def assert_occurs(source: str, needle: str, count: int) -> None:
    actual = source.count(needle)
    if actual != count:
        raise AssertionError(f"{needle!r}: esperado {count}, encontrado {actual}")


def assert_notebook_49(source: str) -> None:
    cfg_match = re.search(r"\bCFG\s*=\s*TrainConfig\(\)", source)
    if cfg_match is None:
        raise AssertionError("notebook 49 no inicializa CFG")
    cfg_pos = cfg_match.start()
    first_cfg_use = source.find("CFG.")
    if first_cfg_use != -1 and first_cfg_use < cfg_pos:
        raise AssertionError("notebook 49 usa CFG antes de crearlo")

    tree = ast.parse(source)
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    for name in ["train_model", "metrics_from_predictions", "preflight"]:
        if function_names.count(name) != 1:
            raise AssertionError(f"notebook 49 debe definir {name} exactamente una vez")

    forbidden = [
        "evaluate_test_once(",
        'loaders["test"]',
        "test_metrics",
        "quality_gate(",
        "AXIAL_FINAL_TEST_CONFIRMATION",
        'RUN_MODE", "full"',
        'Literal["preflight", "smoke", "train", "full"]',
    ]
    for needle in forbidden:
        if needle in source:
            raise AssertionError(f"notebook 49 contiene texto prohibido: {needle}")

    required = [
        'RUN_MODE: Literal["preflight", "smoke", "train"]',
        'AXIAL_RAW0_WEIGHT_BOOST", "1.0"',
        'AXIAL_MONITOR_METRIC", "dice_macro_foreground"',
        'ALLOWED_MONITOR_METRICS = {"dice_macro_foreground", "dice_macro_excluding_raw0"}',
        "RESUME_DIR = CFG.RESUME_ROOT / CFG.RUN_ID",
        "axial_t2_alkafri_v2.last_checkpoint.pt",
        "axial_t2_alkafri_v2.best_checkpoint.pt",
        "testEvaluated",
        "False",
        "dice_macro_foreground",
        "raw0Precision",
        "raw0Recall",
        "truePositive",
        "falsePositive",
        "falseNegative",
        "trueNegative",
        "{250: 0, 0: 1, 50: 2, 100: 3, 150: 4, 200: 5}",
    ]
    for needle in required:
        if needle not in source:
            raise AssertionError(f"notebook 49 falta requerido: {needle}")

    if not re.search(r"MAX_EPOCHS:\s*int\s*=\s*int\(os\.getenv\([^)]*80", source):
        raise AssertionError("notebook 49 no tiene max_epochs 80 configurable")


def assert_notebook_50(source: str) -> None:
    forbidden = ["optimizer", ".backward(", "def train_model"]
    for needle in forbidden:
        if needle in source:
            raise AssertionError(f"notebook 50 contiene entrenamiento o modo prohibido: {needle}")

    required = [
        "AXIAL_FINAL_TEST_CONFIRMATION",
        "axial-final-v2",
        "axial_t2_alkafri_v2.best_checkpoint.pt",
        "test_evaluated_once.json",
        "evaluate_test_once",
        "axial_t2_alkafri_final_v2_best.pt",
        "axial_t2_alkafri_final_v2_candidate.pt",
        "artifact_path.name",
        "dice_macro_foreground",
        "raw0Precision",
        "raw0Recall",
        "[1, 6, 256, 256]",
        "The held-out test partition was previously evaluated for the axial-full-v1 baseline.",
    ]
    for needle in required:
        if needle not in source:
            raise AssertionError(f"notebook 50 falta requerido: {needle}")


def main() -> int:
    nb49 = load_notebook(NOTEBOOK_49)
    nb50 = load_notebook(NOTEBOOK_50)
    source49 = code_source(nb49)
    source50 = code_source(nb50)
    assert_notebook_49(source49)
    assert_notebook_50(source50)
    print("validate_axial_v2_notebooks: OK notebooks=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
