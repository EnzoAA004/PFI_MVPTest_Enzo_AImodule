from __future__ import annotations

import ast
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumbar_mri.axial_v3.guards import find_forbidden_test_references


NOTEBOOKS = [
    ROOT / "notebooks" / "51_axial_v3_iteration_a_raw0_audit.ipynb",
    ROOT / "notebooks" / "52_axial_v3_iteration_b_low_cost.ipynb",
    ROOT / "notebooks" / "53_axial_v3_iteration_c_2_5d.ipynb",
    ROOT / "notebooks" / "54_axial_v3_iteration_d_architectures.ipynb",
]
REQUIRED_DOC = ROOT / "docs" / "axial_v3_improvement_plan.md"
REQUIRED_CONFIG = ROOT / "config" / "axial_v3_low_cost_grid.json"
REQUIRED_REGISTRY = ROOT / "outputs" / "axial_v3" / "experiment_registry.csv"


def _source(nb) -> str:
    return "\n\n".join(cell.source for cell in nb.cells)


def validate_notebook(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    ids: set[str] = set()
    code_source = []
    for index, cell in enumerate(nb.cells):
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
            ast.parse(cell.source)
            code_source.append(cell.source)
        elif cell.cell_type == "markdown":
            if "outputs" in cell or "execution_count" in cell:
                raise AssertionError(f"{path.name}: markdown cell {index} contiene metadata de ejecucion")
    text = _source(nb)
    code_text = "\n\n".join(code_source)
    forbidden = find_forbidden_test_references(code_text)
    if forbidden:
        raise AssertionError(f"{path.name}: referencias test prohibidas {forbidden}")
    if "require_train_val_only" not in code_text:
        raise AssertionError(f"{path.name}: falta guard train/validation")
    if "C:\\Users\\" in text or "/Users/" in text:
        raise AssertionError(f"{path.name}: ruta de usuario hardcodeada")


def validate_documentation() -> None:
    if not REQUIRED_DOC.exists():
        raise AssertionError(f"falta documentacion {REQUIRED_DOC}")
    text = REQUIRED_DOC.read_text(encoding="utf-8")
    required = [
        "qualityGatePassed | false",
        "raw_0",
        "dice_macro_excluding_raw0",
        "Axial v3 no puede seleccionarse usando el test ya observado.",
        "Orden recomendado de ejecucion",
        "The held-out test partition was previously evaluated",
        "axial_t2_alkafri_final_v2_candidate.pt",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise AssertionError(f"documentacion incompleta: {missing}")


def validate_registry_schema() -> None:
    if not REQUIRED_CONFIG.exists():
        raise AssertionError(f"falta config {REQUIRED_CONFIG}")
    if not REQUIRED_REGISTRY.exists():
        raise AssertionError(f"falta registry {REQUIRED_REGISTRY}")
    header = REQUIRED_REGISTRY.read_text(encoding="utf-8").splitlines()[0].split(",")
    expected = [
        "experimentId",
        "iteration",
        "runId",
        "createdAtUtc",
        "gitCommit",
        "seed",
        "configPath",
        "splitSha256",
        "trainingStatus",
        "selectedEpoch",
        "monitorMetric",
    ]
    missing = [column for column in expected if column not in header]
    if missing:
        raise AssertionError(f"registry sin columnas requeridas: {missing}")


def main() -> int:
    for path in NOTEBOOKS:
        validate_notebook(path)
    validate_documentation()
    validate_registry_schema()
    print("validate_axial_v3_notebooks: OK")
    print(f"notebooks={len(NOTEBOOKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
