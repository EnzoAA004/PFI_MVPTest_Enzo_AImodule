from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumbar_mri.axial_v3.guards import find_forbidden_test_references
from lumbar_mri.axial_v3.experiments import expand_low_cost_experiments, load_low_cost_grid
from lumbar_mri.axial_v3.registry import REGISTRY_COLUMNS, registry_row_from_config, upsert_registry_row, read_registry


NOTEBOOKS = [
    ROOT / "notebooks" / "51_axial_v3_iteration_a_raw0_audit.ipynb",
    ROOT / "notebooks" / "52_axial_v3_iteration_b_low_cost.ipynb",
    ROOT / "notebooks" / "53_axial_v3_iteration_c_2_5d.ipynb",
    ROOT / "notebooks" / "54_axial_v3_iteration_d_architectures.ipynb",
]
REQUIRED_DOC = ROOT / "docs" / "axial_v3_improvement_plan.md"
REQUIRED_CONFIG = ROOT / "config" / "axial_v3_low_cost_grid.json"


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
    if path.name.startswith("51_") and "run_iteration_a" not in code_text:
        raise AssertionError("notebook 51 debe orquestar run_iteration_a")
    if path.name.startswith("52_") and "run_training" not in code_text:
        raise AssertionError("notebook 52 debe orquestar runner real B")
    if path.name.startswith("53_") and "require_reliable_slice_order" not in code_text:
        raise AssertionError("notebook 53 debe mantener blocking rule 2.5D")
    if path.name.startswith("54_") and "planned_architecture_metadata" not in code_text:
        raise AssertionError("notebook 54 debe usar metadata de arquitectura segura")
    if "PFI_RUN_AXIAL_V3" in code_text and '== "1"' not in code_text:
        raise AssertionError(f"{path.name}: ejecucion automatica no controlada")
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


def validate_config_and_registry_schema() -> None:
    if not REQUIRED_CONFIG.exists():
        raise AssertionError(f"falta config {REQUIRED_CONFIG}")
    payload = load_low_cost_grid(REQUIRED_CONFIG)
    expanded = expand_low_cost_experiments(payload)
    if len(expanded) < 7:
        raise AssertionError("expansion B0-B6 incompleta")
    with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
        registry_path = Path(temp_dir) / "experiment_registry.csv"
        row = registry_row_from_config(
            {
                "experimentId": "B0",
                "iteration": "B",
                "experimentType": "B0",
                "runId": "axial-v3-B0",
                "createdAtUtc": "2026-07-23T00:00:00Z",
                "updatedAtUtc": "2026-07-23T00:00:00Z",
                "gitCommit": "synthetic",
                "aiServiceCommit": "synthetic",
                "seed": 2026,
                "configPath": str(REQUIRED_CONFIG),
                "configSha256": "synthetic",
                "splitSha256": "synthetic",
                "trainingStatus": "planned",
                "smokeOnly": False,
                "selectedEpoch": None,
                "monitorMetric": "dice_macro_foreground",
            }
        )
        upsert_registry_row(registry_path, row)
        header = registry_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        missing = [column for column in REGISTRY_COLUMNS if column not in header]
        if missing:
            raise AssertionError(f"registry temporal sin columnas requeridas: {missing}")
        loaded = read_registry(registry_path)
        if loaded[0]["experimentId"] != "B0":
            raise AssertionError("registry temporal no recupera fila sintetica")


def main() -> int:
    for path in NOTEBOOKS:
        validate_notebook(path)
    validate_documentation()
    validate_config_and_registry_schema()
    print("validate_axial_v3_notebooks: OK")
    print(f"notebooks={len(NOTEBOOKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
