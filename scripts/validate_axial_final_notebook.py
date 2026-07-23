from __future__ import annotations

import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "48_axial_final_training_patient_split.ipynb"

REQUIRED = [
    'RUN_MODE: Literal["preflight", "smoke", "full"] = os.getenv("RUN_MODE", "preflight")',
    "RAW_TO_CLASS_INDEX",
    "patientId",
    "validate_no_duplicate_leakage",
    "dice_macro_foreground",
    "iou_macro_foreground",
    "raw_0",
    "qualityGatePassed",
    "humanReviewRequired",
    "notClinicalDiagnosis",
]

FORBIDDEN = [
    r"C:\\Users\\",
    r"gcloud compute instances create",
    r"upload_from_filename",
    r"delete_blob",
    r"create_bucket",
    r"huggingface_hub",
    r"hf_hub_upload",
    r"RUN_MODE\s*=\s*[\"']full[\"']",
]


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    ids = []
    text_parts = []
    for index, cell in enumerate(nb.cells):
        if not cell.get("id"):
            raise AssertionError(f"cell {index} sin id")
        ids.append(cell["id"])
        if cell.cell_type == "code":
            if cell.get("outputs") != []:
                raise AssertionError(f"cell {index} tiene outputs")
            if cell.get("execution_count") is not None:
                raise AssertionError(f"cell {index} tiene execution_count")
            compile(cell.source, f"cell_{index}", "exec")
        if cell.cell_type == "markdown":
            if "outputs" in cell or "execution_count" in cell:
                raise AssertionError(f"markdown cell {index} tiene campos de ejecucion")
        text_parts.append(cell.source)
    if len(ids) != len(set(ids)):
        raise AssertionError("ids duplicados")
    text = "\n".join(text_parts)
    for item in REQUIRED:
        if item not in text:
            raise AssertionError(f"falta contenido requerido: {item}")
    for pattern in FORBIDDEN:
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise AssertionError(f"patron prohibido: {pattern}")
    if "0.344" in text or "0.659" in text or "0.817" in text:
        raise AssertionError("metricas historicas no deben quedar hardcodeadas en el notebook final")
    print("validate_axial_final_notebook: OK")
    print(f"cells={len(nb.cells)} code={sum(c.cell_type == 'code' for c in nb.cells)} markdown={sum(c.cell_type == 'markdown' for c in nb.cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
