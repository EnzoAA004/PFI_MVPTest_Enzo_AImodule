from __future__ import annotations

import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "48_axial_final_training_patient_split.ipynb"

REQUIRED = [
    'RUN_MODE: Literal["preflight", "smoke", "full"] = os.getenv("RUN_MODE", "preflight")',
    '"nibabel": "nibabel"',
    "Montaje idempotente de Google Drive",
    "PFI_USE_GOOGLE_DRIVE",
    "preflight_report.json",
    "preflight_ground_truth_overlay.png",
    "class_weight_report",
    "dice_macro_excluding_raw0",
    "iou_macro_excluding_raw0",
    "TEST_EVALUATED_IN_MEMORY",
    "build_axial_model_from_manifest",
    "round_trip_model_from_manifest",
    "humanReviewRequired",
    "notClinicalDiagnosis",
    "fileSize",
    "resolvedPath",
    "sameSizeRows",
    "sha256_stream(Path(df.at[idx, \"resolvedPath\"]))",
    "sliceId",
    "duplicateImages",
    "duplicatePairs",
    "repeatedMasks",
    "maskOnlyWarnings",
    "misma mascara SHA-256 asociada a imagenes diferentes",
    "save_split_snapshot",
    "split_snapshot.csv",
    "required_cell_value",
    "slice_id_from_paths",
    "metrics_from_predictions",
    "_merge_class_counts",
    "confusionMatrix",
    "\"dicePerClass\": {k: v.get(\"dice\")",
    "\"iouPerClass\": {k: v.get(\"iou\")",
    "return manifest",
    "checkpoint_compatibility_errors",
    "RESUME_MODE=auto",
    "dice_macro_excluding_raw0 no es evaluable",
    "ENABLE_SPATIAL_AUGMENTATION",
    "ENABLE_HORIZONTAL_FLIP",
    "AUGMENTATION_EPOCH",
    "smoke selection",
    "RESUME_DIR = CFG.RESUME_ROOT / CFG.RUN_ID",
    "RESUME_DIR / \"axial_t2_alkafri_final.last_checkpoint.pt\"",
    "RESUME_DIR / \"axial_t2_alkafri_final.best_checkpoint.pt\"",
    '"artifactFile": artifact_path.name',
    'f"{artifact_path.name}.manifest.json"',
    'f"{artifact_path.name}.modelcard.md"',
]

FORBIDDEN = [
    r"C:\\Users\\",
    r"gcloud",
    r"gsutil",
    r"upload_from_filename",
    r"delete_blob",
    r"create_bucket",
    r"huggingface_hub",
    r"hf_hub_upload",
    r"RUN_MODE\s*=\s*[\"']full[\"']",
    r"np\.rot90",
    r'"dicePerClass":\s*\{\s*\}',
    r'"iouPerClass":\s*\{\s*\}',
]


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    ids = []
    text = ""
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
        if cell.cell_type == "markdown" and ("outputs" in cell or "execution_count" in cell):
            raise AssertionError(f"markdown cell {index} tiene campos de ejecucion")
        text += "\n" + cell.source
    if len(ids) != len(set(ids)):
        raise AssertionError("ids duplicados")
    for item in REQUIRED:
        if item not in text:
            raise AssertionError(f"falta contenido requerido: {item}")
    for pattern in FORBIDDEN:
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise AssertionError(f"patron prohibido: {pattern}")
    if "0.344" in text or "0.659" in text or "0.817" in text:
        raise AssertionError("metricas historicas no deben quedar hardcodeadas")
    print("validate_axial_final_notebook: OK")
    print(f"cells={len(nb.cells)} code={sum(c.cell_type == 'code' for c in nb.cells)} markdown={sum(c.cell_type == 'markdown' for c in nb.cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
