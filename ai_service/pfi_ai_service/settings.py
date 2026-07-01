from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ServiceSettings:
    """Rutas centralizadas del servicio IA.

    En Colab se asume /content/drive/MyDrive/PFI_MVP.
    En backend/MVP se puede configurar con la variable PFI_ROOT.
    """

    pfi_root: Path
    models_root: Path
    results_root: Path
    figures_root: Path
    docs_root: Path
    output_dir: Path

    sagittal_model_path: Path
    axial_model_path: Path

    e13_results_root: Path
    e14_results_root: Path


def get_settings() -> ServiceSettings:
    pfi_root = Path(os.getenv("PFI_ROOT", "/content/drive/MyDrive/PFI_MVP"))
    output_dir = Path(os.getenv("PFI_OUTPUT_DIR", "outputs"))

    models_root = pfi_root / "models"
    results_root = pfi_root / "results"
    figures_root = pfi_root / "figures"
    docs_root = pfi_root / "docs"

    return ServiceSettings(
        pfi_root=pfi_root,
        models_root=models_root,
        results_root=results_root,
        figures_root=figures_root,
        docs_root=docs_root,
        output_dir=output_dir,
        sagittal_model_path=models_root / "E12_sagittal_multiclass_final_best.pt",
        axial_model_path=models_root / "E10_axial_t2_final_training_clean_best.pt",
        e13_results_root=results_root / "E13_multiplanar_inference_pipeline",
        e14_results_root=results_root / "E14_ai_agent_orchestrator",
    )


MODEL_REGISTRY = {
    "sagittal_spider": {
        "plane": "sagittal",
        "num_classes": 4,
        "class_names": {
            0: "background",
            1: "vertebra_group",
            2: "canal",
            3: "disc_group",
        },
        "human_review_required": True,
    },
    "axial_t2_alkafri": {
        "plane": "axial",
        "num_classes": 6,
        "class_names": {
            0: "background_250",
            1: "raw_0",
            2: "raw_50",
            3: "raw_100",
            4: "raw_150",
            5: "raw_200",
        },
        "human_review_required": True,
    },
}
