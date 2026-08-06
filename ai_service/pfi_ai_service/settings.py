from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ServiceSettings:
    """Rutas centralizadas del servicio IA.

    En cloud se espera que los modelos finales esten en models/final,
    configurable con PFI_MODEL_DIR. PFI_ROOT se mantiene para compatibilidad
    con Colab y para resultados/evidencia externos.
    """

    pfi_root: Path
    models_root: Path
    results_root: Path
    figures_root: Path
    docs_root: Path
    output_dir: Path

    sagittal_model_path: Path
    axial_model_path: Path
    sagittal_model_uri: str | None
    axial_model_uri: str | None
    sagittal_manifest_uri: str | None
    axial_manifest_uri: str | None
    model_download_token: str | None
    gcp_project_id: str | None
    sagittal_release_uri: str | None
    sagittal_release_content_sha256: str | None
    sagittal_release_manifest_sha256: str | None
    sagittal_model_sha256: str | None

    subarticular_checkpoint_path: Path | None
    subarticular_device: str | None

    e13_results_root: Path
    e14_results_root: Path


def optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def optional_env_path(name: str, default: Path) -> Path:
    value = optional_env(name)
    return Path(value) if value else default


def get_settings() -> ServiceSettings:
    pfi_root = Path(os.getenv("PFI_ROOT", "/content/drive/MyDrive/PFI_MVP"))
    output_dir = Path(os.getenv("PFI_OUTPUT_DIR", "outputs"))

    models_root = Path(os.getenv("PFI_MODEL_DIR", "models/final"))
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
        sagittal_model_path=optional_env_path(
            "PFI_SAGITTAL_MODEL_PATH",
            models_root / "sagittal_spider_multiclass_final_best.pt",
        ),
        axial_model_path=optional_env_path(
            "PFI_AXIAL_MODEL_PATH",
            models_root / "axial_t2_alkafri_final_v2_candidate.pt",
        ),
        sagittal_model_uri=optional_env("PFI_SAGITTAL_MODEL_URI"),
        axial_model_uri=optional_env("PFI_AXIAL_MODEL_URI"),
        sagittal_manifest_uri=optional_env("PFI_SAGITTAL_MANIFEST_URI"),
        axial_manifest_uri=optional_env("PFI_AXIAL_MANIFEST_URI"),
        model_download_token=optional_env("PFI_MODEL_DOWNLOAD_TOKEN") or optional_env("HF_TOKEN"),
        gcp_project_id=optional_env("PFI_GCP_PROJECT_ID"),
        sagittal_release_uri=optional_env("PFI_SAGITTAL_RELEASE_URI"),
        sagittal_release_content_sha256=optional_env("PFI_SAGITTAL_RELEASE_CONTENT_SHA256"),
        sagittal_release_manifest_sha256=optional_env("PFI_SAGITTAL_RELEASE_MANIFEST_SHA256"),
        sagittal_model_sha256=optional_env("PFI_SAGITTAL_MODEL_SHA256"),
        subarticular_checkpoint_path=(
            Path(value) if (value := optional_env("PFI_SUBARTICULAR_CHECKPOINT_PATH")) else None
        ),
        subarticular_device=optional_env("PFI_SUBARTICULAR_DEVICE") or optional_env("PFI_INFERENCE_DEVICE"),
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
            # Los nombres son los valores de gris de la mascara original, no
            # nombres clinicos, y se dejan asi a proposito: el manifest del artefacto
            # declara esta misma lista y el validador la compara contra el registro.
            # Renombrarlos aca invalidaba el manifest y deshabilitaba el modelo, que
            # es el guard funcionando: el codigo y el artefacto tienen que decir lo
            # mismo sobre lo que se entreno.
            #
            # La traduccion clinica vive en el frontend (clinicalDisplay), que es
            # donde ya se traducen las clases del sagital. Correspondencia del
            # dataset Al-Kafri: raw_0=fondo, raw_50=disco intervertebral,
            # raw_100=elemento posterior, raw_150=saco tecal, raw_200=area
            # anteroposterior. Ojo: el dataset Mendeley "Medical Lumbar Spine 3D
            # Axial MRI" usa otro orden para los mismos valores.
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
