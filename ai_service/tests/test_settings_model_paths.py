from __future__ import annotations

from pathlib import Path

from pfi_ai_service.model_manifest import manifest_path_for_artifact
from pfi_ai_service.settings import get_settings


def test_model_paths_can_be_overridden_independently(monkeypatch, tmp_path: Path) -> None:
    models_root = tmp_path / "models" / "final"
    sagittal_path = models_root / "custom_sagittal.pt"
    axial_path = models_root / "axial_t2_alkafri_final_v2_candidate.pt"
    monkeypatch.setenv("PFI_MODEL_DIR", str(models_root))
    monkeypatch.setenv("PFI_SAGITTAL_MODEL_PATH", str(sagittal_path))
    monkeypatch.setenv("PFI_AXIAL_MODEL_PATH", str(axial_path))

    settings = get_settings()

    assert settings.models_root == models_root
    assert settings.sagittal_model_path == sagittal_path
    assert settings.axial_model_path == axial_path
    assert manifest_path_for_artifact(settings.axial_model_path).name == "axial_t2_alkafri_final_v2_candidate.pt.manifest.json"


def test_model_paths_default_to_model_dir(monkeypatch, tmp_path: Path) -> None:
    models_root = tmp_path / "models" / "final"
    monkeypatch.setenv("PFI_MODEL_DIR", str(models_root))
    monkeypatch.delenv("PFI_SAGITTAL_MODEL_PATH", raising=False)
    monkeypatch.delenv("PFI_AXIAL_MODEL_PATH", raising=False)

    settings = get_settings()

    assert settings.sagittal_model_path == models_root / "sagittal_spider_multiclass_final_best.pt"
    assert settings.axial_model_path == models_root / "axial_t2_alkafri_final_best.pt"
