import json

import numpy as np
import torch

from pfi_ai_service.model_architectures import SagittalUNet2D
from pfi_ai_service.pipeline import PipelineRunRequest, run_pipeline
from pfi_ai_service.real_inference_runtime import clear_model_cache


def test_pipeline_executes_real_baseline(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "outputs"
    model_dir.mkdir()
    artifact = model_dir / "sagittal_spider_multiclass_final_best.pt"
    model = SagittalUNet2D(num_classes=4, base_channels=2)
    torch.save({"model_state_dict": model.state_dict(), "num_classes": 4, "base_channels": 2, "target_size": [32, 32], "sagittal_axis": 2}, artifact)
    artifact.with_suffix(".pt.manifest.json").write_text(json.dumps({
        "modelKey": "sagittal_spider",
        "version": "test-real-baseline-v1",
        "artifactFile": artifact.name,
        "dataset": "test-dataset",
        "task": "lumbar_mri_segmentation",
        "inputPlane": "sagittal",
        "classes": ["background", "vertebra_group", "canal", "disc_group"],
        "metrics": {"dice": 0.8, "iou": 0.7},
        "trainingStatus": "baseline_evaluated"
    }), encoding="utf-8")
    image_path = tmp_path / "case.npy"
    np.save(image_path, np.random.default_rng(42).normal(size=(40, 40, 9)).astype(np.float32))
    monkeypatch.setenv("PFI_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()

    response = run_pipeline(PipelineRunRequest(
        caseId="CASE-REAL-001",
        plane="sagittal",
        modelKey="sagittal_spider",
        inputPath=str(image_path),
        metadata={"inferenceMode": "real_baseline", "allowContractFallback": False, "traceId": "trace-real-test"},
    ))

    assert response["aiOutput"]["inferenceMode"] == "real_baseline"
    assert response["aiOutput"]["runtime"] == "pytorch"
    assert response["traceId"] == "trace-real-test"
    assert response["metadata"]["outputFiles"]["maskPath"]
    assert response["metadata"]["outputFiles"]["overlayPath"]
