from __future__ import annotations

import torch

from pfi_ai_service.model_architectures import SagittalUNet2D
from pfi_ai_service.model_artifacts import model_artifact_path


def test_sagittal_final_checkpoint_loads_strict_and_eval() -> None:
    artifact_path = model_artifact_path("sagittal_spider")
    assert artifact_path is not None
    assert artifact_path.exists()

    checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    model = SagittalUNet2D(num_classes=4, base_channels=16)

    result = model.load_state_dict(state, strict=True)
    model.eval()

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert model.training is False

    # Smoke-test only: synthetic tensor, not a real input and not a real baseline.
    with torch.inference_mode():
        output = model(torch.randn(1, 1, 256, 256))

    assert tuple(output.shape) == (1, 4, 256, 256)
