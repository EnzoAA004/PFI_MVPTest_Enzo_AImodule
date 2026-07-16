from __future__ import annotations

import torch

from pfi_ai_service.model_architectures import AxialUNet2D, checkpoint_state_dict
from pfi_ai_service.model_artifacts import model_artifact_path


def test_axial_final_checkpoint_loads_strict_and_eval() -> None:
    artifact_path = model_artifact_path("axial_t2_alkafri")
    assert artifact_path is not None
    assert artifact_path.exists()

    checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=False)
    state = checkpoint_state_dict(checkpoint)
    model = AxialUNet2D(num_classes=6, base_channels=16)

    result = model.load_state_dict(state, strict=True)
    model.eval()

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert model.training is False

    # Smoke-test only: synthetic tensor with provisional size, not target_size evidence and not real_baseline.
    with torch.inference_mode():
        output = model(torch.randn(1, 1, 256, 256))

    assert tuple(output.shape) == (1, 6, 256, 256)
