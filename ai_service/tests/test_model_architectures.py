import torch

from pfi_ai_service.model_architectures import AxialUNet2D, SagittalUNet2D, build_checkpoint_model


def test_builds_sagittal_checkpoint_model():
    original = SagittalUNet2D(num_classes=4, base_channels=2)
    checkpoint = {"model_state_dict": original.state_dict(), "num_classes": 4, "base_channels": 2, "target_size": [32, 32]}
    model, metadata = build_checkpoint_model("sagittal_spider", checkpoint)
    output = model(torch.zeros(1, 1, 32, 32))
    assert tuple(output.shape) == (1, 4, 32, 32)
    assert metadata["baseChannels"] == 2


def test_builds_axial_checkpoint_model():
    original = AxialUNet2D(num_classes=6, base_channels=2)
    checkpoint = {"model_state_dict": original.state_dict(), "num_classes": 6, "base_channels": 2, "target_size": [32, 32]}
    model, metadata = build_checkpoint_model("axial_t2_alkafri", checkpoint)
    output = model(torch.zeros(1, 1, 32, 32))
    assert tuple(output.shape) == (1, 6, 32, 32)
    assert metadata["numClasses"] == 6
