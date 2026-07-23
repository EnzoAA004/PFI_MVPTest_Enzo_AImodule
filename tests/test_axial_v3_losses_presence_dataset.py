from __future__ import annotations

import numpy as np
import pytest
import torch

from lumbar_mri.axial_v3.architectures import ArchitectureConfig, build_axial_v3_model, convert_first_conv_1ch_to_3ch
from lumbar_mri.axial_v3.dataset25d import AxialSegmentationDataset25D, SliceRecord25D, neighbor_indices, require_reliable_slice_order
from lumbar_mri.axial_v3.losses import combined_segmentation_loss, tversky_loss
from lumbar_mri.axial_v3.losses import v2_baseline_segmentation_loss
from lumbar_mri.axial_v3.presence import Raw0PresenceHead, presence_targets, total_loss_with_presence


def test_tversky_loss_and_presence_head_are_finite() -> None:
    logits = torch.randn(2, 6, 8, 8, requires_grad=True)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[0, 1:3, 1:3] = 1
    loss = tversky_loss(logits, target, class_index=1, alpha=0.7, beta=0.3)
    assert torch.isfinite(loss)
    total = combined_segmentation_loss(logits, target, raw0_tversky_weight=0.5, raw0_fp_penalty_weight=0.2)
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None

    features = torch.randn(2, 16, 4, 4)
    head = Raw0PresenceHead(16)
    presence_logits = head(features)
    assert list(presence_logits.shape) == [2]
    assert presence_targets(target).tolist() == [1.0, 0.0]
    combined = total_loss_with_presence(total.detach(), presence_logits, target, lambda_presence=0.25)
    assert torch.isfinite(combined)

    b0_total, ce, dice = v2_baseline_segmentation_loss(logits.detach(), target)
    assert torch.isfinite(b0_total)
    assert b0_total == ce + dice


def test_25d_dataset_edges_and_cross_patient_protection() -> None:
    records = [
        SliceRecord25D("p1_0", "m1_0", "train", "p1", "s1", "0", 0, "curated_index"),
        SliceRecord25D("p1_1", "m1_1", "train", "p1", "s1", "1", 1, "curated_index"),
        SliceRecord25D("p2_0", "m2_0", "val", "p2", "s1", "0", 0, "curated_index"),
    ]
    assert neighbor_indices(records, 0) == (0, 0, 1)
    assert neighbor_indices(records, 1) == (0, 1, 1)
    assert neighbor_indices(records, 2) == (2, 2, 2)

    def load_image(record: SliceRecord25D) -> np.ndarray:
        return np.full((4, 4), float(record.order_index), dtype=np.float32)

    def load_mask(record: SliceRecord25D) -> np.ndarray:
        return np.zeros((4, 4), dtype=np.int64)

    dataset = AxialSegmentationDataset25D(records, load_image, load_mask)
    first = dataset[0]
    assert first["image"].shape == (3, 4, 4)
    assert first["patientId"] == "p1"
    assert np.all(first["image"][0] == first["image"][1])


def test_25d_blocks_unknown_or_duplicated_order() -> None:
    missing = [SliceRecord25D("a", "m", "train", "p", "s", "0", None, "curated_index")]
    with pytest.raises(ValueError, match="missing reliable slice order"):
        require_reliable_slice_order(missing)
    duplicated = [
        SliceRecord25D("a", "m", "train", "p", "s", "0", 0, "curated_index"),
        SliceRecord25D("b", "m", "train", "p", "s", "1", 0, "curated_index"),
    ]
    with pytest.raises(ValueError, match="duplicated order index"):
        require_reliable_slice_order(duplicated)
    with pytest.raises(ValueError, match="unreliable order source"):
        require_reliable_slice_order([SliceRecord25D("a", "m", "train", "p", "s", "0", 0, "lexicographic_filename")])


def test_model_runtime_shape_and_first_conv_conversion() -> None:
    model = build_axial_v3_model(ArchitectureConfig(in_channels=3))
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 256, 256))
    assert list(output.shape) == [1, 6, 256, 256]
    assert torch.isfinite(output).all()

    state = {"e1.net.0.weight": torch.ones(4, 1, 3, 3)}
    converted = convert_first_conv_1ch_to_3ch(state)
    assert list(converted["e1.net.0.weight"].shape) == [4, 3, 3, 3]
    assert converted["e1.net.0.weight"].sum().item() == pytest.approx(state["e1.net.0.weight"].sum().item())

    presence_model = build_axial_v3_model(ArchitectureConfig(presence_head=True))
    out = presence_model(torch.randn(2, 1, 64, 64))
    assert set(out) == {"segmentation_logits", "raw0_presence_logits"}
    assert list(out["raw0_presence_logits"].shape) == [2]
