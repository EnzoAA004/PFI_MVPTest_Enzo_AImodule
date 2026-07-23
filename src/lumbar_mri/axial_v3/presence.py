"""Optional raw_0 presence head for axial v3 experiments."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Raw0PresenceHead(nn.Module):
    """Small binary head over encoder or bottleneck feature maps."""

    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


def presence_targets(mask: torch.Tensor, *, raw0_index: int = 1) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"mask must be [N,H,W], got {tuple(mask.shape)}")
    return (mask == raw0_index).flatten(1).any(dim=1).to(dtype=torch.float32)


def total_loss_with_presence(
    segmentation_loss: torch.Tensor,
    presence_logits: torch.Tensor | None,
    mask: torch.Tensor,
    *,
    lambda_presence: float = 0.0,
    raw0_index: int = 1,
) -> torch.Tensor:
    if lambda_presence == 0 or presence_logits is None:
        return segmentation_loss
    target = presence_targets(mask, raw0_index=raw0_index).to(device=presence_logits.device)
    return segmentation_loss + lambda_presence * F.binary_cross_entropy_with_logits(presence_logits, target)
