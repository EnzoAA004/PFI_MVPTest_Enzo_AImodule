"""Small architecture registry for axial v3 validation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


class AxialDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AxialUNet2DV3(nn.Module):
    """Axial U-Net compatible with v2 shape, optionally using 2.5D input channels."""

    def __init__(self, in_channels: int = 1, num_classes: int = 6, base_channels: int = 16) -> None:
        super().__init__()
        b = base_channels
        self.e1 = AxialDoubleConv(in_channels, b)
        self.e2 = AxialDoubleConv(b, b * 2)
        self.e3 = AxialDoubleConv(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)
        self.mid = AxialDoubleConv(b * 4, b * 8)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.d3 = AxialDoubleConv(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.d2 = AxialDoubleConv(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.d1 = AxialDoubleConv(b * 2, b)
        self.out = nn.Conv2d(b, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        mid = self.mid(self.pool(e3))
        x = self.d3(torch.cat([self.u3(mid), e3], dim=1))
        x = self.d2(torch.cat([self.u2(x), e2], dim=1))
        x = self.d1(torch.cat([self.u1(x), e1], dim=1))
        return self.out(x)


@dataclass(frozen=True)
class ArchitectureConfig:
    name: Literal["axial_unet2d", "attention_unet_light", "unetpp_reduced"] = "axial_unet2d"
    in_channels: int = 1
    num_classes: int = 6
    base_channels: int = 16


def build_axial_v3_model(config: ArchitectureConfig) -> nn.Module:
    if config.name != "axial_unet2d":
        raise NotImplementedError(
            f"{config.name} is documented for Iteration D but intentionally not enabled by default"
        )
    return AxialUNet2DV3(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        base_channels=config.base_channels,
    )


def convert_first_conv_1ch_to_3ch(state_dict: dict[str, torch.Tensor], key: str = "e1.net.0.weight") -> dict[str, torch.Tensor]:
    """Replicate a 1-channel first convolution across three input channels."""

    converted = dict(state_dict)
    weight = converted.get(key)
    if weight is None:
        raise KeyError(key)
    if weight.ndim != 4 or weight.shape[1] != 1:
        raise ValueError(f"expected first conv weight [out,1,k,k], got {tuple(weight.shape)}")
    converted[key] = weight.repeat(1, 3, 1, 1) / 3.0
    return converted
