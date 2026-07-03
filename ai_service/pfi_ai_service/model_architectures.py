from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class SagittalDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SagittalUNet2D(nn.Module):
    """Arquitectura exacta del checkpoint sagital E5/E12."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = SagittalDoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = SagittalDoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = SagittalDoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = SagittalDoubleConv(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = SagittalDoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = SagittalDoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = SagittalDoubleConv(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bottleneck = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


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


class AxialUNet2D(nn.Module):
    """Arquitectura exacta del entrenamiento axial E10."""

    def __init__(self, num_classes: int = 6, base_channels: int = 16) -> None:
        super().__init__()
        b = base_channels
        self.e1 = AxialDoubleConv(1, b)
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


def checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return normalize_state_dict(value)
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return normalize_state_dict(checkpoint)
    raise ValueError("El checkpoint no contiene model_state_dict/state_dict utilizable")


def normalize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = str(key)
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        normalized[clean_key] = value
    return normalized


def infer_base_channels(model_key: str, checkpoint: Any, state_dict: Mapping[str, torch.Tensor]) -> int:
    if isinstance(checkpoint, Mapping) and checkpoint.get("base_channels") is not None:
        return int(checkpoint["base_channels"])
    weight_key = "enc1.block.0.weight" if model_key == "sagittal_spider" else "e1.net.0.weight"
    weight = state_dict.get(weight_key)
    return int(weight.shape[0]) if weight is not None else 16


def infer_num_classes(model_key: str, checkpoint: Any, state_dict: Mapping[str, torch.Tensor]) -> int:
    if isinstance(checkpoint, Mapping) and checkpoint.get("num_classes") is not None:
        return int(checkpoint["num_classes"])
    weight_key = "out_conv.weight" if model_key == "sagittal_spider" else "out.weight"
    weight = state_dict.get(weight_key)
    fallback = 4 if model_key == "sagittal_spider" else 6
    return int(weight.shape[0]) if weight is not None else fallback


def build_checkpoint_model(model_key: str, checkpoint: Any) -> tuple[nn.Module, dict[str, Any]]:
    state_dict = checkpoint_state_dict(checkpoint)
    base_channels = infer_base_channels(model_key, checkpoint, state_dict)
    num_classes = infer_num_classes(model_key, checkpoint, state_dict)
    if model_key == "sagittal_spider":
        model: nn.Module = SagittalUNet2D(num_classes=num_classes, base_channels=base_channels)
    elif model_key == "axial_t2_alkafri":
        model = AxialUNet2D(num_classes=num_classes, base_channels=base_channels)
    else:
        raise KeyError(f"Arquitectura no registrada para model_key={model_key}")
    model.load_state_dict(state_dict, strict=True)
    target_size = (256, 256)
    if isinstance(checkpoint, Mapping) and checkpoint.get("target_size") is not None:
        raw_size = checkpoint["target_size"]
        target_size = (int(raw_size[0]), int(raw_size[1]))
    return model, {
        "baseChannels": base_channels,
        "numClasses": num_classes,
        "targetSize": target_size,
        "checkpointKeys": sorted(str(key) for key in checkpoint.keys()) if isinstance(checkpoint, Mapping) else [],
    }
