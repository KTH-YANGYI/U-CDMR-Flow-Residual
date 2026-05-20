from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ucdmr_flow_residual_plus.models.blocks import ConvBlock


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    channels: tuple[int, int, int, int]


class SimpleEncoder(nn.Module):
    def __init__(self, *, base_channels: int = 48) -> None:
        super().__init__()
        c = base_channels
        self.spec = EncoderSpec("simple", (c, c * 2, c * 4, c * 8))
        self.s1 = ConvBlock(3, c)
        self.s2 = ConvBlock(c, c * 2)
        self.s3 = ConvBlock(c * 2, c * 4)
        self.s4 = ConvBlock(c * 4, c * 8)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.s1(x)
        f2 = self.s2(F.avg_pool2d(f1, 2))
        f3 = self.s3(F.avg_pool2d(f2, 2))
        f4 = self.s4(F.avg_pool2d(f3, 2))
        return [f1, f2, f3, f4]


class ResNetEncoder(nn.Module):
    def __init__(self, name: str, *, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import torchvision.models as tvm
        except ModuleNotFoundError as exc:
            raise RuntimeError("torchvision is required for ResNet encoders") from exc
        if name == "resnet50":
            weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
            try:
                model = tvm.resnet50(weights=weights)
            except Exception as exc:
                if pretrained:
                    raise RuntimeError("Could not load pretrained resnet50 weights; cache them on the node or pass --no-pretrained.") from exc
                raise
            channels = (256, 512, 1024, 2048)
        else:
            weights = tvm.ResNet34_Weights.DEFAULT if pretrained else None
            try:
                model = tvm.resnet34(weights=weights)
            except Exception as exc:
                if pretrained:
                    raise RuntimeError("Could not load pretrained resnet34 weights; cache them on the node or pass --no-pretrained.") from exc
                raise
            channels = (64, 128, 256, 512)
        self.spec = EncoderSpec(name, channels)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.pool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stem = self.stem(x)
        f1 = self.layer1(self.pool(stem))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


def build_encoder(name: str, *, pretrained: bool, base_channels: int = 48) -> nn.Module:
    key = name.lower()
    if key in {"simple", "none", "random"}:
        return SimpleEncoder(base_channels=base_channels)
    if key in {"resnet34", "resnet50"}:
        return ResNetEncoder(key, pretrained=pretrained)
    if key in {"convnext_tiny", "convnext", "segformer_b0", "dinov2_small"}:
        raise NotImplementedError(f"Encoder {name} is planned but not implemented in this repo yet; use resnet34/resnet50 or --encoder simple.")
    raise ValueError(f"Unsupported encoder: {name}")
