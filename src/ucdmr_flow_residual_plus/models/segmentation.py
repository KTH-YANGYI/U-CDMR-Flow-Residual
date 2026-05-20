from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ucdmr_flow_residual_plus.models.blocks import ConvBlock, resize_like
from ucdmr_flow_residual_plus.models.encoders import build_encoder


class SegmenterPlus(nn.Module):
    """Pretrained-encoder binary segmentation decoder."""

    def __init__(
        self,
        *,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        base_channels: int = 48,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(encoder_name, pretrained=pretrained, base_channels=base_channels)
        channels = self.encoder.spec.channels
        self.mid = ConvBlock(channels[-1], channels[-1])
        self.up3 = ConvBlock(channels[-1] + channels[-2], channels[-2])
        self.up2 = ConvBlock(channels[-2] + channels[-3], channels[-3])
        self.up1 = ConvBlock(channels[-3] + channels[-4], channels[-4])
        self.out = nn.Sequential(
            ConvBlock(channels[-4], max(base_channels, 32)),
            nn.Conv2d(max(base_channels, 32), out_channels, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(image)
        x = self.mid(feats[-1])
        x = resize_like(x, feats[-2])
        x = self.up3(torch.cat([x, feats[-2]], dim=1))
        x = resize_like(x, feats[-3])
        x = self.up2(torch.cat([x, feats[-3]], dim=1))
        x = resize_like(x, feats[-4])
        x = self.up1(torch.cat([x, feats[-4]], dim=1))
        x = F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(x)
