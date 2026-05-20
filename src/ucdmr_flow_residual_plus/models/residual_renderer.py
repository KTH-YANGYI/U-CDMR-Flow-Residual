from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ucdmr_flow_residual_plus.models.blocks import ConvBlock, resize_like
from ucdmr_flow_residual_plus.models.encoders import build_encoder


class ResidualRendererPlus(nn.Module):
    """Pretrained-encoder residual renderer with mask/style conditioning."""

    def __init__(
        self,
        *,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        base_channels: int = 48,
        condition_channels: int = 7,
        domain_count: int = 3,
        style_dim: int = 16,
        max_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(encoder_name, pretrained=pretrained, base_channels=base_channels)
        channels = self.encoder.spec.channels
        self.max_delta = float(max_delta)
        self.style_dim = int(style_dim)
        bottleneck = channels[-1]
        self.condition_proj = nn.Sequential(
            nn.Conv2d(condition_channels, min(128, bottleneck), kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(min(128, bottleneck), bottleneck, kernel_size=1),
        )
        self.domain_embed = nn.Embedding(domain_count, bottleneck)
        self.style_proj = nn.Linear(style_dim, bottleneck)
        self.mid = ConvBlock(bottleneck, bottleneck)
        self.up3 = ConvBlock(bottleneck + channels[-2], channels[-2])
        self.up2 = ConvBlock(channels[-2] + channels[-3], channels[-3])
        self.up1 = ConvBlock(channels[-3] + channels[-4], channels[-4])
        self.out = nn.Sequential(
            ConvBlock(channels[-4], max(base_channels, 32)),
            nn.Conv2d(max(base_channels, 32), 3, kernel_size=1),
        )

    def forward(
        self,
        image: torch.Tensor,
        condition: torch.Tensor,
        domain_idx: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self.encoder(image)
        if style is None:
            style = image.new_zeros((image.shape[0], self.style_dim))
        cond = F.interpolate(condition, size=feats[-1].shape[-2:], mode="bilinear", align_corners=False)
        bottleneck = feats[-1] + self.condition_proj(cond)
        bottleneck = bottleneck + self.domain_embed(domain_idx).to(bottleneck.dtype)[:, :, None, None]
        bottleneck = bottleneck + self.style_proj(style.to(bottleneck.dtype))[:, :, None, None]
        x = self.mid(bottleneck)
        x = resize_like(x, feats[-2])
        x = self.up3(torch.cat([x, feats[-2]], dim=1))
        x = resize_like(x, feats[-3])
        x = self.up2(torch.cat([x, feats[-3]], dim=1))
        x = resize_like(x, feats[-4])
        x = self.up1(torch.cat([x, feats[-4]], dim=1))
        x = F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return torch.tanh(self.out(x)) * self.max_delta


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
