from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from ucdmr_flow_residual_plus.models.blocks import ConvBlock, resize_like


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim == 1:
        t = t[:, None]
    t = t.reshape(t.shape[0], 1)
    half = dim // 2
    if half == 0:
        return t
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype)
        * -(math.log(10000.0) / max(half - 1, 1))
    )
    args = t * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb


class ResidualFlowUNet(nn.Module):
    """Rectified-flow velocity model for mask-gated crack residuals."""

    def __init__(
        self,
        *,
        residual_channels: int = 3,
        context_channels: int = 3,
        condition_channels: int = 7,
        base_channels: int = 48,
        domain_count: int = 3,
        style_dim: int = 16,
        time_dim: int = 128,
        max_velocity: float = 0.0,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        self.style_dim = int(style_dim)
        self.time_dim = int(time_dim)
        self.max_velocity = float(max_velocity)
        in_channels = residual_channels + context_channels + condition_channels
        self.in_block = ConvBlock(in_channels, c)
        self.down1 = ConvBlock(c, c * 2)
        self.down2 = ConvBlock(c * 2, c * 4)
        self.down3 = ConvBlock(c * 4, c * 8)
        bottleneck = c * 8
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, bottleneck),
            nn.SiLU(inplace=True),
            nn.Linear(bottleneck, bottleneck),
        )
        self.domain_embed = nn.Embedding(domain_count, bottleneck)
        self.style_proj = nn.Linear(style_dim, bottleneck)
        self.mid = ConvBlock(bottleneck, bottleneck)
        self.up2 = ConvBlock(bottleneck + c * 4, c * 4)
        self.up1 = ConvBlock(c * 4 + c * 2, c * 2)
        self.up0 = ConvBlock(c * 2 + c, c)
        self.out = nn.Conv2d(c, residual_channels, kernel_size=1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        condition: torch.Tensor,
        domain_idx: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if style is None:
            style = x_t.new_zeros((x_t.shape[0], self.style_dim))
        x = torch.cat([x_t, context, condition], dim=1)
        f0 = self.in_block(x)
        f1 = self.down1(F.avg_pool2d(f0, 2))
        f2 = self.down2(F.avg_pool2d(f1, 2))
        f3 = self.down3(F.avg_pool2d(f2, 2))
        emb = self.time_proj(sinusoidal_embedding(t.to(dtype=f3.dtype), self.time_dim))
        emb = emb + self.domain_embed(domain_idx).to(dtype=f3.dtype)
        emb = emb + self.style_proj(style.to(dtype=f3.dtype))
        h = self.mid(f3 + emb[:, :, None, None])
        h = resize_like(h, f2)
        h = self.up2(torch.cat([h, f2], dim=1))
        h = resize_like(h, f1)
        h = self.up1(torch.cat([h, f1], dim=1))
        h = resize_like(h, f0)
        h = self.up0(torch.cat([h, f0], dim=1))
        out = self.out(h)
        if self.max_velocity > 0:
            out = torch.tanh(out) * self.max_velocity
        return out
