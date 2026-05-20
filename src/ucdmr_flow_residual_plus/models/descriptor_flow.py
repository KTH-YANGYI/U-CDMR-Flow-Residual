from __future__ import annotations

import torch
from torch import nn


class DescriptorFlowMLP(nn.Module):
    def __init__(self, *, descriptor_dim: int, condition_dim: int = 3, hidden_dim: int = 128, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = descriptor_dim + condition_dim + 1
        for idx in range(depth):
            layers.append(nn.Linear(in_dim if idx == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, descriptor_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([x_t, t, condition], dim=1))

