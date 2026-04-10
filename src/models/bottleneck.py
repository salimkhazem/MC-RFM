"""Trainable bottleneck projection and feature split."""

from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckProjector(nn.Module):
    def __init__(self, in_dim: int, bottleneck_dim: int, dh: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.dh = int(dh)
        if self.dh < 0 or self.dh > self.bottleneck_dim:
            raise ValueError(f"Invalid dh={self.dh} for bottleneck_dim={self.bottleneck_dim}")
        self.de = self.bottleneck_dim - self.dh
        self.proj = nn.Linear(self.in_dim, self.bottleneck_dim)
        self.norm = nn.LayerNorm(self.bottleneck_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.norm(self.proj(x))
        uh = u[..., : self.dh]
        ue = u[..., self.dh :]
        return u, uh, ue

