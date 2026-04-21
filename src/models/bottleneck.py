"""Trainable bottleneck projection and feature split."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _inv_softplus(value: float) -> float:
    value = max(float(value), 1.0e-8)
    return math.log(math.expm1(value))


class BottleneckProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int,
        dh: int,
        hyperbolic_scale_init: float = 0.05,
        hyperbolic_scale_min: float = 0.01,
        hyperbolic_scale_max: float = 0.25,
        euclidean_scale_init: float = 1.0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.dh = int(dh)
        if self.dh < 0 or self.dh > self.bottleneck_dim:
            raise ValueError(f"Invalid dh={self.dh} for bottleneck_dim={self.bottleneck_dim}")
        self.de = self.bottleneck_dim - self.dh
        self.hyperbolic_scale_min = float(hyperbolic_scale_min)
        self.hyperbolic_scale_max = float(hyperbolic_scale_max)
        if self.hyperbolic_scale_min <= 0.0 or self.hyperbolic_scale_max <= self.hyperbolic_scale_min:
            raise ValueError("Expected 0 < hyperbolic_scale_min < hyperbolic_scale_max")
        hyper_init = float(min(max(hyperbolic_scale_init, self.hyperbolic_scale_min), self.hyperbolic_scale_max))
        ratio = (hyper_init - self.hyperbolic_scale_min) / (self.hyperbolic_scale_max - self.hyperbolic_scale_min)
        ratio = min(max(ratio, 1.0e-5), 1.0 - 1.0e-5)
        self.hyper_scale_raw = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32))
        self.euclidean_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(euclidean_scale_init), dtype=torch.float32))

        self.hyper_proj = nn.Linear(self.in_dim, self.dh) if self.dh > 0 else None
        self.euclid_proj = nn.Linear(self.in_dim, self.de) if self.de > 0 else None
        self.euclid_norm = nn.LayerNorm(self.de) if self.de > 0 else None

    def hyperbolic_scale(self) -> torch.Tensor:
        ratio = torch.sigmoid(self.hyper_scale_raw)
        return self.hyperbolic_scale_min + (self.hyperbolic_scale_max - self.hyperbolic_scale_min) * ratio

    def euclidean_scale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.euclidean_scale_raw)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.hyper_proj is not None:
            uh_raw = self.hyper_proj(x)
            uh_norm = torch.linalg.vector_norm(uh_raw, dim=-1, keepdim=True).clamp_min(1.0e-12)
            uh = self.hyperbolic_scale().to(dtype=uh_raw.dtype, device=uh_raw.device) * (uh_raw / uh_norm)
        else:
            uh = x.new_zeros(x.shape[:-1] + (0,))

        if self.euclid_proj is not None and self.euclid_norm is not None:
            ue_raw = self.euclid_proj(x)
            ue = self.euclid_norm(ue_raw)
            ue = ue * self.euclidean_scale().to(dtype=ue.dtype, device=ue.device)
        else:
            ue = x.new_zeros(x.shape[:-1] + (0,))

        u = torch.cat([uh, ue], dim=-1)
        return u, uh, ue
