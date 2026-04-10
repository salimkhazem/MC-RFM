"""MC-FM vector field with sinusoidal time embedding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(
            0,
            math.log(10000.0),
            steps=max(half, 1),
            device=t.device,
            dtype=t.dtype,
        )
        * (-1.0)
    )
    args = t * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        pad = torch.zeros((emb.shape[0], dim - emb.shape[-1]), device=t.device, dtype=t.dtype)
        emb = torch.cat([emb, pad], dim=-1)
    return emb


class VectorFieldNet(nn.Module):
    def __init__(
        self,
        dh: int,
        de: int,
        hidden_dim: int = 512,
        layers: int = 3,
        time_dim: int = 32,
        decoupled_heads: bool = True,
    ):
        super().__init__()
        self.dh = int(dh)
        self.de = int(de)
        self.time_dim = int(time_dim)
        self.decoupled_heads = bool(decoupled_heads)
        in_dim = self.dh + self.de + self.time_dim

        blocks = []
        d = in_dim
        for _ in range(max(layers - 1, 1)):
            blocks.append(nn.Linear(d, hidden_dim))
            blocks.append(nn.LayerNorm(hidden_dim))
            blocks.append(nn.GELU())
            d = hidden_dim
        self.trunk = nn.Sequential(*blocks)

        if self.decoupled_heads:
            self.head_h = nn.Linear(d, self.dh) if self.dh > 0 else None
            self.head_e = nn.Linear(d, self.de) if self.de > 0 else None
            self.head_joint = None
        else:
            self.head_h = None
            self.head_e = None
            self.head_joint = nn.Linear(d, self.dh + self.de)

    def forward(self, zh: torch.Tensor, ze: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temb = sinusoidal_time_embedding(t, dim=self.time_dim)
        x = torch.cat([zh, ze, temb], dim=-1)
        h = self.trunk(x)
        if self.decoupled_heads:
            vh = self.head_h(h) if self.head_h is not None else torch.zeros_like(zh)
            ve = self.head_e(h) if self.head_e is not None else torch.zeros_like(ze)
            return vh, ve
        assert self.head_joint is not None
        out = self.head_joint(h)
        vh = out[..., : self.dh] if self.dh > 0 else torch.zeros_like(zh)
        ve = out[..., self.dh :] if self.de > 0 else torch.zeros_like(ze)
        return vh, ve

