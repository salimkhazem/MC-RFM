"""MLP blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers = []
        d_in = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeConcatMLP(nn.Module):
    """Concatenate scalar time t to latent state and apply MLP."""

    def __init__(self, dim: int, out_dim: int, hidden_dim: int = 512, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(input_dim=dim + 1, output_dim=out_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        while t.dim() < z.dim():
            t = t.unsqueeze(-1)
        t = t.expand(z.shape[0], 1)
        return self.mlp(torch.cat([z, t], dim=-1))

