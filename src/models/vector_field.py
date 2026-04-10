"""Vector field models for flow matching."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.geometry.poincare import project_tangent_orthogonal, radial_direction
from src.models.mlp import TimeConcatMLP


class EuclideanVectorField(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.net = TimeConcatMLP(dim=dim, out_dim=dim, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(z, t)


class DecoupledHyperbolicVectorField(nn.Module):
    """
    PD-HFM vector field with radial/angular decomposition.

    v(z, t) = a(z, t) * u_r(z) + P_perp(z) b(z, t)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        coupled: bool = False,
    ):
        super().__init__()
        self.coupled = coupled
        if coupled:
            self.coupled_head = TimeConcatMLP(dim=dim, out_dim=dim, hidden_dim=hidden_dim, num_layers=num_layers)
        else:
            self.radial_head = TimeConcatMLP(dim=dim, out_dim=1, hidden_dim=hidden_dim, num_layers=num_layers)
            self.angular_head = TimeConcatMLP(dim=dim, out_dim=dim, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.coupled:
            return self.coupled_head(z, t)
        a = self.radial_head(z, t)  # [B,1]
        b = self.angular_head(z, t)  # [B,D]
        ur = radial_direction(z)
        angular = project_tangent_orthogonal(z, b)
        return a * ur + angular

