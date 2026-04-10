"""Mixed-curvature flow-matching adapter."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.geometry.product_manifold import product_project, project_product
from src.models.bottleneck import BottleneckProjector
from src.models.mcfm_vector_field import VectorFieldNet
from src.ode.solver import solve_ode


class CurvatureParameter(nn.Module):
    def __init__(self, initial_c: float, learnable: bool, c_min: float = 1e-3):
        super().__init__()
        self.c_min = float(c_min)
        if learnable:
            inv = torch.log(torch.exp(torch.tensor(initial_c - self.c_min)) - 1.0)
            self.raw = nn.Parameter(inv.clone().detach())
        else:
            self.register_buffer("fixed_c", torch.tensor(float(initial_c)))
            self.raw = None

    def value(self) -> torch.Tensor:
        if self.raw is None:
            return self.fixed_c
        return torch.nn.functional.softplus(self.raw) + self.c_min


class MCRFMAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int,
        dh: int,
        hidden_dim: int,
        layers: int,
        curvature: float,
        learnable_curvature: bool,
        curvature_min: float,
        decoupled_heads: bool,
    ):
        super().__init__()
        self.projector = BottleneckProjector(in_dim=in_dim, bottleneck_dim=bottleneck_dim, dh=dh)
        self.de = bottleneck_dim - dh
        self.curvature_param = CurvatureParameter(
            initial_c=float(curvature),
            learnable=bool(learnable_curvature),
            c_min=float(curvature_min),
        )
        self.vf = VectorFieldNet(
            dh=dh,
            de=self.de,
            hidden_dim=hidden_dim,
            layers=layers,
            decoupled_heads=decoupled_heads,
        )

    def curvature(self) -> float:
        return float(self.curvature_param.value().detach().item())

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, uh, ue = self.projector(features)
        c = float(self.curvature_param.value().detach().item())
        zh, ze = project_product(u, dh=self.projector.dh, c=c)
        return zh, ze, u

    def field(self, zh: torch.Tensor, ze: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vf(zh, ze, t)

    @torch.no_grad()
    def transport(self, zh0: torch.Tensor, ze0: torch.Tensor, solver: str, nfe: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        c = float(self.curvature_param.value().detach().item())
        zh, ze, evals = solve_ode(
            vf=self.field,
            zh0=zh0,
            ze0=ze0,
            solver=solver,
            nfe=nfe,
            curvature=c,
        )
        return zh, ze, evals
