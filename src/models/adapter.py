"""Mixed-curvature flow-matching adapter."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.geometry.poincare import log_map_0
from src.geometry.product_manifold import project_product
from src.models.bottleneck import BottleneckProjector
from src.models.mcfm_vector_field import VectorFieldNet
from src.models.task_conditioning import TaskSignatureEncoder
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
        geometry_mode: str,
        hyperbolic_scale_init: float,
        hyperbolic_scale_min: float,
        hyperbolic_scale_max: float,
        euclidean_scale_init: float,
        vector_field_h_input: str,
        task_conditioning: bool = False,
        task_context_dim: int = 128,
        task_context_hidden_dim: int = 128,
    ):
        super().__init__()
        self.geometry_mode = str(geometry_mode).lower()
        self.vector_field_h_input = str(vector_field_h_input).lower()
        self.task_conditioning = bool(task_conditioning)
        self.task_context_dim = int(task_context_dim) if self.task_conditioning else 0
        self.projector = BottleneckProjector(
            in_dim=in_dim,
            bottleneck_dim=bottleneck_dim,
            dh=dh,
            hyperbolic_scale_init=hyperbolic_scale_init,
            hyperbolic_scale_min=hyperbolic_scale_min,
            hyperbolic_scale_max=hyperbolic_scale_max,
            euclidean_scale_init=euclidean_scale_init,
        )
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
            context_dim=self.task_context_dim,
        )
        self.task_encoder = (
            TaskSignatureEncoder(dh=dh, de=self.de, context_dim=self.task_context_dim, hidden_dim=int(task_context_hidden_dim))
            if self.task_conditioning
            else None
        )

    def curvature(self) -> float:
        return float(self.curvature_param.value().detach().item())

    def hyperbolic_scale(self) -> float:
        return float(self.projector.hyperbolic_scale().detach().item())

    def euclidean_scale(self) -> float:
        return float(self.projector.euclidean_scale().detach().item())

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, uh, ue = self.projector(features)
        c = float(self.curvature_param.value().detach().item())
        zh, ze = project_product(u, dh=self.projector.dh, c=c)
        return zh, ze, u

    def encode_task_context(self, proto_h: torch.Tensor, proto_e: torch.Tensor) -> torch.Tensor | None:
        if self.task_encoder is None:
            return None
        return self.task_encoder(proto_h, proto_e, c=self.curvature())

    def chart_h(self, zh: torch.Tensor) -> torch.Tensor:
        return log_map_0(zh, c=self.curvature())

    def field(
        self,
        zh: torch.Tensor,
        ze: torch.Tensor,
        t: torch.Tensor,
        task_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_h = self.chart_h(zh) if self.vector_field_h_input == "logmap0" else zh
        vh, ve = self.vf(input_h, ze, t, context=task_context)
        if self.geometry_mode in {"euclidean", "remove_hyper"}:
            vh = torch.zeros_like(vh)
        if self.geometry_mode == "hyperbolic":
            ve = torch.zeros_like(ve)
        return vh, ve

    def transport(
        self,
        zh0: torch.Tensor,
        ze0: torch.Tensor,
        solver: str,
        nfe: int,
        task_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        c = float(self.curvature_param.value().detach().item())
        def _vf(zh: torch.Tensor, ze: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.field(zh, ze, t, task_context=task_context)
        zh, ze, evals = solve_ode(
            vf=_vf,
            zh0=zh0,
            ze0=ze0,
            solver=solver,
            nfe=nfe,
            curvature=c,
            hyper_state_mode=self.vector_field_h_input,
        )
        return zh, ze, evals
