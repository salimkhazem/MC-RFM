"""Classifier heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.geometry.poincare import log_map_0
from src.models.prototypes import product_distance_components2


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _inv_softplus(value: float) -> float:
    value = max(float(value), 1.0e-8)
    return math.log(math.expm1(value))


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(value / (1.0 - value))


class MCRFMClassifier(nn.Module):
    def __init__(
        self,
        dh: int,
        de: int,
        num_classes: int,
        mode: str = "hybrid",
        gamma_h_init: float = 0.1,
        gamma_e_init: float = 1.0,
        beta_init: float = 0.5,
        adaptive_branch_gate: bool = False,
        branch_gate_init: float = 0.5,
        adaptive_beta: bool = False,
        gate_hidden_dim: int = 128,
        task_context_dim: int = 0,
    ):
        super().__init__()
        self.dh = int(dh)
        self.de = int(de)
        self.mode = str(mode).lower()
        self.adaptive_branch_gate = bool(adaptive_branch_gate)
        self.adaptive_beta = bool(adaptive_beta)
        self.task_context_dim = int(task_context_dim)
        self.norm_h = nn.LayerNorm(self.dh) if self.dh > 0 else None
        self.norm_e = nn.LayerNorm(self.de) if self.de > 0 else None
        self.linear = LinearClassifier(self.dh + self.de, num_classes=num_classes)
        self.rho_h = nn.Parameter(torch.tensor(_inv_softplus(gamma_h_init), dtype=torch.float32))
        self.rho_e = nn.Parameter(torch.tensor(_inv_softplus(gamma_e_init), dtype=torch.float32))
        self.raw_beta = nn.Parameter(torch.tensor(_logit(beta_init), dtype=torch.float32))
        self.raw_branch_gate = nn.Parameter(torch.tensor(_logit(branch_gate_init), dtype=torch.float32))
        gate_in_dim = self.dh + self.de
        if gate_in_dim > 0 and self.adaptive_branch_gate:
            self.branch_gate_delta = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden_dim),
                nn.LayerNorm(gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            nn.init.zeros_(self.branch_gate_delta[-1].weight)
            nn.init.zeros_(self.branch_gate_delta[-1].bias)
        else:
            self.branch_gate_delta = None
        self.branch_gate_task_bias = nn.Linear(self.task_context_dim, 1) if self.task_context_dim > 0 else None
        if gate_in_dim > 0 and self.adaptive_beta and self.mode == "hybrid":
            self.beta_delta = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden_dim),
                nn.LayerNorm(gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            nn.init.zeros_(self.beta_delta[-1].weight)
            nn.init.zeros_(self.beta_delta[-1].bias)
        else:
            self.beta_delta = None
        self.beta_task_bias = nn.Linear(self.task_context_dim, 1) if self.task_context_dim > 0 and self.mode == "hybrid" else None

    def gamma_h(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.rho_h)

    def gamma_e(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.rho_e)

    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_beta)

    def branch_gate_prior(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_branch_gate)

    def _normalized_branch_features(self, zh: torch.Tensor, ze: torch.Tensor, c: float) -> tuple[torch.Tensor, torch.Tensor]:
        target_dtype = ze.dtype if ze.numel() > 0 else zh.dtype
        xh = log_map_0(zh, c=c).to(dtype=target_dtype)
        if self.norm_h is not None and xh.shape[-1] > 0:
            xh = self.norm_h(xh)
        if self.norm_e is not None and ze.shape[-1] > 0:
            ze = self.norm_e(ze)
        return xh, ze

    def _prepare_task_context(self, task_context: torch.Tensor | None, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if task_context is None or self.task_context_dim <= 0:
            return None
        if task_context.dim() == 1:
            task_context = task_context.unsqueeze(0).expand(batch, -1)
        elif task_context.shape[0] == 1:
            task_context = task_context.expand(batch, -1)
        return task_context.to(device=device, dtype=dtype)

    def _gating_input(self, zh: torch.Tensor, ze: torch.Tensor, c: float, geometry_mode: str) -> torch.Tensor:
        xh, ze = self._normalized_branch_features(zh, ze, c=c)
        mode = geometry_mode.lower()
        if mode in {"euclidean", "remove_hyper"}:
            xh = torch.zeros_like(xh)
        if mode == "hyperbolic":
            ze = torch.zeros_like(ze)
        return torch.cat([xh, ze], dim=-1)

    def branch_gates(
        self,
        zh: torch.Tensor,
        ze: torch.Tensor,
        c: float,
        geometry_mode: str,
        task_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode = geometry_mode.lower()
        if mode == "hyperbolic":
            return torch.ones((zh.shape[0], 1), device=zh.device, dtype=ze.dtype if ze.numel() > 0 else zh.dtype)
        if mode in {"euclidean", "remove_hyper"}:
            return torch.zeros((zh.shape[0], 1), device=zh.device, dtype=ze.dtype if ze.numel() > 0 else zh.dtype)
        if self.branch_gate_delta is None:
            gate = self.branch_gate_prior().to(device=zh.device, dtype=ze.dtype if ze.numel() > 0 else zh.dtype).expand(zh.shape[0], 1)
            context = self._prepare_task_context(task_context, zh.shape[0], zh.device, gate.dtype)
            if context is not None and self.branch_gate_task_bias is not None:
                gate = torch.sigmoid(torch.logit(gate.clamp(1.0e-5, 1.0 - 1.0e-5)) + self.branch_gate_task_bias(context))
            return gate
        feats = self._gating_input(zh, ze, c=c, geometry_mode=geometry_mode)
        raw = self.raw_branch_gate.to(device=feats.device, dtype=feats.dtype) + self.branch_gate_delta(feats).squeeze(-1)
        context = self._prepare_task_context(task_context, zh.shape[0], feats.device, feats.dtype)
        if context is not None and self.branch_gate_task_bias is not None:
            raw = raw + self.branch_gate_task_bias(context).squeeze(-1)
        return torch.sigmoid(raw).unsqueeze(-1)

    def branch_multipliers(
        self,
        zh: torch.Tensor,
        ze: torch.Tensor,
        c: float,
        geometry_mode: str,
        task_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.branch_gates(zh, ze, c=c, geometry_mode=geometry_mode, task_context=task_context)
        mode = geometry_mode.lower()
        if mode == "hyperbolic":
            return torch.ones_like(gate), torch.zeros_like(gate)
        if mode in {"euclidean", "remove_hyper"}:
            return torch.zeros_like(gate), torch.ones_like(gate)
        return 2.0 * gate, 2.0 * (1.0 - gate)

    def beta_values(
        self,
        zh: torch.Tensor,
        ze: torch.Tensor,
        c: float,
        geometry_mode: str,
        task_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode != "hybrid":
            return torch.ones((zh.shape[0], 1), device=zh.device, dtype=ze.dtype if ze.numel() > 0 else zh.dtype)
        if self.beta_delta is None:
            beta = self.beta().to(device=zh.device, dtype=ze.dtype if ze.numel() > 0 else zh.dtype).expand(zh.shape[0], 1)
            context = self._prepare_task_context(task_context, zh.shape[0], zh.device, beta.dtype)
            if context is not None and self.beta_task_bias is not None:
                beta = torch.sigmoid(torch.logit(beta.clamp(1.0e-5, 1.0 - 1.0e-5)) + self.beta_task_bias(context))
            return beta
        feats = self._gating_input(zh, ze, c=c, geometry_mode=geometry_mode)
        raw = self.raw_beta.to(device=feats.device, dtype=feats.dtype) + self.beta_delta(feats).squeeze(-1)
        context = self._prepare_task_context(task_context, zh.shape[0], feats.device, feats.dtype)
        if context is not None and self.beta_task_bias is not None:
            raw = raw + self.beta_task_bias(context).squeeze(-1)
        return torch.sigmoid(raw).unsqueeze(-1)

    def feature_representation(
        self,
        zh: torch.Tensor,
        ze: torch.Tensor,
        c: float,
        geometry_mode: str,
        task_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xh, ze = self._normalized_branch_features(zh, ze, c=c)
        mult_h, mult_e = self.branch_multipliers(zh, ze, c=c, geometry_mode=geometry_mode, task_context=task_context)
        xh = xh * mult_h.sqrt()
        ze = ze * mult_e.sqrt()
        return torch.cat([xh, ze], dim=-1)

    def prototype_logits(
        self,
        zh: torch.Tensor,
        ze: torch.Tensor,
        proto_h: torch.Tensor,
        proto_e: torch.Tensor,
        c: float,
        geometry_mode: str,
        task_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dh2, de2 = product_distance_components2(zh, ze, proto_h, proto_e, c=c)
        mult_h, mult_e = self.branch_multipliers(zh, ze, c=c, geometry_mode=geometry_mode, task_context=task_context)
        gamma_h = self.gamma_h().to(dtype=dh2.dtype, device=dh2.device)
        gamma_e = self.gamma_e().to(dtype=de2.dtype, device=de2.device)
        return -(mult_h.to(dtype=dh2.dtype) * gamma_h * dh2 + mult_e.to(dtype=de2.dtype) * gamma_e * de2)

    def linear_logits(
        self,
        zh: torch.Tensor,
        ze: torch.Tensor,
        c: float,
        geometry_mode: str,
        task_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self.feature_representation(zh, ze, c=c, geometry_mode=geometry_mode, task_context=task_context)
        return self.linear(feats)

    def forward(
        self,
        zh: torch.Tensor,
        ze: torch.Tensor,
        proto_h: torch.Tensor,
        proto_e: torch.Tensor,
        c: float,
        geometry_mode: str,
        task_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        proto = self.prototype_logits(zh, ze, proto_h, proto_e, c=c, geometry_mode=geometry_mode, task_context=task_context)
        if self.mode == "prototype":
            return proto
        linear = self.linear_logits(zh, ze, c=c, geometry_mode=geometry_mode, task_context=task_context)
        if self.mode == "linear":
            return linear
        beta = self.beta_values(zh, ze, c=c, geometry_mode=geometry_mode, task_context=task_context).to(dtype=linear.dtype, device=linear.device)
        return beta * proto + (1.0 - beta) * linear
