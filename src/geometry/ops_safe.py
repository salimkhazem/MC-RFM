"""Numerical safety helpers for geometry."""

from __future__ import annotations

import torch


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1.0e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=eps))


def safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return num / torch.clamp(den, min=eps)


def safe_atanh(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    clipped = torch.clamp(x, min=-1.0 + eps, max=1.0 - eps)
    return 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))


def has_nan_or_inf(t: torch.Tensor) -> bool:
    return bool((~torch.isfinite(t)).any().item())

