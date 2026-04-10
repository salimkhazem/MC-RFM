"""Euclidean path/field helpers for ablations."""

from __future__ import annotations

import torch


def linear_interpolate(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    while t.dim() < z0.dim():
        t = t.unsqueeze(-1)
    return (1.0 - t) * z0 + t * z1


def constant_target_field(z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    return z1 - z0

