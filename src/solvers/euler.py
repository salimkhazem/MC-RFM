"""Euler solver."""

from __future__ import annotations

from typing import Callable

import torch


def solve_euler(
    vf: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    nfe: int,
    project_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, int]:
    if nfe < 1:
        raise ValueError("nfe must be >= 1")
    dt = 1.0 / float(nfe)
    z = z0
    evals = 0
    for i in range(nfe):
        t_scalar = float(i) / float(nfe)
        t = torch.full((z.shape[0],), t_scalar, device=z.device, dtype=z.dtype)
        v = vf(z, t)
        z = z + dt * v
        evals += 1
        if project_fn is not None:
            z = project_fn(z)
    return z, evals

