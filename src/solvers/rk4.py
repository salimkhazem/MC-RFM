"""RK4 solver."""

from __future__ import annotations

from typing import Callable

import torch


def solve_rk4(
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
        t0 = float(i) / float(nfe)
        t = torch.full((z.shape[0],), t0, device=z.device, dtype=z.dtype)
        half_t = torch.full((z.shape[0],), t0 + 0.5 * dt, device=z.device, dtype=z.dtype)
        t1 = torch.full((z.shape[0],), t0 + dt, device=z.device, dtype=z.dtype)
        k1 = vf(z, t)
        k2 = vf(z + 0.5 * dt * k1, half_t)
        k3 = vf(z + 0.5 * dt * k2, half_t)
        k4 = vf(z + dt * k3, t1)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        evals += 4
        if project_fn is not None:
            z = project_fn(z)
    return z, evals

