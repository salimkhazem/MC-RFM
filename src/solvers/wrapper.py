"""Solver dispatch."""

from __future__ import annotations

from typing import Callable

import torch

from src.solvers.euler import solve_euler
from src.solvers.rk4 import solve_rk4


def solve_ode(
    solver: str,
    vf: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    nfe: int,
    project_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, int]:
    s = solver.lower()
    if s == "euler":
        return solve_euler(vf=vf, z0=z0, nfe=nfe, project_fn=project_fn)
    if s == "rk4":
        return solve_rk4(vf=vf, z0=z0, nfe=nfe, project_fn=project_fn)
    raise ValueError(f"Unknown solver: {solver}")

