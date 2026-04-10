"""Reference checks used for manual debugging."""

from __future__ import annotations

import torch

from src.geometry.poincare import expmap0, logmap0, project_to_ball


def exp_log_roundtrip_error(x: torch.Tensor, c: float) -> torch.Tensor:
    x = project_to_ball(x, c=c, eps=1.0e-4)
    v = logmap0(x, c=c)
    xr = expmap0(v, c=c)
    return (x - xr).norm(dim=-1)

