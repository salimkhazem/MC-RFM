"""Numerical stability helpers for mixed-curvature training."""

from __future__ import annotations

import torch


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite tensor detected: {name}")


def safe_time_uniform(batch_size: int, device: torch.device, dtype: torch.dtype, eps: float) -> torch.Tensor:
    t = torch.rand(batch_size, device=device, dtype=dtype)
    return torch.clamp(t, min=0.0, max=1.0 - eps)

