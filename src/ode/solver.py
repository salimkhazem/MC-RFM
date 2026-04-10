"""Fixed-step ODE solvers for product manifold dynamics."""

from __future__ import annotations

from typing import Callable

import torch

from src.geometry.product_manifold import product_project


FieldFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _time(batch: int, t_scalar: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((batch,), float(t_scalar), device=device, dtype=dtype)


def solve_euler(vf: FieldFn, zh0: torch.Tensor, ze0: torch.Tensor, nfe: int, curvature: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    dt = 1.0 / float(max(nfe, 1))
    zh, ze = zh0, ze0
    evals = 0
    for i in range(max(nfe, 1)):
        t = _time(zh.shape[0], i / max(nfe, 1), zh.device, zh.dtype)
        vh, ve = vf(zh, ze, t)
        zh = zh + dt * vh
        ze = ze + dt * ve
        zh, ze = product_project(zh, ze, c=curvature)
        evals += 1
    return zh, ze, evals


def solve_rk4(vf: FieldFn, zh0: torch.Tensor, ze0: torch.Tensor, nfe: int, curvature: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    dt = 1.0 / float(max(nfe, 1))
    zh, ze = zh0, ze0
    evals = 0
    for i in range(max(nfe, 1)):
        t0 = i / max(nfe, 1)
        t = _time(zh.shape[0], t0, zh.device, zh.dtype)
        t_half = _time(zh.shape[0], t0 + 0.5 * dt, zh.device, zh.dtype)
        t1 = _time(zh.shape[0], t0 + dt, zh.device, zh.dtype)
        k1h, k1e = vf(zh, ze, t)
        k2h, k2e = vf(zh + 0.5 * dt * k1h, ze + 0.5 * dt * k1e, t_half)
        k3h, k3e = vf(zh + 0.5 * dt * k2h, ze + 0.5 * dt * k2e, t_half)
        k4h, k4e = vf(zh + dt * k3h, ze + dt * k3e, t1)
        zh = zh + (dt / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h)
        ze = ze + (dt / 6.0) * (k1e + 2 * k2e + 2 * k3e + k4e)
        zh, ze = product_project(zh, ze, c=curvature)
        evals += 4
    return zh, ze, evals


def solve_ode(
    vf: FieldFn,
    zh0: torch.Tensor,
    ze0: torch.Tensor,
    solver: str,
    nfe: int,
    curvature: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    name = solver.lower()
    if name == "euler":
        return solve_euler(vf=vf, zh0=zh0, ze0=ze0, nfe=nfe, curvature=curvature)
    if name == "rk4":
        return solve_rk4(vf=vf, zh0=zh0, ze0=ze0, nfe=nfe, curvature=curvature)
    raise ValueError(f"Unsupported solver: {solver}")

