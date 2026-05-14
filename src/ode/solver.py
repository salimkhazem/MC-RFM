"""Fixed-step ODE solvers for product manifold dynamics."""

from __future__ import annotations

from typing import Callable

import torch

from src.geometry.product_manifold import product_project
from src.geometry.poincare import exp_map_0, log_map_0


FieldFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _time(batch: int, t_scalar: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((batch,), float(t_scalar), device=device, dtype=dtype)


def _state_to_ball(xh: torch.Tensor, zh: torch.Tensor, curvature: float, hyper_state_mode: str) -> torch.Tensor:
    if hyper_state_mode == "logmap0":
        return exp_map_0(xh, c=curvature)
    return zh


def solve_euler(
    vf: FieldFn,
    zh0: torch.Tensor,
    ze0: torch.Tensor,
    nfe: int,
    curvature: float,
    hyper_state_mode: str = "ball",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    dt = 1.0 / float(max(nfe, 1))
    zh, ze = zh0, ze0
    xh = log_map_0(zh0, c=curvature) if hyper_state_mode == "logmap0" else zh0
    evals = 0
    for i in range(max(nfe, 1)):
        t = _time(zh.shape[0], i / max(nfe, 1), zh.device, zh.dtype)
        vh, ve = vf(zh, ze, t)
        if hyper_state_mode == "logmap0":
            xh = xh + dt * vh
            zh = exp_map_0(xh, c=curvature)
        else:
            zh = zh + dt * vh
        ze = ze + dt * ve
        zh, ze = product_project(zh, ze, c=curvature)
        if hyper_state_mode == "logmap0":
            xh = log_map_0(zh, c=curvature)
        evals += 1
    return zh, ze, evals


def solve_rk4(
    vf: FieldFn,
    zh0: torch.Tensor,
    ze0: torch.Tensor,
    nfe: int,
    curvature: float,
    hyper_state_mode: str = "ball",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    dt = 1.0 / float(max(nfe, 1))
    zh, ze = zh0, ze0
    xh = log_map_0(zh0, c=curvature) if hyper_state_mode == "logmap0" else zh0
    evals = 0
    for i in range(max(nfe, 1)):
        t0 = i / max(nfe, 1)
        t = _time(zh.shape[0], t0, zh.device, zh.dtype)
        t_half = _time(zh.shape[0], t0 + 0.5 * dt, zh.device, zh.dtype)
        t1 = _time(zh.shape[0], t0 + dt, zh.device, zh.dtype)
        zh_k1 = _state_to_ball(xh, zh, curvature, hyper_state_mode)
        k1h, k1e = vf(zh_k1, ze, t)
        zh_k2 = _state_to_ball(xh + 0.5 * dt * k1h, zh + 0.5 * dt * k1h, curvature, hyper_state_mode)
        k2h, k2e = vf(zh_k2, ze + 0.5 * dt * k1e, t_half)
        zh_k3 = _state_to_ball(xh + 0.5 * dt * k2h, zh + 0.5 * dt * k2h, curvature, hyper_state_mode)
        k3h, k3e = vf(zh_k3, ze + 0.5 * dt * k2e, t_half)
        zh_k4 = _state_to_ball(xh + dt * k3h, zh + dt * k3h, curvature, hyper_state_mode)
        k4h, k4e = vf(zh_k4, ze + dt * k3e, t1)
        if hyper_state_mode == "logmap0":
            xh = xh + (dt / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h)
            zh = exp_map_0(xh, c=curvature)
        else:
            zh = zh + (dt / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h)
        ze = ze + (dt / 6.0) * (k1e + 2 * k2e + 2 * k3e + k4e)
        zh, ze = product_project(zh, ze, c=curvature)
        if hyper_state_mode == "logmap0":
            xh = log_map_0(zh, c=curvature)
        evals += 4
    return zh, ze, evals


def solve_ode(
    vf: FieldFn,
    zh0: torch.Tensor,
    ze0: torch.Tensor,
    solver: str,
    nfe: int,
    curvature: float,
    hyper_state_mode: str = "ball",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    name = solver.lower()
    if name == "euler":
        return solve_euler(
            vf=vf,
            zh0=zh0,
            ze0=ze0,
            nfe=nfe,
            curvature=curvature,
            hyper_state_mode=hyper_state_mode,
        )
    if name == "rk4":
        return solve_rk4(
            vf=vf,
            zh0=zh0,
            ze0=ze0,
            nfe=nfe,
            curvature=curvature,
            hyper_state_mode=hyper_state_mode,
        )
    raise ValueError(f"Unsupported solver: {solver}")
