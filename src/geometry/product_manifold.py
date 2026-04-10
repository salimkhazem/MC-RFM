"""Product manifold operations for D_c^{dh} x R^{de}."""

from __future__ import annotations

import torch

from src.geometry.poincare import exp_map_0, geodesic, log_map, project_to_ball


def split_features(x: torch.Tensor, dh: int) -> tuple[torch.Tensor, torch.Tensor]:
    if dh < 0 or dh > x.shape[-1]:
        raise ValueError(f"Invalid dh={dh} for shape {tuple(x.shape)}")
    return x[..., :dh], x[..., dh:]


def join_features(xh: torch.Tensor, xe: torch.Tensor) -> torch.Tensor:
    return torch.cat([xh, xe], dim=-1)


def project_product(u: torch.Tensor, dh: int, c: float, eps: float = 1.0e-5) -> tuple[torch.Tensor, torch.Tensor]:
    uh, ue = split_features(u, dh=dh)
    zh = exp_map_0(uh, c=c, eps=eps)
    return zh, ue


def product_geodesic(
    xh: torch.Tensor,
    xe: torch.Tensor,
    yh: torch.Tensor,
    ye: torch.Tensor,
    t: torch.Tensor,
    c: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    zt_h = geodesic(xh, yh, t=t, c=c)
    while t.dim() < xe.dim():
        t = t.unsqueeze(-1)
    zt_e = (1.0 - t) * xe + t * ye
    return zt_h, zt_e


def product_log_map(
    xh: torch.Tensor,
    xe: torch.Tensor,
    yh: torch.Tensor,
    ye: torch.Tensor,
    c: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    vh = log_map(xh, yh, c=c)
    ve = ye - xe
    return vh, ve


def product_project(xh: torch.Tensor, xe: torch.Tensor, c: float, eps: float = 1.0e-5) -> tuple[torch.Tensor, torch.Tensor]:
    return project_to_ball(xh, c=c, eps=eps), xe


def product_distance2(xh: torch.Tensor, xe: torch.Tensor, yh: torch.Tensor, ye: torch.Tensor, c: float) -> torch.Tensor:
    from src.geometry.poincare import dist

    dh2 = dist(xh, yh, c=c).pow(2)
    de2 = torch.sum((xe - ye) ** 2, dim=-1)
    return dh2 + de2

