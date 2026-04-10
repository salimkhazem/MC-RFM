"""Flow-matching targets and loss on mixed-curvature product manifold."""

from __future__ import annotations

import torch

from src.geometry.product_manifold import product_geodesic, product_log_map
from src.geometry.stability import assert_finite, safe_time_uniform


def sample_times(batch_size: int, device: torch.device, dtype: torch.dtype, eps: float) -> torch.Tensor:
    return safe_time_uniform(batch_size, device=device, dtype=dtype, eps=eps)


def interpolate_product(
    zh0: torch.Tensor,
    ze0: torch.Tensor,
    zh1: torch.Tensor,
    ze1: torch.Tensor,
    t: torch.Tensor,
    c: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return product_geodesic(zh0, ze0, zh1, ze1, t=t, c=c)


def target_field(
    zth: torch.Tensor,
    zte: torch.Tensor,
    zh1: torch.Tensor,
    ze1: torch.Tensor,
    t: torch.Tensor,
    c: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    uh, ue = product_log_map(zth, zte, zh1, ze1, c=c)
    den = torch.clamp(1.0 - t, min=eps).unsqueeze(-1)
    return uh / den, ue / den


def hyperbolic_metric_scale(z_h: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
    z2 = (z_h * z_h).sum(dim=-1, keepdim=True)
    den = torch.clamp(1.0 - c * z2, min=eps)
    lam = 2.0 / den
    return lam * lam


def flow_matching_loss(
    v_h: torch.Tensor,
    v_e: torch.Tensor,
    u_h: torch.Tensor,
    u_e: torch.Tensor,
    z_h: torch.Tensor,
    c: float,
    lambda_e: float,
    geometry_mode: str = "mixed",
) -> torch.Tensor:
    mode = geometry_mode.lower()
    loss_h = torch.tensor(0.0, device=v_h.device, dtype=v_h.dtype)
    loss_e = torch.tensor(0.0, device=v_e.device, dtype=v_e.dtype)

    if mode in {"mixed", "hyperbolic"} and v_h.numel() > 0:
        scale = hyperbolic_metric_scale(z_h, c=c).to(v_h.dtype)
        loss_h = (scale * (v_h - u_h).pow(2)).mean()
    if mode in {"mixed", "euclidean", "remove_hyper"} and v_e.numel() > 0:
        loss_e = (v_e - u_e).pow(2).mean()

    loss = loss_h + float(lambda_e) * loss_e
    assert_finite("flow_matching_loss", loss)
    return loss

