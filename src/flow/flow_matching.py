"""Flow-matching targets and loss on mixed-curvature product manifold."""

from __future__ import annotations

import torch

from src.geometry.product_manifold import product_geodesic, product_log_map
from src.geometry.poincare import log_map_0
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


def target_field_origin_chart(
    zth: torch.Tensor,
    zte: torch.Tensor,
    zh1: torch.Tensor,
    ze1: torch.Tensor,
    t: torch.Tensor,
    c: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    den = torch.clamp(1.0 - t, min=eps).unsqueeze(-1)
    xth = log_map_0(zth, c=c)
    xh1 = log_map_0(zh1, c=c)
    return (xh1 - xth) / den, (ze1 - zte) / den


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
    hyper_weight: float | torch.Tensor = 1.0,
    euclidean_weight: float | torch.Tensor = 1.0,
    hyperbolic_loss_weighting: str = "none",
    riemannian_scale_clip: float = 10.0,
) -> torch.Tensor:
    mode = geometry_mode.lower()
    loss_h = torch.tensor(0.0, device=v_h.device, dtype=v_h.dtype)
    loss_e = torch.tensor(0.0, device=v_e.device, dtype=v_e.dtype)

    if mode in {"mixed", "hyperbolic"} and v_h.numel() > 0:
        if str(hyperbolic_loss_weighting).lower() == "clipped_riemannian":
            scale = hyperbolic_metric_scale(z_h.double(), c=c).detach().clamp(max=float(riemannian_scale_clip)).to(v_h.dtype)
            diff_h = (scale * (v_h - u_h).pow(2)).mean(dim=-1)
        else:
            diff_h = (v_h - u_h).pow(2).mean(dim=-1)
        if torch.is_tensor(hyper_weight):
            w_h = hyper_weight.to(device=diff_h.device, dtype=diff_h.dtype).reshape(-1)
            loss_h = (diff_h * w_h).mean()
        else:
            loss_h = float(hyper_weight) * diff_h.mean()
    if mode in {"mixed", "euclidean", "remove_hyper"} and v_e.numel() > 0:
        diff_e = (v_e - u_e).pow(2).mean(dim=-1)
        if torch.is_tensor(euclidean_weight):
            w_e = euclidean_weight.to(device=diff_e.device, dtype=diff_e.dtype).reshape(-1)
            loss_e = (diff_e * w_e).mean()
        else:
            loss_e = float(euclidean_weight) * diff_e.mean()

    loss = loss_h + float(lambda_e) * loss_e
    assert_finite("flow_matching_loss", loss)
    return loss


def flow_matching_breakdown(
    v_h: torch.Tensor,
    v_e: torch.Tensor,
    u_h: torch.Tensor,
    u_e: torch.Tensor,
    z_h: torch.Tensor,
    c: float,
    geometry_mode: str = "mixed",
    hyperbolic_loss_weighting: str = "none",
    riemannian_scale_clip: float = 10.0,
) -> dict[str, torch.Tensor]:
    mode = geometry_mode.lower()
    device = z_h.device
    dtype = z_h.dtype
    zero = torch.zeros((), device=device, dtype=dtype)

    loss_h = zero
    loss_e = zero
    metric_scale_mean = zero
    boundary_margin_min = zero
    u_h_norm_mean = zero
    v_h_norm_mean = zero
    zh_norm_mean = zero
    logmap0_zh_norm_mean = zero

    if z_h.numel() > 0:
        scale = hyperbolic_metric_scale(z_h.double(), c=c).to(dtype)
        metric_scale_mean = scale.mean()
        boundary_margin = 1.0 - float(c) * (z_h.double() * z_h.double()).sum(dim=-1)
        boundary_margin_min = boundary_margin.min().to(dtype)
        u_h_norm_mean = u_h.double().norm(dim=-1).mean().to(dtype)
        v_h_norm_mean = v_h.double().norm(dim=-1).mean().to(dtype)
        zh_norm_mean = z_h.double().norm(dim=-1).mean().to(dtype)
        logmap0_zh_norm_mean = log_map_0(z_h.double(), c=c).norm(dim=-1).mean().to(dtype)

    if mode in {"mixed", "hyperbolic"} and v_h.numel() > 0:
        if str(hyperbolic_loss_weighting).lower() == "clipped_riemannian":
            scale = hyperbolic_metric_scale(z_h.double(), c=c).detach().clamp(max=float(riemannian_scale_clip)).to(v_h.dtype)
            loss_h = (scale * (v_h - u_h).pow(2)).mean()
        else:
            loss_h = (v_h - u_h).pow(2).mean()
    if mode in {"mixed", "euclidean", "remove_hyper"} and v_e.numel() > 0:
        loss_e = (v_e - u_e).pow(2).mean()

    return {
        "loss_h": loss_h,
        "loss_e": loss_e,
        "metric_scale_mean": metric_scale_mean,
        "u_h_norm_mean": u_h_norm_mean,
        "v_h_norm_mean": v_h_norm_mean,
        "min_boundary_margin": boundary_margin_min,
        "mean_zh_norm": zh_norm_mean,
        "mean_logmap0_zh_norm": logmap0_zh_norm_mean,
    }
