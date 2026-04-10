"""Stable Poincaré-ball geometry operations."""

from __future__ import annotations

import torch

from src.geometry.ops_safe import safe_atanh, safe_div, safe_norm


def ball_radius(c: float) -> float:
    if c <= 0:
        raise ValueError(f"Curvature c must be > 0, got {c}")
    return 1.0 / (c**0.5)


def project_to_ball(x: torch.Tensor, c: float, eps: float = 1.0e-5) -> torch.Tensor:
    """Project points to the interior of the ball with boundary margin."""
    radius = (1.0 - eps) * ball_radius(c)
    norm = safe_norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(radius / norm, max=1.0)
    return x * scale


def conformal_factor(x: torch.Tensor, c: float, eps: float = 1.0e-8) -> torch.Tensor:
    """Conformal factor lambda_x = 2 / (1 - c ||x||^2)."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    den = torch.clamp(1.0 - c * x2, min=eps)
    return 2.0 / den


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1.0e-8) -> torch.Tensor:
    """Möbius addition on the Poincaré ball."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    cxy = 2.0 * c * xy
    num = (1.0 + cxy + c * y2) * x + (1.0 - c * x2) * y
    den = 1.0 + cxy + (c * c) * x2 * y2
    out = safe_div(num, den, eps=eps)
    return project_to_ball(out, c=c, eps=1.0e-5)


def mobius_scalar_mul(r: torch.Tensor | float, x: torch.Tensor, c: float, eps: float = 1.0e-8) -> torch.Tensor:
    """Möbius scalar multiplication r ⊗_c x."""
    sqrt_c = c**0.5
    x = project_to_ball(x, c=c, eps=1.0e-5)
    xnorm = safe_norm(x, dim=-1, keepdim=True, eps=eps)
    if not torch.is_tensor(r):
        r = torch.tensor(r, device=x.device, dtype=x.dtype)
    while r.dim() < x.dim():
        r = r.unsqueeze(-1)
    inner = safe_atanh(torch.clamp(sqrt_c * xnorm, max=1.0 - 1.0e-6))
    out = torch.tanh(r * inner) * safe_div(x, sqrt_c * xnorm, eps=eps)
    return project_to_ball(out, c=c, eps=1.0e-5)


def exp_map_0(v: torch.Tensor, c: float, eps: float = 1.0e-8) -> torch.Tensor:
    """Exponential map at the origin."""
    sqrt_c = c**0.5
    vnorm = safe_norm(v, dim=-1, keepdim=True, eps=eps)
    scaled = torch.tanh(sqrt_c * vnorm) * safe_div(v, sqrt_c * vnorm, eps=eps)
    return project_to_ball(scaled, c=c, eps=1.0e-5)


def log_map_0(x: torch.Tensor, c: float, eps: float = 1.0e-8) -> torch.Tensor:
    """Logarithmic map at the origin."""
    sqrt_c = c**0.5
    x = project_to_ball(x, c=c, eps=1.0e-5)
    xnorm = safe_norm(x, dim=-1, keepdim=True, eps=eps)
    arg = sqrt_c * xnorm
    factor = safe_div(safe_atanh(arg), sqrt_c * xnorm, eps=eps)
    return factor * x


def log_map(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1.0e-8) -> torch.Tensor:
    """Approximate log map log_x(y) via scaled Möbius displacement."""
    x = project_to_ball(x, c=c, eps=1.0e-5)
    y = project_to_ball(y, c=c, eps=1.0e-5)
    delta = mobius_add(-x, y, c=c, eps=eps)
    lam_x = conformal_factor(x, c=c, eps=eps)
    return (2.0 / lam_x) * log_map_0(delta, c=c, eps=eps)


def geodesic(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, c: float) -> torch.Tensor:
    """Geodesic interpolation in Poincaré ball."""
    delta = mobius_add(-x, y, c=c)
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    step = mobius_scalar_mul(t, delta, c=c)
    return mobius_add(x, step, c=c)


def dist0(x: torch.Tensor, c: float, eps: float = 1.0e-8) -> torch.Tensor:
    """Distance to origin."""
    sqrt_c = c**0.5
    x = project_to_ball(x, c=c, eps=1.0e-5)
    xnorm = safe_norm(x, dim=-1, keepdim=False, eps=eps)
    return 2.0 * safe_atanh(sqrt_c * xnorm) / sqrt_c


def dist(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1.0e-8) -> torch.Tensor:
    """Hyperbolic distance d_c(x, y)."""
    delta = mobius_add(-x, y, c=c, eps=eps)
    return dist0(delta, c=c, eps=eps)


def hyperbolic_norm(v: torch.Tensor, x: torch.Tensor, c: float) -> torch.Tensor:
    """Riemannian norm of tangent vector v at x."""
    lam = conformal_factor(x, c=c)
    return lam * safe_norm(v, dim=-1, keepdim=True)


def riemannian_norm(x: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Riemannian norm induced by conformal metric."""
    lam = conformal_factor(x, c=c)
    vnorm = safe_norm(v, dim=-1, keepdim=True)
    return lam * vnorm


def radial_direction(x: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Unit radial direction using Euclidean proxy."""
    return x / torch.clamp(safe_norm(x, dim=-1, keepdim=True), min=eps)


def project_tangent_orthogonal(x: torch.Tensor, v: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Project vector v to subspace orthogonal to x (Euclidean proxy)."""
    u = radial_direction(x, eps=eps)
    dot = (v * u).sum(dim=-1, keepdim=True)
    return v - dot * u


# Backward-compatible aliases.
expmap0 = exp_map_0
logmap0 = log_map_0

