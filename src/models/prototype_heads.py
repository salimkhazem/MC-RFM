"""Prototype utilities for few-shot classification."""

from __future__ import annotations

import torch

from src.geometry.ops_safe import safe_norm
from src.geometry.poincare import mobius_add
from src.geometry.ops_safe import safe_atanh


def compute_class_prototypes(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    dim = features.shape[-1]
    device = features.device
    protos = torch.zeros(num_classes, dim, device=device, dtype=features.dtype)
    counts = torch.zeros(num_classes, device=device, dtype=features.dtype)
    protos.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
    counts = torch.clamp(counts.unsqueeze(-1), min=1.0)
    return protos / counts


def euclidean_nearest_prototype_logits(x: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    # Negative squared distances as logits.
    x2 = (x * x).sum(-1, keepdim=True)
    p2 = (protos * protos).sum(-1).unsqueeze(0)
    xp = x @ protos.t()
    d2 = x2 + p2 - 2.0 * xp
    return -d2


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1.0e-6) -> torch.Tensor:
    """
    Pairwise distance d_c(x, y) on Poincaré ball.
    x: [B, D], y: [K, D]
    returns: [B, K]
    """
    xb = x.unsqueeze(1)  # [B,1,D]
    yb = y.unsqueeze(0)  # [1,K,D]
    delta = mobius_add(-xb, yb, c=c, eps=eps)
    norm = safe_norm(delta, dim=-1, keepdim=False, eps=eps)
    return (2.0 / (c**0.5)) * safe_atanh((c**0.5) * norm, eps=eps)


def hyperbolic_nearest_prototype_logits(x: torch.Tensor, protos: torch.Tensor, c: float) -> torch.Tensor:
    d = hyperbolic_distance(x, protos, c=c)
    return -d
