"""Class prototype utilities for product-manifold classification."""

from __future__ import annotations

import torch

from src.geometry.product_manifold import product_distance2
from src.geometry.poincare import exp_map_0


def compute_euclidean_prototypes(u: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    dim = u.shape[-1]
    protos = torch.zeros(num_classes, dim, device=u.device, dtype=u.dtype)
    counts = torch.zeros(num_classes, 1, device=u.device, dtype=u.dtype)
    protos.index_add_(0, labels, u)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=u.dtype).unsqueeze(-1))
    counts = torch.clamp(counts, min=1.0)
    return protos / counts


def compute_product_prototypes(
    uh: torch.Tensor,
    ue: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    c: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ph = compute_euclidean_prototypes(uh, labels, num_classes)
    pe = compute_euclidean_prototypes(ue, labels, num_classes)
    zh = exp_map_0(ph, c=c)
    return zh, pe


def product_logits(
    zh: torch.Tensor,
    ze: torch.Tensor,
    proto_h: torch.Tensor,
    proto_e: torch.Tensor,
    c: float,
) -> torch.Tensor:
    # Compute pairwise product distances and negate as logits.
    b = zh.shape[0]
    k = proto_h.shape[0]
    zh_b = zh.unsqueeze(1).expand(b, k, -1)
    ze_b = ze.unsqueeze(1).expand(b, k, -1)
    ph_b = proto_h.unsqueeze(0).expand(b, k, -1)
    pe_b = proto_e.unsqueeze(0).expand(b, k, -1)
    d2 = product_distance2(zh_b, ze_b, ph_b, pe_b, c=c)
    return -d2

