"""Class prototype utilities for product-manifold classification."""

from __future__ import annotations

import torch

from src.geometry.poincare import dist, exp_map_0


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
    shrinkage: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    ph = compute_euclidean_prototypes(uh, labels, num_classes)
    pe = compute_euclidean_prototypes(ue, labels, num_classes)
    if shrinkage > 0.0:
        global_h = uh.mean(dim=0, keepdim=True)
        global_e = ue.mean(dim=0, keepdim=True)
        tau = float(shrinkage)
        ph = (1.0 - tau) * ph + tau * global_h
        pe = (1.0 - tau) * pe + tau * global_e
    zh = exp_map_0(ph, c=c)
    return zh, pe


def product_distance_components2(
    zh: torch.Tensor,
    ze: torch.Tensor,
    proto_h: torch.Tensor,
    proto_e: torch.Tensor,
    c: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    b = zh.shape[0]
    k = proto_h.shape[0]
    if proto_h.shape[-1] > 0:
        zh_b = zh.unsqueeze(1).expand(b, k, -1)
        ph_b = proto_h.unsqueeze(0).expand(b, k, -1)
        dh2 = dist(zh_b, ph_b, c=c).pow(2)
    else:
        dh2 = zh.new_zeros((b, k))
    if proto_e.shape[-1] > 0:
        ze_b = ze.unsqueeze(1).expand(b, k, -1)
        pe_b = proto_e.unsqueeze(0).expand(b, k, -1)
        de2 = torch.sum((ze_b - pe_b) ** 2, dim=-1)
    else:
        de2 = ze.new_zeros((b, k))
    return dh2, de2


def product_logits(
    zh: torch.Tensor,
    ze: torch.Tensor,
    proto_h: torch.Tensor,
    proto_e: torch.Tensor,
    c: float,
    geometry_mode: str = "mixed",
    gamma_h: float | torch.Tensor = 1.0,
    gamma_e: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    dh2, de2 = product_distance_components2(zh, ze, proto_h, proto_e, c=c)
    mode = geometry_mode.lower()
    gamma_h_t = gamma_h if torch.is_tensor(gamma_h) else dh2.new_tensor(float(gamma_h))
    gamma_e_t = gamma_e if torch.is_tensor(gamma_e) else de2.new_tensor(float(gamma_e))
    if mode == "hyperbolic":
        return -gamma_h_t.to(dtype=dh2.dtype, device=dh2.device) * dh2
    if mode in {"euclidean", "remove_hyper"}:
        return -gamma_e_t.to(dtype=de2.dtype, device=de2.device) * de2
    return -(
        gamma_h_t.to(dtype=dh2.dtype, device=dh2.device) * dh2
        + gamma_e_t.to(dtype=de2.dtype, device=de2.device) * de2
    )
