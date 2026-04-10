"""Helpers around backbone features."""

from __future__ import annotations

import torch


@torch.no_grad()
def infer_feature_dim(model: torch.nn.Module, image_size: int = 224, device: torch.device | None = None) -> int:
    device = device or torch.device("cpu")
    x = torch.randn(2, 3, image_size, image_size, device=device)
    y = model(x)
    if y.dim() != 2:
        raise RuntimeError(f"Feature extractor must output [B, D], got {tuple(y.shape)}")
    return int(y.shape[-1])

