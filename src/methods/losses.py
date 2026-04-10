"""Loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def flow_matching_loss(
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    velocity_reg: float = 0.0,
) -> torch.Tensor:
    mse = F.mse_loss(pred_v, target_v)
    reg = velocity_reg * (pred_v * pred_v).mean()
    return mse + reg


def boundary_penalty(z: torch.Tensor, max_norm: float, margin: float = 0.95) -> torch.Tensor:
    norm = torch.norm(z, dim=-1)
    threshold = margin * max_norm
    return torch.relu(norm - threshold).pow(2).mean()

