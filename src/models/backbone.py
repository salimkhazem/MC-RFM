"""Frozen timm backbone feature extractor."""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn


class FrozenBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(name, pretrained=pretrained, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def _unpack(self, out: Any) -> torch.Tensor:
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)):
            for item in out:
                if torch.is_tensor(item):
                    return item
        if isinstance(out, dict):
            for key in ["x", "features", "last_hidden_state"]:
                if key in out and torch.is_tensor(out[key]):
                    return out[key]
        raise RuntimeError("Unsupported backbone output format")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_features(x)
        out = self._unpack(out)
        if out.dim() == 4:
            out = out.mean(dim=(2, 3))
        elif out.dim() == 3:
            out = out.mean(dim=1)
        elif out.dim() != 2:
            raise RuntimeError(f"Unexpected feature shape: {tuple(out.shape)}")
        return out.float()

