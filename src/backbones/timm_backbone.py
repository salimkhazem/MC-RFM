"""timm-based frozen backbone feature extractor."""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn


class TimmFeatureExtractor(nn.Module):
    def __init__(self, timm_name: str, pretrained: bool = True, pool: str = "avg"):
        super().__init__()
        self.model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        self.pool = pool

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pool == "cls":
            return tokens[:, 0]
        return tokens.mean(dim=1)

    def _unpack_forward_features(self, out: Any) -> torch.Tensor:
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)):
            for item in out:
                if torch.is_tensor(item):
                    return item
        if isinstance(out, dict):
            for key in ["x", "features", "last_hidden_state"]:
                if key in out and torch.is_tensor(out[key]):
                    return out[key]
        raise RuntimeError("Unsupported forward_features output format")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_features(x)
        out = self._unpack_forward_features(out)
        if out.dim() == 4:
            out = out.mean(dim=(2, 3))
        elif out.dim() == 3:
            out = self._pool_tokens(out)
        elif out.dim() != 2:
            raise RuntimeError(f"Unexpected feature tensor shape: {tuple(out.shape)}")
        return out.float()

