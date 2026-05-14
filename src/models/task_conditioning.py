"""Task-level conditioning modules built from support prototypes."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.geometry.poincare import log_map_0


class TaskSignatureEncoder(nn.Module):
    """Encode a prototype bank into a compact task context vector.

    The goal is not to recover a full semantic tree. The goal is to give the
    transport dynamics and classifier a stable summary of the current task:
    prototype distribution, rough branch spread, and class-set scale.
    """

    def __init__(self, dh: int, de: int, context_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.dh = int(dh)
        self.de = int(de)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        token_dim = self.dh + self.de
        stats_dim = 9
        self.token_proj = nn.Sequential(
            nn.Linear(max(token_dim, 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.context_head = nn.Sequential(
            nn.Linear(hidden_dim + stats_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def _pairwise_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] <= 1 or x.shape[-1] == 0:
            zero = x.new_zeros(())
            return zero, zero
        d = torch.cdist(x, x)
        idx = torch.triu_indices(d.shape[0], d.shape[1], offset=1, device=d.device)
        vals = d[idx[0], idx[1]]
        if vals.numel() == 0:
            zero = x.new_zeros(())
            return zero, zero
        return vals.mean(), vals.std(unbiased=False)

    def forward(self, proto_h: torch.Tensor, proto_e: torch.Tensor, c: float) -> torch.Tensor:
        dtype = proto_e.dtype if proto_e.numel() > 0 else proto_h.dtype
        device = proto_h.device
        xh = log_map_0(proto_h, c=c).to(dtype=dtype) if proto_h.shape[-1] > 0 else proto_h.new_zeros((proto_h.shape[0], 0), dtype=dtype)
        xe = proto_e.to(dtype=dtype) if proto_e.shape[-1] > 0 else proto_e.new_zeros((proto_e.shape[0], 0), dtype=dtype)
        if xh.shape[-1] == 0 and xe.shape[-1] == 0:
            return torch.zeros(self.context_dim, device=device, dtype=dtype)

        tokens = torch.cat([xh, xe], dim=-1)
        if tokens.shape[-1] == 0:
            tokens = torch.zeros((proto_h.shape[0], 1), device=device, dtype=dtype)
        h = self.token_proj(tokens)
        attn = torch.softmax(self.attn_score(h).squeeze(-1), dim=0).unsqueeze(-1)
        pooled = (attn * h).sum(dim=0)

        h_norm = xh.norm(dim=-1) if xh.shape[-1] > 0 else torch.zeros(proto_h.shape[0], device=device, dtype=dtype)
        e_norm = xe.norm(dim=-1) if xe.shape[-1] > 0 else torch.zeros(proto_h.shape[0], device=device, dtype=dtype)
        h_pair_mean, h_pair_std = self._pairwise_stats(xh) if xh.shape[-1] > 0 else (torch.zeros((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype))
        e_pair_mean, e_pair_std = self._pairwise_stats(xe) if xe.shape[-1] > 0 else (torch.zeros((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype))
        stats = torch.stack(
            [
                h_norm.mean(),
                h_norm.std(unbiased=False) if h_norm.numel() > 1 else torch.zeros((), device=device, dtype=dtype),
                e_norm.mean(),
                e_norm.std(unbiased=False) if e_norm.numel() > 1 else torch.zeros((), device=device, dtype=dtype),
                h_pair_mean,
                h_pair_std,
                e_pair_mean,
                e_pair_std,
                proto_h.new_tensor(math.log1p(float(proto_h.shape[0]))).to(dtype=dtype),
            ],
            dim=0,
        )
        return self.context_head(torch.cat([pooled, stats], dim=0))
