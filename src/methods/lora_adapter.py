"""Low-rank feature adapter baseline (LoRA-style proxy)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.methods.base import AdaptationMethod, make_feature_loader
from src.methods.losses import classification_loss
from src.models.classifier import LinearClassifier


class LowRankFeatureAdapter(nn.Module):
    def __init__(self, dim: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / float(rank)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.B(self.drop(self.A(x)))


class LoRAAdapterMethod(AdaptationMethod):
    def __init__(self, in_dim: int, num_classes: int, rank: int, alpha: float, dropout: float):
        self.adapter = LowRankFeatureAdapter(dim=in_dim, rank=rank, alpha=alpha, dropout=dropout)
        self.classifier = LinearClassifier(in_dim=in_dim, num_classes=num_classes)

    def fit(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        cfg: dict[str, Any],
        device: torch.device,
    ) -> dict[str, Any]:
        self.adapter.to(device)
        self.classifier.to(device)
        loader = make_feature_loader(
            train_features,
            train_labels,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=True,
        )
        params = list(self.adapter.parameters()) + list(self.classifier.parameters())
        opt = torch.optim.AdamW(params, lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
        best_val = -1.0
        best_state = None
        history = []
        epochs = int(cfg["training"]["epochs"])
        for epoch in range(epochs):
            self.adapter.train()
            self.classifier.train()
            total_loss = 0.0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = self.classifier(self.adapter(x))
                loss = classification_loss(logits, y, label_smoothing=float(cfg["training"]["label_smoothing"]))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, float(cfg["training"]["grad_clip"]))
                opt.step()
                total_loss += float(loss.detach().cpu().item())
            val_logits = self.predict_logits(val_features, cfg, device)
            val_acc = float((val_logits.argmax(dim=-1) == val_labels.to(device)).float().mean().item())
            history.append({"epoch": epoch, "train_loss": total_loss / max(1, len(loader)), "val_acc": val_acc})
            if val_acc > best_val:
                best_val = val_acc
                best_state = {
                    "adapter": {k: v.detach().cpu().clone() for k, v in self.adapter.state_dict().items()},
                    "classifier": {k: v.detach().cpu().clone() for k, v in self.classifier.state_dict().items()},
                }
        if best_state is not None:
            self.adapter.load_state_dict(best_state["adapter"])
            self.classifier.load_state_dict(best_state["classifier"])
        return {"history": history, "best_val_acc": best_val}

    @torch.no_grad()
    def predict_logits(self, features: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
        self.adapter.to(device)
        self.classifier.to(device)
        self.adapter.eval()
        self.classifier.eval()
        x = features.to(device)
        return self.classifier(self.adapter(x))

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.adapter.parameters() if p.requires_grad) + sum(
            p.numel() for p in self.classifier.parameters() if p.requires_grad
        )

    def state_dict(self) -> dict[str, Any]:
        return {"adapter": self.adapter.state_dict(), "classifier": self.classifier.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.adapter.load_state_dict(state["adapter"])
        self.classifier.load_state_dict(state["classifier"])
