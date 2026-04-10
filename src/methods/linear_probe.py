"""Linear probe baseline."""

from __future__ import annotations

from typing import Any

import torch

from src.methods.base import AdaptationMethod, make_feature_loader
from src.methods.losses import classification_loss
from src.models.classifier import LinearClassifier
from src.models.prototype_heads import euclidean_nearest_prototype_logits


class LinearProbeMethod(AdaptationMethod):
    def __init__(self, in_dim: int, num_classes: int):
        self.model = LinearClassifier(in_dim=in_dim, num_classes=num_classes)
        self.num_classes = num_classes
        self.prototypes: torch.Tensor | None = None

    def fit(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        cfg: dict[str, Any],
        device: torch.device,
    ) -> dict[str, Any]:
        self.model.to(device)
        train_loader = make_feature_loader(
            train_features,
            train_labels,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=True,
        )
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        best_val = -1.0
        best_state = None
        history = []
        epochs = int(cfg["training"]["epochs"])
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                logits = self.model(x)
                loss = classification_loss(logits, y, label_smoothing=float(cfg["training"]["label_smoothing"]))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(cfg["training"]["grad_clip"]))
                opt.step()
                total_loss += float(loss.detach().cpu().item())
            val_acc = self._accuracy_logits(self.predict_logits(val_features, cfg, device), val_labels.to(device))
            history.append({"epoch": epoch, "train_loss": total_loss / max(1, len(train_loader)), "val_acc": val_acc})
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"history": history, "best_val_acc": best_val}

    def _accuracy_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        pred = torch.argmax(logits, dim=-1)
        return float((pred == labels).float().mean().item())

    @torch.no_grad()
    def predict_logits(self, features: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
        self.model.to(device)
        self.model.eval()
        x = features.to(device)
        return self.model(x)

    @torch.no_grad()
    def predict_logits_prototype(
        self,
        features: torch.Tensor,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.prototypes is None:
            from src.models.prototype_heads import compute_class_prototypes

            self.prototypes = compute_class_prototypes(support_features, support_labels, self.num_classes)
        return euclidean_nearest_prototype_logits(features, self.prototypes)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
