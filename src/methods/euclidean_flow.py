"""Euclidean flow matching baseline."""

from __future__ import annotations

from typing import Any

import torch

from src.geometry.euclidean import linear_interpolate
from src.methods.base import AdaptationMethod, make_feature_loader
from src.methods.losses import flow_matching_loss
from src.models.prototype_heads import compute_class_prototypes, euclidean_nearest_prototype_logits
from src.models.vector_field import EuclideanVectorField
from src.solvers.wrapper import solve_ode


class EuclideanFlowMethod(AdaptationMethod):
    def __init__(self, dim: int, num_classes: int, hidden_dim: int = 512, num_layers: int = 3):
        self.vf = EuclideanVectorField(dim=dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.num_classes = num_classes
        self.prototypes: torch.Tensor | None = None
        self.dim = dim

    def fit(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        cfg: dict[str, Any],
        device: torch.device,
    ) -> dict[str, Any]:
        self.vf.to(device)
        tr_loader = make_feature_loader(
            train_features,
            train_labels,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=True,
        )
        self.prototypes = compute_class_prototypes(train_features.to(device), train_labels.to(device), self.num_classes)
        opt = torch.optim.AdamW(self.vf.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))

        epochs = int(cfg["training"]["epochs"])
        best_val = -1.0
        best_state = None
        history = []
        velocity_reg = float(cfg["method"]["loss"].get("velocity_reg", 0.0))
        for epoch in range(epochs):
            self.vf.train()
            total_loss = 0.0
            for x, y in tr_loader:
                x = x.to(device)
                y = y.to(device)
                z0 = x
                z1 = self.prototypes[y]
                t = torch.rand(x.shape[0], device=device, dtype=x.dtype)
                zt = linear_interpolate(z0, z1, t)
                target_v = z1 - z0
                pred_v = self.vf(zt, t)
                loss = flow_matching_loss(pred_v, target_v, velocity_reg=velocity_reg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vf.parameters(), float(cfg["training"]["grad_clip"]))
                opt.step()
                total_loss += float(loss.detach().cpu().item())

            val_logits = self.predict_logits(val_features, cfg, device)
            val_acc = float((val_logits.argmax(dim=-1) == val_labels.to(device)).float().mean().item())
            history.append({"epoch": epoch, "train_loss": total_loss / max(1, len(tr_loader)), "val_acc": val_acc})
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.vf.state_dict().items()}

        if best_state is not None:
            self.vf.load_state_dict(best_state)
        return {"history": history, "best_val_acc": best_val}

    @torch.no_grad()
    def _transport(self, features: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, int]:
        self.vf.to(device)
        self.vf.eval()
        z0 = features.to(device)
        nfe = int(cfg["method"]["flow"].get("nfe_eval", cfg["evaluation"]["nfe"]))
        solver = str(cfg["method"]["flow"].get("solver", "euler"))
        zt, evals = solve_ode(solver=solver, vf=self.vf, z0=z0, nfe=nfe, project_fn=None)
        return zt, evals

    @torch.no_grad()
    def predict_logits(self, features: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
        if self.prototypes is None:
            raise RuntimeError("Prototypes are not initialized; call fit first.")
        z, _ = self._transport(features, cfg, device)
        return euclidean_nearest_prototype_logits(z, self.prototypes)

    @torch.no_grad()
    def infer_nfe(self, features: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> int:
        _, evals = self._transport(features[:2], cfg, device)
        return int(evals)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.vf.parameters() if p.requires_grad)

    def state_dict(self) -> dict[str, Any]:
        return {"vf": self.vf.state_dict(), "prototypes": self.prototypes}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.vf.load_state_dict(state["vf"])
        self.prototypes = state["prototypes"]
