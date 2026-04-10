"""Path-Decoupled Hyperbolic Flow Matching (PD-HFM)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry.poincare import expmap0, logmap0, mobius_add, project_to_ball
from src.methods.base import AdaptationMethod, make_feature_loader
from src.methods.losses import boundary_penalty, flow_matching_loss
from src.models.classifier import LinearClassifier
from src.models.prototype_heads import compute_class_prototypes, hyperbolic_nearest_prototype_logits
from src.models.vector_field import DecoupledHyperbolicVectorField
from src.solvers.wrapper import solve_ode


class PDHFMMethod(AdaptationMethod):
    def __init__(self, dim: int, num_classes: int, cfg_method: dict[str, Any]):
        self.dim = dim
        self.num_classes = num_classes
        self.c = float(cfg_method["geometry"]["curvature"])
        self.ball_eps = float(cfg_method["projection"]["ball_eps"])
        self.feature_layernorm = bool(cfg_method["projection"].get("feature_layernorm", True))
        self.feature_l2norm = bool(cfg_method["projection"].get("feature_l2norm", True))
        self.feature_scale = float(cfg_method["projection"].get("feature_scale", 1.0))
        self.target_field_mode = str(cfg_method["flow"].get("target_field_mode", "log_origin_delta"))

        vf_cfg = cfg_method["vector_field"]
        self.vf = DecoupledHyperbolicVectorField(
            dim=dim,
            hidden_dim=int(vf_cfg.get("hidden_dim", 512)),
            num_layers=int(vf_cfg.get("num_layers", 3)),
            coupled=bool(vf_cfg.get("coupled", False)),
        )
        self.prototypes: torch.Tensor | None = None
        self.linear_head: nn.Module | None = None

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        return project_to_ball(z, c=self.c, eps=self.ball_eps)

    def _to_hyperbolic(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        if self.feature_layernorm:
            z = F.layer_norm(z, normalized_shape=(z.shape[-1],))
        if self.feature_l2norm:
            z = F.normalize(z, p=2, dim=-1)
        z = self.feature_scale * z
        z = expmap0(z, c=self.c)
        z = self._project(z)
        return z

    def _path_interpolant(self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        while t.dim() < z0.dim():
            t = t.unsqueeze(-1)
        zt = (1.0 - t) * z0 + t * z1
        return self._project(zt)

    def _target_field(self, zt: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        if self.target_field_mode == "log_origin_delta":
            delta = mobius_add(-zt, z1, c=self.c)
            return logmap0(delta, c=self.c)
        return z1 - z0

    def _fit_linear_head(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        cfg: dict[str, Any],
        device: torch.device,
    ) -> None:
        z_train, _ = self._transport(train_features, cfg, device)
        self.linear_head = LinearClassifier(in_dim=self.dim, num_classes=self.num_classes).to(device)
        opt = torch.optim.AdamW(self.linear_head.parameters(), lr=5.0e-4, weight_decay=1.0e-4)
        loader = make_feature_loader(z_train.detach().cpu(), train_labels.detach().cpu(), batch_size=256, shuffle=True)
        for _ in range(15):
            self.linear_head.train()
            for z, y in loader:
                z = z.to(device)
                y = y.to(device)
                logits = self.linear_head(z)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

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
        z_train = self._to_hyperbolic(train_features.to(device))
        self.prototypes = self._to_hyperbolic(
            compute_class_prototypes(train_features.to(device), train_labels.to(device), self.num_classes)
        )
        loader = make_feature_loader(
            z_train.detach().cpu(),
            train_labels.detach().cpu(),
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=True,
        )
        opt = torch.optim.AdamW(self.vf.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))

        velocity_reg = float(cfg["method"]["loss"].get("velocity_reg", 0.0))
        boundary_w = float(cfg["method"]["loss"].get("boundary_penalty_weight", 0.0))
        max_norm = (1.0 - self.ball_eps) / (self.c**0.5)
        epochs = int(cfg["training"]["epochs"])
        best_val = -1.0
        best_state = None
        history = []

        for epoch in range(epochs):
            self.vf.train()
            total_loss = 0.0
            for z0_cpu, y_cpu in loader:
                z0 = z0_cpu.to(device)
                y = y_cpu.to(device)
                z1 = self.prototypes[y]
                t = torch.rand(z0.shape[0], device=device, dtype=z0.dtype)
                zt = self._path_interpolant(z0, z1, t)
                target_v = self._target_field(zt, z0, z1)
                pred_v = self.vf(zt, t)
                loss = flow_matching_loss(pred_v, target_v, velocity_reg=velocity_reg)
                if boundary_w > 0:
                    loss = loss + boundary_w * boundary_penalty(zt, max_norm=max_norm)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PD-HFM loss detected")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vf.parameters(), float(cfg["training"]["grad_clip"]))
                opt.step()
                total_loss += float(loss.detach().cpu().item())

            val_logits = self.predict_logits(val_features, cfg, device)
            val_acc = float((val_logits.argmax(dim=-1) == val_labels.to(device)).float().mean().item())
            history.append({"epoch": epoch, "train_loss": total_loss / max(1, len(loader)), "val_acc": val_acc})
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.vf.state_dict().items()}

        if best_state is not None:
            self.vf.load_state_dict(best_state)

        if str(cfg["evaluation"].get("classifier", "prototype")) == "linear":
            self._fit_linear_head(train_features, train_labels, cfg, device)

        return {"history": history, "best_val_acc": best_val}

    @torch.no_grad()
    def _transport(self, features: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, int]:
        self.vf.to(device)
        if self.linear_head is not None:
            self.linear_head.to(device)
        self.vf.eval()
        z0 = self._to_hyperbolic(features.to(device))
        nfe = int(cfg["method"]["flow"].get("nfe_eval", cfg["evaluation"]["nfe"]))
        solver = str(cfg["method"]["flow"].get("solver", "euler"))
        zt, evals = solve_ode(solver=solver, vf=self.vf, z0=z0, nfe=nfe, project_fn=self._project)
        return zt, evals

    @torch.no_grad()
    def predict_logits(self, features: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
        if self.prototypes is None:
            raise RuntimeError("Prototypes not initialized; call fit first.")
        z, _ = self._transport(features, cfg, device)
        if str(cfg["evaluation"].get("classifier", "prototype")) == "linear" and self.linear_head is not None:
            self.linear_head.eval()
            return self.linear_head(z)
        return hyperbolic_nearest_prototype_logits(z, self.prototypes, c=self.c)

    @torch.no_grad()
    def infer_nfe(self, features: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> int:
        _, evals = self._transport(features[:2], cfg, device)
        return int(evals)

    def trainable_parameters(self) -> int:
        total = sum(p.numel() for p in self.vf.parameters() if p.requires_grad)
        if self.linear_head is not None:
            total += sum(p.numel() for p in self.linear_head.parameters() if p.requires_grad)
        return total

    def state_dict(self) -> dict[str, Any]:
        payload = {"vf": self.vf.state_dict(), "prototypes": self.prototypes}
        if self.linear_head is not None:
            payload["linear_head"] = self.linear_head.state_dict()
        return payload

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.vf.load_state_dict(state["vf"])
        self.prototypes = state["prototypes"]
        if "linear_head" in state:
            self.linear_head = LinearClassifier(in_dim=self.dim, num_classes=self.num_classes)
            self.linear_head.load_state_dict(state["linear_head"])
