"""Evaluation helper for trained MC-RFM checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from src.engine.trainer import _compute_support_prototypes, _evaluate, build_loaders
from src.models.adapter import MCRFMAdapter
from src.utils.logging import write_json


@torch.no_grad()
def evaluate_checkpoint(cfg: dict, checkpoint_path: str | Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_cfg = ckpt.get("cfg", cfg)
    feature_dim = loaders.train.dataset.feature_dim  # type: ignore[attr-defined]
    model = MCRFMAdapter(
        in_dim=int(feature_dim),
        bottleneck_dim=int(state_cfg["model"]["bottleneck_dim"]),
        dh=int(state_cfg["model"]["dh"]),
        hidden_dim=int(state_cfg["model"]["hidden_dim"]),
        layers=int(state_cfg["model"]["layers"]),
        curvature=float(state_cfg["model"]["curvature"]),
        learnable_curvature=bool(state_cfg["model"]["learnable_curvature"]),
        curvature_min=float(state_cfg["model"]["curvature_min"]),
        decoupled_heads=bool(state_cfg["model"]["decoupled_heads"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    proto_h, proto_e = _compute_support_prototypes(model, loaders.train, device, num_classes=int(cfg["dataset"]["num_classes"]))
    metrics = _evaluate(
        model,
        loaders.test,
        proto_h,
        proto_e,
        cfg,
        device=device,
        num_classes=int(cfg["dataset"]["num_classes"]),
    )
    out_dir = Path(cfg["project"]["output_root"]) / cfg["project"]["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "eval_metrics.json", metrics)
    return metrics

