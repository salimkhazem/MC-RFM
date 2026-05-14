"""Evaluation helper for trained MC-RFM checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from src.engine.trainer import _compute_support_prototypes, _evaluate, build_loaders
from src.models.adapter import MCRFMAdapter
from src.models.classifier import MCRFMClassifier
from src.utils.logging import write_json
from src.utils.speed import measure_throughput


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
        geometry_mode=str(state_cfg["model"]["geometry_mode"]),
        hyperbolic_scale_init=float(state_cfg["model"].get("hyperbolic_scale_init", 0.05)),
        hyperbolic_scale_min=float(state_cfg["model"].get("hyperbolic_scale_min", 0.01)),
        hyperbolic_scale_max=float(state_cfg["model"].get("hyperbolic_scale_max", 0.25)),
        euclidean_scale_init=float(state_cfg["model"].get("euclidean_scale_init", 1.0)),
        vector_field_h_input=str(state_cfg["model"].get("vector_field_h_input", "logmap0")),
    ).to(device)
    classifier = MCRFMClassifier(
        dh=int(state_cfg["model"]["dh"]),
        de=int(state_cfg["model"]["de"]),
        num_classes=int(cfg["dataset"]["num_classes"]),
        mode=str(state_cfg.get("classifier", {}).get("mode", "hybrid")),
        gamma_h_init=float(state_cfg.get("classifier", {}).get("gamma_h_init", 0.1)),
        gamma_e_init=float(state_cfg.get("classifier", {}).get("gamma_e_init", 1.0)),
        beta_init=float(state_cfg.get("classifier", {}).get("beta_init", 0.5)),
        adaptive_branch_gate=bool(state_cfg.get("classifier", {}).get("adaptive_branch_gate", False)),
        branch_gate_init=float(state_cfg.get("classifier", {}).get("branch_gate_init", 0.5)),
        adaptive_beta=bool(state_cfg.get("classifier", {}).get("adaptive_beta", False)),
        gate_hidden_dim=int(state_cfg.get("classifier", {}).get("gate_hidden_dim", 128)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"], strict=False)
    proto_h, proto_e = _compute_support_prototypes(
        model,
        loaders.train,
        device,
        num_classes=int(cfg["dataset"]["num_classes"]),
        shrinkage=float(state_cfg.get("classifier", {}).get("prototype_shrinkage", 0.0)),
    )
    metrics = _evaluate(
        model,
        classifier,
        loaders.test,
        proto_h,
        proto_e,
        cfg,
        device=device,
        num_classes=int(cfg["dataset"]["num_classes"]),
    )

    @torch.no_grad()
    def _fn():
        feats, _, _ = next(iter(loaders.test))
        feats = feats.to(device, non_blocking=True)
        zh0, ze0, _ = model.encode(feats)
        zh, ze, _ = model.transport(
            zh0,
            ze0,
            solver=str(cfg["ode"]["solver"]),
            nfe=int(cfg["eval"]["nfe"]),
        )
        return classifier(
            zh,
            ze,
            proto_h=proto_h,
            proto_e=proto_e,
            c=model.curvature(),
            geometry_mode=str(cfg["model"]["geometry_mode"]).lower(),
        )

    throughput, peak_vram = measure_throughput(_fn, sample_count=int(cfg["eval"]["batch_size"]), device=device)
    metrics = {
        **metrics,
        "nfe": int(cfg["eval"]["nfe"]),
        "throughput_img_s": throughput,
        "peak_vram_gb": peak_vram,
        "trainable_params": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            + sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        ),
        "geometry_mode": str(cfg["model"]["geometry_mode"]),
    }
    out_dir = Path(cfg["project"]["output_root"]) / cfg["project"]["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "eval_metrics.json", metrics)
    return metrics
