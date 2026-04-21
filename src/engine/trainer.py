"""Training engine for Mixed-Curvature Riemannian Flow Matching."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.fewshot_splits import load_or_create_fewshot_indices
from src.data.lmdb_io import LMDBFeatureDataset, FeatureLMDBReader
from src.flow.flow_matching import (
    flow_matching_breakdown,
    flow_matching_loss,
    interpolate_product,
    sample_times,
    target_field,
    target_field_origin_chart,
)
from src.geometry.stability import assert_finite
from src.models.adapter import MCRFMAdapter
from src.models.classifier import MCRFMClassifier
from src.models.prototypes import compute_product_prototypes, product_logits
from src.utils.logging import append_csv, append_jsonl, save_config_snapshot, setup_logger, write_json
from src.utils.metrics import macro_f1, top1_accuracy
from src.utils.speed import measure_throughput


@dataclass
class SplitLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _collate(batch):
    feats = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    idx = torch.tensor([int(b[2]) for b in batch], dtype=torch.long)
    return feats, labels, idx


def _scan_lmdb_labels(path: str | Path) -> list[int]:
    reader = FeatureLMDBReader(path)
    out = []
    for i in range(reader.meta.num_samples):
        _, y = reader.get(i)
        out.append(int(y))
    reader.close()
    return out


def _lmdb_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["cache"]["lmdb_dir"]) / cfg["dataset"]["name"] / cfg["model"]["backbone"]
    return {"train": root / "train.lmdb", "val": root / "val.lmdb", "test": root / "test.lmdb"}


def build_loaders(cfg: dict[str, Any]) -> SplitLoaders:
    paths = _lmdb_paths(cfg)
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing LMDB split '{k}': {p}")

    train_indices = None
    if int(cfg["fewshot"]["shots"]) > 0:
        labels = _scan_lmdb_labels(paths["train"])
        train_indices = load_or_create_fewshot_indices(
            split_dir=cfg["fewshot"]["split_dir"],
            dataset=cfg["dataset"]["name"],
            split="train",
            labels=labels,
            shots=int(cfg["fewshot"]["shots"]),
            seed=int(cfg["fewshot"]["split_seed"]),
        )

    ds_train = LMDBFeatureDataset(paths["train"], indices=train_indices)
    ds_val = LMDBFeatureDataset(paths["val"])
    ds_test = LMDBFeatureDataset(paths["test"])

    tr_loader = DataLoader(
        ds_train,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["device"]["num_workers"]),
        pin_memory=bool(cfg["device"]["pin_memory"]),
        collate_fn=_collate,
    )
    va_loader = DataLoader(
        ds_val,
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["device"]["num_workers"]),
        pin_memory=bool(cfg["device"]["pin_memory"]),
        collate_fn=_collate,
    )
    te_loader = DataLoader(
        ds_test,
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["device"]["num_workers"]),
        pin_memory=bool(cfg["device"]["pin_memory"]),
        collate_fn=_collate,
    )
    return SplitLoaders(train=tr_loader, val=va_loader, test=te_loader)


def _num_classes_from_loader(loader: DataLoader) -> int:
    labels = []
    for _, y, _ in loader:
        labels.append(y)
    all_y = torch.cat(labels, dim=0)
    return int(all_y.max().item() + 1)


def _geometry_mode(cfg: dict[str, Any]) -> str:
    return str(cfg["model"]["geometry_mode"]).lower()


def _hyper_branch_active(cfg: dict[str, Any]) -> bool:
    return _geometry_mode(cfg) in {"mixed", "hyperbolic"}


def _euclidean_branch_active(cfg: dict[str, Any]) -> bool:
    return _geometry_mode(cfg) in {"mixed", "euclidean", "remove_hyper"}


def _hyper_weight(cfg: dict[str, Any], epoch: int) -> float:
    if not _hyper_branch_active(cfg):
        return 0.0
    warmup = int(cfg["training"].get("hyper_loss_warmup_epochs", 0))
    ramp = int(cfg["training"].get("hyper_loss_ramp_epochs", 0))
    if epoch < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    progress = min(max((epoch - warmup + 1) / float(ramp), 0.0), 1.0)
    return progress


@torch.no_grad()
def _compute_support_prototypes(
    model: MCRFMAdapter,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    shrinkage: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    uh_list = []
    ue_list = []
    y_list = []
    c = model.curvature()
    for feats, labels, _ in loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device)
        _, uh, ue = model.projector(feats)
        uh_list.append(uh)
        ue_list.append(ue)
        y_list.append(labels)
    uh_all = torch.cat(uh_list, dim=0)
    ue_all = torch.cat(ue_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    proto_h, proto_e = compute_product_prototypes(
        uh_all,
        ue_all,
        y_all,
        num_classes=num_classes,
        c=c,
        shrinkage=shrinkage,
    )
    return proto_h, proto_e


@torch.no_grad()
def _evaluate(
    model: MCRFMAdapter,
    classifier: MCRFMClassifier,
    loader: DataLoader,
    proto_h: torch.Tensor,
    proto_e: torch.Tensor,
    task_context: torch.Tensor | None,
    cfg: dict[str, Any],
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    classifier.eval()
    logits_all = []
    labels_all = []
    for feats, labels, _ in loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device)
        zh0, ze0, _ = model.encode(feats)
        zh, ze, _ = model.transport(
            zh0,
            ze0,
            solver=str(cfg["ode"]["solver"]),
            nfe=int(cfg["eval"]["nfe"]),
            task_context=task_context,
        )
        logits = classifier(
            zh,
            ze,
            proto_h=proto_h,
            proto_e=proto_e,
            c=model.curvature(),
            geometry_mode=_geometry_mode(cfg),
            task_context=task_context,
        )
        logits_all.append(logits)
        labels_all.append(labels)
    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    assert_finite("eval_logits", logits)
    return {
        "top1": top1_accuracy(logits, labels),
        "macro_f1": macro_f1(logits, labels, num_classes=num_classes),
    }


def _autocast_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _failure_flags(cfg: dict[str, Any], history_rows: list[dict[str, float]], test_metrics: dict[str, float]) -> dict[str, float | bool]:
    mode = _geometry_mode(cfg)
    post_epoch5 = [row for row in history_rows if int(row["epoch"]) >= 5]
    ratios = [float(row["loss_h_to_e_ratio"]) for row in post_epoch5]
    boundary_margins = [float(row["min_boundary_margin"]) for row in post_epoch5]
    loss_imbalance = mode == "mixed" and bool(ratios) and statistics.median(ratios) > 100.0
    boundary_hits = sum(margin < 1.0e-4 for margin in boundary_margins)
    boundary_risk = (
        mode in {"mixed", "hyperbolic"}
        and bool(boundary_margins)
        and boundary_hits >= max(3, math.ceil(0.25 * len(boundary_margins)))
    )
    collapse = (
        str(cfg["dataset"]["name"]).lower() == "cifar100"
        and int(cfg["fewshot"]["shots"]) == 4
        and mode == "hyperbolic"
        and float(test_metrics["top1"]) < 0.05
    )
    return {
        "collapse_flag": bool(collapse),
        "loss_imbalance_flag": bool(loss_imbalance),
        "boundary_risk_flag": bool(boundary_risk),
        "median_loss_h_to_e_ratio_post5": float(statistics.median(ratios)) if ratios else 0.0,
        "boundary_margin_hits_post5": float(boundary_hits),
        "min_boundary_margin_overall": float(min((float(row["min_boundary_margin"]) for row in history_rows), default=0.0)),
    }


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(cfg)
    num_classes = int(cfg["dataset"]["num_classes"]) or _num_classes_from_loader(loaders.train)

    feature_dim = loaders.train.dataset.feature_dim  # type: ignore[attr-defined]
    model = MCRFMAdapter(
        in_dim=int(feature_dim),
        bottleneck_dim=int(cfg["model"]["bottleneck_dim"]),
        dh=int(cfg["model"]["dh"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        layers=int(cfg["model"]["layers"]),
        curvature=float(cfg["model"]["curvature"]),
        learnable_curvature=bool(cfg["model"]["learnable_curvature"]),
        curvature_min=float(cfg["model"]["curvature_min"]),
        decoupled_heads=bool(cfg["model"]["decoupled_heads"]),
        geometry_mode=_geometry_mode(cfg),
        hyperbolic_scale_init=float(cfg["model"].get("hyperbolic_scale_init", 0.05)),
        hyperbolic_scale_min=float(cfg["model"].get("hyperbolic_scale_min", 0.01)),
        hyperbolic_scale_max=float(cfg["model"].get("hyperbolic_scale_max", 0.25)),
        euclidean_scale_init=float(cfg["model"].get("euclidean_scale_init", 1.0)),
        vector_field_h_input=str(cfg["model"].get("vector_field_h_input", "logmap0")),
        task_conditioning=bool(cfg["model"].get("task_conditioning", False)),
        task_context_dim=int(cfg["model"].get("task_context_dim", 128)),
        task_context_hidden_dim=int(cfg["model"].get("task_context_hidden_dim", 128)),
    ).to(device)
    classifier = MCRFMClassifier(
        dh=int(cfg["model"]["dh"]),
        de=int(cfg["model"]["de"]),
        num_classes=num_classes,
        mode=str(cfg.get("classifier", {}).get("mode", "hybrid")),
        gamma_h_init=float(cfg.get("classifier", {}).get("gamma_h_init", 0.1)),
        gamma_e_init=float(cfg.get("classifier", {}).get("gamma_e_init", 1.0)),
        beta_init=float(cfg.get("classifier", {}).get("beta_init", 0.5)),
        adaptive_branch_gate=bool(cfg.get("classifier", {}).get("adaptive_branch_gate", False)),
        branch_gate_init=float(cfg.get("classifier", {}).get("branch_gate_init", 0.5)),
        adaptive_beta=bool(cfg.get("classifier", {}).get("adaptive_beta", False)),
        gate_hidden_dim=int(cfg.get("classifier", {}).get("gate_hidden_dim", 128)),
        task_context_dim=int(cfg["model"].get("task_context_dim", 0)) if bool(cfg["model"].get("task_conditioning", False)) else 0,
    ).to(device)
    ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)))

    params = [p for p in model.parameters() if p.requires_grad] + [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    total_steps = int(cfg["training"]["epochs"]) * max(len(loaders.train), 1)
    warmup = int(cfg["training"]["warmup_epochs"]) * max(len(loaders.train), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(warmup, 1))
        progress = (step - warmup) / float(max(total_steps - warmup, 1))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["device"]["amp"]) and _autocast_dtype(cfg["device"]["amp_dtype"]) == torch.float16)

    run_name = cfg["project"]["run_name"]
    out_dir = Path(cfg["project"]["output_root"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "stdout.log")
    save_config_snapshot(cfg, out_dir)

    accum = max(int(cfg["training"]["effective_batch_size"]) // max(int(cfg["training"]["batch_size"]), 1), 1)
    use_amp = bool(cfg["device"]["amp"]) and torch.cuda.is_available()
    amp_dtype = _autocast_dtype(cfg["device"]["amp_dtype"])
    best_val = -1.0
    history_path = out_dir / "train_history.jsonl"
    history_rows: list[dict[str, float]] = []

    global_step = 0
    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        classifier.train()
        proto_h, proto_e = _compute_support_prototypes(
            model,
            loaders.train,
            device=device,
            num_classes=num_classes,
            shrinkage=float(cfg.get("classifier", {}).get("prototype_shrinkage", 0.0)),
        )
        running = 0.0
        diag_sums = {
            "loss_h": 0.0,
            "loss_e": 0.0,
            "cls_loss": 0.0,
            "metric_scale_mean": 0.0,
            "u_h_norm_mean": 0.0,
            "v_h_norm_mean": 0.0,
            "mean_zh_norm": 0.0,
            "mean_logmap0_zh_norm": 0.0,
            "hyper_mult_mean": 0.0,
            "euclidean_mult_mean": 0.0,
            "branch_gate_mean": 0.0,
            "beta_effective_mean": 0.0,
            "task_context_norm": 0.0,
        }
        min_boundary_margin = float("inf")
        epoch_hyper_weight = _hyper_weight(cfg, epoch)
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loaders.train, desc=f"train_epoch_{epoch}", leave=False)
        for batch_idx, (feats, labels, _) in enumerate(pbar):
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device)
            task_context = model.encode_task_context(proto_h, proto_e)
            zh0, ze0, _ = model.encode(feats)
            zh1 = proto_h[labels]
            ze1 = proto_e[labels]
            t = sample_times(feats.shape[0], device=device, dtype=zh0.dtype, eps=float(cfg["training"]["epsilon_t"]))

            # Hyperbolic geometry ops in float64 for stability.
            zth64, zte = interpolate_product(zh0.double(), ze0, zh1.double(), ze1, t.double(), c=model.curvature())
            if str(cfg["model"].get("vector_field_h_input", "logmap0")).lower() == "logmap0":
                uh64, ue = target_field_origin_chart(
                    zth64,
                    zte,
                    zh1.double(),
                    ze1,
                    t.double(),
                    c=model.curvature(),
                    eps=float(cfg["training"]["epsilon_t"]),
                )
            else:
                uh64, ue = target_field(
                    zth64,
                    zte,
                    zh1.double(),
                    ze1,
                    t.double(),
                    c=model.curvature(),
                    eps=float(cfg["training"]["epsilon_t"]),
                )
            zth = zth64.float()
            zte = zte.float()
            uh = uh64.float()
            ue = ue.float()
            with torch.no_grad():
                hyper_mult, euclidean_mult = classifier.branch_multipliers(
                    zth,
                    zte,
                    c=model.curvature(),
                    geometry_mode=_geometry_mode(cfg),
                    task_context=task_context,
                )
                branch_gate = classifier.branch_gates(
                    zth,
                    zte,
                    c=model.curvature(),
                    geometry_mode=_geometry_mode(cfg),
                    task_context=task_context,
                )

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                vh, ve = model.field(zth, zte, t, task_context=task_context)
                fm_loss = flow_matching_loss(
                    v_h=vh,
                    v_e=ve,
                    u_h=uh,
                    u_e=ue,
                    z_h=zth,
                    c=model.curvature(),
                    lambda_e=float(cfg["training"]["lambda_euclidean"]),
                    geometry_mode=_geometry_mode(cfg),
                    hyper_weight=epoch_hyper_weight * hyper_mult.detach().reshape(-1),
                    euclidean_weight=euclidean_mult.detach().reshape(-1),
                    hyperbolic_loss_weighting=str(cfg["training"].get("hyperbolic_loss_weighting", "none")),
                    riemannian_scale_clip=float(cfg["training"].get("riemannian_scale_clip", 10.0)),
                )
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=False):
                zh_cls, ze_cls, _ = model.transport(
                    zh0.float(),
                    ze0.float(),
                    solver=str(cfg["ode"]["solver"]),
                    nfe=int(cfg["ode"]["nfe"]),
                    task_context=task_context,
                )
                logits = classifier(
                    zh_cls,
                    ze_cls,
                    proto_h=proto_h,
                    proto_e=proto_e,
                    c=model.curvature(),
                    geometry_mode=_geometry_mode(cfg),
                    task_context=task_context,
                )
                beta_values = classifier.beta_values(
                    zh_cls,
                    ze_cls,
                    c=model.curvature(),
                    geometry_mode=_geometry_mode(cfg),
                    task_context=task_context,
                )
                cls_loss = ce_loss(logits, labels)
                loss = fm_loss + float(cfg["training"].get("lambda_cls", 0.0)) * cls_loss
                loss = loss / accum

            with torch.no_grad():
                batch_diag = flow_matching_breakdown(
                    v_h=vh.detach().float(),
                    v_e=ve.detach().float(),
                    u_h=uh.detach().float(),
                    u_e=ue.detach().float(),
                    z_h=zth64.detach(),
                    c=model.curvature(),
                    geometry_mode=_geometry_mode(cfg),
                    hyperbolic_loss_weighting=str(cfg["training"].get("hyperbolic_loss_weighting", "none")),
                    riemannian_scale_clip=float(cfg["training"].get("riemannian_scale_clip", 10.0)),
                )
                for key in diag_sums:
                    if key == "cls_loss":
                        diag_sums[key] += float(cls_loss.detach().cpu().item())
                    elif key == "hyper_mult_mean":
                        diag_sums[key] += float(hyper_mult.detach().mean().cpu().item())
                    elif key == "euclidean_mult_mean":
                        diag_sums[key] += float(euclidean_mult.detach().mean().cpu().item())
                    elif key == "branch_gate_mean":
                        diag_sums[key] += float(branch_gate.detach().mean().cpu().item())
                    elif key == "beta_effective_mean":
                        diag_sums[key] += float(beta_values.detach().mean().cpu().item())
                    elif key == "task_context_norm":
                        diag_sums[key] += float(task_context.detach().norm().cpu().item()) if task_context is not None else 0.0
                    else:
                        diag_sums[key] += float(batch_diag[key].detach().cpu().item())
                min_boundary_margin = min(
                    min_boundary_margin,
                    float(batch_diag["min_boundary_margin"].detach().cpu().item()),
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.detach().cpu().item()) * accum

            if (batch_idx + 1) % accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, float(cfg["training"]["grad_clip"]))
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            if (batch_idx + 1) % int(cfg["training"]["log_interval"]) == 0:
                pbar.set_postfix({"loss": f"{running / (batch_idx + 1):.4f}"})

        if len(loaders.train) % accum != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, float(cfg["training"]["grad_clip"]))
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        proto_h_val, proto_e_val = _compute_support_prototypes(
            model,
            loaders.train,
            device=device,
            num_classes=num_classes,
            shrinkage=float(cfg.get("classifier", {}).get("prototype_shrinkage", 0.0)),
        )
        task_context_val = model.encode_task_context(proto_h_val, proto_e_val)
        val_metrics = _evaluate(
            model,
            classifier,
            loaders.val,
            proto_h_val,
            proto_e_val,
            task_context_val,
            cfg,
            device=device,
            num_classes=num_classes,
        )
        num_batches = max(len(loaders.train), 1)
        epoch_loss_h = diag_sums["loss_h"] / num_batches
        epoch_loss_e = diag_sums["loss_e"] / num_batches
        row = {
            "epoch": epoch,
            "train_loss": running / num_batches,
            "loss_h": epoch_loss_h,
            "loss_e": epoch_loss_e,
            "cls_loss": diag_sums["cls_loss"] / num_batches,
            "loss_h_to_e_ratio": epoch_loss_h / max(epoch_loss_e, 1.0e-12),
            "metric_scale_mean": diag_sums["metric_scale_mean"] / num_batches,
            "u_h_norm_mean": diag_sums["u_h_norm_mean"] / num_batches,
            "v_h_norm_mean": diag_sums["v_h_norm_mean"] / num_batches,
            "mean_zh_norm": diag_sums["mean_zh_norm"] / num_batches,
            "mean_logmap0_zh_norm": diag_sums["mean_logmap0_zh_norm"] / num_batches,
            "min_boundary_margin": float(min_boundary_margin if math.isfinite(min_boundary_margin) else 0.0),
            "val_top1": val_metrics["top1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": float(scheduler.get_last_lr()[0]),
            "curvature": model.curvature(),
            "hyper_weight": epoch_hyper_weight,
            "alpha_h": model.hyperbolic_scale(),
            "alpha_e": model.euclidean_scale(),
            "gamma_h": float(classifier.gamma_h().detach().cpu().item()),
            "gamma_e": float(classifier.gamma_e().detach().cpu().item()),
            "beta": float(classifier.beta().detach().cpu().item()),
            "branch_gate_prior": float(classifier.branch_gate_prior().detach().cpu().item()),
            "branch_gate_mean": diag_sums["branch_gate_mean"] / num_batches,
            "hyper_mult_mean": diag_sums["hyper_mult_mean"] / num_batches,
            "euclidean_mult_mean": diag_sums["euclidean_mult_mean"] / num_batches,
            "beta_effective_mean": diag_sums["beta_effective_mean"] / num_batches,
            "task_context_norm": diag_sums["task_context_norm"] / num_batches,
        }
        history_rows.append(row)
        append_jsonl(history_path, row)
        logger.info(
            "epoch=%d loss=%.4f cls=%.4f loss_h=%.4f loss_e=%.4f ratio=%.2f scale=%.2f margin=%.3e val_top1=%.4f val_f1=%.4f alpha_h=%.4f beta=%.4f",
            epoch,
            row["train_loss"],
            row["cls_loss"],
            row["loss_h"],
            row["loss_e"],
            row["loss_h_to_e_ratio"],
            row["metric_scale_mean"],
            row["min_boundary_margin"],
            row["val_top1"],
            row["val_macro_f1"],
            row["alpha_h"],
            row["beta_effective_mean"],
        )
        if val_metrics["top1"] > best_val:
            best_val = val_metrics["top1"]
            torch.save({"state_dict": model.state_dict(), "classifier_state_dict": classifier.state_dict(), "cfg": cfg}, ckpt_dir / "best.pt")
        torch.save({"state_dict": model.state_dict(), "classifier_state_dict": classifier.state_dict(), "cfg": cfg}, ckpt_dir / "last.pt")

    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"], strict=False)
    proto_h, proto_e = _compute_support_prototypes(
        model,
        loaders.train,
        device=device,
        num_classes=num_classes,
        shrinkage=float(cfg.get("classifier", {}).get("prototype_shrinkage", 0.0)),
    )
    task_context = model.encode_task_context(proto_h, proto_e)
    test_metrics = _evaluate(
        model,
        classifier,
        loaders.test,
        proto_h,
        proto_e,
        task_context,
        cfg,
        device=device,
        num_classes=num_classes,
    )

    @torch.no_grad()
    def _fn():
        feats, _, _ = next(iter(loaders.test))
        feats = feats.to(device, non_blocking=True)
        zh0, ze0, _ = model.encode(feats)
        zh, ze, _ = model.transport(zh0, ze0, solver=str(cfg["ode"]["solver"]), nfe=int(cfg["eval"]["nfe"]), task_context=task_context)
        return classifier(zh, ze, proto_h=proto_h, proto_e=proto_e, c=model.curvature(), geometry_mode=_geometry_mode(cfg), task_context=task_context)

    throughput, peak_vram = measure_throughput(_fn, sample_count=int(cfg["eval"]["batch_size"]), device=device)
    flags = _failure_flags(cfg, history_rows, test_metrics)
    final_row = history_rows[-1] if history_rows else {}
    out_metrics = {
        **test_metrics,
        "nfe": int(cfg["eval"]["nfe"]),
        "throughput_img_s": throughput,
        "peak_vram_gb": peak_vram,
        "trainable_params": int(sum(p.numel() for p in params)),
        "geometry_mode": _geometry_mode(cfg),
        "best_val_top1": float(best_val),
        "classifier_mode": str(cfg.get("classifier", {}).get("mode", "hybrid")),
        "alpha_h": float(model.hyperbolic_scale()),
        "alpha_e": float(model.euclidean_scale()),
        "gamma_h": float(classifier.gamma_h().detach().cpu().item()),
        "gamma_e": float(classifier.gamma_e().detach().cpu().item()),
        "beta": float(classifier.beta().detach().cpu().item()),
        "branch_gate_prior": float(classifier.branch_gate_prior().detach().cpu().item()),
        "branch_gate_mean": float(final_row.get("branch_gate_mean", 0.0)),
        "hyper_mult_mean": float(final_row.get("hyper_mult_mean", 0.0)),
        "euclidean_mult_mean": float(final_row.get("euclidean_mult_mean", 0.0)),
        "beta_effective_mean": float(final_row.get("beta_effective_mean", 0.0)),
        "task_context_norm": float(final_row.get("task_context_norm", 0.0)),
        "mean_zh_norm": float(final_row.get("mean_zh_norm", 0.0)),
        "mean_logmap0_zh_norm": float(final_row.get("mean_logmap0_zh_norm", 0.0)),
        **flags,
    }
    write_json(out_dir / "metrics.json", out_metrics, indent=int(cfg["logging"]["json_indent"]))
    append_csv(
        cfg["logging"]["summary_csv"],
        {
            "run_name": run_name,
            "dataset": cfg["dataset"]["name"],
            "backbone": cfg["model"]["backbone"],
            "shots": int(cfg["fewshot"]["shots"]),
            "seed": int(cfg["seed"]),
            **out_metrics,
        },
    )
    return {"metrics": out_metrics, "run_dir": str(out_dir)}
