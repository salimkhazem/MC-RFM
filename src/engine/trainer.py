"""Training engine for Mixed-Curvature Riemannian Flow Matching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.fewshot_splits import load_or_create_fewshot_indices
from src.data.lmdb_io import LMDBFeatureDataset, FeatureLMDBReader
from src.flow.flow_matching import (
    flow_matching_loss,
    interpolate_product,
    sample_times,
    target_field,
)
from src.geometry.stability import assert_finite
from src.models.adapter import MCRFMAdapter
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


@torch.no_grad()
def _compute_support_prototypes(
    model: MCRFMAdapter,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
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
    proto_h, proto_e = compute_product_prototypes(uh_all, ue_all, y_all, num_classes=num_classes, c=c)
    return proto_h, proto_e


@torch.no_grad()
def _evaluate(
    model: MCRFMAdapter,
    loader: DataLoader,
    proto_h: torch.Tensor,
    proto_e: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
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
        )
        mode = str(cfg["model"]["geometry_mode"]).lower()
        if mode in {"euclidean", "remove_hyper"}:
            logits = -torch.cdist(ze, proto_e, p=2).pow(2)
        elif mode == "hyperbolic":
            logits = product_logits(zh, torch.zeros_like(ze), proto_h, torch.zeros_like(proto_e), c=model.curvature())
        else:
            logits = product_logits(zh, ze, proto_h, proto_e, c=model.curvature())
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
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
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

    global_step = 0
    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        proto_h, proto_e = _compute_support_prototypes(model, loaders.train, device=device, num_classes=num_classes)
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loaders.train, desc=f"train_epoch_{epoch}", leave=False)
        for batch_idx, (feats, labels, _) in enumerate(pbar):
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device)
            zh0, ze0, _ = model.encode(feats)
            zh1 = proto_h[labels]
            ze1 = proto_e[labels]
            t = sample_times(feats.shape[0], device=device, dtype=zh0.dtype, eps=float(cfg["training"]["epsilon_t"]))

            # Hyperbolic geometry ops in float64 for stability.
            zth64, zte = interpolate_product(zh0.double(), ze0, zh1.double(), ze1, t.double(), c=model.curvature())
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

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                vh, ve = model.field(zth, zte, t)
                loss = flow_matching_loss(
                    v_h=vh,
                    v_e=ve,
                    u_h=uh,
                    u_e=ue,
                    z_h=zth,
                    c=model.curvature(),
                    lambda_e=float(cfg["training"]["lambda_euclidean"]),
                    geometry_mode=str(cfg["model"]["geometry_mode"]),
                )
                loss = loss / accum

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

        val_metrics = _evaluate(model, loaders.val, proto_h, proto_e, cfg, device=device, num_classes=num_classes)
        row = {
            "epoch": epoch,
            "train_loss": running / max(len(loaders.train), 1),
            "val_top1": val_metrics["top1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": float(scheduler.get_last_lr()[0]),
            "curvature": model.curvature(),
        }
        append_jsonl(history_path, row)
        logger.info("epoch=%d loss=%.4f val_top1=%.4f val_f1=%.4f", epoch, row["train_loss"], row["val_top1"], row["val_macro_f1"])
        if val_metrics["top1"] > best_val:
            best_val = val_metrics["top1"]
            torch.save({"state_dict": model.state_dict(), "cfg": cfg}, ckpt_dir / "best.pt")
        torch.save({"state_dict": model.state_dict(), "cfg": cfg}, ckpt_dir / "last.pt")

    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    proto_h, proto_e = _compute_support_prototypes(model, loaders.train, device=device, num_classes=num_classes)
    test_metrics = _evaluate(model, loaders.test, proto_h, proto_e, cfg, device=device, num_classes=num_classes)

    @torch.no_grad()
    def _fn():
        feats, _, _ = next(iter(loaders.test))
        feats = feats.to(device, non_blocking=True)
        zh0, ze0, _ = model.encode(feats)
        zh, ze, _ = model.transport(zh0, ze0, solver=str(cfg["ode"]["solver"]), nfe=int(cfg["eval"]["nfe"]))
        return product_logits(zh, ze, proto_h, proto_e, c=model.curvature())

    throughput, peak_vram = measure_throughput(_fn, sample_count=int(cfg["eval"]["batch_size"]), device=device)
    out_metrics = {
        **test_metrics,
        "nfe": int(cfg["eval"]["nfe"]),
        "throughput_img_s": throughput,
        "peak_vram_gb": peak_vram,
        "trainable_params": int(sum(p.numel() for p in params)),
        "geometry_mode": str(cfg["model"]["geometry_mode"]),
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
