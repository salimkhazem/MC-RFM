"""Main CLI entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from src.backbones.factory import build_backbone
from src.backbones.feature_extractors import infer_feature_dim
from src.data.collate import default_collate_with_index
from src.data.datasets import IndexedDataset, build_dataset_bundle, get_targets
from src.data.feature_cache import extract_features, feature_cache_path, load_feature_cache, save_feature_cache
from src.data.fewshot import load_or_create_fewshot_indices
from src.data.transforms import build_transforms
from src.engine.evaluate import run_evaluate
from src.engine.logging_utils import dump_run_metadata, log_metrics, prepare_run_dirs
from src.engine.train import run_train
from src.engine.trainer import FeatureSplits
from src.utils.config_utils import to_container
from src.utils.device import get_device
from src.utils.io import resolve_git_hash
from src.utils.seed import seed_everything

LOGGER = logging.getLogger("pdhfm")


def _configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


def _to_loader(ds, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        IndexedDataset(ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=default_collate_with_index,
    )


def _sort_cache(features: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(indices)
    return features[order], labels[order]


def _load_or_extract_split(
    split_name: str,
    ds,
    cfg: dict[str, Any],
    backbone: torch.nn.Module | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = feature_cache_path(
        cfg["project"]["cache_root"],
        dataset=cfg["data"]["name"],
        backbone=cfg["backbone"]["name"],
        split=split_name,
        cache_tag=str(cfg["task"].get("cache_tag", "default")),
    )
    if path.exists():
        cache = load_feature_cache(path)
        feats, labels = _sort_cache(cache.features, cache.labels, cache.indices)
        return feats, labels

    if not cfg["task"].get("cache_if_missing", True):
        raise FileNotFoundError(f"Missing feature cache: {path}")
    if backbone is None:
        raise RuntimeError(f"Feature cache missing for split '{split_name}' and no backbone available for extraction.")

    loader = _to_loader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["device"]["num_workers"]),
        pin_memory=bool(cfg["device"]["pin_memory"]),
    )
    cache = extract_features(backbone, loader, device)
    cache.meta = {
        "dataset": cfg["data"]["name"],
        "split": split_name,
        "backbone": cfg["backbone"]["name"],
        "feature_dim": int(cache.features.shape[-1]),
    }
    save_feature_cache(path, cache)
    feats, labels = _sort_cache(cache.features, cache.labels, cache.indices)
    return feats, labels


def load_feature_splits(cfg: dict[str, Any], device: torch.device) -> tuple[FeatureSplits, int]:
    train_tf = build_transforms(
        image_size=int(cfg["data"]["image_size"]),
        train=True,
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
    )
    eval_tf = build_transforms(
        image_size=int(cfg["data"]["image_size"]),
        train=False,
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
    )
    bundle = build_dataset_bundle(cfg["data"], train_transform=train_tf, eval_transform=eval_tf)
    num_classes = int(bundle.num_classes)

    debug_subset = int(cfg["task"].get("debug_subset", 0))
    if debug_subset > 0:
        def _sub(ds, n):
            n = min(n, len(ds))
            return Subset(ds, list(range(n)))

        bundle.train = _sub(bundle.train, debug_subset)
        bundle.val = _sub(bundle.val, max(debug_subset // 2, 1))
        bundle.test = _sub(bundle.test, max(debug_subset // 2, 1))

    cache_tag = str(cfg["task"].get("cache_tag", "default"))
    cache_paths = {
        split: feature_cache_path(
            cfg["project"]["cache_root"],
            dataset=cfg["data"]["name"],
            backbone=cfg["backbone"]["name"],
            split=split,
            cache_tag=cache_tag,
        )
        for split in ["train", "val", "test"]
    }
    backbone: torch.nn.Module | None = None
    if not all(p.exists() for p in cache_paths.values()):
        backbone = build_backbone(cfg["backbone"]).to(device)
        _ = infer_feature_dim(backbone, image_size=int(cfg["data"]["image_size"]), device=device)

    train_features, train_labels = _load_or_extract_split("train", bundle.train, cfg, backbone, device)
    val_features, val_labels = _load_or_extract_split("val", bundle.val, cfg, backbone, device)
    test_features, test_labels = _load_or_extract_split("test", bundle.test, cfg, backbone, device)

    if bool(cfg["fewshot"]["enabled"]):
        train_targets = get_targets(bundle.train)
        fewshot_idx = load_or_create_fewshot_indices(
            cache_dir=cfg["fewshot"]["cache_dir"],
            dataset_name=cfg["data"]["name"],
            split="train",
            labels=train_targets,
            shots=int(cfg["fewshot"]["shots"]),
            seed=int(cfg["fewshot"]["split_seed"]),
        )
        train_features = train_features[fewshot_idx]
        train_labels = train_labels[fewshot_idx]

    return (
        FeatureSplits(
            train_features=train_features.float().contiguous(),
            train_labels=train_labels.long().contiguous(),
            val_features=val_features.float().contiguous(),
            val_labels=val_labels.long().contiguous(),
            test_features=test_features.float().contiguous(),
            test_labels=test_labels.long().contiguous(),
        ),
        num_classes,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed), deterministic=True)
    cfg_dict = to_container(cfg)
    device = get_device()
    run_dirs = prepare_run_dirs(cfg_dict)
    _configure_logging(run_dirs["log_dir"] / "stdout.log")
    git_hash = resolve_git_hash()
    dump_run_metadata(cfg, cfg_dict, run_dirs["out_dir"], git_hash=git_hash)

    splits, num_classes = load_feature_splits(cfg_dict, device=device)
    for name, tensor in [
        ("train_features", splits.train_features),
        ("val_features", splits.val_features),
        ("test_features", splits.test_features),
    ]:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values found in {name}")

    if cfg_dict["mode"] == "train":
        metrics = run_train(
            cfg_dict=cfg_dict,
            splits=splits,
            num_classes=num_classes,
            device=device,
            out_dir=run_dirs["out_dir"],
            ckpt_dir=run_dirs["ckpt_dir"],
        )
        LOGGER.info("train metrics: %s", metrics)
        return

    if cfg_dict["mode"] == "eval":
        ckpt_path_str = str(cfg_dict["evaluation"].get("checkpoint_path", "")).strip()
        ckpt_path = Path(ckpt_path_str) if ckpt_path_str else (run_dirs["ckpt_dir"] / "best.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for eval mode: {ckpt_path}")
        metrics = run_evaluate(
            cfg_dict=cfg_dict,
            splits=splits,
            num_classes=num_classes,
            device=device,
            checkpoint_path=ckpt_path,
        )
        log_metrics(run_dirs["out_dir"], metrics)
        LOGGER.info("eval metrics: %s", metrics)
        return

    raise ValueError(f"Unsupported mode: {cfg_dict['mode']}")


if __name__ == "__main__":
    main()
