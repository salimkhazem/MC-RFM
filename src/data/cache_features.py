"""Cache frozen backbone features into LMDB."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import build_dataset_bundle, extract_targets
from src.data.fewshot_splits import sample_stratified_subset
from src.data.lmdb_io import FeatureLMDBWriter, LMDBMeta
from src.models.backbone import FrozenBackbone
from src.utils.config import load_config
from src.utils.logging import ensure_dir, setup_logger
from src.utils.seed import seed_everything


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, y = self.base[index]
        return x, int(y), int(index)


def _cache_split(
    split_name: str,
    ds,
    backbone: FrozenBackbone,
    cfg: dict,
    device: torch.device,
    out_path: Path,
    logger,
) -> None:
    loader = DataLoader(
        IndexedDataset(ds),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["device"]["num_workers"]),
        pin_memory=bool(cfg["device"]["pin_memory"]),
    )
    writer = FeatureLMDBWriter(out_path, map_size_gb=int(cfg["cache"]["map_size_gb"]))
    feat_dim = None
    for x, y, idx in tqdm(loader, desc=f"cache_{split_name}", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            feat = backbone(x).detach().cpu().numpy().astype(np.float32)
        if feat_dim is None:
            feat_dim = int(feat.shape[-1])
        entries = [(int(idx[j]), feat[j], int(y[j])) for j in range(feat.shape[0])]
        writer.put_many(entries)
    writer.put_meta(
        LMDBMeta(
            num_samples=len(ds),
            feature_dim=int(feat_dim or 0),
            dataset=cfg["dataset"]["name"],
            split=split_name,
            backbone=cfg["model"]["backbone"],
        )
    )
    writer.close()
    logger.info("cached %s -> %s", split_name, out_path)


def run_cache(cfg: dict) -> None:
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["device"]["deterministic"]))
    out_root = ensure_dir(Path(cfg["cache"]["lmdb_dir"]) / cfg["dataset"]["name"] / cfg["model"]["backbone"])
    logger = setup_logger(out_root / "cache.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_dataset_bundle(cfg["dataset"], split_dir=cfg["fewshot"]["split_dir"])
    max_samples = int(cfg["cache"].get("max_samples", 0))
    if max_samples > 0:
        split_offsets = {"train": 0, "val": 1, "test": 2}

        def _sub(ds, split_name: str):
            labels = extract_targets(ds)
            indices = sample_stratified_subset(
                labels,
                max_samples=max_samples,
                seed=int(cfg["seed"]) + split_offsets[split_name],
            )
            return Subset(ds, indices)

        bundle.train = _sub(bundle.train, "train")
        bundle.val = _sub(bundle.val, "val")
        bundle.test = _sub(bundle.test, "test")
    backbone = FrozenBackbone(cfg["model"]["backbone"], pretrained=bool(cfg["model"]["pretrained"])).to(device)

    for split_name, ds in [("train", bundle.train), ("val", bundle.val), ("test", bundle.test)]:
        out_path = out_root / f"{split_name}.lmdb"
        if out_path.exists() and not bool(cfg["cache"]["force_rebuild"]):
            logger.info("skip existing LMDB: %s", out_path)
            continue
        _cache_split(split_name, ds, backbone, cfg, device, out_path, logger)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/mcfm_default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_cache(cfg)


if __name__ == "__main__":
    main()
