"""Frozen feature extraction and disk caching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.io import ensure_dir


@dataclass
class FeatureCache:
    features: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor
    meta: dict[str, Any]


def feature_cache_path(cache_root: str | Path, dataset: str, backbone: str, split: str, cache_tag: str = "default") -> Path:
    cache_root = ensure_dir(cache_root)
    return cache_root / dataset / backbone / cache_tag / f"{split}.pt"


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> FeatureCache:
    model.eval()
    feats = []
    labels = []
    indices = []
    for batch in tqdm(dataloader, desc="extract_features", leave=False):
        x, y, idx = batch
        x = x.to(device, non_blocking=True)
        out = model(x)
        feats.append(out.detach().cpu())
        labels.append(y.detach().cpu())
        indices.append(idx.detach().cpu())
    return FeatureCache(
        features=torch.cat(feats, dim=0),
        labels=torch.cat(labels, dim=0).to(torch.long),
        indices=torch.cat(indices, dim=0).to(torch.long),
        meta={},
    )


def save_feature_cache(path: str | Path, cache: FeatureCache) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "features": cache.features,
        "labels": cache.labels,
        "indices": cache.indices,
        "meta": cache.meta,
    }
    torch.save(payload, path)


def load_feature_cache(path: str | Path) -> FeatureCache:
    payload = torch.load(path, map_location="cpu")
    return FeatureCache(
        features=payload["features"].float(),
        labels=payload["labels"].long(),
        indices=payload["indices"].long(),
        meta=payload.get("meta", {}),
    )
