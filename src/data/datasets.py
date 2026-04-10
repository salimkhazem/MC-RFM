"""Dataset factories for supported benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets as tvd


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    num_classes: int


class IndexedDataset(Dataset):
    """Wrap a dataset to expose indices in each sample."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        return x, y, index


def _get_targets_from_dataset(ds: Dataset) -> list[int]:
    candidates = ["targets", "_labels", "labels", "_target"]
    for attr in candidates:
        if hasattr(ds, attr):
            values = getattr(ds, attr)
            if isinstance(values, list):
                return [int(v) for v in values]
            if isinstance(values, np.ndarray):
                return values.astype(int).tolist()
            if torch.is_tensor(values):
                return values.to(torch.int64).cpu().tolist()
    labels: list[int] = []
    for i in range(len(ds)):
        sample = ds[i]
        if isinstance(sample, tuple):
            labels.append(int(sample[1]))
        else:
            raise RuntimeError("Dataset sample does not return tuple(image, label)")
    return labels


def get_targets(ds: Dataset) -> list[int]:
    if isinstance(ds, Subset):
        base_targets = _get_targets_from_dataset(ds.dataset)
        return [int(base_targets[i]) for i in ds.indices]
    return _get_targets_from_dataset(ds)


def _split_eurosat(ds: Dataset, split: str, seed: int = 0) -> Subset:
    n = len(ds)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    if split == "train":
        chosen = indices[:n_train]
    elif split == "val":
        chosen = indices[n_train : n_train + n_val]
    elif split == "test":
        chosen = indices[n_train + n_val :]
    else:
        raise ValueError(f"Unknown EuroSAT split: {split}")
    return Subset(ds, chosen.tolist())


def _prepare_tiny_imagenet_val(root: str) -> Path:
    """Convert Tiny-ImageNet val images into an ImageFolder-compatible layout."""
    tiny_root = Path(root) / "tiny-imagenet-200"
    val_dir = tiny_root / "val"
    images_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"
    organized = val_dir / "organized_by_class"

    if not images_dir.exists() or not ann_path.exists():
        raise FileNotFoundError(
            f"TinyImageNet val split not found under {val_dir}. Expected images/ and val_annotations.txt."
        )

    organized.mkdir(parents=True, exist_ok=True)
    missing_sources: list[str] = []
    with ann_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name, class_name = parts[0], parts[1]
            src = images_dir / image_name
            if not src.exists():
                missing_sources.append(image_name)
                continue
            class_dir = organized / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            dst = class_dir / image_name
            if dst.is_symlink() and not dst.exists():
                dst.unlink()
            if dst.exists():
                continue
            try:
                dst.symlink_to(src.resolve())
            except Exception:
                shutil.copy2(src, dst)

    if missing_sources:
        preview = ", ".join(missing_sources[:5])
        more = "" if len(missing_sources) <= 5 else f", ... (+{len(missing_sources) - 5} more)"
        raise FileNotFoundError(
            "TinyImageNet val images missing from val/images. "
            f"Examples: {preview}{more}."
        )
    return organized


def _build_single(name: str, root: str, split: str, transform, download: bool) -> Dataset:
    n = name.lower()
    if n == "cifar10":
        train = split == "train"
        return tvd.CIFAR10(root=root, train=train, transform=transform, download=download)
    if n == "cifar100":
        train = split == "train"
        return tvd.CIFAR100(root=root, train=train, transform=transform, download=download)
    if n == "flowers102":
        return tvd.Flowers102(root=root, split=split, transform=transform, download=download)
    if n == "stanford_cars":
        return tvd.StanfordCars(root=root, split=split, transform=transform, download=download)
    if n == "pets":
        return tvd.OxfordIIITPet(root=root, split=split, target_types="category", transform=transform, download=download)
    if n == "food101":
        return tvd.Food101(root=root, split=split, transform=transform, download=download)
    if n == "dtd":
        return tvd.DTD(root=root, split=split, transform=transform, download=download)
    if n == "eurosat":
        base = tvd.EuroSAT(root=root, transform=transform, download=download)
        return _split_eurosat(base, split=split, seed=0)
    if n == "aircraft":
        return tvd.FGVCAircraft(root=root, split=split, transform=transform, download=download)
    if n == "tinyimagenet":
        if download:
            raise ValueError(
                "TinyImageNet is not downloaded automatically. Place 'tiny-imagenet-200' under the dataset root."
            )
        tiny_root = Path(root) / "tiny-imagenet-200"
        if split == "train":
            train_root = tiny_root / "train"
            if not train_root.exists():
                raise FileNotFoundError(
                    f"TinyImageNet train split not found at {train_root}. Expected an extracted tiny-imagenet-200 tree."
                )
            return tvd.ImageFolder(str(train_root), transform=transform)
        if split in {"val", "test"}:
            return tvd.ImageFolder(str(_prepare_tiny_imagenet_val(root)), transform=transform)
        raise ValueError(f"Unknown TinyImageNet split: {split}")
    raise ValueError(f"Unsupported dataset: {name}")


def build_dataset_bundle(cfg_data: dict[str, Any], train_transform, eval_transform) -> DatasetBundle:
    train = _build_single(
        cfg_data["name"],
        cfg_data["root"],
        cfg_data.get("train_split", "train"),
        train_transform,
        cfg_data.get("download", True),
    )
    val = _build_single(
        cfg_data["name"],
        cfg_data["root"],
        cfg_data.get("val_split", "val"),
        eval_transform,
        cfg_data.get("download", True),
    )
    test = _build_single(
        cfg_data["name"],
        cfg_data["root"],
        cfg_data.get("test_split", "test"),
        eval_transform,
        cfg_data.get("download", True),
    )
    return DatasetBundle(train=train, val=val, test=test, num_classes=int(cfg_data["num_classes"]))
