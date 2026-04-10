"""Torchvision-style dataset helpers for MC-RFM."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets as tvd, transforms


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    num_classes: int


DEFAULT_SHARED_DATA_ROOTS = [
    "/mnt/storage_2_10T/skhazem/Projects/eccv/adapter-tune/adaptertune-eccv2026/data",
    "/mnt/storage_2_10T/skhazem/Projects/pdhfm/data",
]

DATASET_MARKERS: dict[str, list[str]] = {
    "cifar10": ["cifar-10-batches-py"],
    "cifar100": ["cifar-100-python"],
    "dtd": ["dtd", "dtd/images"],
    "pets": ["oxford-iiit-pet", "oxford-iiit-pet/images"],
    "aircraft": ["fgvc-aircraft-2013b", "fgvc-aircraft-2013b/data"],
    "flowers102": ["flowers-102", "flowers-102/jpg"],
    "stanford_cars": ["stanford-cars", "stanford_cars", "car_data"],
    "food101": ["food-101", "food-101/images"],
    "eurosat": ["eurosat", "eurosat/2750", "2750", "EuroSAT"],
    "tinyimagenet": ["tiny-imagenet-200", "tiny-imagenet-200/train", "tiny-imagenet-200/val"],
}


def _shared_roots() -> list[Path]:
    roots: list[Path] = []
    env_value = os.environ.get("MC_RFM_SHARED_DATA_ROOTS", "")
    if env_value:
        for item in env_value.split(os.pathsep):
            item = item.strip()
            if item:
                roots.append(Path(item))
    roots.extend(Path(p) for p in DEFAULT_SHARED_DATA_ROOTS)

    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def dataset_available_in_root(dataset_name: str, root: str | Path) -> bool:
    markers = DATASET_MARKERS.get(dataset_name, [dataset_name])
    root_path = Path(root)
    return any((root_path / marker).exists() for marker in markers)


def resolve_dataset_root(dataset_name: str, primary_root: str | None = None) -> tuple[str, bool]:
    primary = Path(primary_root or os.environ.get("MC_RFM_DATA_ROOT", "./data"))
    if dataset_available_in_root(dataset_name, primary):
        return str(primary), True
    for shared_root in _shared_roots():
        if dataset_available_in_root(dataset_name, shared_root):
            return str(shared_root), True
    return str(primary), False


def build_transforms(image_size: int, mean: list[float], std: list[float], train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _eurosat_split(base: Dataset, split: str, seed: int = 0) -> Dataset:
    n = len(base)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    if split == "train":
        chosen = idx[:n_train]
    elif split == "val":
        chosen = idx[n_train : n_train + n_val]
    elif split == "test":
        chosen = idx[n_train + n_val :]
    else:
        raise ValueError(f"Unsupported EuroSAT split: {split}")
    return Subset(base, chosen.tolist())


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


def _make_single(cfg: dict[str, Any], split: str, tfm: transforms.Compose) -> Dataset:
    name = cfg["name"].lower()
    root, found = resolve_dataset_root(name, cfg.get("root"))
    download = bool(cfg.get("download", True)) and not found
    if name == "cifar10":
        return tvd.CIFAR10(root=root, train=(split == "train"), transform=tfm, download=download)
    if name == "cifar100":
        return tvd.CIFAR100(root=root, train=(split == "train"), transform=tfm, download=download)
    if name == "dtd":
        return tvd.DTD(root=root, split=split, transform=tfm, download=download)
    if name == "pets":
        return tvd.OxfordIIITPet(root=root, split=split, target_types="category", transform=tfm, download=download)
    if name == "aircraft":
        return tvd.FGVCAircraft(root=root, split=split, transform=tfm, download=download)
    if name == "flowers102":
        return tvd.Flowers102(root=root, split=split, transform=tfm, download=download)
    if name == "stanford_cars":
        return tvd.StanfordCars(root=root, split=split, transform=tfm, download=download)
    if name == "food101":
        return tvd.Food101(root=root, split=split, transform=tfm, download=download)
    if name == "eurosat":
        base = tvd.EuroSAT(root=root, transform=tfm, download=download)
        return _eurosat_split(base, split=split, seed=0)
    if name == "tinyimagenet":
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
            return tvd.ImageFolder(str(train_root), transform=tfm)
        if split in {"val", "test"}:
            return tvd.ImageFolder(str(_prepare_tiny_imagenet_val(root)), transform=tfm)
        raise ValueError(f"Unsupported TinyImageNet split: {split}")
    raise ValueError(f"Unsupported dataset: {cfg['name']}")


def build_dataset_bundle(cfg: dict[str, Any]) -> DatasetBundle:
    train_t = build_transforms(cfg["image_size"], cfg["mean"], cfg["std"], train=True)
    eval_t = build_transforms(cfg["image_size"], cfg["mean"], cfg["std"], train=False)
    train_ds = _make_single(cfg, cfg.get("train_split", "train"), train_t)
    val_ds = _make_single(cfg, cfg.get("val_split", "val"), eval_t)
    test_ds = _make_single(cfg, cfg.get("test_split", "test"), eval_t)
    return DatasetBundle(train=train_ds, val=val_ds, test=test_ds, num_classes=int(cfg["num_classes"]))


def extract_targets(ds: Dataset) -> list[int]:
    if isinstance(ds, Subset):
        base = extract_targets(ds.dataset)
        return [int(base[i]) for i in ds.indices]
    for attr in ["targets", "_labels", "labels"]:
        if hasattr(ds, attr):
            val = getattr(ds, attr)
            if isinstance(val, list):
                return [int(x) for x in val]
            if isinstance(val, np.ndarray):
                return val.astype(np.int64).tolist()
    labels: list[int] = []
    for i in range(len(ds)):
        _, y = ds[i]
        labels.append(int(y))
    return labels
