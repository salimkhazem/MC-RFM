"""Deterministic few-shot and protocol split generation and caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.utils.logging import ensure_dir, write_json


def sample_fewshot(labels: Iterable[int], shots: int, seed: int) -> list[int]:
    y = np.asarray(list(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    out: list[int] = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        if len(idx) < shots:
            raise ValueError(f"Class {cls} has only {len(idx)} samples, but shots={shots}")
        chosen = rng.choice(idx, size=shots, replace=False)
        out.extend(chosen.tolist())
    out.sort()
    return out


def sample_stratified_train_val(
    labels: Iterable[int],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    y = np.asarray(list(labels), dtype=np.int64)
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        if len(idx) < 2:
            raise ValueError(f"Class {cls} has only {len(idx)} samples; cannot form train/val split")
        idx = idx.copy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * float(val_ratio)))
        n_val = max(1, n_val)
        n_val = min(n_val, len(idx) - 1)
        val_part = np.sort(idx[:n_val]).tolist()
        train_part = np.sort(idx[n_val:]).tolist()
        train_idx.extend(train_part)
        val_idx.extend(val_part)
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def sample_stratified_subset(labels: Iterable[int], max_samples: int, seed: int) -> list[int]:
    y = np.asarray(list(labels), dtype=np.int64)
    if max_samples <= 0 or max_samples >= len(y):
        return list(range(len(y)))
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    by_class: dict[int, np.ndarray] = {}
    for cls in classes:
        idx = np.where(y == cls)[0].copy()
        rng.shuffle(idx)
        by_class[int(cls)] = idx

    base_take = max_samples // max(len(classes), 1)
    chosen: dict[int, int] = {int(cls): min(base_take, len(by_class[int(cls)])) for cls in classes}
    remaining = max_samples - sum(chosen.values())

    while remaining > 0:
        candidates = [int(cls) for cls in classes if chosen[int(cls)] < len(by_class[int(cls)])]
        if not candidates:
            break
        rng.shuffle(candidates)
        candidates.sort(key=lambda cls: len(by_class[cls]) - chosen[cls], reverse=True)
        progress = False
        for cls in candidates:
            if remaining <= 0:
                break
            if chosen[cls] >= len(by_class[cls]):
                continue
            chosen[cls] += 1
            remaining -= 1
            progress = True
        if not progress:
            break

    out: list[int] = []
    for cls in classes:
        cls_idx = by_class[int(cls)]
        out.extend(np.sort(cls_idx[: chosen[int(cls)]]).tolist())
    out.sort()
    return out


def split_cache_path(split_dir: str | Path, dataset: str, split: str, shots: int, seed: int, n: int) -> Path:
    split_dir = ensure_dir(split_dir)
    return split_dir / f"{dataset}_{split}_{shots}shot_seed{seed}_n{n}.json"


def internal_val_cache_path(
    split_dir: str | Path,
    dataset: str,
    source_split: str,
    seed: int,
    val_ratio: float,
    n: int,
) -> Path:
    split_dir = ensure_dir(split_dir)
    ratio_tag = str(val_ratio).replace(".", "p")
    return split_dir / f"{dataset}_{source_split}_internal_val_seed{seed}_ratio{ratio_tag}_n{n}.json"


def load_or_create_fewshot_indices(
    split_dir: str | Path,
    dataset: str,
    split: str,
    labels: Iterable[int],
    shots: int,
    seed: int,
) -> list[int]:
    labels_list = [int(x) for x in labels]
    path = split_cache_path(split_dir, dataset, split, shots, seed, n=len(labels_list))
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return [int(i) for i in payload["indices"]]
    idx = sample_fewshot(labels_list, shots=shots, seed=seed)
    write_json(
        path,
        {
            "dataset": dataset,
            "split": split,
            "shots": int(shots),
            "seed": int(seed),
            "num_samples": int(len(labels_list)),
            "indices": idx,
        },
    )
    return idx


def load_or_create_internal_val_indices(
    split_dir: str | Path,
    dataset: str,
    source_split: str,
    labels: Iterable[int],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    labels_list = [int(x) for x in labels]
    path = internal_val_cache_path(
        split_dir,
        dataset=dataset,
        source_split=source_split,
        seed=seed,
        val_ratio=val_ratio,
        n=len(labels_list),
    )
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return [int(i) for i in payload["train_indices"]], [int(i) for i in payload["val_indices"]]

    train_idx, val_idx = sample_stratified_train_val(labels_list, val_ratio=val_ratio, seed=seed)
    write_json(
        path,
        {
            "dataset": dataset,
            "source_split": source_split,
            "seed": int(seed),
            "val_ratio": float(val_ratio),
            "num_samples": int(len(labels_list)),
            "train_indices": train_idx,
            "val_indices": val_idx,
        },
    )
    return train_idx, val_idx
