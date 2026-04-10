"""Deterministic few-shot split utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from src.utils.io import ensure_dir, read_json, write_json


def sample_fewshot_indices(labels: Iterable[int], shots: int, seed: int) -> list[int]:
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    classes = np.unique(labels_arr)
    rng = np.random.default_rng(seed)
    indices: list[int] = []

    for cls in classes:
        cls_indices = np.where(labels_arr == cls)[0]
        if len(cls_indices) < shots:
            raise ValueError(f"Class {cls} has {len(cls_indices)} samples < requested {shots}")
        selected = rng.choice(cls_indices, size=shots, replace=False)
        indices.extend(selected.tolist())

    indices = sorted(indices)
    return indices


def fewshot_cache_path(
    cache_dir: str | Path,
    dataset_name: str,
    split: str,
    shots: int,
    seed: int,
    num_samples: int | None = None,
) -> Path:
    cache_dir = ensure_dir(cache_dir)
    if num_samples is None:
        return cache_dir / f"{dataset_name}_{split}_{shots}shot_seed{seed}.json"
    return cache_dir / f"{dataset_name}_{split}_{shots}shot_seed{seed}_n{num_samples}.json"


def load_or_create_fewshot_indices(
    cache_dir: str | Path,
    dataset_name: str,
    split: str,
    labels: Iterable[int],
    shots: int,
    seed: int,
) -> list[int]:
    labels_list = [int(x) for x in labels]
    path = fewshot_cache_path(cache_dir, dataset_name, split, shots, seed, num_samples=len(labels_list))
    if path.exists():
        payload = read_json(path)
        return [int(x) for x in payload["indices"]]
    # Backward-compat fallback: older cache naming without sample count.
    legacy_path = fewshot_cache_path(cache_dir, dataset_name, split, shots, seed, num_samples=None)
    if legacy_path.exists():
        payload = read_json(legacy_path)
        indices = [int(x) for x in payload["indices"]]
        if len(indices) > 0 and max(indices) < len(labels_list):
            return indices

    indices = sample_fewshot_indices(labels=labels_list, shots=shots, seed=seed)
    payload = {
        "dataset": dataset_name,
        "split": split,
        "shots": int(shots),
        "seed": int(seed),
        "num_samples": int(len(labels_list)),
        "indices": indices,
    }
    write_json(path, payload)
    return indices
