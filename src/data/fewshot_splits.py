"""Deterministic few-shot split generation and caching."""

from __future__ import annotations

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


def split_cache_path(split_dir: str | Path, dataset: str, split: str, shots: int, seed: int, n: int) -> Path:
    split_dir = ensure_dir(split_dir)
    return split_dir / f"{dataset}_{split}_{shots}shot_seed{seed}_n{n}.json"


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
        import json

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

