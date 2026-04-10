"""Collate functions."""

from __future__ import annotations

from typing import Any

import torch


def default_collate_with_index(batch: list[tuple[Any, int, int]]):
    images = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.tensor([int(x[1]) for x in batch], dtype=torch.long)
    indices = torch.tensor([int(x[2]) for x in batch], dtype=torch.long)
    return images, labels, indices


def feature_collate(batch: list[tuple[torch.Tensor, int]]):
    feats = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.tensor([int(x[1]) for x in batch], dtype=torch.long)
    return feats, labels

