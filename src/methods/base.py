"""Base interface for adaptation methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_feature_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(features, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


class AdaptationMethod(ABC):
    @abstractmethod
    def fit(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        cfg: dict[str, Any],
        device: torch.device,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict_logits(self, features: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def trainable_parameters(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError

