"""Device helpers."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_autocast(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

