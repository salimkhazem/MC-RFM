"""Config utilities for OmegaConf/Hydra."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.utils.io import ensure_dir


def to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def save_resolved_config(cfg: DictConfig, out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    out_path = out_dir / "config_resolved.yaml"
    OmegaConf.save(config=cfg, f=str(out_path), resolve=True)
    return out_path

