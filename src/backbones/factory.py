"""Backbone factory."""

from __future__ import annotations

from typing import Any

from src.backbones.timm_backbone import TimmFeatureExtractor


def build_backbone(cfg_backbone: dict[str, Any]):
    return TimmFeatureExtractor(
        timm_name=cfg_backbone["timm_name"],
        pretrained=bool(cfg_backbone.get("pretrained", True)),
        pool=cfg_backbone.get("pool", "avg"),
    )

