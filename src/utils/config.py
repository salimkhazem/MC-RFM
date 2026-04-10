"""Lightweight YAML config loader with dotted overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dst)
    for k, v in src.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _parse_value(raw: str) -> Any:
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "none" or low == "null":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return yaml.safe_load(raw)
        except Exception:
            return raw
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = _read_yaml(path)
    includes = cfg.pop("include", [])
    merged: dict[str, Any] = {}
    for inc in includes:
        inc_cfg = load_config(inc, overrides=None)
        merged = _deep_merge(merged, inc_cfg)
    cfg = _deep_merge(merged, cfg)

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value: {ov}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k, _parse_value(v))
    return cfg

