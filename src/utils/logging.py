"""Logging and artifact helpers."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(log_file: str | Path, level: int = logging.INFO) -> logging.Logger:
    log_file = Path(log_file)
    ensure_dir(log_file.parent)
    logger = logging.getLogger("mc_rfm")
    logger.setLevel(level)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def write_json(path: str | Path, payload: dict[str, Any], indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, sort_keys=True)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_run_name(cfg: dict[str, Any]) -> str:
    explicit = str(cfg["project"].get("run_name", "")).strip()
    if explicit:
        return explicit
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{now}_{cfg['dataset']['name']}_{cfg['model']['backbone']}_shots{cfg['fewshot']['shots']}_seed{cfg['seed']}"


def run_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    run_name = make_run_name(cfg)
    root = ensure_dir(cfg["project"]["output_root"])
    out_dir = ensure_dir(root / run_name)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    return {"run_name": run_name, "out_dir": out_dir, "ckpt_dir": ckpt_dir}


def save_config_snapshot(cfg: dict[str, Any], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    write_json(out_dir / "config_resolved.json", cfg)

