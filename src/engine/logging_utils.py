"""Run logging utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.utils.config_utils import save_resolved_config
from src.utils.io import append_jsonl, append_summary_csv, ensure_dir, write_json


def prepare_run_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    run_name = cfg["project"]["run_name"]
    out_dir = ensure_dir(Path(cfg["project"]["output_root"]) / run_name)
    log_dir = ensure_dir(Path(cfg["project"]["log_root"]) / run_name)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    return {"out_dir": out_dir, "log_dir": log_dir, "ckpt_dir": ckpt_dir}


def dump_run_metadata(cfg, cfg_dict: dict[str, Any], out_dir: Path, git_hash: str) -> None:
    save_resolved_config(cfg, out_dir)
    (out_dir / "git_hash.txt").write_text(git_hash + "\n", encoding="utf-8")
    write_json(out_dir / "run_meta.json", {"pid": os.getpid(), "git_hash": git_hash})
    write_json(out_dir / "config_resolved.json", cfg_dict)


def log_history(out_dir: Path, history: list[dict[str, Any]]) -> None:
    history_path = out_dir / "train_history.jsonl"
    for row in history:
        append_jsonl(history_path, row)


def log_metrics(out_dir: Path, metrics: dict[str, Any]) -> None:
    write_json(out_dir / "metrics.json", metrics)


def append_summary(summary_csv: str, row: dict[str, Any]) -> None:
    append_summary_csv(summary_csv, row)

