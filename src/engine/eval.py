"""CLI entrypoint for MC-RFM evaluation."""

from __future__ import annotations

import argparse

from src.engine.evaluator import evaluate_checkpoint
from src.utils.config import load_config
from src.utils.logging import make_run_name
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/mcfm_default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if not cfg["project"].get("run_name"):
        cfg["project"]["run_name"] = make_run_name(cfg) + "_eval"
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["device"]["deterministic"]))
    metrics = evaluate_checkpoint(cfg, args.checkpoint)
    print(metrics)


if __name__ == "__main__":
    main()

