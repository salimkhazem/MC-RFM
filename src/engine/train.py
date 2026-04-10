"""CLI entrypoint for MC-RFM training."""

from __future__ import annotations

import argparse

from src.engine.trainer import train
from src.utils.config import load_config
from src.utils.logging import make_run_name
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/mcfm_default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if not cfg["project"].get("run_name"):
        cfg["project"]["run_name"] = make_run_name(cfg)
    cfg["fewshot"]["split_seed"] = int(cfg["seed"])
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["device"]["deterministic"]))
    out = train(cfg)
    print(out["metrics"])


if __name__ == "__main__":
    main()

