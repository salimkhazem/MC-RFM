"""Run ablations for MC-RFM."""

from __future__ import annotations

import argparse
import copy

from src.engine.trainer import train
from src.utils.config import load_config
from src.utils.logging import make_run_name
from src.utils.seed import seed_everything


def _with_run_name(cfg: dict, suffix: str) -> dict:
    out = copy.deepcopy(cfg)
    base = out["project"].get("run_name", "") or make_run_name(out)
    out["project"]["run_name"] = f"{base}_{suffix}"
    return out


def run_ablations(cfg: dict) -> None:
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["device"]["deterministic"]))
    ab = cfg["ablation"]
    jobs = []

    for mode in ab["geometry_modes"]:
        c = _with_run_name(cfg, f"geom_{mode}")
        c["model"]["geometry_mode"] = mode
        jobs.append(c)

    for nfe in ab["nfe_values"]:
        c = _with_run_name(cfg, f"nfe_{nfe}")
        c["eval"]["nfe"] = int(nfe)
        c["ode"]["nfe"] = int(nfe)
        jobs.append(c)

    for dh, de in ab["signatures"]:
        c = _with_run_name(cfg, f"sig_{dh}_{de}")
        c["model"]["dh"] = int(dh)
        c["model"]["de"] = int(de)
        c["model"]["bottleneck_dim"] = int(dh) + int(de)
        jobs.append(c)

    for curv in ab["curvature_values"]:
        c = _with_run_name(cfg, f"curv_{curv}")
        c["model"]["curvature"] = float(curv)
        jobs.append(c)

    for dec in ab["decoupled_heads"]:
        c = _with_run_name(cfg, f"decoupled_{str(dec).lower()}")
        c["model"]["decoupled_heads"] = bool(dec)
        jobs.append(c)

    seen = set()
    for job in jobs:
        name = job["project"]["run_name"]
        if name in seen:
            continue
        seen.add(name)
        print(f"[ablate] running {name}")
        train(job)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/ablations.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_ablations(cfg)


if __name__ == "__main__":
    main()

