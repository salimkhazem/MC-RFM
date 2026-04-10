"""Programmatic ablation launcher."""

from __future__ import annotations

import argparse
import subprocess


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = [
        args.python,
        "-m",
        "src.main",
        "experiment=debug",
        f"data={args.dataset}",
        f"backbone={args.backbone}",
        f"fewshot.shots={args.shots}",
        f"seed={args.seed}",
    ]

    # Ablation A: geometry
    _run(base + ["method=euclidean_fm"])
    _run(base + ["method=pd_hfm"])

    # Ablation B: decoupling
    _run(base + ["method=pd_hfm", "method.vector_field.coupled=true"])
    _run(base + ["method=pd_hfm", "method.vector_field.coupled=false"])

    # Ablation C: solver NFE
    for nfe in [1, 3, 5, 10]:
        _run(base + ["method=pd_hfm", f"method.flow.nfe_eval={nfe}"])

    # Ablation D: curvature
    for c in [0.1, 0.5, 1.0]:
        _run(base + ["method=pd_hfm", f"method.geometry.curvature={c}"])


if __name__ == "__main__":
    main()

