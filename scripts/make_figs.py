"""Generate basic ablation plots from summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results_mc_rfm")
    parser.add_argument("--out", type=str, default="results_mc_rfm/figs")
    args = parser.parse_args()

    summary = Path(args.results_dir) / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary}")
    df = pd.read_csv(summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if "nfe" in df.columns:
        plt.figure(figsize=(6, 4))
        for mode, g in df.groupby("geometry_mode"):
            grouped = g.groupby("nfe")["top1"].mean().sort_index()
            plt.plot(grouped.index, grouped.values, marker="o", label=mode)
        plt.xlabel("NFE")
        plt.ylabel("Top-1")
        plt.title("Accuracy vs NFE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "accuracy_vs_nfe.png", dpi=220)
        plt.savefig(out / "accuracy_vs_nfe.pdf")
        plt.close()

    if "throughput_img_s" in df.columns:
        plt.figure(figsize=(6, 4))
        for mode, g in df.groupby("geometry_mode"):
            plt.scatter(g["throughput_img_s"], g["top1"], label=mode, alpha=0.8)
        plt.xlabel("Throughput (img/s)")
        plt.ylabel("Top-1")
        plt.title("Accuracy vs Throughput")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "accuracy_vs_throughput.png", dpi=220)
        plt.savefig(out / "accuracy_vs_throughput.pdf")
        plt.close()

    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()

