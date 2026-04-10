"""Generate simple plots from summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/summary.csv")
    parser.add_argument("--outdir", type=str, default="results/plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)
    if df.empty:
        print("Summary CSV is empty, skipping plots.")
        return

    plt.figure(figsize=(8, 4))
    for method, g in df.groupby("method"):
        grouped = g.groupby("shots")["top1"].mean().sort_index()
        plt.plot(grouped.index, grouped.values, marker="o", label=method)
    plt.xlabel("Shots")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Few-shot Accuracy vs Shots")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fewshot_top1.png", dpi=200)
    plt.savefig(outdir / "fewshot_top1.pdf")
    plt.close()
    print(f"Wrote: {outdir}")


if __name__ == "__main__":
    main()

