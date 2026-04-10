"""Utility to summarize VRAM from summary CSV """

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/summary.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.summary)
    cols = ["dataset", "backbone", "method", "peak_vram_gb", "trainable_params"]
    print(df[cols].sort_values("peak_vram_gb", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

