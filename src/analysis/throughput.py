"""Utility to summarize throughput from summary CSV."""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/summary.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.summary)
    cols = ["dataset", "backbone", "method", "throughput_samples_per_sec", "nfe"]
    print(df[cols].sort_values("throughput_samples_per_sec", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

