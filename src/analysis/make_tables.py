"""Build summary tables from run CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/summary.csv")
    parser.add_argument("--outdir", type=str, default="results/tables")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)
    group_cols = ["dataset", "backbone", "method", "shots"]
    agg = (
        df.groupby(group_cols)
        .agg(
            top1_mean=("top1", "mean"),
            top1_std=("top1", "std"),
            f1_mean=("macro_f1", "mean"),
            params=("trainable_params", "mean"),
            nfe=("nfe", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(outdir / "main_table.csv", index=False)
    (outdir / "main_table.md").write_text(agg.to_markdown(index=False), encoding="utf-8")
    print(f"Wrote: {outdir / 'main_table.csv'}")


if __name__ == "__main__":
    main()

