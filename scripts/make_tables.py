"""Generate LaTeX and CSV result tables from summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results_mc_rfm")
    parser.add_argument("--out", type=str, default="results_mc_rfm/tables.tex")
    args = parser.parse_args()

    summary = Path(args.results_dir) / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary}")
    df = pd.read_csv(summary)
    grp = (
        df.groupby(["dataset", "backbone", "geometry_mode", "shots"])
        .agg(
            top1_mean=("top1", "mean"),
            top1_std=("top1", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            nfe=("nfe", "mean"),
            throughput=("throughput_img_s", "mean"),
        )
        .reset_index()
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(grp.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    grp.to_csv(out.with_suffix(".csv"), index=False)
    print(f"Wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

