#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}

$PYTHON -m src.analysis.make_tables --summary results/summary.csv --outdir results/tables
$PYTHON -m src.analysis.make_plots --summary results/summary.csv --outdir results/plots
$PYTHON -m src.analysis.throughput --summary results/summary.csv
$PYTHON -m src.analysis.vram --summary results/summary.csv

