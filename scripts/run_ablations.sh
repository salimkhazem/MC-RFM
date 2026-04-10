#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}

$PYTHON -m src.analysis.ablations --python "$PYTHON" --dataset cifar100 --backbone resnet50 --shots 4 --seed 42

