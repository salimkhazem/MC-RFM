#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
CONFIG=${CONFIG:-configs/experiments/mcfm_default.yaml}

$PYTHON -m src.data.cache_features --config "$CONFIG" "$@"
$PYTHON -m src.engine.train --config "$CONFIG" "$@"

