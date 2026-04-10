#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}

$PYTHON -m src.main experiment=debug mode=train data=cifar100 backbone=resnet50 method=linear_probe training.epochs=1 task.use_feature_cache=true task.cache_if_missing=true task.cache_tag=smoke_cache
