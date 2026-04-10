#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}

# Experimental priority order
$PYTHON -m src.main experiment=debug data=cifar100 backbone=resnet50 method=linear_probe fewshot.shots=16 seed=42
$PYTHON -m src.main experiment=debug data=cifar100 backbone=resnet50 method=euclidean_fm fewshot.shots=16 seed=42
$PYTHON -m src.main experiment=debug data=cifar100 backbone=resnet50 method=pd_hfm fewshot.shots=16 seed=42

$PYTHON -m src.main experiment=lowshot data=pets backbone=vit_b16 method=pd_hfm fewshot.shots=16 seed=42
$PYTHON -m src.main experiment=lowshot data=pets backbone=vit_b16 method=pd_hfm fewshot.shots=4 seed=42
$PYTHON -m src.main experiment=lowshot data=flowers102 backbone=vit_b16 method=pd_hfm fewshot.shots=4 seed=42
$PYTHON -m src.main experiment=lowshot data=aircraft backbone=vit_b16 method=pd_hfm fewshot.shots=4 seed=42

