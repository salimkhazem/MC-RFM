# MC-RFM: Mixed-Curvature Riemannian Flow Matching

Official Implementation of **MC-RFM: Mixed-Curvature Riemannian Flow Matching** . An **parameter-efficient vision adaptation** with frozen backbones and continuous-time adapters on a product manifold:

\[
\mathcal{M} = \mathbb{D}_c^{d_h} \times \mathbb{R}^{d_e}
\]

Hyperbolic branch models hierarchical semantics; Euclidean branch models non-hierarchical residual factors.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Root

```bash
export MC_RFM_DATA_ROOT=/path/to/datasets
```

Supported now: `cifar10`, `cifar100`, `dtd`, `pets`, `aircraft`, `flowers102`, `stanford_cars`, `food101`, `eurosat`, `tinyimagenet`.

If a dataset already exists in one of your shared project roots, MC-RFM reuses it directly and skips download. By default it checks if the datasets are available in local .
You can add more search roots with:

```bash
export MC_RFM_SHARED_DATA_ROOTS=/path/one:/path/two
```

For `tinyimagenet`, place an extracted `tiny-imagenet-200/` directory under one of the searched roots. The loader will organize the official validation images into an `ImageFolder`-compatible structure on first use.

## Cache Frozen Features (LMDB)

```bash
python -m src.data.cache_features \
  --config configs/experiments/mcfm_default.yaml \
  dataset.name=cifar100 model.backbone=vit_base_patch16_224
```

LMDB output:

`results_mc_rfm/lmdb/<dataset>/<backbone>/{train,val,test}.lmdb`

## Train

```bash
python -m src.engine.train \
  --config configs/experiments/mcfm_default.yaml \
  dataset.name=cifar100 fewshot.shots=4 seed=42
```

## Evaluate

```bash
python -m src.engine.eval \
  --config configs/experiments/mcfm_default.yaml \
  --checkpoint results_mc_rfm/<run_name>/checkpoints/best.pt
```

## Run Ablations

```bash
python -m src.engine.ablate --config configs/experiments/ablations.yaml
```

- Euclidean-only
- Hyperbolic-only
- Mixed (ours)
- Remove hyperbolic branch
- NFE sensitivity (1/3/5)
- Signature sweep (`d_h/d_e`)
- Curvature sweep + decoupling sweep

## Make Targets

```bash
make install
make test
make cache CONFIG=configs/experiments/mcfm_default.yaml
make train CONFIG=configs/experiments/mcfm_default.yaml
make ablate ABLATE_CONFIG=configs/experiments/ablations.yaml
make tables
make figs
make reproduce
```

`make reproduce` runs: cache -> train -> ablations -> tables -> figures.

## Outputs

Per run (`results_mc_rfm/<run_name>/`):
- `config_resolved.json`
- `stdout.log`
- `train_history.jsonl`
- `metrics.json`
- `checkpoints/best.pt`, `checkpoints/last.pt`

Global:
- `results_mc_rfm/summary.csv`
- `results_mc_rfm/tables.tex` (+ csv)
- `results_mc_rfm/figs/*.png|pdf`

## Numerical Stability Rules

- Poincaré norm clipped strictly inside boundary (`1 - 1e-5` margin).
- Safe `atanh`/division with eps guards.
- Hyperbolic interpolation/target computations in float64 in training loop.
- Projection back to ball after each ODE step.
- NaN/Inf checks fail fast with explicit exceptions.

## Reproducibility

- Seeds: `random`, `numpy`, `torch`, `torch.cuda`.
- Deterministic flags enabled (`cudnn.deterministic=True`).
- Few-shot indices persisted on disk.
- Full resolved config and metrics persisted per run.
