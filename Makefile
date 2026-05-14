PYTHON ?= python
PIP ?= pip
CONFIG ?= configs/experiments/mcfm_default.yaml
ABLATE_CONFIG ?= configs/experiments/ablations.yaml
RESULTS ?= results_mc_rfm
OUT ?= results_mc_rfm

.PHONY: install test cache train eval ablate figs tables verify reproduce

install:
	$(PIP) install -r requirements.txt

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest

cache:
	$(PYTHON) -m src.data.cache_features --config $(CONFIG) $(ARGS)

train:
	$(PYTHON) -m src.engine.train --config $(CONFIG) $(ARGS)

eval:
	@if [ -z "$(CKPT)" ]; then echo "Usage: make eval CKPT=results_mc_rfm/<run>/checkpoints/best.pt"; exit 1; fi
	$(PYTHON) -m src.engine.eval --config $(CONFIG) --checkpoint $(CKPT) $(ARGS)

ablate:
	$(PYTHON) -m src.engine.ablate --config $(ABLATE_CONFIG) $(ARGS)

tables:
	$(PYTHON) -m scripts.make_tables --results_dir $(RESULTS) --out $(OUT)/tables.tex

figs:
	$(PYTHON) -m scripts.make_figs --results_dir $(RESULTS) --out $(OUT)/figs

verify:
	$(PYTHON) -m scripts.verify_mcrfm --config $(CONFIG)

reproduce:
	$(PYTHON) -m src.data.cache_features --config $(CONFIG)
	$(PYTHON) -m src.engine.train --config $(CONFIG)
	$(PYTHON) -m src.engine.ablate --config $(ABLATE_CONFIG)
	$(PYTHON) -m scripts.make_tables --results_dir $(RESULTS) --out $(OUT)/tables.tex
	$(PYTHON) -m scripts.make_figs --results_dir $(RESULTS) --out $(OUT)/figs
