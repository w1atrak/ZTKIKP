.PHONY: venv install test fmt lint verify-gmpy2 data tune select-features train quantize bench-primitives budget bench serve demo eval-encrypted

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
ARGS ?=
export PYTHONPATH := src

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

test:
	$(PY) -m pytest -q

fmt:
	$(VENV)/bin/black src tests scripts
	$(VENV)/bin/ruff check --fix src tests scripts

lint:
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/black --check src tests scripts

verify-gmpy2:
	$(PY) scripts/verify_gmpy2.py

# Placeholders for later milestones (M1+)
data:
	$(PY) scripts/download_data.py

tune:
	$(PY) scripts/train.py --tune

select-features:
	$(PY) scripts/select_features.py

train:
	$(PY) scripts/train.py
	$(PY) scripts/eval_plain.py

quantize:
	$(PY) scripts/quantize.py

bench-primitives:
	$(PY) scripts/bench_primitives.py

budget:
	$(PY) scripts/budget.py

bench:
	@echo "Not implemented yet — see scripts/bench.py"

serve:
	$(PY) -m ppspam.server

demo:
	$(PY) scripts/demo.py $(ARGS)

eval-encrypted:
	$(PY) scripts/eval_encrypted.py $(ARGS)
