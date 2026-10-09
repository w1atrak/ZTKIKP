.PHONY: venv install test fmt lint verify-gmpy2 data tune select-features train bench demo

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

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

bench:
	@echo "Not implemented yet — see scripts/bench.py"

demo:
	@echo "Not implemented yet — see scripts/demo.py"
