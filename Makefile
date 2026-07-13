PYTHON ?= python3.11
VENV ?= .venv
CONSTRAINTS ?= requirements/constraints-py311.txt

.PHONY: setup doctor test validate dry-run demo schemas reference-fetch clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -c $(CONSTRAINTS) -e ".[dev]"

doctor:
	VENV=$(VENV) PYTHON_BIN=$(PYTHON) bash scripts/doctor.sh

test:
	$(VENV)/bin/python -m pytest

validate:
	$(VENV)/bin/embodied-demo list-tasks
	$(VENV)/bin/embodied-demo validate --config configs/runs/tabletop_sorting_mock.yaml
	$(VENV)/bin/embodied-demo validate --config configs/runs/towel_folding_mock.yaml

dry-run:
	$(VENV)/bin/embodied-demo dry-run --config configs/runs/tabletop_sorting_mock.yaml --output runs/tabletop_sorting/resolved.yaml

demo:
	$(VENV)/bin/embodied-demo run --config configs/runs/tabletop_sorting_mock.yaml
	$(VENV)/bin/embodied-demo run --config configs/runs/towel_folding_mock.yaml

schemas:
	$(VENV)/bin/embodied-demo export-schema --output-dir build/schemas

reference-fetch:
	bash scripts/reference/fetch_xpolicylab.sh

clean:
	rm -rf .pytest_cache build dist htmlcov
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
