PYTHON ?= python3.11
VENV ?= .venv

.PHONY: setup test validate dry-run schemas clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e ".[dev]"

test:
	$(VENV)/bin/python -m pytest

validate:
	$(VENV)/bin/embodied-demo list-tasks
	$(VENV)/bin/embodied-demo validate --config configs/runs/tabletop_sorting_mock.yaml
	$(VENV)/bin/embodied-demo validate --config configs/runs/towel_folding_mock.yaml

dry-run:
	$(VENV)/bin/embodied-demo dry-run --config configs/runs/tabletop_sorting_mock.yaml --output runs/tabletop_sorting/resolved.yaml

schemas:
	$(VENV)/bin/embodied-demo export-schema --output-dir schemas

clean:
	rm -rf .pytest_cache build dist htmlcov
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
