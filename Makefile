PYTHON ?= python3.11
VENV ?= .venv
CONSTRAINTS ?= requirements/constraints-py311.txt

.PHONY: setup doctor test validate dry-run demo demo-extended lerobot-check-scripts lerobot-train-smoke fastwam-check-scripts fastwam-train-smoke demo-chain-fastwam schemas reference-fetch clean

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
	$(VENV)/bin/embodied-demo validate --config configs/runs/kitchen_counter_sorting_mock.yaml
	$(VENV)/bin/embodied-demo validate --config configs/runs/drawer_pick_place_mock.yaml

dry-run:
	$(VENV)/bin/embodied-demo dry-run --config configs/runs/tabletop_sorting_mock.yaml --output runs/tabletop_sorting/resolved.yaml

demo:
	$(VENV)/bin/embodied-demo run --config configs/runs/tabletop_sorting_mock.yaml
	$(VENV)/bin/embodied-demo run --config configs/runs/towel_folding_mock.yaml

demo-extended: demo
	$(VENV)/bin/embodied-demo run --config configs/runs/kitchen_counter_sorting_mock.yaml
	$(VENV)/bin/embodied-demo run --config configs/runs/drawer_pick_place_mock.yaml

lerobot-check-scripts:
	bash -n scripts/lerobot/install_lerobot_cluster.sh
	bash -n scripts/lerobot/run_pusht_act_gpu_smoke.sh
	bash -n scripts/lerobot/slurm_pusht_act_gpu_smoke.sbatch
	$(VENV)/bin/python scripts/lerobot/parse_train_log.py --log tests/fixtures/lerobot_train_stdout.log --output-dir build/lerobot-parser-test

lerobot-train-smoke:
	bash scripts/lerobot/run_pusht_act_gpu_smoke.sh

fastwam-check-scripts:
	bash -n scripts/fastwam/prepare_fastwam_overlay.sh
	bash -n scripts/fastwam/run_realrobot_train_eval.sh
	bash -n scripts/fastwam/slurm_realrobot_pilot.sbatch
	$(VENV)/bin/python scripts/fastwam/parse_train_log.py --log tests/fixtures/fastwam_train_stdout.log --output-dir build/fastwam-parser-test

fastwam-train-smoke:
	FASTWAM_MODE=smoke bash scripts/fastwam/run_realrobot_train_eval.sh

demo-chain-fastwam:
	test -n "$(FASTWAM_RUN_DIR)" || (echo "FASTWAM_RUN_DIR is required, e.g. FASTWAM_RUN_DIR=runs/fastwam/<run>/<id> make demo-chain-fastwam" >&2; exit 2)
	$(VENV)/bin/embodied-demo report-fastwam --run-dir "$(FASTWAM_RUN_DIR)" $(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",) $(if $(MOCK_RUN_DIR),--mock-run-dir "$(MOCK_RUN_DIR)",)

schemas:
	$(VENV)/bin/embodied-demo export-schema --output-dir build/schemas

reference-fetch:
	bash scripts/reference/fetch_xpolicylab.sh

clean:
	rm -rf .pytest_cache build dist htmlcov
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
