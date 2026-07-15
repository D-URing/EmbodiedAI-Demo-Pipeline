PYTHON ?= python3.11
VENV ?= .venv
CONSTRAINTS ?= requirements/constraints-py311.txt

.PHONY: help setup doctor test validate dry-run demo demo-extended download-lerobot-artifacts download-lerobot-diffusion-pusht-policy download-fastwam-artifacts lerobot-check-scripts lerobot-data-smoke lerobot-train-smoke lerobot-infer-smoke demo-chain-lerobot-fastwam fastwam-check-scripts fastwam-train-smoke demo-chain-fastwam schemas reference-fetch clean

help:
	@echo "EmbodiedAI Demo Pipeline"
	@echo
	@echo "Core / mock:"
	@echo "  make setup                         Create local core .venv"
	@echo "  make test                          Run unit tests"
	@echo "  make validate                      Validate task/run configs"
	@echo "  make demo                          Run minimal mock demos"
	@echo
	@echo "LeRobot pipeline:"
	@echo "  make download-lerobot-artifacts    Download LeRobot PushT dataset"
	@echo "  make download-lerobot-diffusion-pusht-policy"
	@echo "                                      Download LeRobot diffusion PushT policy"
	@echo "  make lerobot-data-smoke            Inspect LeRobot dataset"
	@echo "  make lerobot-train-smoke           Run ACT/PushT GPU training smoke"
	@echo "  make lerobot-infer-smoke           Run offline policy inference smoke"
	@echo
	@echo "Custom / FastWAM pipeline:"
	@echo "  make download-fastwam-artifacts    Download FastWAM release ckpt/stats"
	@echo "  make fastwam-train-smoke           Run FastWAM custom smoke after overlay setup"
	@echo "  make demo-chain-fastwam            Convert FastWAM run into demo report"
	@echo
	@echo "Start here:"
	@echo "  docs/README.md"
	@echo "  pipelines/lerobot/README.md"
	@echo "  pipelines/custom_fastwam/README.md"

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

download-lerobot-artifacts:
	bash scripts/lerobot/download_artifacts.sh

download-lerobot-diffusion-pusht-policy:
	DOWNLOAD_LEROBOT_DATASET=0 DOWNLOAD_LEROBOT_POLICY=1 \
	LEROBOT_POLICY_TYPE=diffusion \
	LEROBOT_POLICY_REPO_ID=lerobot/diffusion_pusht \
	LEROBOT_POLICY_LOCAL_DIR="$${EMBODIED_MODEL_ROOT:-$$(pwd)/models}/lerobot/diffusion/diffusion_pusht" \
	bash scripts/lerobot/download_artifacts.sh

download-fastwam-artifacts:
	bash scripts/fastwam/download_release_artifacts.sh

lerobot-check-scripts:
	bash -n scripts/lerobot/install_lerobot_cluster.sh
	bash -n scripts/lerobot/download_artifacts.sh
	bash -n scripts/lerobot/run_pusht_act_gpu_smoke.sh
	bash -n scripts/lerobot/run_dataset_smoke.sh
	bash -n scripts/lerobot/run_inference_smoke.sh
	bash -n scripts/lerobot/slurm_pusht_act_gpu_smoke.sbatch
	$(VENV)/bin/python scripts/lerobot/parse_train_log.py --log tests/fixtures/lerobot_train_stdout.log --output-dir build/lerobot-parser-test
	$(VENV)/bin/python scripts/lerobot/generate_data_to_inference_report.py --dataset-profile tests/fixtures/lerobot_dataset_profile.json --inference-evidence tests/fixtures/lerobot_inference_evidence.json --training-summary build/lerobot-parser-test/loss_summary.json --output-dir build/lerobot-chain-report-test

lerobot-data-smoke:
	bash scripts/lerobot/run_dataset_smoke.sh

lerobot-train-smoke:
	bash scripts/lerobot/run_pusht_act_gpu_smoke.sh

lerobot-infer-smoke:
	bash scripts/lerobot/run_inference_smoke.sh

demo-chain-lerobot-fastwam:
	test -n "$(LEROBOT_DATASET_PROFILE)" || (echo "LEROBOT_DATASET_PROFILE is required" >&2; exit 2)
	test -n "$(LEROBOT_INFERENCE_EVIDENCE)" || (echo "LEROBOT_INFERENCE_EVIDENCE is required" >&2; exit 2)
	$(VENV)/bin/python scripts/lerobot/generate_data_to_inference_report.py --dataset-profile "$(LEROBOT_DATASET_PROFILE)" --inference-evidence "$(LEROBOT_INFERENCE_EVIDENCE)" $(if $(LEROBOT_TRAINING_SUMMARY),--training-summary "$(LEROBOT_TRAINING_SUMMARY)",) --output-dir "$(if $(OUTPUT_DIR),$(OUTPUT_DIR),build/lerobot-chain-report)"

fastwam-check-scripts:
	bash -n scripts/fastwam/prepare_fastwam_overlay.sh
	bash -n scripts/fastwam/download_release_artifacts.sh
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
