PYTHON ?= python3.11
VENV ?= .venv
CONSTRAINTS ?= requirements/constraints-py311.txt
LEROBOT_TRAIN_CONFIG ?= configs/lerobot/pusht_act_gpu_smoke.sh
LEROBOT_ACCELERATE_CONFIG ?= configs/lerobot/train/svla_so100_smolvla_8gpu_long.sh
LEROBOT_INFER_CONFIG ?= configs/lerobot/native_pusht_act_pipeline.sh

.PHONY: help setup doctor test validate dry-run demo demo-extended download-lerobot-artifacts download-lerobot-pusht-dataset download-lerobot-svla-so100-pickplace-dataset download-lerobot-diffusion-pusht-policy download-lerobot-smolvla-base-policy download-lerobot-fastwam-libero-policy download-data-rovid20k download-data-rovidx download-data-mdm-depth download-data-xperience10m-sample download-data-abc130k download-data-agibotworld-alpha download-data-interndata-a1 download-fastwam-artifacts prepare-imagewam-upstream download-imagewam-artifacts download-imagewam-flux2-base lerobot-check-scripts fastwam-check-scripts imagewam-check-scripts experiments-check-scripts lerobot-data-smoke lerobot-train-smoke lerobot-train-act lerobot-train-diffusion lerobot-train-smolvla lerobot-train-8gpu-smolvla lerobot-infer-smoke lerobot-infer-diffusion lerobot-infer-smolvla lerobot-infer-fastwam demo-chain-lerobot-fastwam fastwam-train-smoke demo-chain-fastwam imagewam-train-smoke schemas reference-fetch clean

help:
	@echo "EmbodiedAI Demo Pipeline"
	@echo
	@echo "Environment / checks:"
	@echo "  make setup                         Create local core .venv"
	@echo "  make test                          Run unit tests"
	@echo "  make validate                      Validate task/run configs"
	@echo "  make lerobot-check-scripts         Check LeRobot wrapper syntax/parsers"
	@echo "  make fastwam-check-scripts         Check FastWAM wrapper syntax/parsers"
	@echo "  make imagewam-check-scripts        Check ImageWAM wrapper syntax"
	@echo "  make experiments-check-scripts     Check experiment launch/config scripts"
	@echo
	@echo "LeRobot downloads:"
	@echo "  make download-lerobot-artifacts    Download LeRobot PushT dataset"
	@echo "  make download-lerobot-svla-so100-pickplace-dataset"
	@echo "                                      Download SmolVLA SO100 pick-place dataset"
	@echo "  make download-lerobot-diffusion-pusht-policy"
	@echo "                                      Download LeRobot diffusion PushT policy"
	@echo "  make download-lerobot-smolvla-base-policy"
	@echo "                                      Download LeRobot SmolVLA base policy"
	@echo "  make download-lerobot-fastwam-libero-policy"
	@echo "                                      Download LeRobot-compatible FastWAM LIBERO policy"
	@echo
	@echo "Open data shortcuts:"
	@echo "  make download-data-rovid20k        Download practical RoVid-X subset"
	@echo "  make download-data-xperience10m-sample"
	@echo "                                      Download Xperience-10M sample episode"
	@echo "  make download-data-abc130k         Download ABC-130k after HF access approval"
	@echo
	@echo "Custom downloads / upstreams:"
	@echo "  make download-fastwam-artifacts    Download FastWAM release ckpt/stats"
	@echo "  make prepare-imagewam-upstream     Clone/update official ImageWAM repo"
	@echo "  make download-imagewam-artifacts   Download ImageWAM FLUX.2 4B LIBERO release"
	@echo "  make download-imagewam-flux2-base  Download FLUX.2 4B base/AE via official script"
	@echo
	@echo "Training / inference:"
	@echo "  Use experiments/<route>/<experiment>/launch.sh, not make."
	@echo "  Start with experiments/README.md"
	@echo
	@echo "Start here:"
	@echo "  docs/README.md"
	@echo "  pipelines/lerobot/README.md"
	@echo "  pipelines/custom/README.md"
	@echo "  experiments/README.md"

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

download-lerobot-pusht-dataset:
	ARTIFACT_MANIFEST_NAME=lerobot_pusht_dataset_manifest.json \
	LEROBOT_DATASET_REPO_ID=lerobot/pusht \
	LEROBOT_DATASET_LOCAL_DIR="$${EMBODIED_DATA_ROOT:-$$(pwd)/data}/lerobot/pusht" \
	bash scripts/lerobot/download_artifacts.sh

download-lerobot-svla-so100-pickplace-dataset:
	ARTIFACT_MANIFEST_NAME=lerobot_svla_so100_pickplace_dataset_manifest.json \
	LEROBOT_DATASET_REPO_ID=lerobot/svla_so100_pickplace \
	LEROBOT_DATASET_LOCAL_DIR="$${EMBODIED_DATA_ROOT:-$$(pwd)/data}/lerobot/svla_so100_pickplace" \
	bash scripts/lerobot/download_artifacts.sh

download-lerobot-diffusion-pusht-policy:
	DOWNLOAD_LEROBOT_DATASET=0 DOWNLOAD_LEROBOT_POLICY=1 \
	ARTIFACT_MANIFEST_NAME=lerobot_diffusion_pusht_policy_manifest.json \
	LEROBOT_POLICY_TYPE=diffusion \
	LEROBOT_POLICY_REPO_ID=lerobot/diffusion_pusht \
	LEROBOT_POLICY_LOCAL_DIR="$${EMBODIED_MODEL_ROOT:-$$(pwd)/models}/lerobot/diffusion/diffusion_pusht" \
	bash scripts/lerobot/download_artifacts.sh

download-lerobot-smolvla-base-policy:
	DOWNLOAD_LEROBOT_DATASET=0 DOWNLOAD_LEROBOT_POLICY=1 \
	ARTIFACT_MANIFEST_NAME=lerobot_smolvla_base_policy_manifest.json \
	LEROBOT_POLICY_TYPE=smolvla \
	LEROBOT_POLICY_REPO_ID=lerobot/smolvla_base \
	LEROBOT_POLICY_LOCAL_DIR="$${EMBODIED_MODEL_ROOT:-$$(pwd)/models}/lerobot/smolvla/smolvla_base" \
	bash scripts/lerobot/download_artifacts.sh

download-lerobot-fastwam-libero-policy:
	DOWNLOAD_LEROBOT_DATASET=0 DOWNLOAD_LEROBOT_POLICY=1 \
	ARTIFACT_MANIFEST_NAME=lerobot_fastwam_libero_policy_manifest.json \
	LEROBOT_POLICY_TYPE=fastwam \
	LEROBOT_POLICY_REPO_ID=lerobot/fastwam_libero_uncond_2cam224 \
	LEROBOT_POLICY_LOCAL_DIR="$${EMBODIED_MODEL_ROOT:-$$(pwd)/models}/lerobot/fastwam/fastwam_libero_uncond_2cam224" \
	bash scripts/lerobot/download_artifacts.sh

download-data-rovid20k:
	ARTIFACT_FAMILY=open_data ARTIFACT_MANIFEST_NAME=open_data_rovid20k_manifest.json \
	LEROBOT_DATASET_REPO_ID=Perflow-Shuai/RoVid-20K-10s \
	LEROBOT_DATASET_LOCAL_DIR="$${EMBODIED_DATA_ROOT:-$$(pwd)/data}/internet/rovid-20k-10s" \
	bash scripts/lerobot/download_artifacts.sh

download-data-rovidx:
	ARTIFACT_FAMILY=open_data ARTIFACT_MANIFEST_NAME=open_data_rovidx_manifest.json \
	LEROBOT_DATASET_REPO_ID=DAGroup-PKU/RoVid-X \
	LEROBOT_DATASET_LOCAL_DIR="$${EMBODIED_DATA_ROOT:-$$(pwd)/data}/internet/rovid-x" \
	bash scripts/lerobot/download_artifacts.sh

download-data-mdm-depth:
	ARTIFACT_FAMILY=open_data ARTIFACT_MANIFEST_NAME=open_data_mdm_depth_manifest.json \
	LEROBOT_DATASET_REPO_ID=robbyant/mdm_depth \
	LEROBOT_DATASET_LOCAL_DIR="$${EMBODIED_DATA_ROOT:-$$(pwd)/data}/perception/mdm_depth" \
	bash scripts/lerobot/download_artifacts.sh

download-data-xperience10m-sample:
	ARTIFACT_FAMILY=open_data ARTIFACT_MANIFEST_NAME=open_data_xperience10m_sample_manifest.json \
	LEROBOT_DATASET_REPO_ID=ropedia-ai/xperience-10m-sample \
	LEROBOT_DATASET_LOCAL_DIR="$${EMBODIED_DATA_ROOT:-$$(pwd)/data}/human/xperience-10m-sample" \
	bash scripts/lerobot/download_artifacts.sh

download-data-abc130k:
	ARTIFACT_FAMILY=open_data ARTIFACT_MANIFEST_NAME=open_data_abc130k_manifest.json \
	LEROBOT_DATASET_REPO_ID=XDOF/ABC-130k \
	LEROBOT_DATASET_LOCAL_DIR="$${EMBODIED_DATA_ROOT:-$$(pwd)/data}/vla/abc-130k" \
	bash scripts/lerobot/download_artifacts.sh

download-data-agibotworld-alpha:
	ARTIFACT_FAMILY=open_data ARTIFACT_MANIFEST_NAME=open_data_agibotworld_alpha_manifest.json \
	LEROBOT_DATASET_REPO_ID=agibot-world/AgiBotWorld-Alpha \
	LEROBOT_DATASET_LOCAL_DIR="$${EMBODIED_DATA_ROOT:-$$(pwd)/data}/vla/agibotworld-alpha" \
	bash scripts/lerobot/download_artifacts.sh

download-data-interndata-a1:
	ARTIFACT_FAMILY=open_data ARTIFACT_MANIFEST_NAME=open_data_interndata_a1_manifest.json \
	LEROBOT_DATASET_REPO_ID=InternRobotics/InternData-A1 \
	LEROBOT_DATASET_LOCAL_DIR="$${EMBODIED_DATA_ROOT:-$$(pwd)/data}/simulation/interndata-a1" \
	bash scripts/lerobot/download_artifacts.sh

download-fastwam-artifacts:
	bash scripts/fastwam/download_release_artifacts.sh

prepare-imagewam-upstream:
	bash scripts/imagewam/prepare_imagewam_upstream.sh

download-imagewam-artifacts:
	bash scripts/imagewam/download_artifacts.sh

download-imagewam-flux2-base: prepare-imagewam-upstream
	cd upstreams/ImageWAM && \
	MODEL_ROOT="$${IMAGEWAM_MODEL_ROOT:-$${EMBODIED_MODEL_ROOT:-$$(pwd)/../../models}/imagewam}" \
	DOWNLOAD_9B="$${IMAGEWAM_DOWNLOAD_9B:-false}" \
	bash scripts/flux2/prepare_flux2_files.sh

lerobot-check-scripts:
	bash -n scripts/lerobot/install_lerobot_cluster.sh
	bash -n scripts/lerobot/download_artifacts.sh
	bash -n scripts/lerobot/run_pusht_act_gpu_smoke.sh
	bash -n scripts/lerobot/run_dataset_smoke.sh
	bash -n scripts/lerobot/run_inference_smoke.sh
	bash -n scripts/lerobot/run_train_accelerate.sh
	bash -n scripts/lerobot/slurm_pusht_act_gpu_smoke.sbatch
	bash -n scripts/lerobot/slurm_smolvla_8gpu_long.sbatch
	bash -n configs/lerobot/train/pusht_act.sh
	bash -n configs/lerobot/train/pusht_diffusion.sh
	bash -n configs/lerobot/train/svla_so100_smolvla.sh
	bash -n configs/lerobot/train/svla_so100_smolvla_8gpu_long.sh
	bash -n configs/lerobot/train/aloha_pi0fast_template.sh
	bash -n configs/lerobot/infer/pusht_diffusion.sh
	bash -n configs/lerobot/infer/svla_so100_smolvla.sh
	bash -n configs/lerobot/infer/fastwam_libero.sh
	$(VENV)/bin/python scripts/lerobot/parse_train_log.py --log tests/fixtures/lerobot_train_stdout.log --output-dir build/lerobot-parser-test
	$(VENV)/bin/python scripts/lerobot/generate_data_to_inference_report.py --dataset-profile tests/fixtures/lerobot_dataset_profile.json --inference-evidence tests/fixtures/lerobot_inference_evidence.json --training-summary build/lerobot-parser-test/loss_summary.json --output-dir build/lerobot-chain-report-test

lerobot-data-smoke:
	bash scripts/lerobot/run_dataset_smoke.sh

lerobot-train-smoke:
	bash scripts/lerobot/run_pusht_act_gpu_smoke.sh $(LEROBOT_TRAIN_CONFIG)

lerobot-train-act:
	bash scripts/lerobot/run_pusht_act_gpu_smoke.sh configs/lerobot/train/pusht_act.sh

lerobot-train-diffusion:
	bash scripts/lerobot/run_pusht_act_gpu_smoke.sh configs/lerobot/train/pusht_diffusion.sh

lerobot-train-smolvla:
	bash scripts/lerobot/run_pusht_act_gpu_smoke.sh configs/lerobot/train/svla_so100_smolvla.sh

lerobot-train-8gpu-smolvla:
	bash scripts/lerobot/run_train_accelerate.sh $(LEROBOT_ACCELERATE_CONFIG)

lerobot-infer-smoke:
	bash scripts/lerobot/run_inference_smoke.sh $(LEROBOT_INFER_CONFIG)

lerobot-infer-diffusion:
	bash scripts/lerobot/run_inference_smoke.sh configs/lerobot/infer/pusht_diffusion.sh

lerobot-infer-smolvla:
	bash scripts/lerobot/run_inference_smoke.sh configs/lerobot/infer/svla_so100_smolvla.sh

lerobot-infer-fastwam:
	bash scripts/lerobot/run_inference_smoke.sh configs/lerobot/infer/fastwam_libero.sh

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

imagewam-check-scripts:
	bash -n configs/imagewam/libero_train_eval.sh
	bash -n scripts/imagewam/prepare_imagewam_upstream.sh
	bash -n scripts/imagewam/download_artifacts.sh
	bash -n scripts/imagewam/run_train_eval.sh
	bash -n scripts/imagewam/slurm_libero_pilot.sbatch

experiments-check-scripts:
	find experiments -name '*.sh' -print0 | xargs -0 -n 1 bash -n
	find experiments -name '*.sbatch' -print0 | xargs -0 -n 1 bash -n

imagewam-train-smoke:
	IMAGEWAM_MODE=metadata-smoke IMAGEWAM_REQUIRE_CUDA=0 bash scripts/imagewam/run_train_eval.sh

schemas:
	$(VENV)/bin/embodied-demo export-schema --output-dir build/schemas

reference-fetch:
	bash scripts/reference/fetch_xpolicylab.sh

clean:
	rm -rf .pytest_cache build dist htmlcov
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
