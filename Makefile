PYTHON ?= python3.11
VENV ?= .venv
CONSTRAINTS ?= requirements/constraints-py311.txt

.PHONY: help setup doctor test validate prepare-dirs prepare-assets-lerobot prepare-assets-custom-fastwam prepare-env-custom-fastwam prepare-assets-imagewam check-assets check-assets-core check-assets-lerobot check-assets-custom-fastwam check-assets-imagewam download-lerobot-artifacts download-lerobot-pusht-dataset download-lerobot-svla-so100-pickplace-dataset download-lerobot-fastwam-libero-dataset convert-lerobot-fastwam-libero-v3 download-lerobot-fastwam-base-cache download-lerobot-diffusion-pusht-policy download-lerobot-smolvla-base-policy download-lerobot-fastwam-libero-policy download-data-rovid20k download-data-rovidx download-data-mdm-depth download-data-xperience10m-sample download-data-abc130k download-data-agibotworld-alpha download-data-interndata-a1 download-custom-fastwam-libero-dataset download-fastwam-artifacts prepare-imagewam-upstream download-imagewam-artifacts download-imagewam-flux2-base lerobot-check-scripts fastwam-check-scripts imagewam-check-scripts experiments-check-scripts lerobot-data-smoke schemas clean

help:
	@echo "EmbodiedAI Demo Pipeline"
	@echo
	@echo "Environment / checks:"
	@echo "  make setup                         Create local core .venv"
	@echo "  make test                          Run unit tests"
	@echo "  make validate                      Run script syntax checks and schema export"
	@echo "  make lerobot-check-scripts         Check LeRobot wrapper syntax/parsers"
	@echo "  make fastwam-check-scripts         Check FastWAM wrapper syntax/parsers"
	@echo "  make imagewam-check-scripts        Check ImageWAM wrapper syntax"
	@echo "  make experiments-check-scripts     Check experiment launch/config scripts"
	@echo "  make prepare-dirs                  Create repo-local asset directories"
	@echo "  make check-assets                  Check repo-local data/model/cache assets"
	@echo
	@echo "Bootstrap:"
	@echo "  make prepare-assets-lerobot        Download first LeRobot datasets/policies/cache"
	@echo "  make prepare-assets-custom-fastwam Download FastWAM data/release and prepare overlay"
	@echo "  make prepare-env-custom-fastwam    Create/install the custom FastWAM conda env"
	@echo "  make prepare-assets-imagewam       Prepare ImageWAM upstream and model assets"
	@echo
	@echo "LeRobot downloads:"
	@echo "  make download-lerobot-artifacts    Download LeRobot PushT dataset"
	@echo "  make download-lerobot-svla-so100-pickplace-dataset"
	@echo "                                      Download SmolVLA SO100 pick-place dataset"
	@echo "  make download-lerobot-fastwam-libero-dataset"
	@echo "                                      Download FastWAM LIBERO raw dataset into the LeRobot route"
	@echo "  make convert-lerobot-fastwam-libero-v3"
	@echo "                                      Convert LeRobot-route FastWAM LIBERO v2.1 subsets to v3.0"
	@echo "  make download-lerobot-fastwam-base-cache"
	@echo "                                      Download Wan/T5 cache required by LeRobot FastWAM inference"
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
	@echo "  make download-custom-fastwam-libero-dataset"
	@echo "                                      Download FastWAM LIBERO raw dataset into the custom route"
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
	$(MAKE) lerobot-check-scripts
	$(MAKE) fastwam-check-scripts
	$(MAKE) imagewam-check-scripts
	$(MAKE) experiments-check-scripts
	$(MAKE) schemas

prepare-dirs:
	mkdir -p data models checkpoints runs/artifact_manifests artifacts upstreams hf_cache/hub hf_cache/datasets hf_cache/torch hf_cache/pip

prepare-assets-lerobot: prepare-dirs
	$(MAKE) download-lerobot-pusht-dataset
	$(MAKE) download-lerobot-svla-so100-pickplace-dataset
	$(MAKE) download-lerobot-diffusion-pusht-policy
	$(MAKE) download-lerobot-smolvla-base-policy
	$(MAKE) download-lerobot-fastwam-libero-policy
	$(MAKE) download-lerobot-fastwam-libero-dataset
	$(MAKE) convert-lerobot-fastwam-libero-v3
	$(MAKE) download-lerobot-fastwam-base-cache

prepare-assets-custom-fastwam: prepare-dirs
	$(MAKE) download-custom-fastwam-libero-dataset
	$(MAKE) download-fastwam-artifacts
	bash scripts/fastwam/prepare_fastwam_overlay.sh

prepare-env-custom-fastwam: prepare-dirs
	FASTWAM_CREATE_CONDA=1 FASTWAM_INSTALL=1 bash scripts/fastwam/prepare_fastwam_overlay.sh

prepare-assets-imagewam: prepare-dirs
	$(MAKE) prepare-imagewam-upstream
	$(MAKE) download-imagewam-artifacts
	$(MAKE) download-imagewam-flux2-base

check-assets:
	$(PYTHON) scripts/check_assets.py --profile all

check-assets-core:
	$(PYTHON) scripts/check_assets.py --profile core

check-assets-lerobot:
	$(PYTHON) scripts/check_assets.py --profile lerobot

check-assets-custom-fastwam:
	$(PYTHON) scripts/check_assets.py --profile custom-fastwam

check-assets-imagewam:
	$(PYTHON) scripts/check_assets.py --profile imagewam

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

download-lerobot-fastwam-libero-dataset:
	ARTIFACT_FAMILY=lerobot ARTIFACT_MANIFEST_NAME=lerobot_fastwam_libero_dataset_manifest.json \
	LEROBOT_DATASET_REPO_ID=yuanty/LIBERO-fastwam \
	LEROBOT_DATASET_LOCAL_DIR="$${EMBODIED_DATA_ROOT:-$$(pwd)/data}/lerobot/libero-fastwam/v2.1" \
	bash scripts/lerobot/download_artifacts.sh

convert-lerobot-fastwam-libero-v3:
	bash scripts/lerobot/convert_fastwam_libero_v21_to_v30.sh

download-lerobot-fastwam-base-cache:
	bash scripts/lerobot/download_fastwam_base_cache.sh

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

download-custom-fastwam-libero-dataset:
	ARTIFACT_FAMILY=custom_fastwam ARTIFACT_MANIFEST_NAME=custom_fastwam_libero_dataset_manifest.json \
	LEROBOT_DATASET_REPO_ID=yuanty/LIBERO-fastwam \
	LEROBOT_DATASET_LOCAL_DIR="$${EMBODIED_DATA_ROOT:-$$(pwd)/data}/custom/fastwam/libero-fastwam" \
	bash scripts/lerobot/download_artifacts.sh

download-fastwam-artifacts:
	bash scripts/fastwam/download_release_artifacts.sh

prepare-imagewam-upstream:
	bash scripts/imagewam/prepare_imagewam_upstream.sh

download-imagewam-artifacts:
	bash scripts/imagewam/download_artifacts.sh

download-imagewam-flux2-base: prepare-imagewam-upstream
	cd upstreams/ImageWAM && \
	MODEL_ROOT="$${IMAGEWAM_MODEL_ROOT:-$${EMBODIED_MODEL_ROOT:-$$(pwd)/../../models}/custom/imagewam}" \
	DOWNLOAD_9B="$${IMAGEWAM_DOWNLOAD_9B:-false}" \
	bash scripts/flux2/prepare_flux2_files.sh

lerobot-check-scripts:
	bash -n scripts/lerobot/install_lerobot_cluster.sh
	bash -n scripts/lerobot/download_artifacts.sh
	bash -n scripts/lerobot/convert_fastwam_libero_v21_to_v30.sh
	bash -n scripts/lerobot/download_fastwam_base_cache.sh
	bash -n scripts/lerobot/run_pusht_act_gpu_smoke.sh
	bash -n scripts/lerobot/run_dataset_smoke.sh
	bash -n scripts/lerobot/run_inference_smoke.sh
	bash -n scripts/lerobot/run_train_accelerate.sh
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

fastwam-check-scripts:
	bash -n scripts/fastwam/prepare_fastwam_overlay.sh
	bash -n scripts/fastwam/download_release_artifacts.sh
	bash -n scripts/fastwam/run_realrobot_train_eval.sh
	$(VENV)/bin/python scripts/fastwam/parse_train_log.py --log tests/fixtures/fastwam_train_stdout.log --output-dir build/fastwam-parser-test
	$(VENV)/bin/python scripts/fastwam/run_config.py --config experiments/custom/fastwam_realrobot_single8_random/config.yaml --dry-run --output-shell build/fastwam-config-test/generated.sh >/dev/null

imagewam-check-scripts:
	bash -n configs/imagewam/libero_train_eval.sh
	bash -n scripts/imagewam/prepare_imagewam_upstream.sh
	bash -n scripts/imagewam/download_artifacts.sh
	bash -n scripts/imagewam/run_train_eval.sh

experiments-check-scripts:
	find experiments -name '*.sh' -print0 | xargs -0 -n 1 bash -n
	find experiments -name '*.sbatch' -print0 | xargs -0 -n 1 bash -n

schemas:
	$(VENV)/bin/embodied-demo export-schema --output-dir build/schemas

clean:
	rm -rf .pytest_cache build dist htmlcov
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
