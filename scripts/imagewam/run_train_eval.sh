#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/imagewam/libero_train_eval.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="$IMAGEWAM_RUN_ROOT/$IMAGEWAM_RUN_NAME/$timestamp"
mkdir -p "$run_dir"

cuda_visible="${CUDA_VISIBLE_DEVICES:-}"
if [[ "$IMAGEWAM_REQUIRE_CUDA" == "1" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$run_dir/gpus.txt" || true
  else
    echo "nvidia-smi not found. Set IMAGEWAM_REQUIRE_CUDA=0 only for metadata checks." >&2
    exit 3
  fi
fi

python_bin="${PYTHON_BIN:-python3}"
"$python_bin" - "$run_dir/backend_manifest.json" <<PY
import json
import os
from pathlib import Path

manifest = {
    "backend": "imagewam",
    "mode": os.environ.get("IMAGEWAM_MODE"),
    "variant": os.environ.get("IMAGEWAM_VARIANT"),
    "task_type": os.environ.get("IMAGEWAM_TASK_TYPE"),
    "task_suite": os.environ.get("IMAGEWAM_TASK_SUITE"),
    "workdir": os.environ.get("IMAGEWAM_WORKDIR"),
    "workdir_exists": Path(os.environ.get("IMAGEWAM_WORKDIR", "")).exists(),
    "data_root": os.environ.get("IMAGEWAM_DATA_ROOT"),
    "data_root_exists": Path(os.environ.get("IMAGEWAM_DATA_ROOT", "")).exists(),
    "policy_local_dir": os.environ.get("IMAGEWAM_POLICY_LOCAL_DIR"),
    "policy_local_dir_exists": Path(os.environ.get("IMAGEWAM_POLICY_LOCAL_DIR", "")).exists(),
    "release_ckpt_path": os.environ.get("IMAGEWAM_RELEASE_CKPT_PATH"),
    "release_ckpt_exists": Path(os.environ.get("IMAGEWAM_RELEASE_CKPT_PATH", "")).exists(),
    "dataset_stats_path": os.environ.get("IMAGEWAM_DATASET_STATS_PATH"),
    "dataset_stats_exists": Path(os.environ.get("IMAGEWAM_DATASET_STATS_PATH", "")).exists(),
    "flux2_src": os.environ.get("IMAGEWAM_FLUX2_SRC"),
    "flux2_src_exists": Path(os.environ.get("IMAGEWAM_FLUX2_SRC", "")).exists(),
    "flux2_model_path": os.environ.get("IMAGEWAM_FLUX2_MODEL_PATH"),
    "flux2_model_exists": Path(os.environ.get("IMAGEWAM_FLUX2_MODEL_PATH", "")).exists(),
    "flux2_ae_model_path": os.environ.get("IMAGEWAM_FLUX2_AE_MODEL_PATH"),
    "flux2_ae_model_exists": Path(os.environ.get("IMAGEWAM_FLUX2_AE_MODEL_PATH", "")).exists(),
    "train_entrypoint": os.environ.get("IMAGEWAM_TRAIN_ENTRYPOINT"),
    "train_entrypoint_exists": Path(os.environ.get("IMAGEWAM_TRAIN_ENTRYPOINT", "")).exists(),
    "eval_entrypoint": os.environ.get("IMAGEWAM_EVAL_ENTRYPOINT"),
    "eval_entrypoint_exists": Path(os.environ.get("IMAGEWAM_EVAL_ENTRYPOINT", "")).exists(),
    "gpus_per_node": os.environ.get("IMAGEWAM_GPUS_PER_NODE"),
    "cuda_visible_devices": "$cuda_visible",
    "precompute_qwen3_cache": os.environ.get("IMAGEWAM_PRECOMPUTE_QWEN3_CACHE"),
}
Path("$run_dir/backend_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY

if [[ "$IMAGEWAM_MODE" == "metadata-smoke" ]]; then
  echo "IMAGEWAM_METADATA_SMOKE_COMPLETE run_dir=$run_dir"
  exit 0
fi

if [[ ! -d "$IMAGEWAM_WORKDIR/.git" ]]; then
  echo "ImageWAM upstream is missing: $IMAGEWAM_WORKDIR" >&2
  echo "Run: make prepare-imagewam-upstream" >&2
  exit 4
fi

if [[ ! -f "$IMAGEWAM_TRAIN_ENTRYPOINT" ]]; then
  echo "ImageWAM train entrypoint not found: $IMAGEWAM_TRAIN_ENTRYPOINT" >&2
  echo "Inspect upstream scripts and override IMAGEWAM_TRAIN_ENTRYPOINT if the official layout changed." >&2
  exit 5
fi

(
  cd "$IMAGEWAM_WORKDIR"
  export DATA_ROOT="$IMAGEWAM_DATA_ROOT"
  export MODEL_ROOT="$IMAGEWAM_MODEL_ROOT"
  export OUTPUT_ROOT="$run_dir"
  export CKPT_PATH="$IMAGEWAM_RELEASE_CKPT_PATH"
  export DATASET_STATS_PATH="$IMAGEWAM_DATASET_STATS_PATH"
  export OUTPUT_DIR="$run_dir"
  export TASK_TYPE="$IMAGEWAM_TASK_TYPE"
  export TASK_SUITE="$IMAGEWAM_TASK_SUITE"
  export GPUS_PER_NODE="$IMAGEWAM_GPUS_PER_NODE"
  export GPU_PER_NODE="$IMAGEWAM_GPUS_PER_NODE"
  export NUM_GPUS="$IMAGEWAM_GPUS_PER_NODE"
  export FLUX2_VARIANT="$IMAGEWAM_FLUX2_VARIANT"
  export FLUX2_SRC="$IMAGEWAM_FLUX2_SRC"
  export FLUX2_MODEL_PATH="$IMAGEWAM_FLUX2_MODEL_PATH"
  export FLUX2_AE_MODEL_PATH="$IMAGEWAM_FLUX2_AE_MODEL_PATH"
  export FLUX2_QWEN3_MODEL_SPEC="$IMAGEWAM_FLUX2_QWEN3_MODEL_SPEC"
  export ACTION_INIT="$IMAGEWAM_ACTION_INIT"
  export PRECOMPUTE_QWEN3_CACHE="$IMAGEWAM_PRECOMPUTE_QWEN3_CACHE"

  echo "Running ImageWAM train entrypoint: $IMAGEWAM_TRAIN_ENTRYPOINT"
  echo "Output dir: $run_dir"
  bash "$IMAGEWAM_TRAIN_ENTRYPOINT" $IMAGEWAM_EXTRA_ARGS 2>&1 | tee "$run_dir/train_stdout.log"
)

echo "IMAGEWAM_TRAIN_COMPLETE run_dir=$run_dir"
