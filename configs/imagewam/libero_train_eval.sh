#!/usr/bin/env bash
# ImageWAM / LIBERO default profile.
#
# This file is intentionally shell-sourceable. Override any variable from the
# command line or cluster job script, then call scripts/imagewam/run_train_eval.sh.

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "configs/imagewam/libero_train_eval.sh must be sourced by bash" >&2
  return 2 2>/dev/null || exit 2
fi

_imagewam_config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$_imagewam_config_dir/../.." && pwd)}"

export EMBODIED_DATA_ROOT="${EMBODIED_DATA_ROOT:-$PROJECT_ROOT/data}"
export EMBODIED_MODEL_ROOT="${EMBODIED_MODEL_ROOT:-$PROJECT_ROOT/models}"
export EMBODIED_RUN_ROOT="${EMBODIED_RUN_ROOT:-$PROJECT_ROOT/runs}"
export EMBODIED_UPSTREAM_ROOT="${EMBODIED_UPSTREAM_ROOT:-$PROJECT_ROOT/upstreams}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

export IMAGEWAM_OFFICIAL_REPO="${IMAGEWAM_OFFICIAL_REPO:-https://github.com/yuyangalin/ImageWAM.git}"
export IMAGEWAM_OFFICIAL_REF="${IMAGEWAM_OFFICIAL_REF:-main}"
export IMAGEWAM_WORKDIR="${IMAGEWAM_WORKDIR:-$EMBODIED_UPSTREAM_ROOT/ImageWAM}"

export IMAGEWAM_MODE="${IMAGEWAM_MODE:-metadata-smoke}"
export IMAGEWAM_VARIANT="${IMAGEWAM_VARIANT:-flux2_4b}"
export IMAGEWAM_TASK_TYPE="${IMAGEWAM_TASK_TYPE:-libero}"
export IMAGEWAM_TASK_SUITE="${IMAGEWAM_TASK_SUITE:-libero_spatial}"

export IMAGEWAM_DATA_ROOT="${IMAGEWAM_DATA_ROOT:-$EMBODIED_DATA_ROOT/fastwam/libero-fastwam}"
export IMAGEWAM_MODEL_ROOT="${IMAGEWAM_MODEL_ROOT:-$EMBODIED_MODEL_ROOT/imagewam}"
export IMAGEWAM_POLICY_REPO_ID="${IMAGEWAM_POLICY_REPO_ID:-yuyangalin/ImageWAM-FLUX.2-4B-LIBERO}"
export IMAGEWAM_POLICY_LOCAL_DIR="${IMAGEWAM_POLICY_LOCAL_DIR:-$IMAGEWAM_MODEL_ROOT/flux2_klein_4b_libero}"
export IMAGEWAM_RELEASE_CKPT_PATH="${IMAGEWAM_RELEASE_CKPT_PATH:-$IMAGEWAM_POLICY_LOCAL_DIR/model.pt}"
export IMAGEWAM_DATASET_STATS_PATH="${IMAGEWAM_DATASET_STATS_PATH:-$IMAGEWAM_POLICY_LOCAL_DIR/dataset_stats.json}"

export IMAGEWAM_FLUX2_VARIANT="${IMAGEWAM_FLUX2_VARIANT:-4b}"
export IMAGEWAM_FLUX2_ROOT="${IMAGEWAM_FLUX2_ROOT:-$IMAGEWAM_MODEL_ROOT/flux2}"
export IMAGEWAM_FLUX2_SRC="${IMAGEWAM_FLUX2_SRC:-$IMAGEWAM_WORKDIR/third_party/flux2}"
export IMAGEWAM_FLUX2_MODEL_PATH="${IMAGEWAM_FLUX2_MODEL_PATH:-$IMAGEWAM_FLUX2_ROOT/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
export IMAGEWAM_FLUX2_AE_MODEL_PATH="${IMAGEWAM_FLUX2_AE_MODEL_PATH:-$IMAGEWAM_FLUX2_ROOT/FLUX.2-dev/ae.safetensors}"
export IMAGEWAM_FLUX2_QWEN3_MODEL_SPEC="${IMAGEWAM_FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
export IMAGEWAM_DOWNLOAD_9B="${IMAGEWAM_DOWNLOAD_9B:-false}"
export IMAGEWAM_ACTION_INIT="${IMAGEWAM_ACTION_INIT:-$IMAGEWAM_MODEL_ROOT/action_init/action_dit_flux2_${IMAGEWAM_FLUX2_VARIANT}_${IMAGEWAM_TASK_TYPE}_init.pt}"

export IMAGEWAM_RUN_ROOT="${IMAGEWAM_RUN_ROOT:-$EMBODIED_RUN_ROOT/imagewam}"
export IMAGEWAM_RUN_NAME="${IMAGEWAM_RUN_NAME:-imagewam_${IMAGEWAM_VARIANT}_${IMAGEWAM_TASK_TYPE}_${IMAGEWAM_MODE}}"

# Official ImageWAM uses its own shell wrappers. Keep this configurable because
# the upstream repo is external and may change.
export IMAGEWAM_TRAIN_ENTRYPOINT="${IMAGEWAM_TRAIN_ENTRYPOINT:-$IMAGEWAM_WORKDIR/scripts/flux2/run_train_flux2_klein_imagewam.sh}"
export IMAGEWAM_EVAL_ENTRYPOINT="${IMAGEWAM_EVAL_ENTRYPOINT:-$IMAGEWAM_WORKDIR/scripts/flux2/run_eval_flux2_libero.sh}"

export IMAGEWAM_GPUS_PER_NODE="${IMAGEWAM_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
export IMAGEWAM_REQUIRE_CUDA="${IMAGEWAM_REQUIRE_CUDA:-1}"
export IMAGEWAM_PRECOMPUTE_QWEN3_CACHE="${IMAGEWAM_PRECOMPUTE_QWEN3_CACHE:-${IMAGEWAM_PRECOMPUTE_CACHE:-false}}"
export IMAGEWAM_EXTRA_ARGS="${IMAGEWAM_EXTRA_ARGS:-}"

unset _imagewam_config_dir
