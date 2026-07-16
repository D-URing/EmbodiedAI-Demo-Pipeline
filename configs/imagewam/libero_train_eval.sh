#!/usr/bin/env bash
# ImageWAM / LIBERO 默认配置。
#
# 当前状态：候选路线/预留路线，不是主线交付。
# 主线请优先使用：
#   1. custom/FastWAM 真实训练；
#   2. LeRobot 官方生态训练/推理。
#
# 本文件可以被 shell source；集群脚本可先覆盖变量，再调用 scripts/imagewam/run_train_eval.sh。

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

# IMAGEWAM_MODE 说明：
#   metadata-smoke = 只检查元数据/目录/配置，不代表训练；
#   pilot/full     = 后续真实训练入口，需要上游 ImageWAM 环境和资产完整。
export IMAGEWAM_MODE="${IMAGEWAM_MODE:-metadata-smoke}"
export IMAGEWAM_VARIANT="${IMAGEWAM_VARIANT:-flux2_4b}"
export IMAGEWAM_TASK_TYPE="${IMAGEWAM_TASK_TYPE:-libero}"
export IMAGEWAM_TASK_SUITE="${IMAGEWAM_TASK_SUITE:-libero_spatial}"

# 复用 custom FastWAM 的 LIBERO 数据副本，避免项目内重复下载两份大数据。
# 如果后续 ImageWAM 需要专有格式，再单独扩展转换脚本。
export IMAGEWAM_DATA_ROOT="${IMAGEWAM_DATA_ROOT:-$EMBODIED_DATA_ROOT/custom/fastwam/libero-fastwam}"
export IMAGEWAM_MODEL_ROOT="${IMAGEWAM_MODEL_ROOT:-$EMBODIED_MODEL_ROOT/custom/imagewam}"
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
# 默认不下载 9B，避免第一阶段资产过重。
export IMAGEWAM_DOWNLOAD_9B="${IMAGEWAM_DOWNLOAD_9B:-false}"
export IMAGEWAM_ACTION_INIT="${IMAGEWAM_ACTION_INIT:-$IMAGEWAM_MODEL_ROOT/action_init/action_dit_flux2_${IMAGEWAM_FLUX2_VARIANT}_${IMAGEWAM_TASK_TYPE}_init.pt}"

# 本项目镜像输出目录。ImageWAM upstream 如果产生自己的输出，也需要在 run manifest 里记录。
export IMAGEWAM_RUN_ROOT="${IMAGEWAM_RUN_ROOT:-$EMBODIED_RUN_ROOT/manual/imagewam}"
export IMAGEWAM_RUN_NAME="${IMAGEWAM_RUN_NAME:-imagewam_${IMAGEWAM_VARIANT}_${IMAGEWAM_TASK_TYPE}_${IMAGEWAM_MODE}}"

# ImageWAM 官方仓库使用自己的 shell wrapper。这里保持可配置，因为上游入口可能变化。
export IMAGEWAM_TRAIN_ENTRYPOINT="${IMAGEWAM_TRAIN_ENTRYPOINT:-$IMAGEWAM_WORKDIR/scripts/flux2/run_train_flux2_klein_imagewam.sh}"
export IMAGEWAM_EVAL_ENTRYPOINT="${IMAGEWAM_EVAL_ENTRYPOINT:-$IMAGEWAM_WORKDIR/scripts/flux2/run_eval_flux2_libero.sh}"

# CUDA 和缓存开关。metadata-smoke 可临时 IMAGEWAM_REQUIRE_CUDA=0；真实训练必须有 CUDA。
export IMAGEWAM_GPUS_PER_NODE="${IMAGEWAM_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
export IMAGEWAM_REQUIRE_CUDA="${IMAGEWAM_REQUIRE_CUDA:-1}"
export IMAGEWAM_PRECOMPUTE_QWEN3_CACHE="${IMAGEWAM_PRECOMPUTE_QWEN3_CACHE:-${IMAGEWAM_PRECOMPUTE_CACHE:-false}}"
export IMAGEWAM_EXTRA_ARGS="${IMAGEWAM_EXTRA_ARGS:-}"

unset _imagewam_config_dir
