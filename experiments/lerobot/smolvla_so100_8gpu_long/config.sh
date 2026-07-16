#!/usr/bin/env bash
# shellcheck shell=bash

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/configs/lerobot/train/svla_so100_smolvla_8gpu_long.sh"

# 单机 8 卡 SmolVLA 长训实验。这里覆盖的是实验身份和常用规模参数。
# 如果要续训，请设置 LEROBOT_RESUME=1 和 LEROBOT_RESUME_CONFIG_PATH。
export EXPERIMENT_ROUTE="lerobot"
export EXPERIMENT_NAME="smolvla_so100_8gpu_long"
export LEROBOT_RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$PROJECT_ROOT/runs/experiments/$EXPERIMENT_ROUTE}"
export LEROBOT_RUN_NAME="${EXPERIMENT_RUN_NAME:-$EXPERIMENT_NAME}"

export LEROBOT_NUM_PROCESSES="${LEROBOT_NUM_PROCESSES:-8}"
# 下面几个值是长训默认值；提交集群任务时可以通过环境变量覆盖。
export LEROBOT_BATCH_SIZE="${LEROBOT_BATCH_SIZE:-8}"
export LEROBOT_STEPS="${LEROBOT_STEPS:-20000}"
export LEROBOT_SAVE_FREQ="${LEROBOT_SAVE_FREQ:-1000}"
