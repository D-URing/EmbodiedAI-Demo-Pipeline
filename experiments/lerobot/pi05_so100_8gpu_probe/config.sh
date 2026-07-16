#!/usr/bin/env bash
# shellcheck shell=bash

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/configs/lerobot/train/svla_so100_pi05_8gpu_probe.sh"

# pi05 / SO100 训练与测速探针。默认 200 steps，适合先确认 loss 与吞吐；
# 长训时通过环境变量覆盖 LEROBOT_STEPS、LEROBOT_BATCH_SIZE、LEROBOT_SAVE_FREQ。
export EXPERIMENT_ROUTE="lerobot"
export EXPERIMENT_NAME="pi05_so100_8gpu_probe"
export LEROBOT_RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$PROJECT_ROOT/runs/experiments/$EXPERIMENT_ROUTE}"
export LEROBOT_RUN_NAME="${EXPERIMENT_RUN_NAME:-$EXPERIMENT_NAME}"
