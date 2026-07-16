#!/usr/bin/env bash
# shellcheck shell=bash

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/configs/lerobot/infer/fastwam_libero.sh"

# LeRobot 路线的 FastWAM 离线推理实验。不要和 custom/FastWAM 训练路线混用数据目录。
export EXPERIMENT_ROUTE="lerobot"
export EXPERIMENT_NAME="fastwam_libero_infer"
export LEROBOT_RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$PROJECT_ROOT/runs/experiments/$EXPERIMENT_ROUTE}"
export LEROBOT_RUN_NAME="${EXPERIMENT_RUN_NAME:-$EXPERIMENT_NAME}"
