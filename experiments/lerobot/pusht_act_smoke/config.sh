#!/usr/bin/env bash
# shellcheck shell=bash

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

# 复用 configs/lerobot/train/pusht_act.sh 的稳定 ACT/PushT 配置，
# 这里只负责声明“这是哪个实验”和“输出到哪里”。
# shellcheck disable=SC1091
source "$PROJECT_ROOT/configs/lerobot/train/pusht_act.sh"

export EXPERIMENT_ROUTE="lerobot"
export EXPERIMENT_NAME="pusht_act_smoke"
export LEROBOT_RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$PROJECT_ROOT/runs/experiments/$EXPERIMENT_ROUTE}"
export LEROBOT_RUN_NAME="${EXPERIMENT_RUN_NAME:-$EXPERIMENT_NAME}"

# 保持为快速 proof run。需要更长训练时从 shell 覆盖 LEROBOT_STEPS/SAVE_FREQ，
# 不建议为了某次试验直接改公共 profile。
export LEROBOT_STEPS="${LEROBOT_STEPS:-1000}"
export LEROBOT_SAVE_FREQ="${LEROBOT_SAVE_FREQ:-1000}"
