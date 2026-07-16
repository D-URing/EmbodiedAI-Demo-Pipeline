#!/usr/bin/env bash
# shellcheck shell=bash

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/configs/lerobot/train/svla_so100_pi05_8gpu_probe.sh"

# pi05 / SO100 训练与测速探针的兼容 shell config。
# 日常实验主入口是同目录 config.yaml；run.py 会把 YAML 渲染为 generated shell config。
export EXPERIMENT_ROUTE="lerobot"
export EXPERIMENT_NAME="pi05_so100_8gpu_probe"
export LEROBOT_RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$PROJECT_ROOT/runs/experiments/$EXPERIMENT_ROUTE}"
export LEROBOT_RUN_NAME="${EXPERIMENT_RUN_NAME:-$EXPERIMENT_NAME}"
