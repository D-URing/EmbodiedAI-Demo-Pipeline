#!/usr/bin/env bash
# shellcheck shell=bash

# ImageWAM custom 路线配置。
#
# 重要状态：这条线目前仍是候选/实验性路线，不是当前主线交付。
# 当前主线是 custom/FastWAM 和 LeRobot。ImageWAM 这里保留用于后续接入，
# 但不要拿 metadata-smoke 当成真实训练结果对外汇报。

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/configs/imagewam/libero_train_eval.sh"

export EXPERIMENT_ROUTE="custom"
export EXPERIMENT_NAME="imagewam_flux2_4b_libero_pilot"

# ImageWAM 产物目录。保留到 runs/experiments/custom 下，避免和 FastWAM 混在一起。
export IMAGEWAM_RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$PROJECT_ROOT/runs/experiments/$EXPERIMENT_ROUTE}"
export IMAGEWAM_RUN_NAME="${EXPERIMENT_RUN_NAME:-$EXPERIMENT_NAME}"

# IMAGEWAM_MODE=pilot 表示预留真实训练入口；metadata-smoke 只检查元数据/接线。
export IMAGEWAM_MODE="${IMAGEWAM_MODE:-pilot}"
export IMAGEWAM_VARIANT="${IMAGEWAM_VARIANT:-flux2_4b}"
export IMAGEWAM_FLUX2_VARIANT="${IMAGEWAM_FLUX2_VARIANT:-4b}"
export IMAGEWAM_TASK_TYPE="${IMAGEWAM_TASK_TYPE:-libero}"

# Qwen3 cache 预计算属于后续真实训练准备项；如果只检查目录/配置，可临时关掉。
export IMAGEWAM_PRECOMPUTE_QWEN3_CACHE="${IMAGEWAM_PRECOMPUTE_QWEN3_CACHE:-true}"
