#!/usr/bin/env bash
# shellcheck shell=bash

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/configs/imagewam/libero_train_eval.sh"

export EXPERIMENT_ROUTE="custom"
export EXPERIMENT_NAME="imagewam_flux2_4b_libero_pilot"
export IMAGEWAM_RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$PROJECT_ROOT/runs/experiments/$EXPERIMENT_ROUTE}"
export IMAGEWAM_RUN_NAME="${EXPERIMENT_RUN_NAME:-$EXPERIMENT_NAME}"

export IMAGEWAM_MODE="${IMAGEWAM_MODE:-pilot}"
export IMAGEWAM_VARIANT="${IMAGEWAM_VARIANT:-flux2_4b}"
export IMAGEWAM_FLUX2_VARIANT="${IMAGEWAM_FLUX2_VARIANT:-4b}"
export IMAGEWAM_TASK_TYPE="${IMAGEWAM_TASK_TYPE:-libero}"
export IMAGEWAM_PRECOMPUTE_QWEN3_CACHE="${IMAGEWAM_PRECOMPUTE_QWEN3_CACHE:-true}"
