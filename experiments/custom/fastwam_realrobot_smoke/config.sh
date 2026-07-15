#!/usr/bin/env bash
# shellcheck shell=bash

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/configs/fastwam/realrobot_train_eval.sh"

export EXPERIMENT_ROUTE="custom"
export EXPERIMENT_NAME="fastwam_realrobot_smoke"
export FASTWAM_RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$PROJECT_ROOT/runs/experiments/$EXPERIMENT_ROUTE}"
export FASTWAM_RUN_NAME="${EXPERIMENT_RUN_NAME:-$EXPERIMENT_NAME}"
export FASTWAM_MODE="${FASTWAM_MODE:-smoke}"
