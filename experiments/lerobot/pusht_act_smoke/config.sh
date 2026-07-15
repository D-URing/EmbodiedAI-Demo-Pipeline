#!/usr/bin/env bash
# shellcheck shell=bash

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

# Reuse the stable LeRobot ACT/PushT profile, then override experiment identity.
# shellcheck disable=SC1091
source "$PROJECT_ROOT/configs/lerobot/train/pusht_act.sh"

export EXPERIMENT_ROUTE="lerobot"
export EXPERIMENT_NAME="pusht_act_smoke"
export LEROBOT_RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$PROJECT_ROOT/runs/experiments/$EXPERIMENT_ROUTE}"
export LEROBOT_RUN_NAME="${EXPERIMENT_RUN_NAME:-$EXPERIMENT_NAME}"

# Keep this as a quick proof run. Override from shell for longer runs.
export LEROBOT_STEPS="${LEROBOT_STEPS:-1000}"
export LEROBOT_SAVE_FREQ="${LEROBOT_SAVE_FREQ:-1000}"
