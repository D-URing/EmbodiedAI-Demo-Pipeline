#!/usr/bin/env bash
# Multi-node FastWAM launcher.
#
# This is kept for the later 8-node route. For the current real single-node
# 8-GPU validation, prefer:
#   python experiments/custom/fastwam_realrobot_single8_random/run.py
#
# When used without Slurm, every node must set FASTWAM_NNODES, FASTWAM_NODE_RANK,
# FASTWAM_MASTER_ADDR, FASTWAM_MASTER_PORT, and the same FASTWAM_RUN_ID.
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$EXPERIMENT_DIR/config.sh}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

cd "$PROJECT_ROOT"
bash scripts/fastwam/run_realrobot_train_eval.sh "$CONFIG_PATH"
