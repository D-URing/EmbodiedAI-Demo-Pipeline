#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$EXPERIMENT_DIR/config.sh}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

cd "$PROJECT_ROOT"
bash scripts/lerobot/run_pusht_act_gpu_smoke.sh "$CONFIG_PATH"
