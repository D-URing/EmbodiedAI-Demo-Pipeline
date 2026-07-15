#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$EXPERIMENT_DIR/config.sh}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

cd "$PROJECT_ROOT"
bash scripts/imagewam/run_train_eval.sh "$CONFIG_PATH"
