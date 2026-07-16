#!/usr/bin/env bash
# ImageWAM 启动入口。
# 当前不是主线交付路径；优先使用 custom/FastWAM。保留本脚本是为了后续接入 ImageWAM 时
# 复用相同的 config.sh + launch.sh + slurm.sbatch 项目结构。
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$EXPERIMENT_DIR/config.sh}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

cd "$PROJECT_ROOT"
bash scripts/imagewam/run_train_eval.sh "$CONFIG_PATH"
