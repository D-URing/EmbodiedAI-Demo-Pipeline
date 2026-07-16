#!/usr/bin/env bash
# FastWAM 多机启动入口。
#
# 当前推荐：先用单机 8 卡 YAML 入口跑通，再使用本脚本。
#   python experiments/custom/fastwam_realrobot_single8_random/run.py
#
# 不通过 Slurm 手动多机时，每个节点必须设置：
#   FASTWAM_NNODES、FASTWAM_NODE_RANK、FASTWAM_MASTER_ADDR、FASTWAM_MASTER_PORT、FASTWAM_RUN_ID。
# FASTWAM_RUN_ID 必须所有节点一致，否则 checkpoint/log 会分裂到不同目录。
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$EXPERIMENT_DIR/config.sh}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

cd "$PROJECT_ROOT"
bash scripts/fastwam/run_realrobot_train_eval.sh "$CONFIG_PATH"
