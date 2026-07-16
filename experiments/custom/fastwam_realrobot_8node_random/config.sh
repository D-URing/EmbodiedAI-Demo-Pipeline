#!/usr/bin/env bash
# shellcheck shell=bash
#
# FastWAM 多机 8 节点随机初始化配置。
#
# 当前优先级：低于 single8 YAML 入口。也就是说：
#   1. 日常调试/给同事演示，先用 fastwam_realrobot_single8_random/config.yaml；
#   2. 确认单机 8 卡稳定后，再用本目录做 8 节点长实验。
#
# 这个文件会被 launch.sh 和 slurm.sbatch source。
# 它默认 8 nodes x 8 GPUs，并且默认 init=random，只证明训练可扩展，不代表 release 微调。

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../../.." && pwd)}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/configs/fastwam/realrobot_train_eval.sh"

export EXPERIMENT_ROUTE="custom"
export EXPERIMENT_NAME="fastwam_realrobot_8node_random"

# 本项目镜像 run 目录：runs/experiments/custom/<run_name>/<run_id>/。
# FastWAM 原生 checkpoint 仍写入 upstreams/FastWAM-realrobot/runs/<task>/<run_id>/。
export FASTWAM_RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$PROJECT_ROOT/runs/experiments/$EXPERIMENT_ROUTE}"
export FASTWAM_RUN_NAME="${EXPERIMENT_RUN_NAME:-$EXPERIMENT_NAME}"

# 训练规模和初始化方式。
# 多机默认跑 pilot，避免第一次就启动过长任务。
# 如果要正式微调，请把 FASTWAM_INIT 改成 release/base，并明确 resume checkpoint。
export FASTWAM_MODE="${FASTWAM_MODE:-pilot}"
export FASTWAM_RECIPE="${FASTWAM_RECIPE:-v6_scratch}"
export FASTWAM_INIT="${FASTWAM_INIT:-random}"

# 分布式拓扑。Slurm 会覆盖 NNODES/NODE_RANK；手动多机时必须每台机器自己设置。
export FASTWAM_GPUS_PER_NODE="${FASTWAM_GPUS_PER_NODE:-8}"
export FASTWAM_NNODES="${FASTWAM_NNODES:-${NNODES:-8}}"
export FASTWAM_NODE_RANK="${FASTWAM_NODE_RANK:-${NODE_RANK:-0}}"

# pilot 训练参数。正式长实验建议在提交作业时通过环境变量覆盖，而不是直接改共享配置。
export FASTWAM_PILOT_MAX_STEPS="${FASTWAM_PILOT_MAX_STEPS:-200}"
export FASTWAM_PILOT_BATCH_SIZE="${FASTWAM_PILOT_BATCH_SIZE:-1}"
export FASTWAM_PILOT_NUM_WORKERS="${FASTWAM_PILOT_NUM_WORKERS:-4}"
export FASTWAM_PILOT_SAVE_EVERY="${FASTWAM_PILOT_SAVE_EVERY:-50}"

# 默认关闭 wandb，避免集群外网/权限问题。需要记录长期实验时再显式打开。
export FASTWAM_WANDB_ENABLE="${FASTWAM_WANDB_ENABLE:-false}"
