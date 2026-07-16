# shellcheck shell=bash

# LeRobot 训练配置 1：PushT + ACT。
# 用途：最轻量、最稳定的 LeRobot 真实训练入口，用来快速证明 dataset -> train -> checkpoint。
# 适合第一次检查 loss 是否下降；不代表最终 VLA 能力。

export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-lerobot/pusht}"
export LEROBOT_DATASET_ROOT="${LEROBOT_DATASET_ROOT:-$EMBODIED_REPO_ROOT/data/lerobot/pusht}"
export LEROBOT_DATASET_VIDEO_BACKEND="${LEROBOT_DATASET_VIDEO_BACKEND:-pyav}"

export LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-act}"
export LEROBOT_POLICY_REPO_ID="${LEROBOT_POLICY_REPO_ID:-local/pusht_act}"
export LEROBOT_POLICY_PUSH_TO_HUB="${LEROBOT_POLICY_PUSH_TO_HUB:-false}"
export LEROBOT_POLICY_DEVICE="${LEROBOT_POLICY_DEVICE:-cuda}"

# 训练规模。默认 1000 step 是短训；如果只想快速检查链路，可在实验 config.sh 覆盖更小。
export LEROBOT_STEPS="${LEROBOT_STEPS:-1000}"
export LEROBOT_BATCH_SIZE="${LEROBOT_BATCH_SIZE:-8}"
export LEROBOT_NUM_WORKERS="${LEROBOT_NUM_WORKERS:-4}"
export LEROBOT_LOG_FREQ="${LEROBOT_LOG_FREQ:-20}"
export LEROBOT_SAVE_FREQ="${LEROBOT_SAVE_FREQ:-1000}"
export LEROBOT_SEED="${LEROBOT_SEED:-1000}"

export LEROBOT_ENV_EVAL_FREQ="${LEROBOT_ENV_EVAL_FREQ:-0}"
export LEROBOT_EVAL_STEPS="${LEROBOT_EVAL_STEPS:-0}"
export LEROBOT_WANDB_ENABLE="${LEROBOT_WANDB_ENABLE:-false}"

export LEROBOT_RUN_NAME="${LEROBOT_RUN_NAME:-pusht_act_train}"
export LEROBOT_RUN_ROOT="${LEROBOT_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/lerobot}"
export TORCH_HOME="${TORCH_HOME:-$EMBODIED_REPO_ROOT/hf_cache/torch}"
