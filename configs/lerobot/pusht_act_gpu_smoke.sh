# shellcheck shell=bash

# 固定到 references/upstreams.yaml 记录的 LeRobot commit，保证团队复现一致。
export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export LEROBOT_UPSTREAM_COMMIT="${LEROBOT_UPSTREAM_COMMIT:-e40b58a8dfa9e7b86918c374791599d070518d11}"

# LeRobot 轻量真实训练配置：PushT + ACT。
# 名字里保留 smoke 是历史命名；这里不是 mock，会真实训练并产出 loss/checkpoint。
# 集群上需要更长训练时，从 shell 或实验 config.sh 覆盖这些变量。
export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-lerobot/pusht}"
export LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-act}"
export LEROBOT_POLICY_REPO_ID="${LEROBOT_POLICY_REPO_ID:-local/pusht_act_gpu_smoke}"
export LEROBOT_POLICY_PUSH_TO_HUB="${LEROBOT_POLICY_PUSH_TO_HUB:-false}"

# SCUT gpu11 的宿主 glibc 偏旧，torchcodec + conda-forge ffmpeg 容易踩 ABI 问题。
# PushT 这条轻量训练用 pyav 更稳，也避免运行时下载额外视频依赖。
export LEROBOT_DATASET_VIDEO_BACKEND="${LEROBOT_DATASET_VIDEO_BACKEND:-pyav}"

# 默认规模足够短，适合第一次集群检查；但仍会产生真实 loss 日志。
export LEROBOT_STEPS="${LEROBOT_STEPS:-1000}"
export LEROBOT_BATCH_SIZE="${LEROBOT_BATCH_SIZE:-8}"
export LEROBOT_NUM_WORKERS="${LEROBOT_NUM_WORKERS:-4}"
export LEROBOT_LOG_FREQ="${LEROBOT_LOG_FREQ:-20}"
export LEROBOT_SAVE_FREQ="${LEROBOT_SAVE_FREQ:-1000}"
export LEROBOT_SEED="${LEROBOT_SEED:-1000}"

# CUDA-only。runner 会拒绝 CPU fallback。
export LEROBOT_POLICY_DEVICE="${LEROBOT_POLICY_DEVICE:-cuda}"

# 默认关闭仿真评测和 wandb；这条入口目标是训练 loss 和 checkpoint，不声明任务成功率。
export LEROBOT_ENV_EVAL_FREQ="${LEROBOT_ENV_EVAL_FREQ:-0}"
export LEROBOT_EVAL_STEPS="${LEROBOT_EVAL_STEPS:-0}"
export LEROBOT_WANDB_ENABLE="${LEROBOT_WANDB_ENABLE:-false}"

export LEROBOT_RUN_NAME="${LEROBOT_RUN_NAME:-pusht_act_gpu_smoke}"
export LEROBOT_RUN_ROOT="${LEROBOT_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/lerobot}"
export TORCH_HOME="${TORCH_HOME:-$EMBODIED_REPO_ROOT/hf_cache/torch}"
