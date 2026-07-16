# shellcheck shell=bash

# LeRobot 推理配置：PushT + Diffusion policy。
# 用途：加载本地 policy，对本地 dataset 的一个 sample 做离线 action 推理。
# 默认不允许运行时下载，避免计算节点没网时卡住。

export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-lerobot/pusht}"
export LEROBOT_DATASET_ROOT="${LEROBOT_DATASET_ROOT:-$EMBODIED_REPO_ROOT/data/lerobot/pusht}"
export LEROBOT_SAMPLE_INDEX="${LEROBOT_SAMPLE_INDEX:-0}"
export LEROBOT_ALLOW_DOWNLOAD="${LEROBOT_ALLOW_DOWNLOAD:-0}"

# policy 必须已经下载到 models/lerobot/...；不要在推理脚本里临时拉远端模型。
export LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-diffusion}"
export LEROBOT_POLICY_CLASS="${LEROBOT_POLICY_CLASS:-lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy}"
export LEROBOT_POLICY_PATH="${LEROBOT_POLICY_PATH:-$EMBODIED_REPO_ROOT/models/lerobot/diffusion/diffusion_pusht}"
export LEROBOT_INFERENCE_DEVICE="${LEROBOT_INFERENCE_DEVICE:-cuda}"

export LEROBOT_RUN_NAME="${LEROBOT_RUN_NAME:-infer_pusht_diffusion}"
export LEROBOT_RUN_ROOT="${LEROBOT_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/lerobot_infer}"
