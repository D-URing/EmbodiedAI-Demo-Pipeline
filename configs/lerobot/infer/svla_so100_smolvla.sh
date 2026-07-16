# shellcheck shell=bash

# LeRobot 推理配置：SO100 pick-place + SmolVLA。
# 用途：验证本地 SmolVLA policy/base 能读一个 LeRobot sample 并输出 action。
# 这不是环境交互评测，只是 data-to-policy inference chain check。

export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-lerobot/svla_so100_pickplace}"
export LEROBOT_DATASET_ROOT="${LEROBOT_DATASET_ROOT:-$EMBODIED_REPO_ROOT/data/lerobot/svla_so100_pickplace}"
export LEROBOT_SAMPLE_INDEX="${LEROBOT_SAMPLE_INDEX:-0}"
export LEROBOT_ALLOW_DOWNLOAD="${LEROBOT_ALLOW_DOWNLOAD:-0}"

# 默认加载 base policy；如果要测自己训练出的 checkpoint，把这里覆盖到 checkpoint 目录。
export LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-smolvla}"
export LEROBOT_POLICY_CLASS="${LEROBOT_POLICY_CLASS:-lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy}"
export LEROBOT_POLICY_PATH="${LEROBOT_POLICY_PATH:-$EMBODIED_REPO_ROOT/models/lerobot/smolvla/smolvla_base}"
export LEROBOT_INFERENCE_DEVICE="${LEROBOT_INFERENCE_DEVICE:-cuda}"

export LEROBOT_RUN_NAME="${LEROBOT_RUN_NAME:-infer_svla_so100_smolvla}"
export LEROBOT_RUN_ROOT="${LEROBOT_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/lerobot_infer}"
