# shellcheck shell=bash

# LeRobot 推理配置：SO100 pick-place + pi05。
#
# 用途：从本地 SO100 数据集中取一个 sample，加载本地 pi05 base/checkpoint，
# 跑一次离线 action 预测并写出 inference_evidence.json。
# 这不是环境交互评测，只验证 data -> policy -> action 的推理链路。

export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-lerobot/svla_so100_pickplace}"
export LEROBOT_DATASET_ROOT="${LEROBOT_DATASET_ROOT:-$EMBODIED_REPO_ROOT/data/lerobot/svla_so100_pickplace}"
export LEROBOT_SAMPLE_INDEX="${LEROBOT_SAMPLE_INDEX:-0}"
export LEROBOT_ALLOW_DOWNLOAD="${LEROBOT_ALLOW_DOWNLOAD:-0}"

# 默认加载 pi05 base。若要测训练 checkpoint，把 LEROBOT_POLICY_PATH 覆盖到 checkpoint 目录。
export LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-pi05}"
export LEROBOT_POLICY_CLASS="${LEROBOT_POLICY_CLASS:-lerobot.policies.pi05.modeling_pi05.PI05Policy}"
export LEROBOT_POLICY_PATH="${LEROBOT_POLICY_PATH:-$EMBODIED_REPO_ROOT/models/lerobot/pi05/pi05_base}"
export LEROBOT_INFERENCE_DEVICE="${LEROBOT_INFERENCE_DEVICE:-cuda}"

export LEROBOT_RUN_NAME="${LEROBOT_RUN_NAME:-infer_svla_so100_pi05}"
export LEROBOT_RUN_ROOT="${LEROBOT_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/lerobot_infer}"
export HF_HOME="${HF_HOME:-$EMBODIED_REPO_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
