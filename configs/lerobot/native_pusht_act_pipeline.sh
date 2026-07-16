# shellcheck shell=bash

# LeRobot-native data-to-inference 配置。
# 用途：读取一个本地 PushT sample，加载本地 ACT policy/checkpoint，做离线推理。
# 默认不下载远端资产；只有显式设置 LEROBOT_ALLOW_DOWNLOAD=1 才允许联网。

export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export LEROBOT_UPSTREAM_COMMIT="${LEROBOT_UPSTREAM_COMMIT:-e40b58a8dfa9e7b86918c374791599d070518d11}"

export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-lerobot/pusht}"
export LEROBOT_DATASET_ROOT="${LEROBOT_DATASET_ROOT:-}"
export LEROBOT_DATASET_SPLIT="${LEROBOT_DATASET_SPLIT:-}"
export LEROBOT_SAMPLE_INDEX="${LEROBOT_SAMPLE_INDEX:-0}"
export LEROBOT_ALLOW_DOWNLOAD="${LEROBOT_ALLOW_DOWNLOAD:-0}"

export LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-act}"
export LEROBOT_POLICY_CLASS="${LEROBOT_POLICY_CLASS:-lerobot.policies.act.modeling_act.ACTPolicy}"
export LEROBOT_POLICY_PATH="${LEROBOT_POLICY_PATH:-}"
export LEROBOT_INFERENCE_DEVICE="${LEROBOT_INFERENCE_DEVICE:-cuda}"

export LEROBOT_RUN_NAME="${LEROBOT_RUN_NAME:-lerobot_native_pusht_act}"
export LEROBOT_RUN_ROOT="${LEROBOT_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/lerobot_native}"
