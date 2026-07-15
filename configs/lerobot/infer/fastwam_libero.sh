# shellcheck shell=bash

export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# The raw FastWAM LIBERO release data under data/custom/fastwam/libero-fastwam is LeRobot v2.1.
# Point LEROBOT_DATASET_ROOT at a converted v3 subset before using this inference profile
# with current LeRobot loaders.
export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-yuanty/LIBERO-fastwam}"
export LEROBOT_DATASET_ROOT="${LEROBOT_DATASET_ROOT:-$EMBODIED_REPO_ROOT/data/custom/fastwam/libero-fastwam/libero_10_no_noops_lerobot_v3}"
export LEROBOT_SAMPLE_INDEX="${LEROBOT_SAMPLE_INDEX:-0}"
export LEROBOT_ALLOW_DOWNLOAD="${LEROBOT_ALLOW_DOWNLOAD:-0}"

export LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-fastwam}"
export LEROBOT_POLICY_CLASS="${LEROBOT_POLICY_CLASS:-lerobot.policies.fastwam.modeling_fastwam.FastWAMPolicy}"
export LEROBOT_POLICY_PATH="${LEROBOT_POLICY_PATH:-$EMBODIED_REPO_ROOT/models/lerobot/fastwam/fastwam_libero_uncond_2cam224}"
export LEROBOT_INFERENCE_DEVICE="${LEROBOT_INFERENCE_DEVICE:-cuda}"

export LEROBOT_RUN_NAME="${LEROBOT_RUN_NAME:-infer_fastwam_libero}"
export LEROBOT_RUN_ROOT="${LEROBOT_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/lerobot_infer}"
