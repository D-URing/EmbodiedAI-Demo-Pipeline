# shellcheck shell=bash

export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# Keep LeRobot-route LIBERO data separate from the custom/FastWAM route.
# data/lerobot/libero-fastwam/v2.1 may contain a copied raw release, while
# data/lerobot/libero-fastwam/v3 is the expected converted layout for current
# LeRobot loaders.
export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-yuanty/LIBERO-fastwam}"
export LEROBOT_FASTWAM_LIBERO_ROOT="${LEROBOT_FASTWAM_LIBERO_ROOT:-$EMBODIED_REPO_ROOT/data/lerobot/libero-fastwam}"
export LEROBOT_FASTWAM_LIBERO_VERSION="${LEROBOT_FASTWAM_LIBERO_VERSION:-v3}"
export LEROBOT_FASTWAM_LIBERO_SUBSET="${LEROBOT_FASTWAM_LIBERO_SUBSET:-libero_10_no_noops_lerobot}"
export LEROBOT_DATASET_ROOT="${LEROBOT_DATASET_ROOT:-$LEROBOT_FASTWAM_LIBERO_ROOT/$LEROBOT_FASTWAM_LIBERO_VERSION/$LEROBOT_FASTWAM_LIBERO_SUBSET}"
export LEROBOT_SAMPLE_INDEX="${LEROBOT_SAMPLE_INDEX:-0}"
export LEROBOT_ALLOW_DOWNLOAD="${LEROBOT_ALLOW_DOWNLOAD:-0}"

export LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-fastwam}"
export LEROBOT_POLICY_CLASS="${LEROBOT_POLICY_CLASS:-lerobot.policies.fastwam.modeling_fastwam.FastWAMPolicy}"
export LEROBOT_POLICY_PATH="${LEROBOT_POLICY_PATH:-$EMBODIED_REPO_ROOT/models/lerobot/fastwam/fastwam_libero_uncond_2cam224}"
export LEROBOT_INFERENCE_DEVICE="${LEROBOT_INFERENCE_DEVICE:-cuda}"

export LEROBOT_RUN_NAME="${LEROBOT_RUN_NAME:-infer_fastwam_libero}"
export LEROBOT_RUN_ROOT="${LEROBOT_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/lerobot_infer}"
