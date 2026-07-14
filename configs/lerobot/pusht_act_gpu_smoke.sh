# shellcheck shell=bash

# Pinned to the LeRobot commit recorded in references/upstreams.yaml.
export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export LEROBOT_UPSTREAM_COMMIT="${LEROBOT_UPSTREAM_COMMIT:-e40b58a8dfa9e7b86918c374791599d070518d11}"

# Official lightweight LeRobot smoke target. Override these on the cluster when needed.
export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-lerobot/pusht}"
export LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-act}"

# Keep the first run short enough for a cluster smoke job while still producing loss logs.
export LEROBOT_STEPS="${LEROBOT_STEPS:-1000}"
export LEROBOT_BATCH_SIZE="${LEROBOT_BATCH_SIZE:-8}"
export LEROBOT_NUM_WORKERS="${LEROBOT_NUM_WORKERS:-4}"
export LEROBOT_LOG_FREQ="${LEROBOT_LOG_FREQ:-20}"
export LEROBOT_SAVE_FREQ="${LEROBOT_SAVE_FREQ:-1000}"
export LEROBOT_SEED="${LEROBOT_SEED:-1000}"

# CUDA-only. The runner refuses to execute if torch.cuda.is_available() is false.
export LEROBOT_POLICY_DEVICE="${LEROBOT_POLICY_DEVICE:-cuda}"

# Disable simulator eval and wandb for the smoke run. The goal is training loss and checkpoint.
export LEROBOT_ENV_EVAL_FREQ="${LEROBOT_ENV_EVAL_FREQ:-0}"
export LEROBOT_EVAL_STEPS="${LEROBOT_EVAL_STEPS:-0}"
export LEROBOT_WANDB_ENABLE="${LEROBOT_WANDB_ENABLE:-false}"

export LEROBOT_RUN_NAME="${LEROBOT_RUN_NAME:-pusht_act_gpu_smoke}"
export LEROBOT_RUN_ROOT="${LEROBOT_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/lerobot}"
