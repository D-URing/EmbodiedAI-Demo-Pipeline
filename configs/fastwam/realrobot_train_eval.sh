# shellcheck shell=bash

# FastWAM real-robot training/evaluation backend defaults.
#
# This config intentionally describes an external, CUDA-only policy/training
# environment. It is not sourced by the lightweight core demo environment.

# Official FastWAM base repository plus the private internal real-robot overlay.
export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export FASTWAM_OFFICIAL_REPO="${FASTWAM_OFFICIAL_REPO:-https://github.com/yuantianyuan01/FastWAM.git}"
export FASTWAM_OFFICIAL_REF="${FASTWAM_OFFICIAL_REF:-45d8e1458921d83f8ad6cf9ce993d371208dabd0}"
export FASTWAM_OVERLAY_REPO="${FASTWAM_OVERLAY_REPO:-https://github.com/D-URing/fastwam-realrobot-pipeline.git}"
export FASTWAM_OVERLAY_REF="${FASTWAM_OVERLAY_REF:-5b9791f7d49956b96e0694786f46ff94e8214eca}"

# Source layout. FASTWAM_WORKDIR is the overlaid runnable FastWAM tree.
export FASTWAM_CACHE_ROOT="${FASTWAM_CACHE_ROOT:-$EMBODIED_REPO_ROOT/upstreams}"
export FASTWAM_WORKDIR="${FASTWAM_WORKDIR:-$FASTWAM_CACHE_ROOT/FastWAM-realrobot}"
export FASTWAM_OVERLAY_DIR="${FASTWAM_OVERLAY_DIR:-$FASTWAM_CACHE_ROOT/fastwam-realrobot-pipeline}"
export FASTWAM_RESET_WORKDIR="${FASTWAM_RESET_WORKDIR:-0}"

# Model/checkpoint locations. Defaults are repo-local because the project itself
# is expected to live on shared storage during early cluster testing.
export FASTWAM_MODEL_BASE="${FASTWAM_MODEL_BASE:-$EMBODIED_REPO_ROOT/models}"
export FASTWAM_RELEASE_DIR="${FASTWAM_RELEASE_DIR:-$FASTWAM_MODEL_BASE/custom/fastwam/release}"
export FASTWAM_RELEASE_CKPT="${FASTWAM_RELEASE_CKPT:-$FASTWAM_RELEASE_DIR/libero_uncond_2cam224.pt}"
export FASTWAM_RELEASE_DATASET_STATS="${FASTWAM_RELEASE_DATASET_STATS:-$FASTWAM_RELEASE_DIR/libero_uncond_2cam224_dataset_stats.json}"
export FASTWAM_ACTION_DIT_BACKBONE="${FASTWAM_ACTION_DIT_BACKBONE:-$EMBODIED_REPO_ROOT/checkpoints/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"

# Optional pinned normalization stats for V6 multi-node recipes. Leave empty for
# non-V6 smoke/pilot runs or when the FastWAM config should compute/read its own.
export FASTWAM_PIN_STATS="${FASTWAM_PIN_STATS:-}"

# What to run.
#   FASTWAM_MODE: smoke | pilot | full
#   FASTWAM_RECIPE:
#     joint_base | pose_base | v6_clean | v6_decision | v6_codebook |
#     v6_scratch | v6_discrim | v6_dagger | v6_robust
#   FASTWAM_TASK_NAME can override the recipe-to-task mapping directly.
export FASTWAM_MODE="${FASTWAM_MODE:-smoke}"
export FASTWAM_RECIPE="${FASTWAM_RECIPE:-joint_base}"
export FASTWAM_TASK_NAME="${FASTWAM_TASK_NAME:-}"

# CUDA-only. The runner refuses to execute without CUDA by default.
export FASTWAM_REQUIRE_CUDA="${FASTWAM_REQUIRE_CUDA:-1}"
export FASTWAM_GPUS_PER_NODE="${FASTWAM_GPUS_PER_NODE:-}"
export FASTWAM_NNODES="${FASTWAM_NNODES:-${NNODES:-1}}"
export FASTWAM_NODE_RANK="${FASTWAM_NODE_RANK:-${NODE_RANK:-0}}"
export FASTWAM_MIXED_PRECISION="${FASTWAM_MIXED_PRECISION:-bf16}"
export FASTWAM_WANDB_ENABLE="${FASTWAM_WANDB_ENABLE:-false}"
export FASTWAM_MASTER_ADDR="${FASTWAM_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
export FASTWAM_MASTER_PORT="${FASTWAM_MASTER_PORT:-${MASTER_PORT:-29500}}"
export FASTWAM_INIT="${FASTWAM_INIT:-release}"
export FASTWAM_MODEL_ID="${FASTWAM_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
export FASTWAM_TOKENIZER_MODEL_ID="${FASTWAM_TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
export FASTWAM_REDIRECT_COMMON_FILES="${FASTWAM_REDIRECT_COMMON_FILES:-false}"

# FastWAM reads cached T5/Wan text embeddings during training. The cache is a
# real prerequisite, not a smoke-test artifact. Keep it enabled by default so a
# training run first materializes missing prompt embeddings under the upstream
# runnable tree, which points back to project-local data/ through a symlink.
# Values: auto|1|true|yes to run, 0|false|no to require pre-existing cache.
export FASTWAM_PRECOMPUTE_TEXT_EMBEDS="${FASTWAM_PRECOMPUTE_TEXT_EMBEDS:-auto}"
export FASTWAM_TEXT_EMBED_GPUS="${FASTWAM_TEXT_EMBED_GPUS:-}"
export FASTWAM_TEXT_EMBED_OVERWRITE="${FASTWAM_TEXT_EMBED_OVERWRITE:-false}"
export FASTWAM_TEXT_EMBED_WAIT_TIMEOUT="${FASTWAM_TEXT_EMBED_WAIT_TIMEOUT:-3600}"
export FASTWAM_TEXT_EMBED_MASTER_ADDR="${FASTWAM_TEXT_EMBED_MASTER_ADDR:-127.0.0.1}"
export FASTWAM_TEXT_EMBED_MASTER_PORT="${FASTWAM_TEXT_EMBED_MASTER_PORT:-29517}"

# Manual run artifact mirror owned by this demo pipeline. Experiment launchers
# override this to runs/experiments/custom/<experiment>/.
# FastWAM still writes its native checkpoints under FASTWAM_WORKDIR/runs/<task>/<run_id>/.
export FASTWAM_RUN_ROOT="${FASTWAM_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/manual/fastwam}"
export FASTWAM_RUN_NAME="${FASTWAM_RUN_NAME:-realrobot_${FASTWAM_RECIPE}_${FASTWAM_MODE}}"
export FASTWAM_RUN_ID="${FASTWAM_RUN_ID:-}"

# Mode presets. These are intentionally modest; use environment overrides on the
# cluster rather than editing this shared config.
export FASTWAM_SMOKE_MAX_STEPS="${FASTWAM_SMOKE_MAX_STEPS:-1}"
export FASTWAM_SMOKE_BATCH_SIZE="${FASTWAM_SMOKE_BATCH_SIZE:-1}"
export FASTWAM_SMOKE_NUM_WORKERS="${FASTWAM_SMOKE_NUM_WORKERS:-0}"

export FASTWAM_PILOT_MAX_STEPS="${FASTWAM_PILOT_MAX_STEPS:-200}"
export FASTWAM_PILOT_BATCH_SIZE="${FASTWAM_PILOT_BATCH_SIZE:-2}"
export FASTWAM_PILOT_NUM_WORKERS="${FASTWAM_PILOT_NUM_WORKERS:-4}"
export FASTWAM_PILOT_SAVE_EVERY="${FASTWAM_PILOT_SAVE_EVERY:-50}"

export FASTWAM_FULL_BATCH_SIZE="${FASTWAM_FULL_BATCH_SIZE:-4}"
export FASTWAM_FULL_NUM_WORKERS="${FASTWAM_FULL_NUM_WORKERS:-8}"
export FASTWAM_FULL_NUM_EPOCHS="${FASTWAM_FULL_NUM_EPOCHS:-5}"
export FASTWAM_FULL_SAVE_EVERY="${FASTWAM_FULL_SAVE_EVERY:-500}"

# FASTWAM_INIT:
#   release: use recipe defaults; non-scratch recipes require the public release ckpt.
#   base:    do not resume release ckpt; keep Wan/ActionDiT base initialization.
#   random:  do not resume release ckpt; skip Wan/ActionDiT pretrained loading.
#
# Additional Hydra overrides, space-separated. Example:
#   FASTWAM_EXTRA_OVERRIDES='learning_rate=3e-5 keep_last_n_checkpoints=5'
export FASTWAM_EXTRA_OVERRIDES="${FASTWAM_EXTRA_OVERRIDES:-}"
