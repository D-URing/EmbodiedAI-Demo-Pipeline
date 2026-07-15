#!/usr/bin/env bash
#SBATCH --job-name=fastwam-8node-random
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm/fastwam-8node-random-%j.out
#SBATCH --error=runs/slurm/fastwam-8node-random-%j.err

set -euo pipefail

PROJECT="${PROJECT:-$PWD}"
cd "$PROJECT"
mkdir -p runs/slurm

if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
elif [[ -n "${BASE:-}" && -f "$BASE/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$BASE/miniconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${FASTWAM_CONDA_ENV:-fastwam}"
fi

export FASTWAM_NNODES="${SLURM_NNODES:-8}"
export FASTWAM_GPUS_PER_NODE="${FASTWAM_GPUS_PER_NODE:-8}"
export FASTWAM_MASTER_ADDR="${FASTWAM_MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export FASTWAM_MASTER_PORT="${FASTWAM_MASTER_PORT:-29500}"
export FASTWAM_RUN_ID="${FASTWAM_RUN_ID:-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-slurm}}"
export MASTER_ADDR="$FASTWAM_MASTER_ADDR"
export MASTER_PORT="$FASTWAM_MASTER_PORT"

srun --ntasks="$FASTWAM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -lc 'export FASTWAM_NODE_RANK="${SLURM_NODEID:-0}"; bash experiments/custom/fastwam_realrobot_8node_random/launch.sh'
