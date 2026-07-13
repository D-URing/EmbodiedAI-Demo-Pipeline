#!/usr/bin/env bash
#SBATCH --job-name=fastwam-realrobot-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=runs/fastwam/slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${FASTWAM_CONDA_ENV:-fastwam}"
fi

export FASTWAM_MODE="${FASTWAM_MODE:-pilot}"
export FASTWAM_RECIPE="${FASTWAM_RECIPE:-joint_base}"
export FASTWAM_GPUS_PER_NODE="${FASTWAM_GPUS_PER_NODE:-8}"

bash scripts/fastwam/run_realrobot_train_eval.sh "${FASTWAM_CONFIG:-configs/fastwam/realrobot_train_eval.sh}"
