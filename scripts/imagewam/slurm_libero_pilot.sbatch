#!/usr/bin/env bash
#SBATCH --job-name=imagewam-libero-pilot
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/imagewam-libero-pilot-%j.out
#SBATCH --error=runs/slurm/imagewam-libero-pilot-%j.err

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
  conda activate "${IMAGEWAM_CONDA_ENV:-imagewam}"
fi

export IMAGEWAM_MODE="${IMAGEWAM_MODE:-pilot}"
export IMAGEWAM_VARIANT="${IMAGEWAM_VARIANT:-flux2_4b}"
export IMAGEWAM_TASK_TYPE="${IMAGEWAM_TASK_TYPE:-libero}"
export IMAGEWAM_PRECOMPUTE_CACHE="${IMAGEWAM_PRECOMPUTE_CACHE:-true}"

bash scripts/imagewam/run_train_eval.sh configs/imagewam/libero_train_eval.sh
