#!/usr/bin/env bash
#SBATCH --job-name=lerobot-pusht-act-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runs/lerobot/slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${LEROBOT_CONDA_BASE:-}" ]]; then
  # shellcheck disable=SC1091
  source "$LEROBOT_CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${LEROBOT_CONDA_ENV:-lerobot}"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${LEROBOT_CONDA_ENV:-lerobot}"
fi

bash scripts/lerobot/run_pusht_act_gpu_smoke.sh "${LEROBOT_CONFIG:-configs/lerobot/pusht_act_gpu_smoke.sh}"
