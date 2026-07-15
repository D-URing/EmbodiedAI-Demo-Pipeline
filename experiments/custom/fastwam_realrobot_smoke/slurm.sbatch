#!/usr/bin/env bash
#SBATCH --job-name=fastwam-realrobot-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/fastwam-realrobot-%j.out
#SBATCH --error=runs/slurm/fastwam-realrobot-%j.err

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

bash experiments/custom/fastwam_realrobot_smoke/launch.sh
