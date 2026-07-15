#!/usr/bin/env bash
#SBATCH --job-name=lerobot-smolvla-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=runs/lerobot/slurm-smolvla-8gpu-%j.out

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

export LEROBOT_NUM_PROCESSES="${LEROBOT_NUM_PROCESSES:-8}"
export LEROBOT_NUM_MACHINES="${LEROBOT_NUM_MACHINES:-1}"
export LEROBOT_MACHINE_RANK="${LEROBOT_MACHINE_RANK:-0}"
export LEROBOT_MAIN_PROCESS_IP="${LEROBOT_MAIN_PROCESS_IP:-127.0.0.1}"
export LEROBOT_MAIN_PROCESS_PORT="${LEROBOT_MAIN_PROCESS_PORT:-29501}"

bash scripts/lerobot/run_train_accelerate.sh \
  "${LEROBOT_CONFIG:-configs/lerobot/train/svla_so100_smolvla_8gpu_long.sh}"
