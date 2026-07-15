#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LEROBOT_REF="${LEROBOT_REF:-e40b58a8dfa9e7b86918c374791599d070518d11}"
LEROBOT_SOURCE_DIR="${LEROBOT_SOURCE_DIR:-$REPO_ROOT/upstreams/lerobot}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-training,pusht,smolvla,pi}"
CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_CHANNEL_ARGS="${CONDA_CHANNEL_ARGS:---override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge}"

if [[ "${LEROBOT_CREATE_CONDA:-0}" == "1" ]]; then
  if ! command -v "$CONDA_EXE" >/dev/null 2>&1; then
    echo "ERROR: LEROBOT_CREATE_CONDA=1 but CONDA_EXE is not available: $CONDA_EXE" >&2
    exit 2
  fi
  LEROBOT_CONDA_ENV="${LEROBOT_CONDA_ENV:-lerobot}"
  # shellcheck disable=SC2086
  "$CONDA_EXE" create -y -n "$LEROBOT_CONDA_ENV" $CONDA_CHANNEL_ARGS python=3.12 pip
  # shellcheck disable=SC1091
  source "$("$CONDA_EXE" info --base)/etc/profile.d/conda.sh"
  conda activate "$LEROBOT_CONDA_ENV"
fi

python - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(
        f"ERROR: LeRobot requires Python >=3.12 for this setup; got {sys.version.split()[0]}"
    )
print(f"Python OK: {sys.version.split()[0]}")
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision

mkdir -p "$(dirname "$LEROBOT_SOURCE_DIR")"
if [[ ! -d "$LEROBOT_SOURCE_DIR/.git" ]]; then
  git clone https://github.com/huggingface/lerobot.git "$LEROBOT_SOURCE_DIR"
fi
git -C "$LEROBOT_SOURCE_DIR" fetch --depth 1 origin "$LEROBOT_REF"
git -C "$LEROBOT_SOURCE_DIR" checkout --detach "$LEROBOT_REF"

python -m pip install -e "${LEROBOT_SOURCE_DIR}[${LEROBOT_EXTRAS}]"

python - <<'PY'
import importlib.metadata
import torch
print(f"torch={torch.__version__}")
print(f"lerobot={importlib.metadata.version('lerobot')}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

command -v lerobot-train >/dev/null
lerobot-train --help >/dev/null

echo "LeRobot cluster environment is ready."
echo "Source checkout: $LEROBOT_SOURCE_DIR"
echo "Pinned commit: $LEROBOT_REF"
