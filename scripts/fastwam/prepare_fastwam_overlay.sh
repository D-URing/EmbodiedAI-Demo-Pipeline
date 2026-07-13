#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/fastwam/realrobot_train_eval.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: FastWAM config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

FASTWAM_INSTALL="${FASTWAM_INSTALL:-0}"
FASTWAM_CREATE_CONDA="${FASTWAM_CREATE_CONDA:-0}"
FASTWAM_CONDA_ENV="${FASTWAM_CONDA_ENV:-fastwam}"
FASTWAM_TORCH_INDEX_URL="${FASTWAM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

command -v git >/dev/null || { echo "ERROR: git is required." >&2; exit 2; }
command -v rsync >/dev/null || { echo "ERROR: rsync is required." >&2; exit 2; }

mkdir -p "$FASTWAM_CACHE_ROOT"

if [[ ! -d "$FASTWAM_WORKDIR/.git" ]]; then
  git clone "$FASTWAM_OFFICIAL_REPO" "$FASTWAM_WORKDIR"
fi
git -C "$FASTWAM_WORKDIR" fetch origin "$FASTWAM_OFFICIAL_REF"
if [[ -n "$(git -C "$FASTWAM_WORKDIR" status --porcelain)" ]]; then
  if [[ "${FASTWAM_RESET_WORKDIR}" == "1" ]]; then
    git -C "$FASTWAM_WORKDIR" reset --hard
    git -C "$FASTWAM_WORKDIR" clean -fd
  else
    echo "ERROR: FASTWAM_WORKDIR has local changes: $FASTWAM_WORKDIR" >&2
    echo "This is expected after an overlay. Use FASTWAM_RESET_WORKDIR=1 to rebuild this generated workspace, or choose a new FASTWAM_WORKDIR." >&2
    exit 2
  fi
fi
git -C "$FASTWAM_WORKDIR" checkout --detach "$FASTWAM_OFFICIAL_REF"

if [[ ! -d "$FASTWAM_OVERLAY_DIR/.git" ]]; then
  git clone "$FASTWAM_OVERLAY_REPO" "$FASTWAM_OVERLAY_DIR"
fi
git -C "$FASTWAM_OVERLAY_DIR" fetch origin "$FASTWAM_OVERLAY_REF"
git -C "$FASTWAM_OVERLAY_DIR" checkout --detach "$FASTWAM_OVERLAY_REF"

rsync -a \
  --exclude ".git/" \
  --exclude "runs/" \
  --exclude "data/" \
  --exclude "checkpoints/" \
  --exclude "evaluate_results/" \
  "$FASTWAM_OVERLAY_DIR"/ "$FASTWAM_WORKDIR"/

echo "FastWAM overlay prepared."
echo "Official FastWAM: $FASTWAM_OFFICIAL_REF -> $FASTWAM_WORKDIR"
echo "Private overlay:   $FASTWAM_OVERLAY_REF -> $FASTWAM_OVERLAY_DIR"

if [[ "$FASTWAM_INSTALL" != "1" ]]; then
  echo "FASTWAM_INSTALL=0, skipped Python/CUDA package installation."
  echo "To install in an active CUDA environment: FASTWAM_INSTALL=1 bash $0 $CONFIG_PATH"
  exit 0
fi

if [[ "$FASTWAM_CREATE_CONDA" == "1" ]]; then
  command -v conda >/dev/null || {
    echo "ERROR: FASTWAM_CREATE_CONDA=1 but conda is not available." >&2
    exit 2
  }
  conda create -y -n "$FASTWAM_CONDA_ENV" python=3.10
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$FASTWAM_CONDA_ENV"
fi

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"ERROR: FastWAM environment should use Python 3.10, got {sys.version.split()[0]}")
print(f"Python OK: {sys.version.split()[0]}")
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url "$FASTWAM_TORCH_INDEX_URL"
python -m pip install -e "$FASTWAM_WORKDIR"

python - <<'PY'
import importlib.metadata
import torch
print(f"torch={torch.__version__}")
try:
    print(f"fastwam={importlib.metadata.version('fastwam')}")
except importlib.metadata.PackageNotFoundError:
    print("fastwam=editable-source")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

echo "FastWAM CUDA environment is ready."
