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
FASTWAM_SOURCE_MODE="${FASTWAM_SOURCE_MODE:-sync}"
CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_CHANNEL_ARGS="${CONDA_CHANNEL_ARGS:---override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge}"

command -v git >/dev/null || { echo "ERROR: git is required." >&2; exit 2; }
if [[ "$FASTWAM_SOURCE_MODE" == "sync" ]]; then
  command -v rsync >/dev/null || { echo "ERROR: rsync is required." >&2; exit 2; }
fi

mkdir -p "$FASTWAM_CACHE_ROOT"

case "$FASTWAM_SOURCE_MODE" in
  sync)
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
    ;;
  reuse)
    if [[ ! -f "$FASTWAM_WORKDIR/scripts/train_zero1.sh" ]]; then
      echo "ERROR: FASTWAM_SOURCE_MODE=reuse but runnable FastWAM tree is missing:" >&2
      echo "  $FASTWAM_WORKDIR/scripts/train_zero1.sh" >&2
      echo "Prepare it on a networked/login node first:" >&2
      echo "  FASTWAM_SOURCE_MODE=sync bash scripts/fastwam/prepare_fastwam_overlay.sh" >&2
      exit 2
    fi
    echo "FastWAM overlay reused: $FASTWAM_WORKDIR"
    ;;
  *)
    echo "ERROR: FASTWAM_SOURCE_MODE must be sync|reuse, got ${FASTWAM_SOURCE_MODE}" >&2
    exit 2
    ;;
esac

if [[ "$FASTWAM_INSTALL" != "1" ]]; then
  if [[ "$FASTWAM_SOURCE_MODE" == "sync" ]]; then
    echo "FASTWAM_INSTALL=0, skipped Python/CUDA package installation."
    echo "To install in an active CUDA environment: FASTWAM_SOURCE_MODE=reuse FASTWAM_INSTALL=1 bash $0 $CONFIG_PATH"
  else
    echo "FASTWAM_INSTALL=0, skipped Python/CUDA package installation."
  fi
  exit 0
fi

if [[ "$FASTWAM_CREATE_CONDA" == "1" ]]; then
  command -v "$CONDA_EXE" >/dev/null || {
    echo "ERROR: FASTWAM_CREATE_CONDA=1 but CONDA_EXE is not available: $CONDA_EXE" >&2
    exit 2
  }
  if "$CONDA_EXE" env list | awk '{print $1}' | grep -Fxq "$FASTWAM_CONDA_ENV"; then
    echo "Conda env already exists, reusing: $FASTWAM_CONDA_ENV"
  else
    # shellcheck disable=SC2086
    "$CONDA_EXE" create -y -n "$FASTWAM_CONDA_ENV" $CONDA_CHANNEL_ARGS python=3.10 pip
  fi
  # shellcheck disable=SC1091
  source "$("$CONDA_EXE" info --base)/etc/profile.d/conda.sh"
  conda activate "$FASTWAM_CONDA_ENV"
fi

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"ERROR: FastWAM environment should use Python 3.10, got {sys.version.split()[0]}")
print(f"Python OK: {sys.version.split()[0]}")
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install PyYAML
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
