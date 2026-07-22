#!/usr/bin/env bash
# 百舸 Pro6000/sm120 FastWAM 环境准备脚本。
#
# 使用平台默认 Python / PyTorch / CUDA / NCCL，只准备 FastWAM overlay
# 和上层 Python 依赖；不覆盖 torch / torchvision。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "BAIGE_FASTWAM_SYSTEM_PYTHON_PREPARE repo=$REPO_ROOT"

export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"

python - <<'PY'
import sys
import torch

print("python", sys.executable)
print("python_version", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
    print("cuda_capability", torch.cuda.get_device_capability(0))
    print("torch_arch_list", torch.cuda.get_arch_list())
PY

export FASTWAM_SOURCE_MODE="${FASTWAM_SOURCE_MODE:-reuse}"
export FASTWAM_INSTALL=1
export FASTWAM_CREATE_CONDA=0
export FASTWAM_SKIP_TORCH_INSTALL=1
export FASTWAM_SKIP_PIP_BOOTSTRAP=1
export FASTWAM_INSTALL_NVCC=0
export FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH=1

bash scripts/fastwam/prepare_fastwam_overlay.sh

python - <<'PY'
import importlib
import torch

mods = [
    "yaml",
    "accelerate",
    "diffusers",
    "transformers",
    "fastwam",
]
for name in mods:
    mod = importlib.import_module(name)
    print(name, "OK", getattr(mod, "__version__", ""))

print("torch", torch.__version__, torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
PY

echo "BAIGE_FASTWAM_SYSTEM_PYTHON_READY"
