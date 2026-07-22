#!/usr/bin/env bash
# 百舸 Pro6000/sm120 环境准备脚本。
#
# 使用场景：
#   官方镜像已经提供正确的系统 Python / PyTorch / CUDA / NCCL；
#   我们只安装 LeRobot 训练所需的上层 Python 依赖；
#   严禁让 pip 覆盖 torch / torchvision。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "BAIGE_LEROBOT_SYSTEM_PYTHON_PREPARE repo=$REPO_ROOT"

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

export LEROBOT_CREATE_CONDA=0
export LEROBOT_SKIP_TORCH_INSTALL=1
export LEROBOT_INSTALL_NO_DEPS=1
export LEROBOT_FORCE_OPENCV_HEADLESS=1
export LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-training,pusht,smolvla,pi,fastwam}"
export LEROBOT_SOURCE_DIR="${LEROBOT_SOURCE_DIR:-$REPO_ROOT/upstreams/lerobot}"

bash scripts/lerobot/install_lerobot_cluster.sh

# install_lerobot_cluster.sh 会根据 LeRobot pyproject 动态安装 extras。
# 下面这组是当前 pi05/SO100 训练路径已经踩到或常见会踩到的运行时依赖，
# 用于抹平不同官方镜像里的系统包差异；仍然不安装 torch/torchvision。
python -m pip install --upgrade --no-cache-dir \
  "termcolor" \
  "gymnasium>=0.29,<1.3" \
  "draccus==0.10.0" \
  "numpy>=2.0,<2.3" \
  "packaging>=24.2,<26" \
  "setuptools>=71,<81" \
  "datasets>=4,<5" \
  "pyarrow>=18,<22" \
  "av>=14,<16" \
  "accelerate" \
  "transformers<5" \
  "safetensors" \
  "huggingface_hub" \
  "einops" \
  "scipy>=1.14,<1.16" \
  "scikit-learn>=1.5,<1.8" \
  "opencv-python-headless>=4.9,<4.14" \
  "imageio" \
  "tqdm" \
  "wandb" \
  "rich"

python - <<'PY'
import importlib
import inspect

mods = [
    "termcolor",
    "gymnasium",
    "draccus",
    "numpy",
    "packaging",
    "setuptools",
    "datasets",
    "pyarrow",
    "av",
    "accelerate",
    "transformers",
    "safetensors",
    "huggingface_hub",
    "einops",
    "scipy",
    "sklearn",
    "cv2",
    "imageio",
    "tqdm",
    "wandb",
    "lerobot",
    "lerobot.scripts.lerobot_train",
]
for name in mods:
    mod = importlib.import_module(name)
    print(name, "OK", getattr(mod, "__version__", ""))

import av
from datasets.packaged_modules.parquet.parquet import ParquetConfig

print("av_has_option", hasattr(av, "option"))
print("parquet_config_supports_filters", "filters" in inspect.signature(ParquetConfig).parameters)
PY

echo "BAIGE_LEROBOT_SYSTEM_PYTHON_READY"
