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

# 官方镜像的系统 Python 由 Debian/Ubuntu 和镜像共同管理；直接升级 wheel
# 可能触发 "uninstall-no-record-file"。这里显式允许 pip 往系统环境补包，
# 但不升级/替换平台默认 torch、torchvision、CUDA、NCCL。
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

export LEROBOT_CREATE_CONDA=0
export LEROBOT_SKIP_TORCH_INSTALL=1
export LEROBOT_SKIP_PIP_BOOTSTRAP=1
export LEROBOT_INSTALL_NO_DEPS=1
export LEROBOT_FORCE_OPENCV_HEADLESS=1
export LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-training,pusht,smolvla,pi,fastwam}"
export LEROBOT_SOURCE_DIR="${LEROBOT_SOURCE_DIR:-$REPO_ROOT/upstreams/lerobot}"
# diffusers 的 Wan 模块在部分版本里会触发 logger 未定义 bug。
# 当前百舸系统 Python 环境已验证下面的默认 pip spec 可修复；如平台需要固定版本可覆盖：
#   LEROBOT_DIFFUSERS_SPEC='diffusers==0.xx.y' bash scripts/cluster/baige_prepare_lerobot_system_python.sh
export LEROBOT_DIFFUSERS_SPEC="${LEROBOT_DIFFUSERS_SPEC:-diffusers>=0.36.1}"

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

python -m pip install --break-system-packages --no-cache-dir -U "$LEROBOT_DIFFUSERS_SPEC"

python - <<'PY'
from __future__ import annotations

import importlib
from pathlib import Path

import diffusers


def can_import_wan() -> bool:
    try:
        from diffusers import AutoencoderKLWan  # noqa: F401
        return True
    except Exception as exc:
        print(f"diffusers AutoencoderKLWan import failed before patch: {type(exc).__name__}: {exc}")
        return False


def patch_missing_logger(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if "logger" not in text or "logger = logging.get_logger(__name__)" in text:
        return False

    if "from ...utils import logging" in text:
        marker = "from ...utils import logging\n"
        replacement = marker + "\nlogger = logging.get_logger(__name__)\n"
    elif "from diffusers.utils import logging" in text:
        marker = "from diffusers.utils import logging\n"
        replacement = marker + "\nlogger = logging.get_logger(__name__)\n"
    else:
        # autoencoder_kl_wan.py lives in diffusers/models/autoencoders, so
        # three-dot relative import reaches diffusers.utils.
        marker = "import torch\n"
        replacement = marker + "from ...utils import logging\n\nlogger = logging.get_logger(__name__)\n"

    if marker not in text:
        return False
    path.write_text(text.replace(marker, replacement, 1), encoding="utf-8")
    print(f"Patched missing diffusers logger in {path}")
    return True


if not can_import_wan():
    root = Path(diffusers.__file__).resolve().parent
    candidates = [
        root / "models" / "autoencoders" / "autoencoder_kl_wan.py",
        root / "quantizers" / "pipe_quant_config.py",
        root / "quantizers" / "torchao_quantizer.py",
    ]
    patched = False
    for candidate in candidates:
        patched = patch_missing_logger(candidate) or patched
    importlib.invalidate_caches()
    if not patched:
        print("WARNING: no diffusers logger patch was applied; import validation will show the original error.")
PY

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
    "diffusers",
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
from diffusers import AutoencoderKLWan
from datasets.packaged_modules.parquet.parquet import ParquetConfig

print("av_has_option", hasattr(av, "option"))
print("diffusers_autoencoder_kl_wan", AutoencoderKLWan)
print("parquet_config_supports_filters", "filters" in inspect.signature(ParquetConfig).parameters)
PY

echo "BAIGE_LEROBOT_SYSTEM_PYTHON_READY"
