#!/usr/bin/env bash
# 安装 LeRobot 集群环境。
#
# 只应在有网络的管理节点/登录节点运行；计算节点如果不能联网，只激活已经安装好的共享环境。
# 默认固定 LeRobot commit，安装 training/pusht/smolvla/pi/fastwam extras。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LEROBOT_REF="${LEROBOT_REF:-e40b58a8dfa9e7b86918c374791599d070518d11}"
LEROBOT_SOURCE_DIR="${LEROBOT_SOURCE_DIR:-$REPO_ROOT/upstreams/lerobot}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
LEROBOT_TORCH_SPEC="${LEROBOT_TORCH_SPEC:-torch}"
LEROBOT_TORCHVISION_SPEC="${LEROBOT_TORCHVISION_SPEC:-torchvision}"
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-training,pusht,smolvla,pi,fastwam}"
LEROBOT_INSTALL_NO_DEPS="${LEROBOT_INSTALL_NO_DEPS:-0}"
LEROBOT_SKIP_TORCH_INSTALL="${LEROBOT_SKIP_TORCH_INSTALL:-0}"
LEROBOT_FORCE_OPENCV_HEADLESS="${LEROBOT_FORCE_OPENCV_HEADLESS:-1}"
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
if [[ "$LEROBOT_SKIP_TORCH_INSTALL" == "1" ]]; then
  echo "Skip torch/torchvision install; using platform-provided PyTorch."
else
  python -m pip install --index-url "$TORCH_INDEX_URL" "$LEROBOT_TORCH_SPEC" "$LEROBOT_TORCHVISION_SPEC"
fi

mkdir -p "$(dirname "$LEROBOT_SOURCE_DIR")"
if [[ ! -d "$LEROBOT_SOURCE_DIR/.git" ]]; then
  git clone https://github.com/huggingface/lerobot.git "$LEROBOT_SOURCE_DIR"
fi
git -C "$LEROBOT_SOURCE_DIR" fetch --depth 1 origin "$LEROBOT_REF"
git -C "$LEROBOT_SOURCE_DIR" checkout --detach "$LEROBOT_REF"

if [[ "$LEROBOT_INSTALL_NO_DEPS" == "1" ]]; then
  REQUIREMENTS_FILE="$(mktemp)"
  python - "$LEROBOT_SOURCE_DIR" "$LEROBOT_EXTRAS" "$REQUIREMENTS_FILE" <<'PY'
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

source_dir = Path(sys.argv[1])
extras = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
output_path = Path(sys.argv[3])

pyproject = tomllib.loads((source_dir / "pyproject.toml").read_text(encoding="utf-8"))
requirements: list[str] = list(pyproject.get("project", {}).get("dependencies", []))
optional = pyproject.get("project", {}).get("optional-dependencies", {})
for extra in extras:
    requirements.extend(optional.get(extra, []))

skip_names = {"torch", "torchvision", "lerobot"}
seen: set[str] = set()
filtered: list[str] = []


def parse_name_and_extras(requirement: str) -> tuple[str, list[str]]:
    requirement = requirement.strip()
    raw_name = re.split(r"[<>=!~;]", requirement, maxsplit=1)[0].strip()
    if "[" not in raw_name:
        return raw_name.lower().replace("_", "-"), []
    name, extras_part = raw_name.split("[", 1)
    extras_value = extras_part.rstrip("]")
    return name.lower().replace("_", "-"), [item.strip() for item in extras_value.split(",") if item.strip()]


def add_requirement(requirement: str) -> None:
    # Good enough for PEP 508 requirement strings used by LeRobot; avoids pulling
    # LeRobot's torch/torchvision upper bounds on clusters that need a newer CUDA wheel.
    requirement = requirement.strip()
    name, nested_extras = parse_name_and_extras(requirement)
    if name == "lerobot" and nested_extras:
        print(f"Expand LeRobot self-referential extra: {requirement}")
        for nested_extra in nested_extras:
            for nested_requirement in optional.get(nested_extra, []):
                add_requirement(nested_requirement)
        return
    if name in skip_names:
        print(f"Skip dependency managed by cluster install script: {requirement}")
        return
    if requirement not in seen:
        seen.add(requirement)
        filtered.append(requirement)


for requirement in requirements:
    add_requirement(requirement)

output_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
print(f"Wrote {len(filtered)} non-torch LeRobot requirements to {output_path}")
PY
  python -m pip install -r "$REQUIREMENTS_FILE"
  rm -f "$REQUIREMENTS_FILE"
  python -m pip install --no-deps -e "${LEROBOT_SOURCE_DIR}[${LEROBOT_EXTRAS}]"
else
  python -m pip install -e "${LEROBOT_SOURCE_DIR}[${LEROBOT_EXTRAS}]"
fi

if [[ "$LEROBOT_FORCE_OPENCV_HEADLESS" == "1" ]]; then
  # Cluster nodes often do not provide libGL.so.1. gym-pusht may pull
  # opencv-python, but LeRobot only needs cv2 APIs; headless avoids the system
  # GUI/GL runtime dependency.
  python -m pip uninstall -y opencv-python >/dev/null 2>&1 || true
  python -m pip install "opencv-python-headless<4.14.0,>=4.9.0"
fi

python - <<'PY'
import importlib.metadata
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"torch_arch_list={torch.cuda.get_arch_list() if torch.cuda.is_available() else []}")
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
