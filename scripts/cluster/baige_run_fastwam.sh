#!/usr/bin/env bash
# 百舸 PyTorchJob 的 FastWAM 测速入口。
#
# 在 PyTorchJob command 中使用：
#   bash scripts/cluster/baige_run_fastwam.sh
#
# 平台会在 master/worker 同时执行本脚本；本脚本不 ssh、不手动拉 worker。
set -euo pipefail

REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# Pro6000/sm120 当前官方口径：无 RDMA，走 PCIe + TCP。
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH="${FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH:-1}"

python experiments/custom/fastwam_baige_pytorchjob_probe/run.py
