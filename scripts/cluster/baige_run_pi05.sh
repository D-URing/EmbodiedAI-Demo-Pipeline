#!/usr/bin/env bash
# 百舸 PyTorchJob 的 pi05 正式入口。
#
# 在 PyTorchJob command 中使用：
#   bash scripts/cluster/baige_run_pi05.sh
#
# 平台会在 master/worker 同时执行本脚本；本脚本不 ssh、不手动拉 worker。
set -euo pipefail

REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# Pro6000/sm120 当前官方口径：无 RDMA，走 PCIe + TCP。
# 不指定 NCCL_SOCKET_IFNAME，让平台/系统默认路由决定 socket 网卡。
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

python experiments/lerobot/pi05_baige_pytorchjob_probe/run.py
