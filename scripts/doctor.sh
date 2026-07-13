#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV="${VENV:-.venv}"

if [[ "${VENV}" != /* ]]; then
  VENV="${ROOT_DIR}/${VENV}"
fi

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} not found; read docs/ENVIRONMENT.md"

"${PYTHON_BIN}" -c '
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ is required, found {sys.version.split()[0]}")
print(f"system_python={sys.version.split()[0]}")
'

[[ -x "${VENV}/bin/python" ]] || fail "virtual environment missing; run make setup"

"${VENV}/bin/python" -c '
import platform
import sys
print(f"venv_python={sys.version.split()[0]}")
print(f"platform={platform.system()}-{platform.machine()}")
'

"${VENV}/bin/python" -m pip check
"${VENV}/bin/embodied-demo" --version
"${VENV}/bin/embodied-demo" validate \
  --config "${ROOT_DIR}/configs/runs/tabletop_sorting_mock.yaml"
"${VENV}/bin/embodied-demo" validate \
  --config "${ROOT_DIR}/configs/runs/towel_folding_mock.yaml"

if command -v gh >/dev/null 2>&1; then
  printf 'gh=%s\n' "$(gh --version | head -1)"
else
  printf 'INFO: gh is optional for runtime and required only for GitHub publishing.\n'
fi

if [[ -n "${HTTPS_PROXY:-${https_proxy:-}}" ]]; then
  printf 'shell_proxy=configured\n'
else
  printf 'INFO: shell proxy is not configured; this is valid on direct-connect networks.\n'
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia_smi=available\n'
else
  printf 'INFO: NVIDIA runtime not detected; this is expected for core/mock development.\n'
fi

printf 'environment_status=OK\n'
