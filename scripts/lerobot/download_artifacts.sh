#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EMBODIED_DATA_ROOT="${EMBODIED_DATA_ROOT:-$HOME/.cache/embodied-demo/data}"
EMBODIED_MODEL_ROOT="${EMBODIED_MODEL_ROOT:-$HOME/.cache/embodied-demo/models}"
EMBODIED_RUN_ROOT="${EMBODIED_RUN_ROOT:-$REPO_ROOT/runs}"

LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-lerobot/pusht}"
LEROBOT_DATASET_NAME="${LEROBOT_DATASET_NAME:-${LEROBOT_DATASET_REPO_ID#*/}}"
LEROBOT_DATASET_LOCAL_DIR="${LEROBOT_DATASET_LOCAL_DIR:-$EMBODIED_DATA_ROOT/lerobot/$LEROBOT_DATASET_NAME}"

LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-act}"
LEROBOT_POLICY_REPO_ID="${LEROBOT_POLICY_REPO_ID:-}"
LEROBOT_POLICY_NAME="${LEROBOT_POLICY_NAME:-${LEROBOT_POLICY_REPO_ID##*/}}"
LEROBOT_POLICY_LOCAL_DIR="${LEROBOT_POLICY_LOCAL_DIR:-$EMBODIED_MODEL_ROOT/lerobot/$LEROBOT_POLICY_TYPE/${LEROBOT_POLICY_NAME:-manual_checkpoint}}"

DOWNLOAD_LEROBOT_DATASET="${DOWNLOAD_LEROBOT_DATASET:-1}"
DOWNLOAD_LEROBOT_POLICY="${DOWNLOAD_LEROBOT_POLICY:-0}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_CLI_BIN="${HF_CLI_BIN:-}"

if [[ -n "$HF_CLI_BIN" ]]; then
  if ! command -v "$HF_CLI_BIN" >/dev/null 2>&1; then
    echo "HF_CLI_BIN=$HF_CLI_BIN is not executable or not on PATH." >&2
    exit 127
  fi
  HF_DOWNLOAD_CMD=("$HF_CLI_BIN" download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD_CMD=(huggingface-cli download)
elif command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD_CMD=(hf download)
else
  echo "Hugging Face CLI is required. Install with: python3 -m pip install -U huggingface_hub hf_transfer" >&2
  echo "Expected either 'huggingface-cli' or the newer 'hf' command on PATH." >&2
  exit 127
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN is required to write artifact manifests. Set PYTHON_BIN=/path/to/python if needed." >&2
  exit 127
fi

mkdir -p "$LEROBOT_DATASET_LOCAL_DIR" "$EMBODIED_MODEL_ROOT" "$EMBODIED_RUN_ROOT/artifact_manifests"

dataset_downloaded=false
policy_downloaded=false

if [[ "$DOWNLOAD_LEROBOT_DATASET" == "1" ]]; then
  echo "[download] LeRobot dataset: $LEROBOT_DATASET_REPO_ID -> $LEROBOT_DATASET_LOCAL_DIR"
  HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
    "${HF_DOWNLOAD_CMD[@]}" "$LEROBOT_DATASET_REPO_ID" \
      --repo-type dataset \
      --local-dir "$LEROBOT_DATASET_LOCAL_DIR"
  dataset_downloaded=true
else
  echo "[skip] DOWNLOAD_LEROBOT_DATASET=$DOWNLOAD_LEROBOT_DATASET"
fi

if [[ "$DOWNLOAD_LEROBOT_POLICY" == "1" ]]; then
  if [[ -z "$LEROBOT_POLICY_REPO_ID" ]]; then
    echo "LEROBOT_POLICY_REPO_ID is required when DOWNLOAD_LEROBOT_POLICY=1." >&2
    echo "Example: LEROBOT_POLICY_REPO_ID=<org>/<repo> DOWNLOAD_LEROBOT_POLICY=1 make download-lerobot-artifacts" >&2
    exit 2
  fi

  mkdir -p "$LEROBOT_POLICY_LOCAL_DIR"
  echo "[download] LeRobot policy: $LEROBOT_POLICY_REPO_ID -> $LEROBOT_POLICY_LOCAL_DIR"
  HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
    "${HF_DOWNLOAD_CMD[@]}" "$LEROBOT_POLICY_REPO_ID" \
      --local-dir "$LEROBOT_POLICY_LOCAL_DIR"
  policy_downloaded=true
else
  echo "[skip] DOWNLOAD_LEROBOT_POLICY=$DOWNLOAD_LEROBOT_POLICY"
fi

MANIFEST_PATH="$EMBODIED_RUN_ROOT/artifact_manifests/lerobot_artifacts_manifest.json"
LEROBOT_DATASET_REPO_ID="$LEROBOT_DATASET_REPO_ID" \
LEROBOT_DATASET_LOCAL_DIR="$LEROBOT_DATASET_LOCAL_DIR" \
LEROBOT_POLICY_REPO_ID="$LEROBOT_POLICY_REPO_ID" \
LEROBOT_POLICY_TYPE="$LEROBOT_POLICY_TYPE" \
LEROBOT_POLICY_LOCAL_DIR="$LEROBOT_POLICY_LOCAL_DIR" \
dataset_downloaded="$dataset_downloaded" \
policy_downloaded="$policy_downloaded" \
"$PYTHON_BIN" - "$MANIFEST_PATH" <<PY
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

manifest = {
    "artifact_family": "lerobot",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "dataset": {
        "repo_id": os.environ["LEROBOT_DATASET_REPO_ID"],
        "local_dir": os.environ["LEROBOT_DATASET_LOCAL_DIR"],
        "downloaded": os.environ["dataset_downloaded"] == "true",
    },
    "policy": {
        "repo_id": os.environ.get("LEROBOT_POLICY_REPO_ID", ""),
        "policy_type": os.environ["LEROBOT_POLICY_TYPE"],
        "local_dir": os.environ["LEROBOT_POLICY_LOCAL_DIR"],
        "downloaded": os.environ["policy_downloaded"] == "true",
    },
    "next_env": {
        "LEROBOT_DATASET_ROOT": os.environ["LEROBOT_DATASET_LOCAL_DIR"],
        "LEROBOT_POLICY_PATH": os.environ["LEROBOT_POLICY_LOCAL_DIR"] if os.environ["policy_downloaded"] == "true" else "",
    },
}

path = sys.argv[1]
with open(path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
    f.write("\\n")
PY

echo "[manifest] $MANIFEST_PATH"
echo
echo "Next dataset smoke:"
echo "  export LEROBOT_DATASET_ROOT=\"$LEROBOT_DATASET_LOCAL_DIR\""
echo "  make lerobot-data-smoke"
if [[ "$policy_downloaded" == "true" ]]; then
  echo
  echo "Next inference smoke:"
  echo "  export LEROBOT_POLICY_PATH=\"$LEROBOT_POLICY_LOCAL_DIR\""
  echo "  make lerobot-infer-smoke"
fi
