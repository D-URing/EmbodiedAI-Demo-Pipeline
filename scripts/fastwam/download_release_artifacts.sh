#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EMBODIED_MODEL_ROOT="${EMBODIED_MODEL_ROOT:-$HOME/.cache/embodied-demo/models}"
EMBODIED_RUN_ROOT="${EMBODIED_RUN_ROOT:-$REPO_ROOT/runs}"

FASTWAM_RELEASE_REPO_ID="${FASTWAM_RELEASE_REPO_ID:-yuanty/fastwam}"
FASTWAM_RELEASE_LOCAL_DIR="${FASTWAM_RELEASE_LOCAL_DIR:-$EMBODIED_MODEL_ROOT/fastwam_release}"
FASTWAM_RELEASE_FILES="${FASTWAM_RELEASE_FILES:-libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required. Install with: python3 -m pip install -U huggingface_hub hf_transfer" >&2
  exit 127
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN is required to write artifact manifests. Set PYTHON_BIN=/path/to/python if needed." >&2
  exit 127
fi

mkdir -p "$FASTWAM_RELEASE_LOCAL_DIR" "$EMBODIED_RUN_ROOT/artifact_manifests"

read -r -a release_files <<< "$FASTWAM_RELEASE_FILES"
if [[ "${#release_files[@]}" -eq 0 ]]; then
  echo "FASTWAM_RELEASE_FILES must contain at least one filename." >&2
  exit 2
fi

echo "[download] FastWAM release: $FASTWAM_RELEASE_REPO_ID -> $FASTWAM_RELEASE_LOCAL_DIR"
HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
  huggingface-cli download "$FASTWAM_RELEASE_REPO_ID" \
    "${release_files[@]}" \
    --local-dir "$FASTWAM_RELEASE_LOCAL_DIR"

MANIFEST_PATH="$EMBODIED_RUN_ROOT/artifact_manifests/fastwam_release_artifacts_manifest.json"
FASTWAM_RELEASE_REPO_ID="$FASTWAM_RELEASE_REPO_ID" \
FASTWAM_RELEASE_LOCAL_DIR="$FASTWAM_RELEASE_LOCAL_DIR" \
FASTWAM_RELEASE_FILES="$FASTWAM_RELEASE_FILES" \
"$PYTHON_BIN" - "$MANIFEST_PATH" <<PY
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

files = os.environ["FASTWAM_RELEASE_FILES"].split()
local_dir = os.environ["FASTWAM_RELEASE_LOCAL_DIR"]

manifest = {
    "artifact_family": "fastwam_release",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "repo_id": os.environ["FASTWAM_RELEASE_REPO_ID"],
    "local_dir": local_dir,
    "files": [{"name": name, "path": f"{local_dir.rstrip('/')}/{name}"} for name in files],
    "next_env": {
        "FASTWAM_MODEL_BASE": os.path.dirname(local_dir.rstrip("/")),
        "FASTWAM_RELEASE_CKPT": f"{local_dir.rstrip('/')}/libero_uncond_2cam224.pt",
        "FASTWAM_RELEASE_DATASET_STATS": f"{local_dir.rstrip('/')}/libero_uncond_2cam224_dataset_stats.json",
    },
}

path = sys.argv[1]
with open(path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
    f.write("\\n")
PY

echo "[manifest] $MANIFEST_PATH"
echo
echo "Next FastWAM overlay env:"
echo "  export FASTWAM_MODEL_BASE=\"$(dirname "$FASTWAM_RELEASE_LOCAL_DIR")\""
echo "  export FASTWAM_RELEASE_CKPT=\"$FASTWAM_RELEASE_LOCAL_DIR/libero_uncond_2cam224.pt\""
echo "  export FASTWAM_RELEASE_DATASET_STATS=\"$FASTWAM_RELEASE_LOCAL_DIR/libero_uncond_2cam224_dataset_stats.json\""
