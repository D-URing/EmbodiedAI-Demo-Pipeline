#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EMBODIED_RUN_ROOT="${EMBODIED_RUN_ROOT:-$REPO_ROOT/runs}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

HF_CLI_BIN="${HF_CLI_BIN:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"
WAN_REPO_ID="${WAN_REPO_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
TOKENIZER_REPO_ID="${TOKENIZER_REPO_ID:-google/umt5-xxl}"
MANIFEST_PATH="${EMBODIED_RUN_ROOT}/artifact_manifests/lerobot_fastwam_base_cache_manifest.json"

if [[ -n "$HF_CLI_BIN" ]]; then
  if ! command -v "$HF_CLI_BIN" >/dev/null 2>&1; then
    echo "HF_CLI_BIN=$HF_CLI_BIN is not executable or not on PATH." >&2
    exit 127
  fi
  HF_DOWNLOAD_CMD=("$HF_CLI_BIN" download)
elif command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD_CMD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD_CMD=(huggingface-cli download)
else
  echo "The Hugging Face CLI is required: install huggingface_hub or set HF_CLI_BIN." >&2
  exit 127
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN=$PYTHON_BIN is not executable or not on PATH." >&2
  exit 127
fi

mkdir -p "$HUGGINGFACE_HUB_CACHE" "$EMBODIED_RUN_ROOT/artifact_manifests"

echo "[artifact] family=lerobot_fastwam_base_cache"
echo "[artifact] wan_repo=$WAN_REPO_ID"
echo "[artifact] tokenizer_repo=$TOKENIZER_REPO_ID"
echo "[artifact] hf_home=$HF_HOME"
echo "[artifact] hf_hub_cache=$HUGGINGFACE_HUB_CACHE"
echo "[artifact] hf_endpoint=$HF_ENDPOINT"
echo "[artifact] hf_hub_disable_xet=$HF_HUB_DISABLE_XET"
echo "[artifact] hf_cli=${HF_DOWNLOAD_CMD[*]}"
echo "[artifact] manifest=$MANIFEST_PATH"

HF_HOME="$HF_HOME" \
HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
HF_ENDPOINT="$HF_ENDPOINT" \
HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
  "${HF_DOWNLOAD_CMD[@]}" "$WAN_REPO_ID" \
    --include "vae/*" \
    --include "text_encoder/*" \
    --cache-dir "$HUGGINGFACE_HUB_CACHE" \
    --max-workers "$HF_MAX_WORKERS"

HF_HOME="$HF_HOME" \
HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
HF_ENDPOINT="$HF_ENDPOINT" \
HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
  "${HF_DOWNLOAD_CMD[@]}" "$TOKENIZER_REPO_ID" \
    --include "tokenizer*" \
    --include "spiece.model" \
    --include "special_tokens_map.json" \
    --include "config.json" \
    --cache-dir "$HUGGINGFACE_HUB_CACHE" \
    --max-workers "$HF_MAX_WORKERS"

WAN_REPO_ID="$WAN_REPO_ID" \
TOKENIZER_REPO_ID="$TOKENIZER_REPO_ID" \
HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
"$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

hub = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
repos = [
    os.environ["WAN_REPO_ID"],
    os.environ["TOKENIZER_REPO_ID"],
]


def cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def du(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


entries = []
for repo_id in repos:
    path = hub / cache_name(repo_id)
    entries.append(
        {
            "repo_id": repo_id,
            "cache_path": str(path),
            "exists": path.exists(),
            "bytes": du(path),
        }
    )

manifest = {
    "artifact_family": "lerobot_fastwam_base_cache",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "hf_hub_cache": str(hub),
    "entries": entries,
    "notes": [
        "Required by LeRobot FastWAM policy: frozen Wan2.2 VAE/text_encoder and UMT5 tokenizer.",
        "Keep this in HF cache layout because upstream FastWAM loads by repo id.",
    ],
}

path = Path(sys.argv[1])
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[manifest] {path}")
for entry in entries:
    print(entry)
PY
