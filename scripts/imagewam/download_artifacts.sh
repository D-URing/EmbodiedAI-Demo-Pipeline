#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/imagewam/libero_train_eval.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

mkdir -p "$IMAGEWAM_POLICY_LOCAL_DIR" "$EMBODIED_RUN_ROOT/artifact_manifests" "$HF_HOME"

export HF_HOME HUGGINGFACE_HUB_CACHE
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

downloaded_by=""
if [[ -n "${HFD_BIN:-}" && -f "$HFD_BIN" ]]; then
  bash "$HFD_BIN" "$IMAGEWAM_POLICY_REPO_ID" \
    --local-dir "$IMAGEWAM_POLICY_LOCAL_DIR" \
    --tool "${HFD_TOOL:-aria2c}" \
    -x "${HFD_THREADS:-10}" \
    -j "${HFD_JOBS:-2}"
  downloaded_by="$HFD_BIN"
elif command -v hfd.sh >/dev/null 2>&1; then
  hfd.sh "$IMAGEWAM_POLICY_REPO_ID" \
    --local-dir "$IMAGEWAM_POLICY_LOCAL_DIR" \
    --tool "${HFD_TOOL:-aria2c}" \
    -x "${HFD_THREADS:-10}" \
    -j "${HFD_JOBS:-2}"
  downloaded_by="hfd.sh"
elif command -v hf >/dev/null 2>&1; then
  hf download "$IMAGEWAM_POLICY_REPO_ID" \
    --local-dir "$IMAGEWAM_POLICY_LOCAL_DIR" \
    --local-dir-use-symlinks False
  downloaded_by="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$IMAGEWAM_POLICY_REPO_ID" \
    --local-dir "$IMAGEWAM_POLICY_LOCAL_DIR" \
    --local-dir-use-symlinks False
  downloaded_by="huggingface-cli"
else
  echo "No Hugging Face downloader found. Install huggingface_hub or set HFD_BIN=/home/scut/hfd.sh" >&2
  exit 127
fi

python_bin="${PYTHON_BIN:-python3}"
"$python_bin" - "$IMAGEWAM_POLICY_REPO_ID" "$IMAGEWAM_POLICY_LOCAL_DIR" "$downloaded_by" "$EMBODIED_RUN_ROOT/artifact_manifests/imagewam_artifacts_manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

repo_id, local_dir, downloaded_by, manifest_path = sys.argv[1:5]
root = Path(local_dir)
files = []
total_bytes = 0
for path in root.rglob("*"):
    if path.is_file():
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        files.append({"path": rel, "bytes": size})
        total_bytes += size

manifest = {
    "family": "imagewam",
    "kind": "model_artifacts",
    "repo_id": repo_id,
    "local_dir": str(root),
    "downloaded_by": downloaded_by,
    "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    "file_count": len(files),
    "total_bytes": total_bytes,
    "files_preview": files[:50],
}
Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY

echo "ImageWAM artifacts ready: $IMAGEWAM_POLICY_LOCAL_DIR"
