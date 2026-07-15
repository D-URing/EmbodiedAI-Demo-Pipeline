#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EMBODIED_DATA_ROOT="${EMBODIED_DATA_ROOT:-$REPO_ROOT/data}"
EMBODIED_RUN_ROOT="${EMBODIED_RUN_ROOT:-$REPO_ROOT/runs}"

LEROBOT_FASTWAM_LIBERO_ROOT="${LEROBOT_FASTWAM_LIBERO_ROOT:-$EMBODIED_DATA_ROOT/lerobot/libero-fastwam}"
LEROBOT_FASTWAM_LIBERO_V21_ROOT="${LEROBOT_FASTWAM_LIBERO_V21_ROOT:-$LEROBOT_FASTWAM_LIBERO_ROOT/v2.1}"
LEROBOT_FASTWAM_LIBERO_V30_ROOT="${LEROBOT_FASTWAM_LIBERO_V30_ROOT:-$LEROBOT_FASTWAM_LIBERO_ROOT/v3}"
LEROBOT_FASTWAM_LIBERO_SUBSETS="${LEROBOT_FASTWAM_LIBERO_SUBSETS:-libero_10_no_noops_lerobot libero_goal_no_noops_lerobot libero_object_no_noops_lerobot libero_spatial_no_noops_lerobot}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LEROBOT_CONVERT_DATA_FILE_SIZE_MB="${LEROBOT_CONVERT_DATA_FILE_SIZE_MB:-256}"
LEROBOT_CONVERT_VIDEO_FILE_SIZE_MB="${LEROBOT_CONVERT_VIDEO_FILE_SIZE_MB:-512}"
LEROBOT_CONVERT_FORCE="${LEROBOT_CONVERT_FORCE:-0}"
LEROBOT_CONVERT_LOG_DIR="${LEROBOT_CONVERT_LOG_DIR:-$EMBODIED_RUN_ROOT/artifact_manifests/lerobot_fastwam_libero_v3_conversion}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required to copy v2.1 subsets into the v3 conversion workspace." >&2
  exit 127
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN=$PYTHON_BIN is not executable or not on PATH." >&2
  exit 127
fi

mkdir -p "$LEROBOT_FASTWAM_LIBERO_V30_ROOT" "$LEROBOT_CONVERT_LOG_DIR"

read -r -a subsets <<< "$LEROBOT_FASTWAM_LIBERO_SUBSETS"
if [[ "${#subsets[@]}" -eq 0 ]]; then
  echo "LEROBOT_FASTWAM_LIBERO_SUBSETS must contain at least one subset." >&2
  exit 2
fi

echo "[convert] root=$LEROBOT_FASTWAM_LIBERO_ROOT"
echo "[convert] v2.1=$LEROBOT_FASTWAM_LIBERO_V21_ROOT"
echo "[convert] v3=$LEROBOT_FASTWAM_LIBERO_V30_ROOT"
echo "[convert] subsets=${subsets[*]}"
echo "[convert] python=$PYTHON_BIN"
echo "[convert] logs=$LEROBOT_CONVERT_LOG_DIR"

version_of() {
  local dataset_dir="$1"
  if [[ ! -f "$dataset_dir/meta/info.json" ]]; then
    echo "missing"
    return
  fi
  "$PYTHON_BIN" - "$dataset_dir/meta/info.json" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text()).get("codebase_version", "unknown"))
PY
}

for subset in "${subsets[@]}"; do
  src="$LEROBOT_FASTWAM_LIBERO_V21_ROOT/$subset"
  dst="$LEROBOT_FASTWAM_LIBERO_V30_ROOT/$subset"
  log_path="$LEROBOT_CONVERT_LOG_DIR/${subset}.log"

  if [[ ! -d "$src" ]]; then
    echo "[error] missing v2.1 subset: $src" >&2
    exit 1
  fi

  current_version="$(version_of "$dst")"
  if [[ "$current_version" == "v3.0" && "$LEROBOT_CONVERT_FORCE" != "1" ]]; then
    echo "[skip] $subset is already v3.0 at $dst"
    continue
  fi

  if [[ "$LEROBOT_CONVERT_FORCE" == "1" ]]; then
    rm -rf "$dst" "${dst}_old" "${dst}_v30"
  fi

  if [[ ! -f "$dst/meta/info.json" ]]; then
    echo "[copy] $src -> $dst"
    rm -rf "$dst"
    mkdir -p "$dst"
    rsync -a "$src"/ "$dst"/
  fi

  current_version="$(version_of "$dst")"
  if [[ "$current_version" != "v2.1" ]]; then
    echo "[error] $dst has codebase_version=$current_version, expected v2.1 before conversion." >&2
    echo "        Use LEROBOT_CONVERT_FORCE=1 to rebuild this subset from v2.1." >&2
    exit 1
  fi

  echo "[run] convert $subset"
  "$PYTHON_BIN" -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id "local/$subset" \
    --root "$dst" \
    --push-to-hub false \
    --data-file-size-in-mb "$LEROBOT_CONVERT_DATA_FILE_SIZE_MB" \
    --video-file-size-in-mb "$LEROBOT_CONVERT_VIDEO_FILE_SIZE_MB" \
    2>&1 | tee "$log_path"

  converted_version="$(version_of "$dst")"
  if [[ "$converted_version" != "v3.0" ]]; then
    echo "[error] conversion did not produce v3.0 for $dst; got $converted_version" >&2
    exit 1
  fi
  echo "[ok] $subset -> v3.0"
done

"$PYTHON_BIN" - "$LEROBOT_FASTWAM_LIBERO_ROOT" "$LEROBOT_CONVERT_LOG_DIR" "${subsets[@]}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
subsets = sys.argv[3:]


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


def version(path: Path) -> str:
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        return "missing"
    return json.loads(info_path.read_text()).get("codebase_version", "unknown")


manifest = {
    "artifact_family": "lerobot_fastwam_libero_v3_conversion",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "root": str(root),
    "log_dir": str(log_dir),
    "subsets": [
        {
            "name": subset,
            "v21_path": str(root / "v2.1" / subset),
            "v21_exists": (root / "v2.1" / subset).exists(),
            "v21_bytes": du(root / "v2.1" / subset),
            "v30_path": str(root / "v3" / subset),
            "v30_exists": (root / "v3" / subset).exists(),
            "v30_bytes": du(root / "v3" / subset),
            "v30_codebase_version": version(root / "v3" / subset),
            "log_path": str(log_dir / f"{subset}.log"),
        }
        for subset in subsets
    ],
}

out = log_dir.parent / "lerobot_fastwam_libero_v3_conversion_manifest.json"
out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[manifest] {out}")
PY
