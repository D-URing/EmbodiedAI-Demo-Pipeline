#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/imagewam/libero_train_eval.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

mkdir -p "$(dirname "$IMAGEWAM_WORKDIR")" "$EMBODIED_RUN_ROOT/artifact_manifests"

if [[ ! -d "$IMAGEWAM_WORKDIR/.git" ]]; then
  git clone "$IMAGEWAM_OFFICIAL_REPO" "$IMAGEWAM_WORKDIR"
else
  git -C "$IMAGEWAM_WORKDIR" fetch origin --prune
fi

if git -C "$IMAGEWAM_WORKDIR" rev-parse --verify "$IMAGEWAM_OFFICIAL_REF" >/dev/null 2>&1; then
  git -C "$IMAGEWAM_WORKDIR" checkout "$IMAGEWAM_OFFICIAL_REF"
else
  git -C "$IMAGEWAM_WORKDIR" checkout -B "local-${IMAGEWAM_OFFICIAL_REF}" "origin/${IMAGEWAM_OFFICIAL_REF}"
fi

revision="$(git -C "$IMAGEWAM_WORKDIR" rev-parse HEAD)"
cat > "$EMBODIED_RUN_ROOT/artifact_manifests/imagewam_upstream_manifest.json" <<JSON
{
  "family": "imagewam",
  "kind": "upstream_source",
  "repo": "$IMAGEWAM_OFFICIAL_REPO",
  "ref": "$IMAGEWAM_OFFICIAL_REF",
  "revision": "$revision",
  "local_dir": "$IMAGEWAM_WORKDIR"
}
JSON

echo "ImageWAM upstream ready: $IMAGEWAM_WORKDIR @ $revision"
