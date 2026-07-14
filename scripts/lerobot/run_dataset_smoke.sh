#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/lerobot/native_pusht_act_pipeline.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: LeRobot native config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

RUN_ID="${LEROBOT_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${LEROBOT_RUN_ROOT}/${LEROBOT_RUN_NAME}/${RUN_ID}"
mkdir -p "$RUN_DIR"

ARGS=(
  --repo-id "$LEROBOT_DATASET_REPO_ID"
  --sample-index "$LEROBOT_SAMPLE_INDEX"
  --output-dir "$RUN_DIR"
)

if [[ -n "${LEROBOT_DATASET_ROOT:-}" ]]; then
  ARGS+=(--root "$LEROBOT_DATASET_ROOT")
fi
if [[ -n "${LEROBOT_DATASET_SPLIT:-}" ]]; then
  ARGS+=(--split "$LEROBOT_DATASET_SPLIT")
fi
if [[ "${LEROBOT_ALLOW_DOWNLOAD}" == "1" ]]; then
  ARGS+=(--allow-download)
else
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

python scripts/lerobot/inspect_dataset.py "${ARGS[@]}"

echo "LEROBOT_DATA_SMOKE_COMPLETE $RUN_DIR"
