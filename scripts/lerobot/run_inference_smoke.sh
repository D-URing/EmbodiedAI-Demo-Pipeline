#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/lerobot/native_pusht_act_pipeline.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: LeRobot native config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

if [[ -z "${LEROBOT_POLICY_PATH:-}" ]]; then
  echo "ERROR: LEROBOT_POLICY_PATH is required and must point to a local checkpoint/pretrained directory." >&2
  echo "Downloads are disabled by default; prepare the checkpoint on the cluster/cache first." >&2
  exit 2
fi

RUN_ID="${LEROBOT_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${LEROBOT_RUN_ROOT}/${LEROBOT_RUN_NAME}/${RUN_ID}"
mkdir -p "$RUN_DIR"
cp "$CONFIG_PATH" "$RUN_DIR/config.sh"

ARGS=(
  --dataset-repo-id "$LEROBOT_DATASET_REPO_ID"
  --sample-index "$LEROBOT_SAMPLE_INDEX"
  --policy-type "$LEROBOT_POLICY_TYPE"
  --policy-path "$LEROBOT_POLICY_PATH"
  --device "$LEROBOT_INFERENCE_DEVICE"
  --output-dir "$RUN_DIR"
)

if [[ -n "${LEROBOT_POLICY_CLASS:-}" ]]; then
  ARGS+=(--policy-class "$LEROBOT_POLICY_CLASS")
fi
if [[ -n "${LEROBOT_DATASET_ROOT:-}" ]]; then
  ARGS+=(--dataset-root "$LEROBOT_DATASET_ROOT")
fi
if [[ "${LEROBOT_ALLOW_DOWNLOAD}" == "1" ]]; then
  ARGS+=(--allow-download)
else
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

python scripts/lerobot/run_policy_inference_smoke.py "${ARGS[@]}"

echo "LEROBOT_INFERENCE_SMOKE_COMPLETE $RUN_DIR"
