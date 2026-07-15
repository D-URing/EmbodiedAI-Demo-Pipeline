#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/lerobot/train/svla_so100_smolvla_8gpu_long.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: LeRobot train config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

if [[ "${LEROBOT_POLICY_DEVICE}" != "cuda" ]]; then
  echo "ERROR: LEROBOT_POLICY_DEVICE must be cuda. CPU fallback is intentionally disabled." >&2
  exit 2
fi

command -v accelerate >/dev/null || {
  echo "ERROR: accelerate is not on PATH. Reinstall LeRobot with training extras." >&2
  exit 2
}
command -v lerobot-train >/dev/null || {
  echo "ERROR: lerobot-train is not on PATH. Run scripts/lerobot/install_lerobot_cluster.sh first." >&2
  exit 2
}

python - <<'PY'
import os
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is required for LeRobot accelerate training.")
visible = torch.cuda.device_count()
print(f"CUDA OK: visible_device_count={visible}")
for idx in range(visible):
    print(f"cuda:{idx}={torch.cuda.get_device_name(idx)}")
print(f"torch={torch.__version__}")
if sys.version_info < (3, 12):
    raise SystemExit(f"ERROR: LeRobot env must use Python >=3.12, got {sys.version.split()[0]}")
print(f"python={sys.version.split()[0]}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
PY

RUN_ID="${LEROBOT_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${LEROBOT_RUN_ROOT}/${LEROBOT_RUN_NAME}/${RUN_ID}"
OUTPUT_DIR="${LEROBOT_OUTPUT_DIR:-${RUN_DIR}/lerobot_output}"

if [[ -e "$OUTPUT_DIR" && "${LEROBOT_RESUME:-0}" != "1" ]]; then
  echo "ERROR: LeRobot output_dir already exists: $OUTPUT_DIR" >&2
  echo "Set LEROBOT_RESUME=1 and LEROBOT_RESUME_CONFIG_PATH=/path/to/train_config.json to resume intentionally." >&2
  exit 2
fi

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_DIR")"

TRAIN_CMD=(
  lerobot-train
  --policy.type="$LEROBOT_POLICY_TYPE"
  --policy.device="$LEROBOT_POLICY_DEVICE"
  --policy.repo_id="$LEROBOT_POLICY_REPO_ID"
  --policy.push_to_hub="$LEROBOT_POLICY_PUSH_TO_HUB"
  --dataset.repo_id="$LEROBOT_DATASET_REPO_ID"
  --dataset.video_backend="$LEROBOT_DATASET_VIDEO_BACKEND"
  --output_dir="$OUTPUT_DIR"
  --job_name="$LEROBOT_RUN_NAME"
  --steps="$LEROBOT_STEPS"
  --batch_size="$LEROBOT_BATCH_SIZE"
  --num_workers="$LEROBOT_NUM_WORKERS"
  --log_freq="$LEROBOT_LOG_FREQ"
  --save_freq="$LEROBOT_SAVE_FREQ"
  --save_checkpoint="$LEROBOT_SAVE_CHECKPOINT"
  --env_eval_freq="$LEROBOT_ENV_EVAL_FREQ"
  --eval_steps="$LEROBOT_EVAL_STEPS"
  --seed="$LEROBOT_SEED"
  --wandb.enable="$LEROBOT_WANDB_ENABLE"
)

append_if_set() {
  local key="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    TRAIN_CMD+=(--"$key"="$value")
  fi
}

append_if_set dataset.root "${LEROBOT_DATASET_ROOT:-}"
append_if_set dataset.eval_split "${LEROBOT_DATASET_EVAL_SPLIT:-}"
append_if_set max_eval_samples "${LEROBOT_MAX_EVAL_SAMPLES:-}"
append_if_set prefetch_factor "${LEROBOT_PREFETCH_FACTOR:-}"
append_if_set persistent_workers "${LEROBOT_PERSISTENT_WORKERS:-}"
append_if_set policy.pretrained_path "${LEROBOT_POLICY_PRETRAINED_PATH:-}"
append_if_set policy.dtype "${LEROBOT_POLICY_DTYPE:-}"
append_if_set policy.compile_model "${LEROBOT_POLICY_COMPILE_MODEL:-}"
append_if_set policy.enable_gradient_checkpointing "${LEROBOT_POLICY_ENABLE_GRADIENT_CHECKPOINTING:-}"
append_if_set policy.gradient_checkpointing "${LEROBOT_POLICY_GRADIENT_CHECKPOINTING:-}"
append_if_set policy.use_gradient_checkpointing "${LEROBOT_POLICY_USE_GRADIENT_CHECKPOINTING:-}"
append_if_set policy.freeze_vision_encoder "${LEROBOT_POLICY_FREEZE_VISION_ENCODER:-}"
append_if_set policy.train_expert_only "${LEROBOT_POLICY_TRAIN_EXPERT_ONLY:-}"
append_if_set policy.num_inference_steps "${LEROBOT_POLICY_NUM_INFERENCE_STEPS:-}"
append_if_set optimizer.lr "${LEROBOT_OPTIM_LR:-}"

if [[ "${LEROBOT_RESUME:-0}" == "1" ]]; then
  if [[ -z "${LEROBOT_RESUME_CONFIG_PATH:-}" ]]; then
    echo "ERROR: LEROBOT_RESUME_CONFIG_PATH is required when LEROBOT_RESUME=1." >&2
    echo "Example: LEROBOT_RESUME_CONFIG_PATH=runs/lerobot/<run>/<id>/lerobot_output/checkpoints/<step>/train_config.json" >&2
    exit 2
  fi
  TRAIN_CMD=(lerobot-train --config_path="$LEROBOT_RESUME_CONFIG_PATH" --resume=true --output_dir="$OUTPUT_DIR")
  if [[ -n "${LEROBOT_TRAIN_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    RESUME_EXTRA_ARGS=( ${LEROBOT_TRAIN_EXTRA_ARGS} )
    TRAIN_CMD+=("${RESUME_EXTRA_ARGS[@]}")
  fi
else
  if [[ -n "${LEROBOT_TRAIN_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${LEROBOT_TRAIN_EXTRA_ARGS} )
    TRAIN_CMD+=("${EXTRA_ARGS[@]}")
  fi
fi

ACCELERATE_CMD=(
  accelerate launch
  --num_processes "$LEROBOT_NUM_PROCESSES"
  --num_machines "$LEROBOT_NUM_MACHINES"
  --machine_rank "$LEROBOT_MACHINE_RANK"
  --main_process_ip "$LEROBOT_MAIN_PROCESS_IP"
  --main_process_port "$LEROBOT_MAIN_PROCESS_PORT"
  --mixed_precision "$LEROBOT_ACCELERATE_MIXED_PRECISION"
)

if [[ "${LEROBOT_NUM_MACHINES}" == "1" ]]; then
  ACCELERATE_CMD+=(--multi_gpu)
fi

FULL_CMD=("${ACCELERATE_CMD[@]}" "${TRAIN_CMD[@]}")

printf "%q " "${FULL_CMD[@]}" > "$RUN_DIR/command.txt"
printf "\n" >> "$RUN_DIR/command.txt"

python - <<PY
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "backend": "lerobot",
    "launcher": "accelerate",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "config_path": "${CONFIG_PATH}",
    "run_id": "${RUN_ID}",
    "run_dir": "${RUN_DIR}",
    "output_dir": "${OUTPUT_DIR}",
    "policy_type": "${LEROBOT_POLICY_TYPE}",
    "dataset_repo_id": "${LEROBOT_DATASET_REPO_ID}",
    "dataset_root": "${LEROBOT_DATASET_ROOT:-}",
    "steps": int("${LEROBOT_STEPS}"),
    "per_process_batch_size": int("${LEROBOT_BATCH_SIZE}"),
    "num_processes": int("${LEROBOT_NUM_PROCESSES}"),
    "effective_batch_size": int("${LEROBOT_BATCH_SIZE}") * int("${LEROBOT_NUM_PROCESSES}"),
    "save_freq": int("${LEROBOT_SAVE_FREQ}"),
    "resume": "${LEROBOT_RESUME:-0}" == "1",
    "resume_config_path": "${LEROBOT_RESUME_CONFIG_PATH:-}",
    "accelerate": {
        "num_machines": int("${LEROBOT_NUM_MACHINES}"),
        "machine_rank": int("${LEROBOT_MACHINE_RANK}"),
        "main_process_ip": "${LEROBOT_MAIN_PROCESS_IP}",
        "main_process_port": int("${LEROBOT_MAIN_PROCESS_PORT}"),
        "mixed_precision": "${LEROBOT_ACCELERATE_MIXED_PRECISION}",
    },
}
Path("${RUN_DIR}/backend_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "LEROBOT_ACCELERATE_TRAIN_START run_dir=$RUN_DIR output_dir=$OUTPUT_DIR"
echo "LEROBOT_ACCELERATE_EFFECTIVE_BATCH_SIZE $(( LEROBOT_BATCH_SIZE * LEROBOT_NUM_PROCESSES ))"

set +e
"${FULL_CMD[@]}" 2>&1 | tee "$RUN_DIR/train_stdout.log"
train_status=${PIPESTATUS[0]}
set -e

if python scripts/lerobot/parse_train_log.py --log "$RUN_DIR/train_stdout.log" --output-dir "$RUN_DIR"; then
  echo "LEROBOT_LOSS_REPORT $RUN_DIR/loss_summary.json"
else
  echo "WARNING: LeRobot loss parser did not find complete train records in $RUN_DIR/train_stdout.log" >&2
fi

if (( train_status != 0 )); then
  echo "ERROR: LeRobot accelerate training failed with status ${train_status}. See $RUN_DIR/train_stdout.log" >&2
  exit "$train_status"
fi

echo "LEROBOT_ACCELERATE_TRAIN_COMPLETE $RUN_DIR"
echo "LEROBOT_OUTPUT $OUTPUT_DIR"
