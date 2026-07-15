#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/lerobot/pusht_act_gpu_smoke.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: LeRobot smoke config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

if [[ "${LEROBOT_POLICY_DEVICE}" != "cuda" ]]; then
  echo "ERROR: LEROBOT_POLICY_DEVICE must be cuda. CPU fallback is intentionally disabled." >&2
  exit 2
fi

command -v lerobot-train >/dev/null || {
  echo "ERROR: lerobot-train is not on PATH. Run scripts/lerobot/install_lerobot_cluster.sh first." >&2
  exit 2
}

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is required for this LeRobot replication demo.")
print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
print(f"torch={torch.__version__}")
if sys.version_info < (3, 12):
    raise SystemExit(f"ERROR: LeRobot env must use Python >=3.12, got {sys.version.split()[0]}")
print(f"python={sys.version.split()[0]}")
PY

RUN_ID="${LEROBOT_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${LEROBOT_RUN_ROOT}/${LEROBOT_RUN_NAME}/${RUN_ID}"
OUTPUT_DIR="${RUN_DIR}/lerobot_output"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: LeRobot output_dir already exists: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"

CMD=(
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
  --env_eval_freq="$LEROBOT_ENV_EVAL_FREQ"
  --eval_steps="$LEROBOT_EVAL_STEPS"
  --seed="$LEROBOT_SEED"
  --wandb.enable="$LEROBOT_WANDB_ENABLE"
)

append_if_set() {
  local key="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    CMD+=(--"$key"="$value")
  fi
}

append_if_set policy.pretrained_path "${LEROBOT_POLICY_PRETRAINED_PATH:-}"
append_if_set policy.dtype "${LEROBOT_POLICY_DTYPE:-}"
append_if_set policy.compile_model "${LEROBOT_POLICY_COMPILE_MODEL:-}"
append_if_set policy.gradient_checkpointing "${LEROBOT_POLICY_GRADIENT_CHECKPOINTING:-}"
append_if_set policy.freeze_vision_encoder "${LEROBOT_POLICY_FREEZE_VISION_ENCODER:-}"
append_if_set policy.train_expert_only "${LEROBOT_POLICY_TRAIN_EXPERT_ONLY:-}"
append_if_set policy.num_inference_steps "${LEROBOT_POLICY_NUM_INFERENCE_STEPS:-}"
append_if_set optimizer.lr "${LEROBOT_OPTIM_LR:-}"

if [[ -n "${LEROBOT_DATASET_ROOT:-}" ]]; then
  CMD+=(--dataset.root="$LEROBOT_DATASET_ROOT")
fi

if [[ -n "${LEROBOT_TRAIN_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${LEROBOT_TRAIN_EXTRA_ARGS} )
  CMD+=("${EXTRA_ARGS[@]}")
fi

cat > "$RUN_DIR/command.txt" <<EOF
${CMD[*]}
EOF

"${CMD[@]}" 2>&1 | tee "$RUN_DIR/train_stdout.log"

python scripts/lerobot/parse_train_log.py \
  --log "$RUN_DIR/train_stdout.log" \
  --output-dir "$RUN_DIR"

echo "LEROBOT_TRAIN_COMPLETE $RUN_DIR"
echo "LOSS_SUMMARY $RUN_DIR/loss_summary.json"
echo "LEROBOT_OUTPUT $OUTPUT_DIR"
