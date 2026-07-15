#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/fastwam/realrobot_train_eval.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: FastWAM config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

resolve_task_name() {
  if [[ -n "${FASTWAM_TASK_NAME}" ]]; then
    echo "${FASTWAM_TASK_NAME}"
    return
  fi

  case "${FASTWAM_RECIPE}" in
    joint_base) echo "real_robot_joint_2cam224_1e-4" ;;
    pose_base) echo "real_robot_uncond_2cam224_1e-4" ;;
    v6_clean) echo "real_robot_joint_2cam224_v6_clean" ;;
    v6_decision) echo "real_robot_joint_2cam224_v6_decision" ;;
    v6_codebook) echo "real_robot_joint_2cam224_v6_codebook" ;;
    v6_scratch) echo "real_robot_joint_2cam224_v6_scratch" ;;
    v6_discrim) echo "real_robot_joint_2cam224_v6_2_discrim" ;;
    v6_dagger) echo "real_robot_joint_2cam224_v6_dagger" ;;
    v6_robust) echo "real_robot_joint_2cam224_v6_1_robust" ;;
    *)
      echo "ERROR: unsupported FASTWAM_RECIPE=${FASTWAM_RECIPE}" >&2
      exit 2
      ;;
  esac
}

detect_gpus_per_node() {
  if [[ -n "${FASTWAM_GPUS_PER_NODE}" ]]; then
    echo "${FASTWAM_GPUS_PER_NODE}"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    python - <<'PY'
import os
visible = [x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
print(len(visible))
PY
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l | tr -d ' '
    return
  fi
  echo "0"
}

require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing ${hint}: ${path}" >&2
    exit 2
  fi
}

TASK_NAME="$(resolve_task_name)"
GPUS_PER_NODE="$(detect_gpus_per_node)"
if [[ ! "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || (( GPUS_PER_NODE < 1 )); then
  echo "ERROR: no visible GPU detected. FastWAM backend has no CPU fallback." >&2
  exit 2
fi

require_file "$FASTWAM_WORKDIR/scripts/train_zero1.sh" "FastWAM train launcher"
require_file "$FASTWAM_WORKDIR/configs/task/${TASK_NAME}.yaml" "FastWAM task config"
if [[ "${FASTWAM_RECIPE}" == "v6_scratch" ]]; then
  require_file "$FASTWAM_ACTION_DIT_BACKBONE" "FastWAM ActionDiT backbone"
else
  require_file "$FASTWAM_RELEASE_CKPT" "FastWAM release checkpoint"
fi

if [[ "${FASTWAM_REQUIRE_CUDA}" == "1" ]]; then
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is required for FastWAM; CPU fallback is intentionally disabled.")
print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
print(f"torch={torch.__version__}")
PY
fi

RUN_ID="${FASTWAM_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${FASTWAM_RUN_ROOT}/${FASTWAM_RUN_NAME}/${RUN_ID}"
FASTWAM_NATIVE_OUTPUT_DIR="${FASTWAM_WORKDIR}/runs/${TASK_NAME}/${RUN_ID}"

if [[ -e "$RUN_DIR" ]]; then
  echo "ERROR: run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
cp "$CONFIG_PATH" "$RUN_DIR/config.sh"

export PYTHONPATH="${FASTWAM_WORKDIR}/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MASTER_ADDR="${FASTWAM_MASTER_ADDR}"
export MASTER_PORT="${FASTWAM_MASTER_PORT}"
export RUN_ID

MODEL_ARGS=(
  "task=${TASK_NAME}"
  model.model_id=Wan2.2-TI2V-5B
  model.tokenizer_model_id=Wan2.2-TI2V-5B
  model.redirect_common_files=false
  model.load_text_encoder=false
  "mixed_precision=${FASTWAM_MIXED_PRECISION}"
  "wandb.enabled=${FASTWAM_WANDB_ENABLE}"
)

case "${FASTWAM_RECIPE}" in
  v6_clean|v6_decision|v6_codebook|v6_discrim|v6_dagger|v6_robust)
    MODEL_ARGS+=(model.skip_dit_load_from_pretrain=false model.action_dit_pretrained_path=null)
    ;;
  v6_scratch)
    MODEL_ARGS+=(model.skip_dit_load_from_pretrain=false)
    ;;
  *)
    MODEL_ARGS+=(model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null)
    ;;
esac

if [[ -n "${FASTWAM_PIN_STATS}" ]]; then
  MODEL_ARGS+=("data.train.pretrained_norm_stats=${FASTWAM_PIN_STATS}")
fi

case "${FASTWAM_MODE}" in
  smoke)
    RUN_ARGS=(
      "max_steps=${FASTWAM_SMOKE_MAX_STEPS}"
      log_every=1
      save_every=1
      eval_every=0
      num_epochs=1
      "batch_size=${FASTWAM_SMOKE_BATCH_SIZE}"
      "num_workers=${FASTWAM_SMOKE_NUM_WORKERS}"
      gradient_accumulation_steps=1
    )
    ;;
  pilot)
    RUN_ARGS=(
      "max_steps=${FASTWAM_PILOT_MAX_STEPS}"
      log_every=5
      "save_every=${FASTWAM_PILOT_SAVE_EVERY}"
      eval_every=0
      num_epochs=1
      "batch_size=${FASTWAM_PILOT_BATCH_SIZE}"
      "num_workers=${FASTWAM_PILOT_NUM_WORKERS}"
      gradient_accumulation_steps=1
    )
    ;;
  full)
    RUN_ARGS=(
      max_steps=null
      log_every=10
      "save_every=${FASTWAM_FULL_SAVE_EVERY}"
      eval_every=0
      "num_epochs=${FASTWAM_FULL_NUM_EPOCHS}"
      "batch_size=${FASTWAM_FULL_BATCH_SIZE}"
      "num_workers=${FASTWAM_FULL_NUM_WORKERS}"
    )
    ;;
  offline-smoke)
    require_file "$FASTWAM_RELEASE_DATASET_STATS" "FastWAM release dataset stats"
    OFFLINE_CMD=(
      python scripts/offline_fastwam_smoke.py
      "task=${TASK_NAME}"
      "offline_smoke.checkpoint_path=${FASTWAM_RELEASE_CKPT}"
      "offline_smoke.dataset_stats_path=${FASTWAM_RELEASE_DATASET_STATS}"
      "offline_smoke.output_json=${PWD}/${RUN_DIR}/offline_fastwam_smoke_result.json"
    )
    printf "%q " "${OFFLINE_CMD[@]}" > "$RUN_DIR/command.txt"
    {
      echo "FASTWAM_OFFLINE_SMOKE task=${TASK_NAME}"
      cd "$FASTWAM_WORKDIR"
      "${OFFLINE_CMD[@]}"
    } 2>&1 | tee "$RUN_DIR/offline_stdout.log"
    echo "$FASTWAM_NATIVE_OUTPUT_DIR" > "$RUN_DIR/fastwam_native_output_dir.txt"
    echo "FASTWAM_OFFLINE_COMPLETE $RUN_DIR"
    exit 0
    ;;
  *)
    echo "ERROR: FASTWAM_MODE must be smoke|pilot|full|offline-smoke, got ${FASTWAM_MODE}" >&2
    exit 2
    ;;
esac

if [[ -n "${FASTWAM_EXTRA_OVERRIDES}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${FASTWAM_EXTRA_OVERRIDES} )
else
  EXTRA_ARGS=()
fi

CMD=(bash scripts/train_zero1.sh "$GPUS_PER_NODE" "${MODEL_ARGS[@]}" "${RUN_ARGS[@]}" "${EXTRA_ARGS[@]}")
printf "%q " "${CMD[@]}" > "$RUN_DIR/command.txt"
echo "$FASTWAM_NATIVE_OUTPUT_DIR" > "$RUN_DIR/fastwam_native_output_dir.txt"

python - <<PY
import json
from pathlib import Path

manifest = {
    "backend": "fastwam-realrobot",
    "mode": "${FASTWAM_MODE}",
    "recipe": "${FASTWAM_RECIPE}",
    "task_name": "${TASK_NAME}",
    "run_id": "${RUN_ID}",
    "run_dir": "${RUN_DIR}",
    "fastwam_workdir": "${FASTWAM_WORKDIR}",
    "fastwam_native_output_dir": "${FASTWAM_NATIVE_OUTPUT_DIR}",
    "official_repo": "${FASTWAM_OFFICIAL_REPO}",
    "official_ref": "${FASTWAM_OFFICIAL_REF}",
    "overlay_repo": "${FASTWAM_OVERLAY_REPO}",
    "overlay_ref": "${FASTWAM_OVERLAY_REF}",
    "gpus_per_node": int("${GPUS_PER_NODE}"),
    "mixed_precision": "${FASTWAM_MIXED_PRECISION}",
    "release_checkpoint": "${FASTWAM_RELEASE_CKPT}",
    "pin_stats": "${FASTWAM_PIN_STATS}",
}
Path("${RUN_DIR}/backend_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "FASTWAM_TRAIN_START task=${TASK_NAME} mode=${FASTWAM_MODE} recipe=${FASTWAM_RECIPE} run_dir=${RUN_DIR}"

set +e
(
  cd "$FASTWAM_WORKDIR"
  NNODES="${NNODES:-1}" \
  NODE_RANK="${NODE_RANK:-0}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${MASTER_PORT}" \
  RUN_ID="${RUN_ID}" \
  "${CMD[@]}"
) 2>&1 | tee "$RUN_DIR/train_stdout.log"
train_status=${PIPESTATUS[0]}
set -e

if python scripts/fastwam/parse_train_log.py --log "$RUN_DIR/train_stdout.log" --output-dir "$RUN_DIR"; then
  echo "FASTWAM_LOSS_REPORT $RUN_DIR/loss_summary.json"
else
  echo "WARNING: FastWAM loss parser did not find complete train records in $RUN_DIR/train_stdout.log" >&2
fi

if (( train_status != 0 )); then
  echo "ERROR: FastWAM training command failed with status ${train_status}. See $RUN_DIR/train_stdout.log" >&2
  exit "$train_status"
fi

echo "FASTWAM_TRAIN_COMPLETE $RUN_DIR"
echo "FASTWAM_NATIVE_OUTPUT $FASTWAM_NATIVE_OUTPUT_DIR"
