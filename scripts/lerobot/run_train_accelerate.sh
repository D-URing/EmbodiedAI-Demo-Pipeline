#!/usr/bin/env bash
# LeRobot accelerate 多卡训练 wrapper。
#
# 主要用于 SmolVLA 单机 8 卡/后续多机训练。它读取 configs/lerobot/train/*.sh，
# 检查 CUDA 和 lerobot-train，然后通过 accelerate 启动真实训练。
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

python - <<'PY'
import importlib

for module in ["accelerate", "lerobot.scripts.lerobot_train"]:
    importlib.import_module(module)
print("LeRobot launcher imports OK: accelerate + lerobot.scripts.lerobot_train")
PY

if [[ "${LEROBOT_ALLOW_BUSY_GPUS:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  busy_gpu_processes="$(
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true
  )"
  if [[ -n "$busy_gpu_processes" ]]; then
    echo "ERROR: found existing GPU compute processes before LeRobot launch." >&2
    echo "This usually means a previous distributed run left orphan processes and may cause CUDA OOM." >&2
    echo "$busy_gpu_processes" >&2
    echo "Stop them first, or set LEROBOT_ALLOW_BUSY_GPUS=1 if sharing GPUs is intentional." >&2
    exit 2
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
PY

RUN_ID="${LEROBOT_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${LEROBOT_RUN_ROOT}/${LEROBOT_RUN_NAME}/${RUN_ID}"
OUTPUT_DIR="${LEROBOT_OUTPUT_DIR:-${RUN_DIR}/lerobot_output}"
MACHINE_RANK="${LEROBOT_MACHINE_RANK:-0}"
NUM_MACHINES="${LEROBOT_NUM_MACHINES:-1}"

if [[ -e "$OUTPUT_DIR" && "${LEROBOT_RESUME:-0}" != "1" && "$MACHINE_RANK" == "0" ]]; then
  echo "ERROR: LeRobot output_dir already exists: $OUTPUT_DIR" >&2
  echo "Set LEROBOT_RESUME=1 and LEROBOT_RESUME_CONFIG_PATH=/path/to/train_config.json to resume intentionally." >&2
  exit 2
fi

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_DIR")"
CONFIG_SNAPSHOT="$RUN_DIR/config.sh"
COMMAND_TXT="$RUN_DIR/command.txt"
BACKEND_MANIFEST_JSON="$RUN_DIR/backend_manifest.json"
ACCELERATE_ENTRY="$RUN_DIR/accelerate_entry.py"
LEROBOT_TRAIN_ENTRY="$RUN_DIR/lerobot_train_entry.py"
if [[ "$NUM_MACHINES" != "1" ]]; then
  CONFIG_SNAPSHOT="$RUN_DIR/config.rank${MACHINE_RANK}.sh"
  COMMAND_TXT="$RUN_DIR/command.rank${MACHINE_RANK}.txt"
  BACKEND_MANIFEST_JSON="$RUN_DIR/backend_manifest.rank${MACHINE_RANK}.json"
  ACCELERATE_ENTRY="$RUN_DIR/accelerate_entry.rank${MACHINE_RANK}.py"
  LEROBOT_TRAIN_ENTRY="$RUN_DIR/lerobot_train_entry.rank${MACHINE_RANK}.py"
fi
cp "$CONFIG_PATH" "$CONFIG_SNAPSHOT"

cat > "$ACCELERATE_ENTRY" <<'PY'
from accelerate.commands.accelerate_cli import main
raise SystemExit(main())
PY

cat > "$LEROBOT_TRAIN_ENTRY" <<'PY'
from lerobot.scripts.lerobot_train import main
raise SystemExit(main())
PY

TRAIN_CMD=(
  "$LEROBOT_TRAIN_ENTRY"
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
  TRAIN_CMD=("$LEROBOT_TRAIN_ENTRY" --config_path="$LEROBOT_RESUME_CONFIG_PATH" --resume=true --output_dir="$OUTPUT_DIR")
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
  python "$ACCELERATE_ENTRY" launch
  --same_network
  --multi_gpu
  --gpu_ids all
  --num_processes "$LEROBOT_NUM_PROCESSES"
  --num_machines "$LEROBOT_NUM_MACHINES"
  --machine_rank "$LEROBOT_MACHINE_RANK"
  --main_process_ip "$LEROBOT_MAIN_PROCESS_IP"
  --main_process_port "$LEROBOT_MAIN_PROCESS_PORT"
  --mixed_precision "$LEROBOT_ACCELERATE_MIXED_PRECISION"
)

FULL_CMD=("${ACCELERATE_CMD[@]}" "${TRAIN_CMD[@]}")

printf "%q " "${FULL_CMD[@]}" > "$COMMAND_TXT"
printf "\n" >> "$COMMAND_TXT"

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
Path("${BACKEND_MANIFEST_JSON}").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "LEROBOT_ACCELERATE_TRAIN_START run_dir=$RUN_DIR output_dir=$OUTPUT_DIR"
echo "LEROBOT_ACCELERATE_EFFECTIVE_BATCH_SIZE $(( LEROBOT_BATCH_SIZE * LEROBOT_NUM_PROCESSES ))"

TRAIN_STDOUT_LOG="$RUN_DIR/train_stdout.log"
if [[ "$NUM_MACHINES" != "1" ]]; then
  TRAIN_STDOUT_LOG="$RUN_DIR/train_stdout.rank${MACHINE_RANK}.log"
fi

run_start_epoch=$(date +%s)
set +e
"${FULL_CMD[@]}" 2>&1 | tee "$TRAIN_STDOUT_LOG"
train_status=${PIPESTATUS[0]}
set -e
run_end_epoch=$(date +%s)
run_elapsed_seconds=$(( run_end_epoch - run_start_epoch ))

if [[ "$MACHINE_RANK" == "0" ]] && python scripts/lerobot/parse_train_log.py --log "$TRAIN_STDOUT_LOG" --output-dir "$RUN_DIR"; then
  echo "LEROBOT_LOSS_REPORT $RUN_DIR/loss_summary.json"
elif [[ "$MACHINE_RANK" != "0" ]]; then
  echo "LEROBOT_LOSS_REPORT skipped on machine_rank=${MACHINE_RANK}; rank0 writes the shared summary."
else
  echo "WARNING: LeRobot loss parser did not find complete train records in $TRAIN_STDOUT_LOG" >&2
fi

if [[ "$MACHINE_RANK" != "0" ]]; then
  if (( train_status != 0 )); then
    echo "ERROR: LeRobot accelerate training failed with status ${train_status}. See $TRAIN_STDOUT_LOG" >&2
    exit "$train_status"
  fi
  echo "LEROBOT_ACCELERATE_TRAIN_COMPLETE $RUN_DIR"
  echo "LEROBOT_OUTPUT $OUTPUT_DIR"
  exit 0
fi

python - <<PY
from __future__ import annotations

import json
from pathlib import Path

loss_summary_path = Path("${RUN_DIR}") / "loss_summary.json"
step_metrics = {}
if loss_summary_path.is_file():
    try:
        step_metrics = json.loads(loss_summary_path.read_text(encoding="utf-8")).get("step_metrics", {})
    except json.JSONDecodeError:
        step_metrics = {}

elapsed_seconds = max(float("${run_elapsed_seconds}"), 1.0)
steps = int("${LEROBOT_STEPS}")
effective_batch_size = int("${LEROBOT_BATCH_SIZE}") * int("${LEROBOT_NUM_PROCESSES}")
completed = int("${train_status}") == 0
summary = {
    "schema_version": "1.0",
    "backend": "lerobot",
    "policy_type": "${LEROBOT_POLICY_TYPE}",
    "dataset_repo_id": "${LEROBOT_DATASET_REPO_ID}",
    "run_dir": "${RUN_DIR}",
    "output_dir": "${OUTPUT_DIR}",
    "completed": completed,
    "configured_steps": steps,
    "per_process_batch_size": int("${LEROBOT_BATCH_SIZE}"),
    "num_processes": int("${LEROBOT_NUM_PROCESSES}"),
    "effective_batch_size": effective_batch_size,
    "wall_time_seconds": elapsed_seconds,
    "approx_step_per_second": steps / elapsed_seconds if completed else None,
    "approx_sample_per_second": steps * effective_batch_size / elapsed_seconds if completed else None,
    "parsed_step_metrics": step_metrics,
    "notes": [
        "step/sample throughput uses configured_steps when training exits successfully.",
        "parsed_step_metrics comes from LeRobot per-step logs and excludes model initialization time.",
        "First run may include torch.compile or kernel warmup overhead if enabled.",
    ],
}
path = Path("${RUN_DIR}") / "speed_summary.json"
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
wall_step_per_second = summary["approx_step_per_second"]
wall_sample_per_second = summary["approx_sample_per_second"]
mean_update_seconds = step_metrics.get("mean_update_seconds")
mean_data_seconds = step_metrics.get("mean_data_seconds")
mean_samples_per_second = step_metrics.get("mean_samples_per_second")
max_memory_gb = step_metrics.get("max_memory_gb")

print(
    "LEROBOT_SPEED_REPORT "
    f"completed={str(completed).lower()} "
    f"wall_step_per_second={wall_step_per_second} "
    f"wall_sample_per_second={wall_sample_per_second} "
    f"train_mean_update_seconds={mean_update_seconds} "
    f"train_mean_sample_per_second={mean_samples_per_second} "
    f"max_memory_gb={max_memory_gb} "
    f"path={path}"
)
print(
    "LEROBOT_SPEED_HUMAN "
    f"end_to_end={wall_step_per_second:.3f} step/s, {wall_sample_per_second:.1f} sample/s; "
    f"train_loop={mean_update_seconds:.3f}s/step, {mean_samples_per_second:.1f} sample/s; "
    f"data={mean_data_seconds:.3f}s/step; gpu_mem={max_memory_gb:.2f}GB"
    if all(v is not None for v in [wall_step_per_second, wall_sample_per_second, mean_update_seconds, mean_samples_per_second, mean_data_seconds, max_memory_gb])
    else "LEROBOT_SPEED_HUMAN unavailable: no parsed step metrics found"
)
PY

if (( train_status != 0 )); then
  echo "ERROR: LeRobot accelerate training failed with status ${train_status}. See $RUN_DIR/train_stdout.log" >&2
  exit "$train_status"
fi

echo "LEROBOT_ACCELERATE_TRAIN_COMPLETE $RUN_DIR"
echo "LEROBOT_OUTPUT $OUTPUT_DIR"
