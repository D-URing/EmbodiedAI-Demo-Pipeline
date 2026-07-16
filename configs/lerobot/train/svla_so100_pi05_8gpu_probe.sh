# shellcheck shell=bash

# LeRobot 训练配置：SO100 pick-place + pi05 微调/测速探针。
#
# 这条线的目标不是替代 SmolVLA，而是验证 LeRobot 官方 pi05 policy 在我们的
# 项目资产布局下能否完成：
#   本地数据 -> 本地 base 权重 -> accelerate 多卡训练 -> loss/throughput 记录。
#
# 默认使用项目内资产：
#   data/lerobot/svla_so100_pickplace
#   models/lerobot/pi05/pi05_base
#
# 注意：
# - pi05 是重型 VLA policy，首次运行应先用较小 steps 做探针测速。
# - batch_size 是“每张卡/每个进程”的 batch size；有效 batch = batch_size * num_processes。
# - A100/A800 推荐 bf16；如果显存不足，优先减小 LEROBOT_BATCH_SIZE，再考虑开启/关闭 compile。

export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-lerobot/svla_so100_pickplace}"
export LEROBOT_DATASET_ROOT="${LEROBOT_DATASET_ROOT:-$EMBODIED_REPO_ROOT/data/lerobot/svla_so100_pickplace}"
export LEROBOT_DATASET_VIDEO_BACKEND="${LEROBOT_DATASET_VIDEO_BACKEND:-pyav}"
export LEROBOT_DATASET_EVAL_SPLIT="${LEROBOT_DATASET_EVAL_SPLIT:-}"

export LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-pi05}"
export LEROBOT_POLICY_REPO_ID="${LEROBOT_POLICY_REPO_ID:-local/svla_so100_pi05_8gpu_probe}"
export LEROBOT_POLICY_PUSH_TO_HUB="${LEROBOT_POLICY_PUSH_TO_HUB:-false}"
export LEROBOT_POLICY_DEVICE="${LEROBOT_POLICY_DEVICE:-cuda}"
export LEROBOT_POLICY_PRETRAINED_PATH="${LEROBOT_POLICY_PRETRAINED_PATH:-$EMBODIED_REPO_ROOT/models/lerobot/pi05/pi05_base}"
export LEROBOT_POLICY_DTYPE="${LEROBOT_POLICY_DTYPE:-bfloat16}"

# pi05 官方配置面里常见的两个加速/省显存开关。compile 首次会有编译开销；
# 如果只是快速排错，可临时 LEROBOT_POLICY_COMPILE_MODEL=false。
export LEROBOT_POLICY_COMPILE_MODEL="${LEROBOT_POLICY_COMPILE_MODEL:-true}"
export LEROBOT_POLICY_GRADIENT_CHECKPOINTING="${LEROBOT_POLICY_GRADIENT_CHECKPOINTING:-true}"

# 默认是测速探针，不是正式长训。正式实验可在启动时覆盖：
#   LEROBOT_STEPS=20000 LEROBOT_BATCH_SIZE=4 bash experiments/lerobot/pi05_so100_8gpu_probe/launch.sh
export LEROBOT_STEPS="${LEROBOT_STEPS:-200}"
export LEROBOT_BATCH_SIZE="${LEROBOT_BATCH_SIZE:-1}"
export LEROBOT_NUM_WORKERS="${LEROBOT_NUM_WORKERS:-4}"
export LEROBOT_PREFETCH_FACTOR="${LEROBOT_PREFETCH_FACTOR:-4}"
export LEROBOT_PERSISTENT_WORKERS="${LEROBOT_PERSISTENT_WORKERS:-true}"
export LEROBOT_LOG_FREQ="${LEROBOT_LOG_FREQ:-10}"
export LEROBOT_SAVE_CHECKPOINT="${LEROBOT_SAVE_CHECKPOINT:-true}"
export LEROBOT_SAVE_FREQ="${LEROBOT_SAVE_FREQ:-100}"
export LEROBOT_SEED="${LEROBOT_SEED:-1005}"

# 当前只测离线微调吞吐和 loss；仿真/真机评测暂不启用。
export LEROBOT_ENV_EVAL_FREQ="${LEROBOT_ENV_EVAL_FREQ:-0}"
export LEROBOT_EVAL_STEPS="${LEROBOT_EVAL_STEPS:-0}"
export LEROBOT_MAX_EVAL_SAMPLES="${LEROBOT_MAX_EVAL_SAMPLES:-}"
export LEROBOT_WANDB_ENABLE="${LEROBOT_WANDB_ENABLE:-false}"

export LEROBOT_RUN_NAME="${LEROBOT_RUN_NAME:-svla_so100_pi05_8gpu_probe}"
export LEROBOT_RUN_ROOT="${LEROBOT_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/lerobot}"
export HF_HOME="${HF_HOME:-$EMBODIED_REPO_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$EMBODIED_REPO_ROOT/hf_cache/torch}"

# Accelerate 分布式参数。默认单机 8 卡；多机时覆盖下面 4 个变量即可。
export LEROBOT_NUM_PROCESSES="${LEROBOT_NUM_PROCESSES:-8}"
export LEROBOT_NUM_MACHINES="${LEROBOT_NUM_MACHINES:-1}"
export LEROBOT_MACHINE_RANK="${LEROBOT_MACHINE_RANK:-0}"
export LEROBOT_MAIN_PROCESS_IP="${LEROBOT_MAIN_PROCESS_IP:-127.0.0.1}"
export LEROBOT_MAIN_PROCESS_PORT="${LEROBOT_MAIN_PROCESS_PORT:-29505}"
export LEROBOT_ACCELERATE_MIXED_PRECISION="${LEROBOT_ACCELERATE_MIXED_PRECISION:-bf16}"

# 断点续训：
#   export LEROBOT_RESUME=1
#   export LEROBOT_RESUME_CONFIG_PATH=runs/lerobot/<run>/<id>/lerobot_output/checkpoints/<step>/train_config.json
export LEROBOT_RESUME="${LEROBOT_RESUME:-0}"
export LEROBOT_RESUME_CONFIG_PATH="${LEROBOT_RESUME_CONFIG_PATH:-}"
