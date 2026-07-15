# LeRobot Pipeline

## 这条线解决什么

LeRobot 是第一阶段的主线。它回答：

> 我们能不能基于开源 LeRobot，把 dataset read → policy train/load → offline inference → evidence report 跑通？

当前默认 demo：

```text
dataset: lerobot/pusht
policy:  ACT
device:  CUDA only
```

这不是 CPU toy trainer，也不是家庭任务最终模型。它是团队后续接新模型、新数据、新仿真的基准管线。

## 当前 SCUT 状态

已验证：

```text
GPU: 8 x NVIDIA A800-SXM4-80GB on gpu11
torch: 2.11.0+cu128
lerobot: 0.6.1
dataset: data/lerobot/pusht
dataset frames: 25650
2-step loss: 96.987 -> 83.351
```

关键产物：

```text
data/lerobot/pusht
hf_cache/torch/hub/checkpoints/resnet18-f37072fd.pth
runs/lerobot/<run_name>/<run_id>/loss_summary.json
```

## 环境

```bash
export BASE=/mnt/gpu11_200T/dingxibo
export PROJECT=$BASE/EmbodiedAI-Demo-Pipeline
export CONDA=$BASE/miniconda3/bin/conda

cd "$PROJECT"
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
```

通用变量：

```bash
export PROJECT_ROOT="$PROJECT"
export EMBODIED_DATA_ROOT="$PROJECT/data"
export EMBODIED_MODEL_ROOT="$PROJECT/models"
export EMBODIED_RUN_ROOT="$PROJECT/runs"
export HF_HOME="$PROJECT/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PROJECT/hf_cache/torch"
export HF_ENDPOINT=https://hf-mirror.com
```

## 下载

PushT 数据：

```bash
export PYTHON_BIN="$BASE/miniconda3/envs/embodied-core/bin/python"
export HFD_BIN=/home/scut/hfd.sh
export HFD_TOOL=aria2c
export HFD_THREADS=10
export HFD_JOBS=4

make download-lerobot-artifacts
```

ResNet18 backbone，如果不存在：

```bash
mkdir -p "$TORCH_HOME/hub/checkpoints"
wget -O "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth" \
  https://download.pytorch.org/models/resnet18-f37072fd.pth
```

## 验证 dataset

```bash
export LEROBOT_ALLOW_DOWNLOAD=0
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/pusht"

make lerobot-data-smoke
```

预期：

```text
SUMMARY repo_id=lerobot/pusht length=25650
```

## 跑 GPU training smoke

快速 2-step 环境检查：

```bash
export LEROBOT_ALLOW_DOWNLOAD=0
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/pusht"
export TORCH_HOME="$PROJECT/hf_cache/torch"

export LEROBOT_STEPS=2
export LEROBOT_BATCH_SIZE=2
export LEROBOT_NUM_WORKERS=0
export LEROBOT_LOG_FREQ=1
export LEROBOT_SAVE_FREQ=2
export LEROBOT_RUN_NAME=pusht_act_gpu_make_check

make lerobot-train-smoke
```

正式一点的 loss 下降观察：

```bash
export LEROBOT_STEPS=1000
export LEROBOT_BATCH_SIZE=8
export LEROBOT_NUM_WORKERS=4
export LEROBOT_LOG_FREQ=20
export LEROBOT_SAVE_FREQ=1000
export LEROBOT_RUN_NAME=pusht_act_gpu_smoke

make lerobot-train-smoke
```

输出：

```text
runs/lerobot/<run_name>/<run_id>/
├── command.txt
├── train_stdout.log
├── loss_summary.json
└── lerobot_output/
```

## 已固化的坑位处理

| 问题 | 当前处理 |
|---|---|
| 计算节点不能访问 `download.pytorch.org` | `TORCH_HOME` 指到项目内 `hf_cache/torch` |
| `torchcodec + ffmpeg` 在旧 glibc 上失败 | 默认 `dataset.video_backend=pyav` |
| LeRobot 要求 `policy.repo_id` | 默认 `local/pusht_act_gpu_smoke` |
| 不希望 push Hub | 默认 `policy.push_to_hub=false` |
| 不允许 CPU fallback | runner 会检查 `torch.cuda.is_available()` |

## 相关文件

```text
configs/lerobot/
scripts/lerobot/
demo_chains/lerobot_fastwam_data_to_inference_v0.yaml
docs/LEROBOT_FIRST_PIPELINE.md
docs/LEROBOT_REPLICATION.md
```

