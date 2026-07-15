# LeRobot Pipeline

## 这条线解决什么

LeRobot 是第一阶段的主线。它回答：

> 我们能不能基于开源 LeRobot，把 dataset read → policy train/load → offline inference → evidence report 跑通？

当前第一组训练 profile：

```text
P0: dataset=lerobot/pusht, policy=ACT
P0: dataset=lerobot/pusht, policy=Diffusion
P1: dataset=lerobot/svla_so100_pickplace, policy=SmolVLA from lerobot/smolvla_base
device: CUDA only
```

资产来自根目录全局池：

```text
data/lerobot/pusht
models/lerobot/<policy>/<name>
hf_cache/torch/hub/checkpoints/resnet18-f37072fd.pth
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
data/lerobot/svla_so100_pickplace                  # SmolVLA fine-tune 数据
models/lerobot/diffusion/diffusion_pusht        # 可选开源预训练 policy
models/lerobot/smolvla/smolvla_base             # SmolVLA base policy
models/lerobot/fastwam/fastwam_libero_uncond_2cam224
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

## 资产层级

LeRobot 主线的资产分四层：

| 层 | 路径 | 作用 |
|---|---|---|
| Dataset | `data/lerobot/pusht` | ACT/PushT 训练和 dataset smoke |
| Backbone cache | `hf_cache/torch/hub/checkpoints/resnet18-f37072fd.pth` | ACT 默认 ResNet18 视觉 backbone |
| Open policy | `models/lerobot/diffusion/diffusion_pusht` | 可直接下载的 LeRobot diffusion PushT 预训练 policy |
| VLA base policy | `models/lerobot/smolvla/smolvla_base` | SmolVLA fine-tune 起点 |
| FastWAM policy | `models/lerobot/fastwam/fastwam_libero_uncond_2cam224` | LeRobot-compatible FastWAM LIBERO 权重 |
| Local checkpoint | `runs/lerobot/<run>/lerobot_output` 或整理到 `models/lerobot/act/pusht/<name>` | 我们自己训练得到的 ACT checkpoint |

注意：ACT/PushT 当前主线优先用于训练 smoke。开源预训练 policy 先用 `lerobot/diffusion_pusht` 作为“可下载、可管理”的 policy 样例，后续再补 ACT 或 FastWAM 的 LeRobot-native checkpoint。

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

SmolVLA SO100 pick-place 数据：

```bash
make download-lerobot-svla-so100-pickplace-dataset
```

LeRobot diffusion PushT 预训练 policy：

```bash
make download-lerobot-diffusion-pusht-policy
```

默认落盘：

```text
models/lerobot/diffusion/diffusion_pusht
```

LeRobot SmolVLA base policy：

```bash
make download-lerobot-smolvla-base-policy
```

默认落盘：

```text
models/lerobot/smolvla/smolvla_base
```

LeRobot-compatible FastWAM LIBERO policy：

```bash
make download-lerobot-fastwam-libero-policy
```

默认落盘：

```text
models/lerobot/fastwam/fastwam_libero_uncond_2cam224
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

## 使用下载的 policy

下载完成后可以指定：

```bash
export LEROBOT_POLICY_PATH="$PROJECT/models/lerobot/diffusion/diffusion_pusht"
```

然后按当前 inference smoke 入口测试：

```bash
make lerobot-infer-smoke
```

如果 policy 类型和 inference 脚本版本不匹配，优先记录错误并修 adapter，不要把 policy 复制到别的位置。

## 跑 GPU training

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

ACT / PushT：

```bash
export LEROBOT_STEPS=1000
export LEROBOT_BATCH_SIZE=8
export LEROBOT_NUM_WORKERS=4
export LEROBOT_LOG_FREQ=20
export LEROBOT_SAVE_FREQ=1000

make lerobot-train-act
```

Diffusion / PushT：

```bash
export LEROBOT_STEPS=1000
export LEROBOT_BATCH_SIZE=8

make lerobot-train-diffusion
```

SmolVLA / SO100：

```bash
export LEROBOT_STEPS=2000
export LEROBOT_BATCH_SIZE=8

make lerobot-train-smolvla
```

## 单机八卡长期训练

真实八卡入口使用 LeRobot 内部的 `accelerate.Accelerator`，而不是手写 DDP。

配置：

```text
configs/lerobot/train/svla_so100_smolvla_8gpu_long.sh
scripts/lerobot/run_train_accelerate.sh
scripts/lerobot/slurm_smolvla_8gpu_long.sbatch
```

直接在 8 卡节点上跑：

```bash
export LEROBOT_STEPS=20000
export LEROBOT_BATCH_SIZE=8        # per-process batch size; effective batch = 8 * 8
export LEROBOT_NUM_PROCESSES=8
export LEROBOT_SAVE_FREQ=1000

make lerobot-train-8gpu-smolvla
```

Slurm：

```bash
sbatch scripts/lerobot/slurm_smolvla_8gpu_long.sbatch
```

长期实验产物：

```text
runs/lerobot/svla_so100_smolvla_8gpu_long/<run_id>/
├── command.txt
├── backend_manifest.json
├── train_stdout.log
├── loss_summary.json
└── lerobot_output/
    └── checkpoints/
```

恢复训练：

```bash
export LEROBOT_RESUME=1
export LEROBOT_RESUME_CONFIG_PATH="$PROJECT/runs/lerobot/<run>/<id>/lerobot_output/checkpoints/<step>/train_config.json"
export LEROBOT_OUTPUT_DIR="$PROJECT/runs/lerobot/<new_or_same_run>/<id>/lerobot_output"
make lerobot-train-8gpu-smolvla
```

注意：LeRobot resume 对 batch size 和 world size 敏感。为了样本顺序完全一致，恢复时尽量保持 `LEROBOT_BATCH_SIZE` 和 `LEROBOT_NUM_PROCESSES` 不变。

## 推理链路

当前是离线单样本推理 smoke：读取本地 dataset sample，加载本地 policy/checkpoint，在 CUDA 上输出 action 形状、latency 和 evidence JSON。

Diffusion / PushT：

```bash
make lerobot-infer-diffusion
```

SmolVLA / SO100：

```bash
make lerobot-infer-smolvla
```

FastWAM / LIBERO：

```bash
make lerobot-infer-fastwam
```

FastWAM 注意事项：`data/fastwam/libero-fastwam` 目前是 LeRobot v2.1 数据。当前 LeRobot v3 loader 直接读需要转换一个 v3 副本，然后把 `LEROBOT_DATASET_ROOT` 指向转换后的 subset。

如果 SmolVLA 显存或 dataloader 压力偏大：

```bash
export LEROBOT_BATCH_SIZE=2
export LEROBOT_NUM_WORKERS=2

make lerobot-train-smolvla
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
docs/LEROBOT_MULTI_MODEL_PLAN.md
docs/OPEN_DATA_AND_EVAL_PLAN.md
```
