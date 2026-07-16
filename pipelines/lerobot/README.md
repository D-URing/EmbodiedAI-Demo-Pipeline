# LeRobot Pipeline

LeRobot 是当前第一阶段主线：验证开源 LeRobot 生态下的 dataset read → policy train/load → offline inference → evidence report。

具体训练/推理命令以 [`../../docs/TRAINING_AND_INFERENCE.md`](../../docs/TRAINING_AND_INFERENCE.md) 为准。本文件只保留路线说明、入口索引和当前状态。

## 当前可用链路

| 链路 | 类型 | 状态 | 入口 |
|---|---|---|---|
| ACT / PushT | 训练 | 已在 SCUT `gpu11` 验证，2-step loss 下降 | `experiments/lerobot/pusht_act_smoke/launch.sh` |
| Diffusion / PushT | 训练 | 入口已准备 | `experiments/lerobot/pusht_diffusion_train/launch.sh` |
| SmolVLA / SO100 | 单机八卡/多机训练 | 入口已准备 | `experiments/lerobot/smolvla_so100_8gpu_long/launch.sh` |
| pi05 / SO100 | 单机八卡训练测速 | 入口已准备，待集群实测 | `experiments/lerobot/pi05_so100_8gpu_probe/run.py` |
| Diffusion / PushT | 推理 | 入口已准备，依赖本地 policy | `experiments/lerobot/diffusion_pusht_infer/launch.sh` |
| SmolVLA / SO100 | 推理 | 入口已准备，依赖本地 policy/base | `experiments/lerobot/smolvla_so100_infer/launch.sh` |
| pi05 / SO100 | 推理 | 入口已准备，依赖本地 pi05 base/checkpoint | `experiments/lerobot/pi05_so100_infer/run.py` |
| FastWAM / LIBERO | 推理 | 已在 SCUT `gpu11` 验证 CUDA inference | `experiments/lerobot/fastwam_libero_infer/launch.sh` |

已验证 FastWAM evidence：

```text
runs/experiments/lerobot/fastwam_libero_infer/20260715-210113/inference_evidence.json
policy_type=fastwam
device=cuda
action.shape=[1, 7]
latency_ms=7931.62
```

## 环境

```bash
export BASE=/mnt/gpu11_200T/dingxibo
export PROJECT=$BASE/EmbodiedAI-Demo-Pipeline
export CONDA=$BASE/miniconda3/bin/conda

cd "$PROJECT"
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot

export PROJECT_ROOT="$PROJECT"
export EMBODIED_DATA_ROOT="$PROJECT/data"
export EMBODIED_MODEL_ROOT="$PROJECT/models"
export EMBODIED_RUN_ROOT="$PROJECT/runs"
export HF_HOME="$PROJECT/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PROJECT/hf_cache/torch"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

重建环境：

```bash
CONDA_EXE="$CONDA" LEROBOT_CREATE_CONDA=1 LEROBOT_CONDA_ENV=lerobot \
bash scripts/lerobot/install_lerobot_cluster.sh
```

新架构 GPU / CUDA 13 wheel 节点可使用：

```bash
CONDA_EXE=/opt/conda/bin/conda \
LEROBOT_CREATE_CONDA=1 \
LEROBOT_CONDA_ENV=lerobot-sm120 \
LEROBOT_INSTALL_NO_DEPS=1 \
LEROBOT_FORCE_OPENCV_HEADLESS=1 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
LEROBOT_TORCH_SPEC='torch==2.13.0+cu130' \
LEROBOT_TORCHVISION_SPEC='torchvision==0.28.0+cu130' \
bash scripts/lerobot/install_lerobot_cluster.sh
```

SCUT `gpu11` 需要注意：

- `ffmpeg=6.*`，避免旧 glibc 节点上 `ffmpeg=8` native ABI 问题；
- FastWAM policy 需要 `transformers` 和 `diffusers`；
- 推理大模型时设置 `HF_HOME=$PROJECT/hf_cache`，避免访问默认用户 cache。

## 资产

```bash
make download-lerobot-pusht-dataset
make download-lerobot-svla-so100-pickplace-dataset
make download-lerobot-diffusion-pusht-policy
make download-lerobot-smolvla-base-policy
make download-lerobot-pi05-base-policy
make download-lerobot-pi05-runtime-cache
make download-lerobot-fastwam-libero-policy
make download-lerobot-fastwam-libero-dataset
make convert-lerobot-fastwam-libero-v3
make download-lerobot-fastwam-base-cache
```

默认路径：

```text
data/lerobot/pusht/
data/lerobot/svla_so100_pickplace/
data/lerobot/libero-fastwam/v2.1/
data/lerobot/libero-fastwam/v3/

models/lerobot/diffusion/diffusion_pusht/
models/lerobot/smolvla/smolvla_base/
models/lerobot/pi05/pi05_base/
models/lerobot/fastwam/fastwam_libero_uncond_2cam224/

hf_cache/hub/models--google--paligemma-3b-pt-224/
hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/
hf_cache/hub/models--google--umt5-xxl/
```

pi05 运行时还会通过 tokenizer processor 读取 `google/paligemma-3b-pt-224` 的 tokenizer/config。该 repo 可能是 gated；如果下载报 `Access denied`，先完成 Hugging Face 访问申请，并在集群侧 `hf auth login` 或设置 `HF_TOKEN` 后重试 `make download-lerobot-pi05-runtime-cache`。

LeRobot 路线不直接读写 custom/FastWAM 数据：

```text
data/custom/fastwam/libero-fastwam/
```

那是 custom pipeline 的输入。

## 训练

ACT / PushT：

```bash
bash experiments/lerobot/pusht_act_smoke/launch.sh
```

Diffusion / PushT：

```bash
bash experiments/lerobot/pusht_diffusion_train/launch.sh
```

SmolVLA / SO100 单机八卡：

```bash
export LEROBOT_NUM_PROCESSES=8
export LEROBOT_BATCH_SIZE=8
export LEROBOT_STEPS=20000

bash experiments/lerobot/smolvla_so100_8gpu_long/launch.sh
```

pi05 / SO100 单机八卡测速探针：

```bash
export LEROBOT_NUM_PROCESSES=8
export LEROBOT_BATCH_SIZE=1
export LEROBOT_STEPS=200

python experiments/lerobot/pi05_so100_8gpu_probe/run.py
```

如果只想先排错，不想把首次 `torch.compile` 编译耗时混入测速：

```bash
LEROBOT_POLICY_COMPILE_MODEL=false python experiments/lerobot/pi05_so100_8gpu_probe/run.py
```

Slurm：

```bash
sbatch experiments/lerobot/smolvla_so100_8gpu_long/slurm.sbatch
```

## 推理

Diffusion / PushT：

```bash
bash experiments/lerobot/diffusion_pusht_infer/launch.sh
```

SmolVLA / SO100：

```bash
bash experiments/lerobot/smolvla_so100_infer/launch.sh
```

pi05 / SO100：

```bash
python experiments/lerobot/pi05_so100_infer/run.py
```

FastWAM / LIBERO：

```bash
export HF_HOME="$PROJECT/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash experiments/lerobot/fastwam_libero_infer/launch.sh
```

## 输出

训练输出：

```text
runs/experiments/lerobot/<experiment>/<run_id>/
├── command.txt
├── backend_manifest.json
├── train_stdout.log
├── loss_summary.json
├── speed_summary.json
└── lerobot_output/
```

推理输出：

```text
runs/experiments/lerobot/<experiment>/<run_id>/
├── config.sh
└── inference_evidence.json
```

## 排障

| 问题 | 处理 |
|---|---|
| `torchcodec` / `libavutil` / `glibc` 报错 | 固定 `ffmpeg=6.*` |
| FastWAM policy 找不到 `transformers` / `diffusers` | 安装 fastwam extra 或 `pip install transformers diffusers` |
| FastWAM policy 离线访问 Wan/T5 | `make download-lerobot-fastwam-base-cache`，并设置 `HF_HOME=$PROJECT/hf_cache` |
| FastWAM LIBERO 数据格式不对 | `make convert-lerobot-fastwam-libero-v3` |
| 计算节点不能联网 | 先在管理节点下载到项目内，再设置 offline 环境变量 |
