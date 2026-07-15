# Bootstrap: prepare a usable workspace

这份文档只回答一个问题：**一个新 checkout 怎么变成可以跑训练/推理的工作区？**

成熟项目通常把准备阶段拆成四步：

1. 准备目录和环境变量；
2. 准备 Python / CUDA 环境；
3. 下载数据、模型和上游源码；
4. 检查资产是否已经落到约定位置。

训练和推理命令见 [`TRAINING_AND_INFERENCE.md`](TRAINING_AND_INFERENCE.md)；这里不启动训练。

## 0. 进入项目

SCUT 共享盘默认：

```bash
export BASE=/mnt/gpu11_200T/dingxibo
export PROJECT=$BASE/EmbodiedAI-Demo-Pipeline
export CONDA=$BASE/miniconda3/bin/conda

cd "$PROJECT"
```

通用项目内路径：

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
export HF_HUB_DISABLE_XET=1
```

先创建项目内资产目录：

```bash
make prepare-dirs
```

## 1. 环境准备

### Core 工具环境

用于本项目轻量工具、schema、report 和测试：

```bash
"$CONDA" run -n embodied-core python -m pytest
```

如果环境不存在，按 [`ENVIRONMENT.md`](ENVIRONMENT.md) 创建。

### LeRobot 环境

用于 LeRobot 训练/推理：

```bash
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
```

如果要重建：

```bash
CONDA_EXE="$CONDA" LEROBOT_CREATE_CONDA=1 LEROBOT_CONDA_ENV=lerobot \
bash scripts/lerobot/install_lerobot_cluster.sh
```

### Custom FastWAM 环境

FastWAM overlay 通常使用独立 `fastwam` 环境：

```bash
CONDA_EXE="$CONDA" make prepare-env-custom-fastwam
conda activate fastwam
```

如果只准备源码 overlay、不安装 Python 包：

```bash
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

## 2. 资产准备：LeRobot 路线

下载 LeRobot 第一批数据、policy、FastWAM/LIBERO 数据和 base cache：

```bash
make prepare-assets-lerobot
```

落盘位置：

```text
data/lerobot/pusht/
data/lerobot/svla_so100_pickplace/
data/lerobot/libero-fastwam/v2.1/
data/lerobot/libero-fastwam/v3/

models/lerobot/diffusion/diffusion_pusht/
models/lerobot/smolvla/smolvla_base/
models/lerobot/fastwam/fastwam_libero_uncond_2cam224/

hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/
hf_cache/hub/models--google--umt5-xxl/
```

检查：

```bash
make check-assets-lerobot
```

## 3. 资产准备：Custom FastWAM 路线

下载 custom FastWAM 数据、release 权重，并准备 overlay：

```bash
make prepare-assets-custom-fastwam
```

落盘位置：

```text
data/custom/fastwam/libero-fastwam/
models/custom/fastwam/release/libero_uncond_2cam224.pt
models/custom/fastwam/release/libero_uncond_2cam224_dataset_stats.json
upstreams/FastWAM-realrobot/
upstreams/fastwam-realrobot-pipeline/
```

检查：

```bash
make check-assets-custom-fastwam
```

## 4. 资产准备：ImageWAM 路线

ImageWAM 部分资产可能受 Hugging Face 权限、网络和上游脚本影响。按需执行：

```bash
make prepare-assets-imagewam
```

落盘位置：

```text
upstreams/ImageWAM/
models/custom/imagewam/flux2_klein_4b_libero/
models/custom/imagewam/flux2/
```

检查：

```bash
make check-assets-imagewam
```

## 5. 一次性检查

如果你希望检查当前工作区已经准备了什么：

```bash
make check-assets-core
make check-assets-lerobot
make check-assets-custom-fastwam
```

全量检查：

```bash
make check-assets
```

注意：`check-assets` 会把 ImageWAM 也算进去；如果 ImageWAM 还没准备，它会报缺失，这是预期行为。

## 6. 准备完成后跑什么

LeRobot 训练/推理：

```bash
bash experiments/lerobot/pusht_act_smoke/launch.sh
bash experiments/lerobot/fastwam_libero_infer/launch.sh
```

Custom FastWAM 单机 8 卡随机初始化，优先用于手动验证和短试验：

```bash
conda activate fastwam
python experiments/custom/fastwam_realrobot_single8_random/run.py --dry-run
python experiments/custom/fastwam_realrobot_single8_random/run.py
```

Custom FastWAM 8 机随机初始化：

```bash
sbatch experiments/custom/fastwam_realrobot_8node_random/slurm.sbatch
```

更多实验入口见 [`experiments/README.md`](../experiments/README.md)。
