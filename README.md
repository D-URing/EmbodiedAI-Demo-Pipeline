# EmbodiedAI Demo Pipeline

这是一个面向具身智能工程验证的公开项目基座。当前目标是把开源生态里的 **数据读取 → 模型训练/加载 → 推理 → 日志/评测证据 → 报告** 跑通，给后续模型开发、集群实验和评测接入留下稳定结构。

项目维护两条主线：

| 主线 | 当前目标 | 入口 |
|---|---|---|
| LeRobot | 复刻 LeRobot data-to-train-to-inference，并真实训练/推理多个 policy | [`pipelines/lerobot/`](pipelines/lerobot/) |
| Custom WAM | 保留自建模型/custom backend 路径，FastWAM 和 ImageWAM 并列接入 | [`pipelines/custom/`](pipelines/custom/) |

## 我应该跑哪个入口？

| 目标 | 推荐入口 | 说明 |
|---|---|---|
| 查看所有常用训练入口 | `./run.py list` | 顶层统一启动入口，别再手写长命令 |
| cluster_120 两节点 LeRobot/pi05 | `./run.py pi05-2node` | 自动选择 config/profile/conda python/run_id |
| cluster_120 两节点 FastWAM smoke | `./run.py fastwam-2node-smoke` | 已实测 2 节点 × 8 卡；默认 1 step 连通性验证 |
| cluster_120 两节点 FastWAM pilot | `./run.py fastwam-2node-pilot` | smoke 通过后再跑短实验 |
| SCUT gpu11 单机 pi05/FastWAM | `./run.py pi05-scut` / `./run.py fastwam-scut` | 单机 8 卡入口 |
| 快速证明 LeRobot 能训练出 loss | `bash experiments/lerobot/pusht_act_smoke/launch.sh` | 轻量 ACT/PushT 真实训练 |
| 跑 LeRobot VLA 微调 | `bash experiments/lerobot/smolvla_so100_8gpu_long/launch.sh` | 单机 8 卡 SmolVLA 长训模板 |
| 跑 LeRobot 离线推理 | `bash experiments/lerobot/*_infer/launch.sh` | 只证明 data-to-policy inference，不是环境评测 |
| 跑 custom FastWAM 真实训练链路 | `python experiments/custom/fastwam_realrobot_single8_random/run.py` | 当前已在 gpu11 跑通 8 卡真实训练 |
| 底层 SSH launcher 排障 | `python scripts/distributed/ssh_launch.py --config ... --profile ...` | 只在需要调试 profile/节点时使用 |
| 做 custom FastWAM release 微调 | 复制 `fastwam_realrobot_single8_random` 新建实验，改 `init=release` 和 `resume=...` | 不要把默认 `init=random` 说成正式微调 |
| 准备多机 FastWAM | `experiments/custom/fastwam_realrobot_8node_random/` | 等单机稳定后再用 |
| 研究 ImageWAM | `experiments/custom/imagewam_flux2_4b_libero_pilot/` | 候选路线，不是当前主线交付 |

## 当前状态

- LeRobot ACT/PushT 已在 SCUT `gpu11` 跑通真实 GPU training smoke，并观察到 2-step loss 下降：`96.987 -> 83.351`；
- LeRobot FastWAM/LIBERO 已在 SCUT `gpu11` 跑通 CUDA inference smoke，输出 `action.shape=[1, 7]`；
- LeRobot 训练/推理入口已覆盖 ACT、Diffusion、SmolVLA、FastWAM/LIBERO；
- Custom FastWAM realrobot overlay 已在 SCUT `gpu11` 跑通单机 8 卡真实训练，支持 `release|base|random` 初始化和多节点启动；
- ImageWAM 已作为 `custom/imagewam` 候选后端接入，默认走 FLUX.2 4B + LIBERO，但不作为当前主线验收；
- 不维护 CPU toy trainer，不维护本地符号 rollout，不声明仿真或真机 closed-loop 成功。

## 阅读顺序

1. [`docs/README.md`](docs/README.md)
2. [`docs/BOOTSTRAP.md`](docs/BOOTSTRAP.md)
3. [`docs/TRAINING_AND_INFERENCE.md`](docs/TRAINING_AND_INFERENCE.md)
4. [`docs/DISTRIBUTED_TRAINING.md`](docs/DISTRIBUTED_TRAINING.md)
5. [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)
6. [`docs/STORAGE_AND_ARTIFACTS.md`](docs/STORAGE_AND_ARTIFACTS.md)
7. [`pipelines/lerobot/README.md`](pipelines/lerobot/README.md)
8. [`pipelines/custom/README.md`](pipelines/custom/README.md)
9. [`experiments/README.md`](experiments/README.md)

## 仓库结构

```text
.
├── pipelines/
│   ├── lerobot/          # LeRobot 主线：dataset -> train/load -> inference -> report
│   └── custom/           # Custom WAM 后端族：FastWAM / ImageWAM / future backends
├── run.py                # 项目顶层统一启动入口：./run.py list / ./run.py pi05-2node
├── experiments/          # 训练/推理启动入口：优先 config.yaml + run.py，必要时保留 launch.sh/slurm
├── configs/
│   ├── launch/           # 顶层启动别名：把 config/profile/python 路径收敛到 YAML
│   ├── lerobot/          # LeRobot 配置
│   ├── fastwam/          # FastWAM/custom 配置
│   ├── imagewam/         # ImageWAM/custom 配置
│   └── distributed/      # SSH 多机 profile
├── scripts/
│   ├── lerobot/          # LeRobot 下载、训练、推理、报告脚本
│   ├── fastwam/          # FastWAM 下载、overlay、训练报告脚本
│   ├── imagewam/         # ImageWAM 下载、上游源码、训练/评测 wrapper
│   ├── distributed/      # trainer0 -> 多节点 SSH 启动器
│   └── reference/        # 外部参考项目
├── src/embodied_demo/    # 轻量 core：schema、CLI、evidence report
├── demo_chains/          # evidence/report 链路定义
├── docs/                 # 文档入口与长说明
└── references/           # 上游 pin、模型 registry
```

本地/集群资产池：

```text
data/        # dataset pool
models/      # model / checkpoint / release weight pool
checkpoints/
runs/
artifacts/
upstreams/
hf_cache/
```

## 环境怎么分

当前有两个已经稳定使用的 conda 环境：

| 环境 | 用途 | 是否含 CUDA 重依赖 |
|---|---|---|
| `embodied-core` | 本项目轻量 Python 工具、schema、报告、测试 | 否 |
| `lerobot` | LeRobot 训练/推理、torch/torchcodec、FastWAM policy extra | 是 |

Custom FastWAM overlay 通常另建 `fastwam` 环境，因为它来自外部 overlay，依赖和 LeRobot 不完全一致。也就是说：

- 本地/CI 检查：`embodied-core`
- LeRobot 训练/推理：`lerobot`
- Custom FastWAM 训练：`fastwam`

这不是重复建设，而是避免把 LeRobot、FastWAM、ImageWAM、未来仿真器依赖塞进一个不可维护的大环境。

## 本地 core 检查

```bash
make setup
make doctor
make test
make validate
```

## 准备数据、模型和 cache

完整准备指引见 [`docs/BOOTSTRAP.md`](docs/BOOTSTRAP.md)。常用入口：

```bash
make prepare-dirs
make prepare-assets-lerobot
make prepare-assets-custom-fastwam
make check-assets-lerobot
make check-assets-custom-fastwam
```

## SCUT / NVIDIA 集群基础变量

```bash
export BASE=/mnt/gpu11_200T/dingxibo
export PROJECT=$BASE/EmbodiedAI-Demo-Pipeline
export CONDA=$BASE/miniconda3/bin/conda

cd "$PROJECT"

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

## LeRobot 常用入口

```bash
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
```

资产准备：

```bash
make download-lerobot-pusht-dataset
make download-lerobot-svla-so100-pickplace-dataset
make download-lerobot-diffusion-pusht-policy
make download-lerobot-smolvla-base-policy
make download-lerobot-fastwam-libero-policy
make download-lerobot-fastwam-libero-dataset
make convert-lerobot-fastwam-libero-v3
make download-lerobot-fastwam-base-cache
```

训练：

```bash
bash experiments/lerobot/pusht_act_smoke/launch.sh
bash experiments/lerobot/pusht_diffusion_train/launch.sh
bash experiments/lerobot/smolvla_so100_8gpu_long/launch.sh
```

推理：

```bash
bash experiments/lerobot/diffusion_pusht_infer/launch.sh
bash experiments/lerobot/smolvla_so100_infer/launch.sh
bash experiments/lerobot/fastwam_libero_infer/launch.sh
```

## Custom FastWAM 真实训练

准备 overlay、数据和 FastWAM conda 环境。计算节点若不能联网，请在管理节点执行这一步，落盘到共享项目目录：

```bash
make prepare-assets-custom-fastwam
CONDA_EXE="$(command -v conda)" make prepare-env-custom-fastwam
conda activate fastwam
```

`make prepare-assets-custom-fastwam` 会把 custom FastWAM 训练需要的真实资产放到项目内，包括 LIBERO 数据、release ckpt/stats、Wan2.2 VAE/T5 text encoder 和 Wan2.1 tokenizer。

单机 8 卡随机初始化，优先用于手动验证：

```bash
python experiments/custom/fastwam_realrobot_single8_random/run.py --dry-run
python experiments/custom/fastwam_realrobot_single8_random/run.py
```

`run.py` 会在训练前自动执行 FastWAM 必需的 text embedding cache 预计算：

```text
runs/experiments/custom/fastwam_realrobot_single8_random/<run_id>/precompute_text_embeds.log
runs/experiments/custom/fastwam_realrobot_single8_random/<run_id>/precompute_text_embeds_command.txt
upstreams/FastWAM-realrobot/data/text_embeds_cache/libero/*.pt
```

已有缓存时会以 `overwrite=false` 跳过，不会重复计算。

Slurm 启动 8 机 × 8 卡：

```bash
sbatch experiments/custom/fastwam_realrobot_8node_random/slurm.sbatch
```

单机八卡实验默认：

```text
FASTWAM_MODE=pilot
FASTWAM_RECIPE=joint_base
FASTWAM_TASK_NAME=libero_joint_2cam224_1e-4
FASTWAM_INIT=random
FASTWAM_NNODES=1
FASTWAM_GPUS_PER_NODE=8
FASTWAM_MODEL_ID=Wan-AI/Wan2.2-TI2V-5B
FASTWAM_TOKENIZER_MODEL_ID=Wan-AI/Wan2.1-T2V-1.3B
FASTWAM_PRECOMPUTE_TEXT_EMBEDS=auto
```

`FASTWAM_INIT=random` 会显式传入：

```text
resume=null
model.skip_dit_load_from_pretrain=true
model.action_dit_pretrained_path=null
```

如果不是 Slurm，多机任务需要每台机器分别启动：

```bash
export FASTWAM_NNODES=8
export FASTWAM_NODE_RANK=<0-7>
export FASTWAM_MASTER_ADDR=<rank0-host-or-ip>
export FASTWAM_MASTER_PORT=29500
export FASTWAM_GPUS_PER_NODE=8
export FASTWAM_RUN_ID=<shared-run-id>

bash experiments/custom/fastwam_realrobot_8node_random/launch.sh
```

结果写到：

```text
runs/experiments/custom/fastwam_realrobot_8node_random/<run_id>/
upstreams/FastWAM-realrobot/runs/<task>/<run_id>/
```

## Make targets

Make 只作为环境、下载、转换和检查入口；训练/推理实验优先使用 `experiments/<route>/<experiment>/run.py`。

| Target | 作用 |
|---|---|
| `make test` | core 单测 |
| `make validate` | 脚本语法检查 + schema export |
| `make prepare-dirs` | 创建项目内资产目录 |
| `make prepare-assets-lerobot` | 下载第一批 LeRobot 数据、policy 和 FastWAM base cache |
| `make prepare-assets-custom-fastwam` | 下载 custom FastWAM 数据、release 权重、Wan runtime assets 并准备 overlay |
| `make prepare-assets-imagewam` | 准备 ImageWAM upstream、checkpoint 和 FLUX.2 base |
| `make check-assets` | 检查 data/models/hf_cache/upstreams 是否按约定落盘 |
| `make lerobot-check-scripts` | 检查 LeRobot wrapper 和 parser |
| `make fastwam-check-scripts` | 检查 FastWAM wrapper 和 parser |
| `make imagewam-check-scripts` | 检查 ImageWAM wrapper |
| `make experiments-check-scripts` | 检查 experiments 启动脚本 |
| `make download-lerobot-pusht-dataset` | 下载 LeRobot PushT |
| `make download-lerobot-svla-so100-pickplace-dataset` | 下载 SmolVLA SO100 pick-place 数据 |
| `make download-lerobot-fastwam-libero-dataset` | 下载 FastWAM LIBERO 原始数据到 LeRobot 路线 |
| `make convert-lerobot-fastwam-libero-v3` | 转换 LeRobot 路线 FastWAM LIBERO v2.1 → v3.0 |
| `make download-lerobot-fastwam-base-cache` | 下载 LeRobot FastWAM 推理所需 Wan/T5 base cache |
| `make download-custom-fastwam-libero-dataset` | 下载 FastWAM LIBERO 原始数据到 custom/FastWAM 路线 |
| `make download-fastwam-artifacts` | 下载 FastWAM release 权重/stats 以及 Wan2.2 VAE/text encoder、Wan2.1 tokenizer |
| `make prepare-imagewam-upstream` | 准备 ImageWAM 官方源码 |
| `make download-imagewam-artifacts` | 下载 ImageWAM release checkpoint |
| `make download-imagewam-flux2-base` | 下载 FLUX.2 4B base / AE |
