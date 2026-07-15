# EmbodiedAI Demo Pipeline

这是一个面向具身智能 demo 的工程基座。当前目标不是从零训练大模型，也不是马上接真机，而是把开源生态里的 **数据读取 → 模型训练/加载 → 推理 → 日志/评测证据 → demo 报告** 跑通，给团队后续开发留下稳定接口。

项目现在按两条主线理解，其中第二条线已经从单一 FastWAM 调整为可扩展的 Custom WAM 后端族：

| 主线 | 当前目标 | 入口 |
|---|---|---|
| LeRobot 主线 | 复刻 LeRobot data-to-train-to-inference，并真实训练多个 policy | [`pipelines/lerobot/`](pipelines/lerobot/) |
| Custom WAM 主线 | 保留自拟模型/custom backend 路径，FastWAM 和 ImageWAM 并列接入 | [`pipelines/custom_wam/`](pipelines/custom_wam/) |

Household/mock demo 仍保留，但它是应用展示层，不是当前训练能力验收主线。

## 当前最重要的状态

- LeRobot ACT/PushT 已在 SCUT `gpu11` 跑通真实 GPU training smoke；
- 已观察到 2-step loss 下降：`96.987 -> 83.351`；
- LeRobot PushT 数据已下载到 `data/lerobot/pusht`；
- LeRobot 多模型训练 profile 已补齐：ACT、Diffusion、SmolVLA；
- LeRobot 开源 policy 下载入口已补齐：Diffusion PushT、SmolVLA base、FastWAM LIBERO；
- FastWAM release 权重已下载到 `models/fastwam_release/`；
- FastWAM LIBERO 数据已下载并解压到 `data/fastwam/libero-fastwam/`；
- ImageWAM 已作为 `custom_wam/imagewam` 后端接入，默认走 FLUX.2 4B + LIBERO；
- FastWAM 私有 realrobot overlay 还需要远端 GitHub 私有仓库权限。

如果你刚接手项目，按这个顺序读：

1. [`docs/README.md`](docs/README.md)；
2. [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)；
3. [`docs/STORAGE_AND_ARTIFACTS.md`](docs/STORAGE_AND_ARTIFACTS.md)；
4. [`pipelines/lerobot/README.md`](pipelines/lerobot/README.md)；
5. [`docs/LEROBOT_MULTI_MODEL_PLAN.md`](docs/LEROBOT_MULTI_MODEL_PLAN.md)；
6. [`docs/OPEN_DATA_AND_EVAL_PLAN.md`](docs/OPEN_DATA_AND_EVAL_PLAN.md)；
7. [`pipelines/custom_wam/README.md`](pipelines/custom_wam/README.md)；
8. [`docs/IMAGEWAM_INTEGRATION.md`](docs/IMAGEWAM_INTEGRATION.md)。

## 仓库结构

```text
.
├── pipelines/
│   ├── lerobot/          # LeRobot 主线：dataset -> train/load -> inference -> report
│   ├── custom_wam/       # 自拟/custom WAM 后端族：FastWAM / ImageWAM / future backends
│   └── custom_fastwam/   # 兼容入口：历史 FastWAM 说明
├── configs/
│   ├── lerobot/          # LeRobot 配置
│   ├── fastwam/          # FastWAM/custom 配置
│   ├── imagewam/         # ImageWAM/custom 配置
│   ├── runs/             # household/mock demo 配置
│   └── profiles/         # smoke/dev/release profile
├── scripts/
│   ├── lerobot/          # LeRobot 下载、训练、推理、报告脚本
│   ├── fastwam/          # FastWAM 下载、overlay、训练报告脚本
│   ├── imagewam/         # ImageWAM 下载、上游源码、训练/评测 wrapper
│   └── reference/        # 外部参考项目
├── src/embodied_demo/    # 本项目 core：schema、CLI、mock runner、report
├── tasks/                # household/mock task 定义
├── demo_chains/          # evidence/report 链路定义
├── docs/                 # 文档入口与长说明
└── references/           # 上游 pin、模型 registry
```

以下目录是本地/集群资产池，pipeline 只引用它们，不拥有它们：

```text
data/        # 全局 dataset 池
models/      # 全局 model / checkpoint / release 权重池
checkpoints/
runs/
artifacts/
upstreams/
hf_cache/
```

其中 `data/README.md` 和 `models/README.md` 会提交 Git，用来说明目录职责；真实数据和权重仍然被忽略。详细规则见 [`docs/STORAGE_AND_ARTIFACTS.md`](docs/STORAGE_AND_ARTIFACTS.md)。

## 本地 core / mock 快速检查

本地 core 环境只用于 schema、mock、报告和测试，不安装 CUDA、LeRobot、FastWAM、Isaac 或真机 SDK。

```bash
make setup
make doctor
make test
make validate
```

运行 mock demo：

```bash
make demo
make demo-extended
```

产物写到：

```text
runs/<run_name>/<run_id>/
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

当前已准备两个 conda 环境：

```text
embodied-core
lerobot
```

验证：

```bash
"$CONDA" run -n embodied-core python -m pytest
```

## LeRobot 主线

进入环境：

```bash
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
```

验证 dataset：

```bash
export LEROBOT_ALLOW_DOWNLOAD=0
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/pusht"

make lerobot-data-smoke
```

下载第一轮 LeRobot 资产：

```bash
make download-lerobot-pusht-dataset
make download-lerobot-svla-so100-pickplace-dataset
make download-lerobot-diffusion-pusht-policy
make download-lerobot-smolvla-base-policy
make download-lerobot-fastwam-libero-policy
```

在 GPU 节点上跑多模型训练：

```bash
make lerobot-train-act
make lerobot-train-diffusion
make lerobot-train-smolvla
```

单机八卡长期实验：

```bash
make lerobot-train-8gpu-smolvla
```

对应配置：

```text
configs/lerobot/train/svla_so100_smolvla_8gpu_long.sh
scripts/lerobot/run_train_accelerate.sh
scripts/lerobot/slurm_smolvla_8gpu_long.sbatch
```

推理：

```bash
make lerobot-infer-diffusion
make lerobot-infer-smolvla
make lerobot-infer-fastwam
```

快速 2-step 验环境见 [`pipelines/lerobot/README.md`](pipelines/lerobot/README.md)。

## Custom WAM 主线

Custom WAM 当前包含 FastWAM 与 ImageWAM。FastWAM 已下载：

```text
models/fastwam_release/libero_uncond_2cam224.pt
models/fastwam_release/libero_uncond_2cam224_dataset_stats.json
data/fastwam/libero-fastwam/
```

LeRobot 预训练 policy 也应放在全局 `models/` 池中，例如：

```text
models/lerobot/diffusion/diffusion_pusht/
```

下载命令：

```bash
make download-lerobot-diffusion-pusht-policy
```

重新下载 release 权重：

```bash
make download-fastwam-artifacts
```

FastWAM LIBERO 数据下载、v2.1/v3 格式说明和 private overlay 权限问题见 [`pipelines/custom_wam/fastwam/README.md`](pipelines/custom_wam/fastwam/README.md)。

ImageWAM 第一阶段默认 FLUX.2 4B + LIBERO：

```bash
make prepare-imagewam-upstream
make download-imagewam-artifacts
make download-imagewam-flux2-base
make imagewam-train-smoke
```

路径：

```text
upstreams/ImageWAM/
models/imagewam/flux2_klein_4b_libero/
models/imagewam/flux2/
runs/imagewam/
```

说明见 [`pipelines/custom_wam/imagewam/README.md`](pipelines/custom_wam/imagewam/README.md)。

## 常用 Make targets

| Target | 作用 |
|---|---|
| `make test` | core 单测 |
| `make validate` | mock task/config 校验 |
| `make demo` | 跑两个 mock demo |
| `make download-lerobot-artifacts` | 下载 LeRobot PushT |
| `make download-lerobot-svla-so100-pickplace-dataset` | 下载 SmolVLA SO100 pick-place 数据 |
| `make download-lerobot-diffusion-pusht-policy` | 下载 LeRobot diffusion PushT 预训练 policy |
| `make download-lerobot-smolvla-base-policy` | 下载 LeRobot SmolVLA base policy |
| `make download-lerobot-fastwam-libero-policy` | 下载 LeRobot-compatible FastWAM LIBERO policy |
| `make lerobot-data-smoke` | LeRobot dataset inspection |
| `make lerobot-train-act` | LeRobot ACT/PushT training |
| `make lerobot-train-diffusion` | LeRobot Diffusion/PushT training |
| `make lerobot-train-smolvla` | LeRobot SmolVLA/SO100 fine-tune |
| `make lerobot-train-8gpu-smolvla` | LeRobot SmolVLA/SO100 单机八卡长期训练 |
| `make lerobot-infer-smoke` | LeRobot offline inference，使用 `LEROBOT_INFER_CONFIG` |
| `make lerobot-infer-diffusion` | Diffusion/PushT offline inference |
| `make lerobot-infer-smolvla` | SmolVLA/SO100 offline inference |
| `make lerobot-infer-fastwam` | FastWAM/LIBERO offline inference，需 v3 数据 |
| `make download-data-rovid20k` | 下载 RoVid-X 实用子集 |
| `make download-data-xperience10m-sample` | 下载 Xperience-10M sample |
| `make download-fastwam-artifacts` | 下载 FastWAM release 权重和 stats |
| `make fastwam-train-smoke` | FastWAM custom smoke，依赖 overlay 准备 |
| `make demo-chain-fastwam` | 将 FastWAM 训练产物整理成报告 |

## 当前边界

- LeRobot ACT/PushT 是当前已验证真实训练链路；
- Diffusion/PushT 与 SmolVLA/SO100 已进入可运行 profile，待集群执行验证；
- FastWAM release 权重和 LIBERO 数据已准备，但 private overlay 还没在 SCUT 完整跑通；
- FastWAM LIBERO 数据是 LeRobot v2.1，当前 LeRobot v3 loader 直接读需要转换；
- household/mock demo 是展示层，不代表模型能力；
- viewer、真实仿真器、真机闭环暂缓。
