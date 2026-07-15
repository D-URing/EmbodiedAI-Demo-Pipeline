# EmbodiedAI Demo Pipeline

这是一个面向具身智能 demo 的工程基座。当前目标不是从零训练大模型，也不是马上接真机，而是把开源生态里的 **数据读取 → 模型训练/加载 → 推理 → 日志/评测证据 → demo 报告** 跑通，给团队后续开发留下稳定接口。

项目现在按两条主线理解：

| 主线 | 当前目标 | 入口 |
|---|---|---|
| LeRobot 主线 | 复刻 LeRobot data-to-train-to-inference，全流程先跑通 | [`pipelines/lerobot/`](pipelines/lerobot/) |
| Custom/FastWAM 主线 | 保留自拟模型/custom backend 路径，以 FastWAM 为第一个例子 | [`pipelines/custom_fastwam/`](pipelines/custom_fastwam/) |

Household/mock demo 仍保留，但它是应用展示层，不是当前训练能力验收主线。

## 当前最重要的状态

- LeRobot ACT/PushT 已在 SCUT `gpu11` 跑通真实 GPU training smoke；
- 已观察到 2-step loss 下降：`96.987 -> 83.351`；
- LeRobot PushT 数据已下载到 `data/lerobot/pusht`；
- FastWAM release 权重已下载到 `models/fastwam_release/`；
- FastWAM LIBERO 数据已下载并解压到 `data/fastwam/libero-fastwam/`；
- FastWAM 私有 realrobot overlay 还需要远端 GitHub 私有仓库权限。

如果你刚接手项目，按这个顺序读：

1. [`docs/README.md`](docs/README.md)；
2. [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)；
3. [`docs/STORAGE_AND_ARTIFACTS.md`](docs/STORAGE_AND_ARTIFACTS.md)；
4. [`pipelines/lerobot/README.md`](pipelines/lerobot/README.md)；
5. [`pipelines/custom_fastwam/README.md`](pipelines/custom_fastwam/README.md)。

## 仓库结构

```text
.
├── pipelines/
│   ├── lerobot/          # LeRobot 主线：dataset -> train/load -> inference -> report
│   └── custom_fastwam/   # 自拟/custom 主线：FastWAM release / overlay / realrobot
├── configs/
│   ├── lerobot/          # LeRobot 配置
│   ├── fastwam/          # FastWAM/custom 配置
│   ├── runs/             # household/mock demo 配置
│   └── profiles/         # smoke/dev/release profile
├── scripts/
│   ├── lerobot/          # LeRobot 下载、训练、推理、报告脚本
│   ├── fastwam/          # FastWAM 下载、overlay、训练报告脚本
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

在 GPU 节点上跑训练 smoke：

```bash
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/pusht"
export TORCH_HOME="$PROJECT/hf_cache/torch"

export LEROBOT_STEPS=1000
export LEROBOT_BATCH_SIZE=8
export LEROBOT_NUM_WORKERS=4
export LEROBOT_LOG_FREQ=20
export LEROBOT_SAVE_FREQ=1000

make lerobot-train-smoke
```

快速 2-step 验环境见 [`pipelines/lerobot/README.md`](pipelines/lerobot/README.md)。

## FastWAM / custom 主线

已下载：

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

FastWAM LIBERO 数据下载、v2.1/v3 格式说明和 private overlay 权限问题见 [`pipelines/custom_fastwam/README.md`](pipelines/custom_fastwam/README.md)。

## 常用 Make targets

| Target | 作用 |
|---|---|
| `make test` | core 单测 |
| `make validate` | mock task/config 校验 |
| `make demo` | 跑两个 mock demo |
| `make download-lerobot-artifacts` | 下载 LeRobot PushT |
| `make download-lerobot-diffusion-pusht-policy` | 下载 LeRobot diffusion PushT 预训练 policy |
| `make lerobot-data-smoke` | LeRobot dataset inspection |
| `make lerobot-train-smoke` | LeRobot ACT/PushT GPU training smoke |
| `make lerobot-infer-smoke` | LeRobot offline inference smoke |
| `make download-fastwam-artifacts` | 下载 FastWAM release 权重和 stats |
| `make fastwam-train-smoke` | FastWAM custom smoke，依赖 overlay 准备 |
| `make demo-chain-fastwam` | 将 FastWAM 训练产物整理成报告 |

## 当前边界

- LeRobot ACT/PushT 是当前可交差的真实训练 demo；
- FastWAM release 权重和 LIBERO 数据已准备，但 private overlay 还没在 SCUT 完整跑通；
- FastWAM LIBERO 数据是 LeRobot v2.1，当前 LeRobot v3 loader 直接读需要转换；
- household/mock demo 是展示层，不代表模型能力；
- viewer、真实仿真器、真机闭环暂缓。
