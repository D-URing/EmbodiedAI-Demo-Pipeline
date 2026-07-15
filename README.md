# EmbodiedAI Demo Pipeline

这是一个面向具身智能 demo 的工程基座。当前目标不是从零训练大模型，也不是马上接真机，而是把开源生态里的 **数据读取 → 模型训练/加载 → 推理 → 日志/评测证据 → demo 报告** 跑通，给团队后续开发留下稳定接口。

项目现在按两条主线理解，其中第二条线已经从单一 FastWAM 调整为可扩展的 Custom WAM 后端族：

| 主线 | 当前目标 | 入口 |
|---|---|---|
| LeRobot 主线 | 复刻 LeRobot data-to-train-to-inference，并真实训练多个 policy | [`pipelines/lerobot/`](pipelines/lerobot/) |
| Custom WAM 主线 | 保留自拟模型/custom backend 路径，FastWAM 和 ImageWAM 并列接入 | [`pipelines/custom/`](pipelines/custom/) |

Household/mock demo 仍保留，但它是应用展示层，不是当前训练能力验收主线。

## 当前最重要的状态

- LeRobot ACT/PushT 已在 SCUT `gpu11` 跑通真实 GPU training smoke；
- 已观察到 2-step loss 下降：`96.987 -> 83.351`；
- LeRobot PushT 数据已下载到 `data/lerobot/pusht`；
- LeRobot 多模型训练 profile 已补齐：ACT、Diffusion、SmolVLA；
- LeRobot 开源 policy 下载入口已补齐：Diffusion PushT、SmolVLA base、FastWAM LIBERO；
- LeRobot FastWAM/LIBERO 已在 SCUT `gpu11` 跑通 CUDA inference smoke，输出 `action.shape=[1, 7]`；
- FastWAM release 权重已下载到 `models/custom/fastwam/release/`；
- FastWAM LIBERO 数据已按路线拆分：`data/custom/fastwam/libero-fastwam/` 给 custom/FastWAM，`data/lerobot/libero-fastwam/` 给 LeRobot；
- LeRobot FastWAM 需要的 Wan/T5 base cache 已下载到 `hf_cache/hub/`；
- ImageWAM 已作为 `custom/imagewam` 后端接入，默认走 FLUX.2 4B + LIBERO；
- FastWAM 私有 realrobot overlay 还需要远端 GitHub 私有仓库权限。

如果你刚接手项目，按这个顺序读：

1. [`docs/README.md`](docs/README.md)；
2. [`docs/TRAINING_AND_INFERENCE.md`](docs/TRAINING_AND_INFERENCE.md)；
3. [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)；
4. [`docs/STORAGE_AND_ARTIFACTS.md`](docs/STORAGE_AND_ARTIFACTS.md)；
5. [`pipelines/lerobot/README.md`](pipelines/lerobot/README.md)；
6. [`pipelines/custom/README.md`](pipelines/custom/README.md)；
7. [`experiments/README.md`](experiments/README.md)。

## 仓库结构

```text
.
├── pipelines/
│   ├── lerobot/          # LeRobot 主线：dataset -> train/load -> inference -> report
│   ├── custom/           # 自拟/custom WAM 后端族：FastWAM / ImageWAM / future backends
├── experiments/          # 训练/推理启动入口：config.sh + launch.sh + 可选 slurm.sbatch
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
embodied-demo run --config configs/runs/tabletop_sorting_mock.yaml
embodied-demo run --config configs/runs/towel_folding_mock.yaml
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

## 当前训练 / 推理入口

完整命令见 [`docs/TRAINING_AND_INFERENCE.md`](docs/TRAINING_AND_INFERENCE.md)。常用入口如下。

### LeRobot 主线

进入环境：

```bash
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
```

准备第一批资产：

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

已验证 FastWAM/LIBERO 推理证据：

```text
runs/experiments/lerobot/fastwam_libero_infer/20260715-210113/inference_evidence.json
policy_type=fastwam
device=cuda
action.shape=[1, 7]
latency_ms=7931.62
```

### Custom WAM 主线

Custom WAM 当前包含 FastWAM 与 ImageWAM。FastWAM 资产：

```text
models/custom/fastwam/release/libero_uncond_2cam224.pt
models/custom/fastwam/release/libero_uncond_2cam224_dataset_stats.json
data/custom/fastwam/libero-fastwam/
```

Custom FastWAM：

```bash
make download-custom-fastwam-libero-dataset
make download-fastwam-artifacts
bash experiments/custom/fastwam_realrobot_smoke/launch.sh
```

ImageWAM：

```bash
make prepare-imagewam-upstream
make download-imagewam-artifacts
make download-imagewam-flux2-base
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```

更多说明见 [`pipelines/custom/README.md`](pipelines/custom/README.md)。

## 常用 Make targets

Make 只作为环境、下载和检查入口；训练/推理实验请使用 `experiments/*/*/launch.sh`。

| Target | 作用 |
|---|---|
| `make test` | core 单测 |
| `make validate` | mock task/config 校验 |
| `make lerobot-check-scripts` | 检查 LeRobot wrapper 和 parser |
| `make fastwam-check-scripts` | 检查 FastWAM wrapper 和 parser |
| `make imagewam-check-scripts` | 检查 ImageWAM wrapper |
| `make experiments-check-scripts` | 检查 experiments 启动脚本 |
| `make download-lerobot-pusht-dataset` | 下载 LeRobot PushT |
| `make download-lerobot-svla-so100-pickplace-dataset` | 下载 SmolVLA SO100 pick-place 数据 |
| `make download-lerobot-fastwam-libero-dataset` | 下载 FastWAM LIBERO 原始数据到 LeRobot 路线 |
| `make convert-lerobot-fastwam-libero-v3` | 转换 LeRobot 路线 FastWAM LIBERO v2.1 → v3.0 |
| `make download-lerobot-fastwam-base-cache` | 下载 LeRobot FastWAM 推理所需 Wan/T5 base cache |
| `make download-lerobot-diffusion-pusht-policy` | 下载 LeRobot diffusion PushT 预训练 policy |
| `make download-lerobot-smolvla-base-policy` | 下载 LeRobot SmolVLA base policy |
| `make download-lerobot-fastwam-libero-policy` | 下载 LeRobot-compatible FastWAM LIBERO policy |
| `make download-data-rovid20k` | 下载 RoVid-X 实用子集 |
| `make download-data-xperience10m-sample` | 下载 Xperience-10M sample |
| `make download-custom-fastwam-libero-dataset` | 下载 FastWAM LIBERO 原始数据到 custom/FastWAM 路线 |
| `make download-fastwam-artifacts` | 下载 FastWAM release 权重和 stats |
| `make prepare-imagewam-upstream` | 准备 ImageWAM 官方源码 |
| `make download-imagewam-artifacts` | 下载 ImageWAM release checkpoint |
| `make download-imagewam-flux2-base` | 下载 FLUX.2 4B base / AE |

## 当前边界

- LeRobot ACT/PushT 是当前已验证真实训练链路；
- LeRobot FastWAM/LIBERO 是当前已验证真实 CUDA 推理链路；
- Diffusion/PushT 与 SmolVLA/SO100 的训练/推理入口已准备，长期实验结果另行记录；
- Custom FastWAM realrobot 训练依赖私有 overlay 权限；
- household/mock demo 是展示层，不代表模型能力；
- viewer、真实仿真器、真机闭环暂缓。
