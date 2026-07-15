# 模型、数据与权重存放规范

> 状态：v0.1<br>
> 日期：2026-07-14<br>
> 目标：明确当前 LeRobot demo 是什么模型，以及未来接入新模型时如何下载、存放、记录和引用。

## 1. 当前 LeRobot demo 是什么

当前仓库里已经封装的 LeRobot-native demo 默认是：

```text
dataset.repo_id = lerobot/pusht
policy.type     = act
policy class    = lerobot.policies.act.modeling_act.ACTPolicy
```

也就是说，当前 LeRobot demo 是 **ACT on PushT** 的 data/train/inference smoke，不是 FastWAM demo。

当前能力边界：

| 环节 | 当前状态 |
|---|---|
| dataset smoke | 已有入口：`make lerobot-data-smoke` |
| training smoke | 已有入口：`bash experiments/lerobot/pusht_act_smoke/launch.sh` |
| offline inference smoke | 已有入口：`bash experiments/lerobot/diffusion_pusht_infer/launch.sh` |
| report | 已有入口：`python scripts/lerobot/generate_data_to_inference_report.py` |
| 大文件下载 | 默认禁用，`LEROBOT_ALLOW_DOWNLOAD=0` |
| 真实闭环 | 未声明 |

第一版之所以选 ACT/PushT，是因为它轻、LeRobot 官方路径稳定、适合验证 data-to-inference 的工程链路。FastWAM 是下一条重点 LeRobot-native policy path。

## 2. 两条模型路径

未来模型接入统一分为两条：

| 路径 | 说明 | 当前代表 | 何时使用 |
|---|---|---|---|
| LeRobot-native path | 模型已能通过 LeRobot dataset/policy/train/inference API 跑通 | ACT/PushT，后续 FastWAM | 第一优先级，适合标准 demo |
| Custom backend path | 需要自定义训练、私有数据、特殊 action head 或未进入 LeRobot 的模型 | FastWAM realrobot overlay | 自研/改模型/真机扩展 |

FastWAM 当前 custom overlay 不是完全从零自拟模型，而是基于 FastWAM 公开结构/权重做 realrobot 数据微调、recipe 适配和离线 probe。它是 custom backend 的第一个样板。

## 3. 大文件原则

仓库永远不提交：

- dataset；
- pretrained weights；
- checkpoint；
- run output；
- Hugging Face cache；
- FastWAM release 权重；
- 真机视频/图像数据；
- 私有 overlay 产物。

仓库只提交：

- 下载/准备脚本；
- config；
- manifest；
- 小型 fixture；
- parser；
- schema；
- report generator；
- 文档。

## 4. 推荐存放位置

### 默认：项目内目录

本项目默认把公开数据、模型权重、训练产物和上游源码都放在仓库目录内。由于整个项目目录会放在共享盘上，这比 `$HOME/.cache` 更直观，也更方便团队复现。

```text
EmbodiedAI-Demo-Pipeline/
├── data/          # LeRobot/Open-X/DROID 等数据
├── models/        # 公开模型权重、release 权重、预训练 policy
├── checkpoints/   # 本项目生成或整理的 checkpoint
├── runs/          # smoke、训练、评测、manifest 和 report
├── artifacts/     # 后续可视化/导出包
├── upstreams/     # LeRobot/FastWAM/XPolicyLab 等上游源码 checkout
└── hf_cache/      # Hugging Face hub/datasets cache
```

这些目录已经在 `.gitignore` 中忽略，不会提交大文件。

脚本默认等价于：

```bash
export PROJECT_ROOT="$PWD"
export EMBODIED_MODEL_ROOT="$PROJECT_ROOT/models"
export EMBODIED_DATA_ROOT="$PROJECT_ROOT/data"
export EMBODIED_RUN_ROOT="$PROJECT_ROOT/runs"
export HF_HOME="$PROJECT_ROOT/hf_cache"
```

### 推荐目录结构

```text
$EMBODIED_MODEL_ROOT/
├── lerobot/
│   ├── act/
│   │   └── pusht/
│   │       └── <checkpoint-or-pretrained-dir>/
│   ├── fastwam/
│   │   └── <checkpoint-or-pretrained-dir>/
│   └── diffusion_policy/
└── custom/
    ├── fastwam/
    │   └── release/
    │       ├── libero_uncond_2cam224.pt
    │       └── libero_uncond_2cam224_dataset_stats.json
    ├── imagewam/
    │   ├── flux2_klein_4b_libero/
    │   └── flux2/
    └── <future_model_name>/

$EMBODIED_DATA_ROOT/
├── lerobot/
│   └── pusht/
├── fastwam_realrobot/
└── custom/

$PROJECT_ROOT/upstreams/
├── lerobot/
├── FastWAM-realrobot/
├── fastwam-realrobot-pipeline/
└── XPolicyLab/

$PROJECT_ROOT/checkpoints/
└── fastwam/
    └── ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

## 5. 当前 ACT/PushT 使用方式

### 下载 PushT dataset

在项目根目录下显式下载公开 dataset：

```bash
make download-lerobot-artifacts
```

在 SCUT 集群上，脚本会优先使用 `/home/scut/hfd.sh`、`hf-mirror.com` 和 `aria2c`；如果该工具不存在，才回退到 `hf download` / `huggingface-cli download`。

默认下载：

```text
repo: lerobot/pusht
target: $EMBODIED_DATA_ROOT/lerobot/pusht
manifest: $EMBODIED_RUN_ROOT/artifact_manifests/lerobot_artifacts_manifest.json
```

如果要下载一个确认过的 LeRobot policy repo：

```bash
export LEROBOT_POLICY_REPO_ID="<org>/<model-repo>"
export LEROBOT_POLICY_TYPE="act"
export LEROBOT_POLICY_LOCAL_DIR="$EMBODIED_MODEL_ROOT/lerobot/act/pusht/<model-repo>"

DOWNLOAD_LEROBOT_DATASET=0 \
DOWNLOAD_LEROBOT_POLICY=1 \
make download-lerobot-artifacts
```

当前仓库不默认绑定 ACT/PushT 的预训练 policy repo；如果没有明确 checkpoint，优先通过 `bash experiments/lerobot/pusht_act_smoke/launch.sh` 训练一个本地 checkpoint，再进入 inference smoke。

### Dataset smoke

如果 dataset 已在项目内：

```bash
export LEROBOT_DATASET_REPO_ID=lerobot/pusht
export LEROBOT_DATASET_ROOT="$EMBODIED_DATA_ROOT/lerobot/pusht"
make lerobot-data-smoke
```

默认不会下载。如果确实要允许 LeRobot/Hugging Face 下载：

```bash
LEROBOT_ALLOW_DOWNLOAD=1 make lerobot-data-smoke
```

### Train smoke

```bash
bash experiments/lerobot/pusht_act_smoke/launch.sh
```

默认是：

```text
policy.type=act
dataset.repo_id=lerobot/pusht
policy.device=cuda
```

### Inference smoke

准备本地 checkpoint 后：

```bash
export LEROBOT_POLICY_PATH="$EMBODIED_MODEL_ROOT/lerobot/act/pusht/<checkpoint-or-pretrained-dir>"
export LEROBOT_DATASET_ROOT="$EMBODIED_DATA_ROOT/lerobot/pusht"
bash experiments/lerobot/diffusion_pusht_infer/launch.sh
```

注意：`LEROBOT_POLICY_PATH` 必须是本地路径。脚本不会默认下载 checkpoint。

## 6. FastWAM 下载与存放

FastWAM 有两类资产：

| 资产 | 建议位置 | 用途 |
|---|---|---|
| LeRobot-native FastWAM checkpoint | `$EMBODIED_MODEL_ROOT/lerobot/fastwam/<name>/` | 后续 LeRobot-native data-to-inference |
| FastWAM release 权重 | `$EMBODIED_MODEL_ROOT/custom/fastwam/release/` | custom overlay 微调初始化 |
| FastWAM LIBERO 数据 | `$EMBODIED_DATA_ROOT/custom/fastwam/libero-fastwam/` | FastWAM LIBERO 训练/复现实验 |
| FastWAM dataset stats | `$EMBODIED_MODEL_ROOT/custom/fastwam/release/` 或数据目录 meta | normalizer / stats |
| custom overlay checkpoint | `$EMBODIED_MODEL_ROOT/custom/fastwam_realrobot/<run>/` | 私有 realrobot recipe 输出 |

FastWAM release 权重按上游说明下载，例如：

```bash
make download-fastwam-artifacts
```

在 SCUT 管理节点上，脚本会自动等价于：

```bash
HF_ENDPOINT=https://hf-mirror.com \
bash /home/scut/hfd.sh yuanty/fastwam \
  --include libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json \
  --local-dir "$EMBODIED_MODEL_ROOT/custom/fastwam/release" \
  --tool aria2c \
  -x 10 -j 2
```

然后运行 custom overlay 时显式指定：

```bash
export FASTWAM_MODEL_BASE="$EMBODIED_MODEL_ROOT"
export FASTWAM_RELEASE_CKPT="$EMBODIED_MODEL_ROOT/custom/fastwam/release/libero_uncond_2cam224.pt"
export FASTWAM_RELEASE_DATASET_STATS="$EMBODIED_MODEL_ROOT/custom/fastwam/release/libero_uncond_2cam224_dataset_stats.json"
```

如需下载其他 FastWAM release 文件：

```bash
FASTWAM_RELEASE_FILES="libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json <another-file>" \
make download-fastwam-artifacts
```

FastWAM 的公开 LIBERO 数据不是放在 `yuanty/fastwam` model repo，而是在 Hugging Face dataset repo：

```text
yuanty/LIBERO-fastwam
```

SCUT 已下载并解压到：

```text
data/custom/fastwam/libero-fastwam/
```

其中 4 个子集分别是：

```text
libero_10_no_noops_lerobot/
libero_goal_no_noops_lerobot/
libero_object_no_noops_lerobot/
libero_spatial_no_noops_lerobot/
```

该 release 是 LeRobot v2.1 格式；当前 LeRobot v3 主线如需直接读取，应先做格式转换。完整命令见 [`CLUSTER_ARTIFACTS_RUNBOOK.md`](CLUSTER_ARTIFACTS_RUNBOOK.md)。

更完整的集群下载、项目内目录和 smoke 验证步骤见 [`CLUSTER_ARTIFACTS_RUNBOOK.md`](CLUSTER_ARTIFACTS_RUNBOOK.md)。

## 7. 新模型接入清单

每接一个新模型，必须补齐：

1. registry 记录：模型名、路径类型、数据要求、是否 LeRobot-native；
2. 下载说明：来源、命令、目标目录；
3. smoke 入口：dataset/train/load/inference 至少一个；
4. evidence 输出：dataset/training/inference/evaluation 至少一种；
5. 边界声明：offline、sim、real 哪个层级已经证明；
6. 不入库说明：哪些权重、数据和产物必须留在项目内 ignored 目录或外部共享盘。

## 8. 近期接入顺序

| 顺序 | 模型/路径 | 类型 | 目标 |
|---|---|---|---|
| 1 | ACT / PushT | LeRobot-native | 已作为第一条轻量 data-to-inference smoke |
| 2 | FastWAM | LeRobot-native | 跑通官方 LeRobot FastWAM policy path |
| 3 | FastWAM realrobot overlay | Custom backend | 跑通私有数据微调和 report |
| 4 | Diffusion Policy / 其他 LeRobot policy | LeRobot-native | 复用同一套 dataset/inference evidence |
| 5 | 自研模型 | Custom backend | 复用 custom backend 模板与 evidence contract |
