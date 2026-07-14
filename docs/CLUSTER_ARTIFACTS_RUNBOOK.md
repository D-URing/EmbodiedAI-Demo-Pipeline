# 集群开源数据与模型下载 Runbook

> 状态：v0.1<br>
> 日期：2026-07-14<br>
> 目标：给 NVIDIA 集群测试准备一套可复制的 artifact 下载、存放和 smoke 验证流程。

这份文档只处理公开开源资产和本项目 smoke 链路，不下载私有真机数据，不提交任何大文件到 git。

## 1. 当前要下载什么

第一阶段按优先级分两类：

| 优先级 | 资产 | 来源 | 用途 | 默认是否下载 |
|---|---|---|---|---|
| P0 | `lerobot/pusht` dataset | Hugging Face dataset | LeRobot ACT/PushT data/train/inference smoke | 是 |
| P0 | ACT/PushT 训练输出 checkpoint | 集群本地训练产生 | `make lerobot-infer-smoke` 输入 | 训练后本地产生 |
| P1 | LeRobot policy checkpoint | Hugging Face model repo 或内部 checkpoint | 跳过训练、直接推理 | 否，需要显式 repo id |
| P1 | FastWAM release 权重与 stats | `yuanty/fastwam` | custom FastWAM overlay 初始化 | 否，按需执行 |
| P2 | 大规模 LeRobot/Open-X/DROID/BridgeData 等数据 | 各上游 | 后续扩展 | 暂不在第一条 smoke 自动下载 |

当前仓库默认 LeRobot demo 是 **ACT on PushT**。FastWAM 是下一条重点接入路径；公开 release 权重用于 custom overlay 或后续 FastWAM 实验，不等同于当前已完成的 LeRobot-native FastWAM smoke。

## 2. 集群路径约定

本项目默认假设整个仓库目录已经放在共享盘上，因此所有公开数据、模型权重、Hugging Face cache、上游源码和运行输出都落在项目内。先进入项目根目录：

```bash
cd /path/to/shared/EmbodiedAI-Demo-Pipeline

export PROJECT_ROOT="$PWD"

export EMBODIED_MODEL_ROOT="$PROJECT_ROOT/models"
export EMBODIED_DATA_ROOT="$PROJECT_ROOT/data"
export EMBODIED_RUN_ROOT="$PROJECT_ROOT/runs"

export HF_HOME="$PROJECT_ROOT/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHON_BIN=python3
```

如果你不设置这些变量，仓库脚本也会默认使用相同的项目内目录。显式 export 的好处是 shell 里更容易看清当前路径。

项目内目录规划：

```text
$PROJECT_ROOT/
├── data/
├── models/
├── checkpoints/
├── runs/
├── artifacts/
├── upstreams/
└── hf_cache/
```

## 3. 准备 Python / LeRobot 环境

在 GPU 节点或带 CUDA 的容器里：

```bash
bash scripts/lerobot/install_lerobot_cluster.sh
```

如果只是预下载 Hugging Face 资产，至少需要：

```bash
python3 -m pip install -U huggingface_hub hf_transfer
```

下载脚本会自动寻找旧版 `huggingface-cli` 或新版 `hf`。如果集群把 `hf` 装在特殊位置，可以显式指定：

```bash
export HF_CLI_BIN=/usr/local/bin/hf
```

## 4. 下载 LeRobot PushT 数据

默认下载 `lerobot/pusht` 到：

```text
$EMBODIED_DATA_ROOT/lerobot/pusht
```

命令：

```bash
make download-lerobot-artifacts
```

等价显式写法：

```bash
export LEROBOT_DATASET_REPO_ID=lerobot/pusht
export LEROBOT_DATASET_LOCAL_DIR="$EMBODIED_DATA_ROOT/lerobot/pusht"
DOWNLOAD_LEROBOT_DATASET=1 make download-lerobot-artifacts
```

脚本会生成：

```text
$EMBODIED_RUN_ROOT/artifact_manifests/lerobot_artifacts_manifest.json
```

下载完成后验证数据能被 LeRobot 读到：

```bash
export LEROBOT_DATASET_ROOT="$EMBODIED_DATA_ROOT/lerobot/pusht"
make lerobot-data-smoke
```

## 5. 可选：下载 LeRobot policy checkpoint

当前仓库不硬编码 ACT/PushT 的预训练 checkpoint，因为不同 LeRobot 版本和社区仓库可能命名不同。若你确认了一个 Hugging Face model repo，可以这样下载：

```bash
export LEROBOT_POLICY_REPO_ID="<org>/<model-repo>"
export LEROBOT_POLICY_TYPE="act"
export LEROBOT_POLICY_LOCAL_DIR="$EMBODIED_MODEL_ROOT/lerobot/act/pusht/<model-repo>"

DOWNLOAD_LEROBOT_DATASET=0 \
DOWNLOAD_LEROBOT_POLICY=1 \
make download-lerobot-artifacts
```

然后推理：

```bash
export LEROBOT_DATASET_ROOT="$EMBODIED_DATA_ROOT/lerobot/pusht"
export LEROBOT_POLICY_PATH="$LEROBOT_POLICY_LOCAL_DIR"
make lerobot-infer-smoke
```

如果没有合适的公开 checkpoint，推荐直接跑训练 smoke 生成本地 checkpoint。

## 6. 训练 ACT/PushT 并观察 loss

在 CUDA 节点上：

```bash
export LEROBOT_DATASET_ROOT="$EMBODIED_DATA_ROOT/lerobot/pusht"
make lerobot-train-smoke
```

训练输出在 `runs/lerobot/...` 下。脚本会解析 stdout 并生成 loss summary，用来回答“loss 是否正常下降”。

训练完成后，找到 LeRobot 输出的 checkpoint/pretrained policy 目录，并设置：

```bash
export LEROBOT_POLICY_PATH="<runs/lerobot/.../lerobot_output/.../pretrained_model-or-checkpoint-dir>"
make lerobot-infer-smoke
```

不同 LeRobot 版本的 checkpoint 目录名可能略有差异，因此这里不把路径写死。判断标准是该目录能被 LeRobot policy load/pretrained loader 读取。

## 7. 下载 FastWAM release 权重

按需执行：

```bash
make download-fastwam-artifacts
```

默认下载：

```text
repo: yuanty/fastwam
files:
  - libero_uncond_2cam224.pt
  - libero_uncond_2cam224_dataset_stats.json
target:
  $EMBODIED_MODEL_ROOT/fastwam_release
```

如果要下载其他 FastWAM release 文件：

```bash
FASTWAM_RELEASE_FILES="libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json <another-file>" \
make download-fastwam-artifacts
```

脚本会生成：

```text
$EMBODIED_RUN_ROOT/artifact_manifests/fastwam_release_artifacts_manifest.json
```

如果没有显式设置 `EMBODIED_MODEL_ROOT`，默认目标路径是：

```text
$PROJECT_ROOT/models/fastwam_release
```

后续 custom overlay 运行时：

```bash
export FASTWAM_MODEL_BASE="$EMBODIED_MODEL_ROOT"
export FASTWAM_RELEASE_CKPT="$EMBODIED_MODEL_ROOT/fastwam_release/libero_uncond_2cam224.pt"
export FASTWAM_RELEASE_DATASET_STATS="$EMBODIED_MODEL_ROOT/fastwam_release/libero_uncond_2cam224_dataset_stats.json"

FASTWAM_MODE=pilot FASTWAM_RECIPE=joint_base \
bash scripts/fastwam/run_realrobot_train_eval.sh
```

## 8. 下载网络问题排查

如果遇到：

```text
Network is unreachable
LocalEntryNotFoundError
```

说明当前节点不能访问 Hugging Face，且项目内 `hf_cache/` 里也没有对应 snapshot。先检查：

```bash
which hf || which huggingface-cli
curl -I "${HF_ENDPOINT:-https://huggingface.co}"
env | grep -E '^(HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|HF_ENDPOINT)='
```

常见处理方式：

```bash
# 方式一：集群有代理
export HTTPS_PROXY=http://<proxy-host>:<proxy-port>
export HTTP_PROXY=http://<proxy-host>:<proxy-port>

# 方式二：集群提供 Hugging Face mirror
export HF_ENDPOINT=https://<your-hf-mirror>

# 方式三：显式指定新版 hf CLI
export HF_CLI_BIN=/usr/local/bin/hf
```

如果集群完全没有外网，就在有外网的机器下载后，把以下文件拷贝到 `$EMBODIED_MODEL_ROOT/fastwam_release/`：

```text
libero_uncond_2cam224.pt
libero_uncond_2cam224_dataset_stats.json
```

然后继续设置：

```bash
export FASTWAM_RELEASE_CKPT="$EMBODIED_MODEL_ROOT/fastwam_release/libero_uncond_2cam224.pt"
export FASTWAM_RELEASE_DATASET_STATS="$EMBODIED_MODEL_ROOT/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
```

## 9. 生成第一版交付报告

LeRobot data-to-inference 链路：

```bash
export LEROBOT_DATASET_PROFILE="<runs/.../dataset_profile.json>"
export LEROBOT_INFERENCE_EVIDENCE="<runs/.../inference_evidence.json>"
export LEROBOT_TRAINING_SUMMARY="<runs/.../loss_summary.json>"  # 可选

make demo-chain-lerobot-fastwam
```

FastWAM custom overlay 链路：

```bash
FASTWAM_RUN_DIR="runs/fastwam/<run_name>/<run_id>" make demo-chain-fastwam
```

## 10. 常见开关

| 变量 | 默认值 | 作用 |
|---|---|---|
| `EMBODIED_DATA_ROOT` | `$PROJECT_ROOT/data` | 开源/私有数据根目录 |
| `EMBODIED_MODEL_ROOT` | `$PROJECT_ROOT/models` | 权重、checkpoint 根目录 |
| `EMBODIED_RUN_ROOT` | `$PROJECT_ROOT/runs` | manifest 和运行输出 |
| `HF_HOME` | `$PROJECT_ROOT/hf_cache` | Hugging Face cache 根目录 |
| `LEROBOT_DATASET_REPO_ID` | `lerobot/pusht` | LeRobot dataset repo |
| `LEROBOT_DATASET_LOCAL_DIR` | `$EMBODIED_DATA_ROOT/lerobot/pusht` | dataset 落盘目录 |
| `DOWNLOAD_LEROBOT_DATASET` | `1` | 是否下载 LeRobot dataset |
| `DOWNLOAD_LEROBOT_POLICY` | `0` | 是否下载 LeRobot policy |
| `LEROBOT_POLICY_REPO_ID` | 空 | policy repo，启用 policy 下载时必填 |
| `FASTWAM_RELEASE_REPO_ID` | `yuanty/fastwam` | FastWAM release repo |
| `FASTWAM_RELEASE_FILES` | LIBERO 权重 + stats | 要下载的 FastWAM 文件 |
| `PYTHON_BIN` | `python3` | 写 manifest 用的 Python，可改成 venv/conda 里的解释器 |
| `HF_CLI_BIN` | 自动检测 | 可显式指定 `/path/to/hf` 或 `/path/to/huggingface-cli` |

## 11. 第一轮集群测试建议

最小测试顺序：

```bash
bash scripts/lerobot/install_lerobot_cluster.sh
make download-lerobot-artifacts

export LEROBOT_DATASET_ROOT="$EMBODIED_DATA_ROOT/lerobot/pusht"
make lerobot-data-smoke
make lerobot-train-smoke

export LEROBOT_POLICY_PATH="<训练输出 checkpoint/pretrained policy dir>"
make lerobot-infer-smoke
```

如果这四步跑通，第一阶段最关键的链路已经成立：公开数据能下载和读取，官方 LeRobot 训练入口能跑，loss 有 summary，checkpoint 能进入 offline inference。

## 12. 上游链接

- LeRobot GitHub：https://github.com/huggingface/lerobot
- LeRobot PushT dataset：https://huggingface.co/datasets/lerobot/pusht
- Hugging Face CLI download 文档：https://huggingface.co/docs/huggingface_hub/guides/cli
- FastWAM release artifacts：https://huggingface.co/yuanty/fastwam
