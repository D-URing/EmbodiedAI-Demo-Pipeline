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
| P0 | ACT/PushT 训练输出 checkpoint | 集群本地训练产生 | `bash experiments/lerobot/diffusion_pusht_infer/launch.sh` 输入 | 训练后本地产生 |
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
export TORCH_HOME="$PROJECT_ROOT/hf_cache/torch"
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

SCUT/A100 集群推荐使用共享盘 Miniconda。先准备 core 环境，确保仓库基础命令和测试可运行：

```bash
export BASE=/mnt/gpu11_200T/dingxibo
export PROJECT=$BASE/EmbodiedAI-Demo-Pipeline
export CONDA=$BASE/miniconda3/bin/conda

cd "$PROJECT"
"$CONDA" create -y -n embodied-core --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  python=3.11 pip setuptools wheel
"$CONDA" run -n embodied-core python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -c requirements/constraints-py311.txt -e ".[dev]"
"$CONDA" run -n embodied-core python -m pytest
```

如果 `$BASE/miniconda3` 尚不存在，先按 [`ENVIRONMENT.md`](ENVIRONMENT.md) 的 SCUT/A100 Miniconda 小节安装。

LeRobot/FastWAM 是独立 CUDA policy 环境，不装进 `embodied-core`。在 GPU 节点或带 CUDA 的容器里：

```bash
CONDA_EXE="$CONDA" LEROBOT_CREATE_CONDA=1 LEROBOT_CONDA_ENV=lerobot \
bash scripts/lerobot/install_lerobot_cluster.sh
```

SCUT `gpu11` 已验证需要额外注意三点：

- LeRobot ACT 会默认加载 torchvision ResNet18 backbone，计算节点不能联网时要预先把 `resnet18-f37072fd.pth` 放到 `$TORCH_HOME/hub/checkpoints/`；
- 当前 `gpu11` host glibc 较老，训练默认使用 `dataset.video_backend=pyav`，避免 `torchcodec + conda-forge ffmpeg` 的 native ABI 问题；
- `policy.repo_id` 必须非空，但第一阶段 smoke 不 push Hub，默认使用 `local/pusht_act_gpu_smoke` 并设置 `policy.push_to_hub=false`。

ResNet18 权重下载命令：

```bash
mkdir -p "$TORCH_HOME/hub/checkpoints"
wget -O "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth" \
  https://download.pytorch.org/models/resnet18-f37072fd.pth
```

管理节点已确认存在 `/home/scut/hfd.sh`，并且 `hf-mirror.com + aria2c` 明显快于 `hf download`。下载脚本会优先使用：

```bash
/home/scut/hfd.sh
```

默认参数：

```text
HF_ENDPOINT=https://hf-mirror.com
HFD_THREADS=10
HFD_JOBS=2   # FastWAM release
HFD_JOBS=4   # LeRobot dataset/policy
```

如果某台机器没有 `/home/scut/hfd.sh`，脚本才会回退到旧版 `huggingface-cli` 或新版 `hf`。此时至少需要：

```bash
python3 -m pip install -U huggingface_hub hf_transfer
```

如果 hfd 脚本在别的位置，可以显式指定：`export HFD_BIN=/path/to/hfd.sh`。

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
bash experiments/lerobot/diffusion_pusht_infer/launch.sh
```

如果没有合适的公开 checkpoint，推荐直接跑训练 smoke 生成本地 checkpoint。

## 6. 训练 ACT/PushT 并观察 loss

在 CUDA 节点上：

```bash
source /mnt/gpu11_200T/dingxibo/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export LEROBOT_DATASET_ROOT="$EMBODIED_DATA_ROOT/lerobot/pusht"
export TORCH_HOME="$PROJECT_ROOT/hf_cache/torch"
bash experiments/lerobot/pusht_act_smoke/launch.sh
```

当前 SCUT `gpu11` 已通过 2-step GPU env check：

```text
CUDA OK: NVIDIA A800-SXM4-80GB
dataset.num_frames=25650
dataset.num_episodes=206
loss: 96.987 -> 83.351
loss_decreased=true, drop=14.06%
```

该检查只证明真实 LeRobot GPU 训练链路能启动并产生 loss，不代表正式收敛结果。正式回答“loss 是否正常下降”建议把 `LEROBOT_STEPS` 提高到 500-1000。

训练输出在 `runs/lerobot/...` 下。脚本会解析 stdout 并生成 loss summary，用来回答“loss 是否正常下降”。

训练完成后，找到 LeRobot 输出的 checkpoint/pretrained policy 目录，并设置：

```bash
export LEROBOT_POLICY_PATH="<runs/lerobot/.../lerobot_output/.../pretrained_model-or-checkpoint-dir>"
bash experiments/lerobot/diffusion_pusht_infer/launch.sh
```

不同 LeRobot 版本的 checkpoint 目录名可能略有差异，因此这里不把路径写死。判断标准是该目录能被 LeRobot policy load/pretrained loader 读取。

## 7. 下载 FastWAM release 权重

按需执行：

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

默认下载：

```text
repo: yuanty/fastwam
files:
  - libero_uncond_2cam224.pt
  - libero_uncond_2cam224_dataset_stats.json
target:
  $EMBODIED_MODEL_ROOT/custom/fastwam/release
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
$PROJECT_ROOT/models/custom/fastwam/release
```

后续 custom overlay 运行时：

```bash
export FASTWAM_MODEL_BASE="$EMBODIED_MODEL_ROOT"
export FASTWAM_RELEASE_CKPT="$EMBODIED_MODEL_ROOT/custom/fastwam/release/libero_uncond_2cam224.pt"
export FASTWAM_RELEASE_DATASET_STATS="$EMBODIED_MODEL_ROOT/custom/fastwam/release/libero_uncond_2cam224_dataset_stats.json"

FASTWAM_MODE=pilot FASTWAM_RECIPE=joint_base \
bash scripts/fastwam/run_realrobot_train_eval.sh
```

## 8. 下载 FastWAM LIBERO 数据

当前 FastWAM release 权重 `libero_uncond_2cam224.pt` 对应的公开预处理数据是：

```text
repo: yuanty/LIBERO-fastwam
format: LeRobot v2.1
target: $EMBODIED_DATA_ROOT/custom/fastwam/libero-fastwam
```

SCUT 管理节点下载命令：

```bash
mkdir -p "$EMBODIED_DATA_ROOT/custom/fastwam/libero-fastwam"
HF_ENDPOINT=https://hf-mirror.com \
bash /home/scut/hfd.sh yuanty/LIBERO-fastwam \
  --dataset \
  --local-dir "$EMBODIED_DATA_ROOT/custom/fastwam/libero-fastwam" \
  --tool aria2c \
  -x 10 -j 4
```

下载后按上游 README 解压：

```bash
cd "$EMBODIED_DATA_ROOT/custom/fastwam/libero-fastwam"
for f in *.tar.gz; do
  tar -xzf "$f"
done
```

当前 SCUT 已下载并解压：

```text
data/custom/fastwam/libero-fastwam/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

数据概况：

```text
format: LeRobot v2.1
subsets: 4
total_frames: 277713
robot: franka
fps: 20
action_shape: [7]
image_keys:
  - observation.images.image
  - observation.images.wrist_image
```

Manifest 已写到：

```text
runs/artifact_manifests/fastwam_libero_dataset_manifest.json
```

注意：当前项目的 LeRobot 主线使用较新的 LeRobot v3 loader，直接 `make lerobot-data-smoke` 读取这些 v2.1 子集会提示需要转换。若要纳入 LeRobot-native 训练/推理，应先使用 LeRobot 自带命令做 v2.1 → v3.0 转换，例如：

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
  --repo-id <local-or-team-repo-id> \
  --root "$EMBODIED_DATA_ROOT/custom/fastwam/libero-fastwam/<subset>" \
  --push-to-hub false
```

如果走 FastWAM 官方/custom overlay 路线，优先按 FastWAM 代码期望的数据格式使用该 v2.1 release，不要盲目转换后覆盖原始目录。

## 9. 下载网络问题排查

如果遇到：

```text
Network is unreachable
LocalEntryNotFoundError
```

说明当前节点不能访问所选 Hugging Face endpoint，或者 hfd/hf fallback 没有找到可用路径。先检查：

```bash
test -f /home/scut/hfd.sh && echo "hfd.sh OK"
command -v aria2c
curl -I "${HF_ENDPOINT:-https://hf-mirror.com}"
env | grep -E '^(HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|HF_ENDPOINT|HFD_BIN)='
```

常见处理方式：

```bash
# 方式一：集群有代理
export HTTPS_PROXY=http://<proxy-host>:<proxy-port>
export HTTP_PROXY=http://<proxy-host>:<proxy-port>

# 方式二：显式使用已验证的 mirror
export HF_ENDPOINT=https://hf-mirror.com

# 方式三：显式指定 hfd 位置
export HFD_BIN=/home/scut/hfd.sh
```

如果集群完全没有外网，就在有外网的机器下载后，把以下文件拷贝到 `$EMBODIED_MODEL_ROOT/custom/fastwam/release/`：

```text
libero_uncond_2cam224.pt
libero_uncond_2cam224_dataset_stats.json
```

然后继续设置：

```bash
export FASTWAM_RELEASE_CKPT="$EMBODIED_MODEL_ROOT/custom/fastwam/release/libero_uncond_2cam224.pt"
export FASTWAM_RELEASE_DATASET_STATS="$EMBODIED_MODEL_ROOT/custom/fastwam/release/libero_uncond_2cam224_dataset_stats.json"
```

## 9. 生成第一版交付报告

LeRobot data-to-inference 链路：

```bash
export LEROBOT_DATASET_PROFILE="<runs/.../dataset_profile.json>"
export LEROBOT_INFERENCE_EVIDENCE="<runs/.../inference_evidence.json>"
export LEROBOT_TRAINING_SUMMARY="<runs/.../loss_summary.json>"  # 可选

python scripts/lerobot/generate_data_to_inference_report.py
```

FastWAM custom overlay 链路：

```bash
FASTWAM_RUN_DIR="runs/experiments/custom/fastwam_realrobot_smoke/<run_id>" embodied-demo report-fastwam
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
| `HFD_BIN` | `/home/scut/hfd.sh` | SCUT 集群推荐下载器，存在时优先使用 |
| `HFD_THREADS` | `10` | hfd/aria2c 单文件线程数 |
| `HFD_JOBS` | FastWAM `2`，LeRobot `4` | hfd 并发文件数 |
| `HFD_TOOL` | `aria2c` | hfd 后端下载工具 |
| `HF_ENDPOINT` | hfd 时默认 `https://hf-mirror.com` | Hugging Face endpoint / mirror |
| `HF_CLI_BIN` | 自动检测 | fallback 时可显式指定 `/path/to/hf` 或 `/path/to/huggingface-cli` |

## 11. 第一轮集群测试建议

最小测试顺序：

```bash
bash scripts/lerobot/install_lerobot_cluster.sh
make download-lerobot-artifacts

export LEROBOT_DATASET_ROOT="$EMBODIED_DATA_ROOT/lerobot/pusht"
make lerobot-data-smoke
bash experiments/lerobot/pusht_act_smoke/launch.sh

export LEROBOT_POLICY_PATH="<训练输出 checkpoint/pretrained policy dir>"
bash experiments/lerobot/diffusion_pusht_infer/launch.sh
```

如果这四步跑通，第一阶段最关键的链路已经成立：公开数据能下载和读取，官方 LeRobot 训练入口能跑，loss 有 summary，checkpoint 能进入 offline inference。

## 12. 上游链接

- LeRobot GitHub：https://github.com/huggingface/lerobot
- LeRobot PushT dataset：https://huggingface.co/datasets/lerobot/pusht
- Hugging Face CLI download 文档：https://huggingface.co/docs/huggingface_hub/guides/cli
- FastWAM release artifacts：https://huggingface.co/yuanty/fastwam
