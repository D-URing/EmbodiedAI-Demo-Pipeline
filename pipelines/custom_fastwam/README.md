# Custom / FastWAM Pipeline

## 这条线解决什么

Custom/FastWAM 是第二条线。它回答：

> 如果未来团队要接自拟模型、自建 policy、私有真机数据，项目要怎么保留一条不被 LeRobot 主线绑死的工程路径？

当前用 FastWAM 作为第一个 custom backend 例子：

```text
official base: yuantianyuan01/FastWAM
private overlay: D-URing/fastwam-realrobot-pipeline
release ckpt: yuanty/fastwam
release data: yuanty/LIBERO-fastwam
```

资产来自根目录全局池：

```text
data/fastwam/libero-fastwam
models/fastwam_release
models/custom/
```

注意：这条线目前更接近“基于 FastWAM release 的微调/扩展路径”，不是从零自研模型。

## 当前 SCUT 状态

已准备：

```text
models/fastwam_release/libero_uncond_2cam224.pt
models/fastwam_release/libero_uncond_2cam224_dataset_stats.json
data/fastwam/libero-fastwam/
runs/artifact_manifests/fastwam_release_artifacts_manifest.json
runs/artifact_manifests/fastwam_libero_dataset_manifest.json
```

FastWAM LIBERO 数据已解压：

```text
data/fastwam/libero-fastwam/
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
fps: 20
action_shape: [7]
image_keys:
  - observation.images.image
  - observation.images.wrist_image
```

尚未完成：

```text
private overlay clone on SCUT
fastwam conda env
FastWAM train/eval smoke
```

原因：SCUT 管理节点目前没有私有 GitHub 仓库访问权限，clone `D-URing/fastwam-realrobot-pipeline` 会失败。

## 下载 release 权重

```bash
export BASE=/mnt/gpu11_200T/dingxibo
export PROJECT=$BASE/EmbodiedAI-Demo-Pipeline
cd "$PROJECT"

export EMBODIED_MODEL_ROOT="$PROJECT/models"
export EMBODIED_RUN_ROOT="$PROJECT/runs"
export HF_HOME="$PROJECT/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHON_BIN="$BASE/miniconda3/envs/embodied-core/bin/python"
export HFD_BIN=/home/scut/hfd.sh
export HFD_TOOL=aria2c
export HFD_THREADS=10
export HFD_JOBS=2

make download-fastwam-artifacts
```

## 下载 LIBERO 数据

```bash
export EMBODIED_DATA_ROOT="$PROJECT/data"

mkdir -p "$EMBODIED_DATA_ROOT/fastwam/libero-fastwam"
HF_ENDPOINT=https://hf-mirror.com \
bash /home/scut/hfd.sh yuanty/LIBERO-fastwam \
  --dataset \
  --local-dir "$EMBODIED_DATA_ROOT/fastwam/libero-fastwam" \
  --tool aria2c \
  -x 10 -j 4

cd "$EMBODIED_DATA_ROOT/fastwam/libero-fastwam"
for f in *.tar.gz; do
  tar -xzf "$f"
done
```

## v2.1 / v3 格式注意

`yuanty/LIBERO-fastwam` 是 LeRobot v2.1 格式。当前 LeRobot 主线使用 v3 loader，所以不能直接拿它跑 `make lerobot-data-smoke`。

如果要走 LeRobot-native 训练/推理，应先转换一份，不要覆盖原始目录：

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
  --repo-id <local-or-team-repo-id> \
  --root "$EMBODIED_DATA_ROOT/fastwam/libero-fastwam/<subset>" \
  --push-to-hub false
```

如果走 FastWAM 官方/custom overlay，优先按 FastWAM 代码要求使用原始 v2.1 release。

## 准备 private overlay

当前脚本：

```bash
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

它会尝试：

```text
upstreams/FastWAM-realrobot/              # official FastWAM + overlay 后工作区
upstreams/fastwam-realrobot-pipeline/     # private overlay checkout
```

在 SCUT 上继续之前，需要先解决私有仓库认证。建议任选一种：

```bash
# 方案 A：配置 SSH key 后改用 git@github.com:D-URing/fastwam-realrobot-pipeline.git

# 方案 B：配置 gh auth / token，让 HTTPS clone 能读私有仓库
```

## 相关文件

```text
configs/fastwam/
scripts/fastwam/
demo_chains/fastwam_realrobot_v0.yaml
docs/FASTWAM_REALROBOT_INTEGRATION.md
```
