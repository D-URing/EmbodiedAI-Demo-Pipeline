# Custom WAM / FastWAM

这是 FastWAM 在 `custom` 结构下的规范入口。后续新自研/自定义后端都按 `pipelines/custom/<backend>` 接入。

## 当前定位

FastWAM 这条线不是“完全从零自研模型”，而是：

```text
FastWAM official release
  + private realrobot overlay
  + 项目级数据/模型/日志约定
  -> custom backend fine-tuning / evaluation evidence
```

## 关键路径

```text
configs/fastwam/realrobot_train_eval.sh
scripts/fastwam/download_release_artifacts.sh
scripts/fastwam/prepare_fastwam_overlay.sh
scripts/fastwam/run_realrobot_train_eval.sh
experiments/custom/fastwam_realrobot_single8_random/config.yaml
experiments/custom/fastwam_realrobot_single8_random/run.py
experiments/custom/fastwam_realrobot_8node_random/
pipelines/custom/fastwam/README.md
```

## 准备资产

```bash
make download-custom-fastwam-libero-dataset
make download-fastwam-artifacts
FASTWAM_SOURCE_MODE=sync bash scripts/fastwam/prepare_fastwam_overlay.sh
```

`download-fastwam-artifacts` 不只下载 release ckpt/stats，也会准备 Wan2.2 VAE、Wan2.2 T5 text encoder 和 Wan2.1 tokenizer。它们是 text embedding cache 预计算的真实依赖。

## 新集群测速准备

如果新集群已经准备好了 Python、torch、torchvision 和 CUDA，不希望本项目脚本重装 torch，可以在已激活的环境里执行：

```bash
FASTWAM_SOURCE_MODE=sync FASTWAM_INSTALL=1 FASTWAM_CREATE_CONDA=0 \
FASTWAM_SKIP_TORCH_INSTALL=1 FASTWAM_INSTALL_NVCC=0 \
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

这会做三件事：

1. 同步官方 FastWAM 和 realrobot overlay；
2. 安装 FastWAM 除 torch/torchvision 之外的 Python 依赖；
3. 以 editable 方式安装 generated workspace。

执行前建议先确认当前环境：

```bash
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
PY
```

如果容器自带的是 Python 3.11，而不是推荐的 Python 3.10，可以先用于测速：

```bash
FASTWAM_SOURCE_MODE=sync FASTWAM_INSTALL=1 FASTWAM_CREATE_CONDA=0 \
FASTWAM_SKIP_TORCH_INSTALL=1 FASTWAM_INSTALL_NVCC=0 \
FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH=1 \
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

注意：这只是快速兼容模式。长期正式实验仍建议使用 Python 3.10 环境，减少 DeepSpeed、CUDA extension 和上游依赖的隐性兼容风险。

## 日志和编译缓存

当前集群上 `torchcodec` 能被 Python 发现，但缺少匹配的 FFmpeg `libavutil`，所以 upstream LeRobot 默认路径会先打印一大段 torchcodec loading traceback，再回退到 `pyav`。本项目默认：

```yaml
fastwam:
  video_backend: pyav
  suppress_video_warnings: true
```

同时 `prepare_fastwam_overlay.sh` 会给 generated workspace 打一个很小的兼容补丁，让 `FASTWAM_VIDEO_BACKEND=pyav` 从源头生效。更新代码后在集群执行一次：

```bash
FASTWAM_SOURCE_MODE=reuse FASTWAM_INSTALL=0 bash scripts/fastwam/prepare_fastwam_overlay.sh
```

训练前的 Torch/DeepSpeed/Triton 扩展编译不能完全省掉；第一次运行或升级 Python/Torch/CUDA 后仍会编译。本项目把缓存固定到：

```text
.cache/torch_extensions/fastwam
.cache/triton/fastwam
```

这些目录在项目内、由 `.gitignore` 忽略。只要共享盘缓存不被删，同一环境的后续实验应复用缓存，不应每次重新编译。

如果同一节点上已有未结束的 `torchrun`，可能占用默认端口。可以不改 YAML，直接用环境变量换端口：

```bash
FASTWAM_MASTER_PORT=29600 FASTWAM_TEXT_EMBED_MASTER_PORT=29617 \
python experiments/custom/fastwam_realrobot_single8_random/run.py
```

如果日志出现 `gpu_arch=sm_120` 但 `torch_supported_arches` 只到 `sm_90`，说明当前 PyTorch wheel 不支持这张 GPU。此时不要继续测速，应该切换到支持该 GPU 架构的 PyTorch/CUDA 环境。

## 启动实验

训练/评测入口放在 `experiments/`，不要用 Makefile 启动：

```bash
python experiments/custom/fastwam_realrobot_single8_random/run.py --dry-run
python experiments/custom/fastwam_realrobot_single8_random/run.py
```

当前默认入口是 `init=random`，用于验证真实训练链路：数据读取、text cache、8 卡训练、loss、checkpoint。它不是 release checkpoint 微调。

如果要做正式微调，建议复制 `fastwam_realrobot_single8_random/` 新建实验，然后在 `config.yaml` 中设置：

```yaml
fastwam:
  init: release
  extra_overrides:
    - resume=/mnt/.../models/custom/fastwam/release/libero_uncond_2cam224.pt
    - learning_rate=3e-5
```

8 机 × 8 卡随机初始化：

```bash
sbatch experiments/custom/fastwam_realrobot_8node_random/slurm.sbatch
```

等价关键开关：

```text
FASTWAM_RECIPE=v6_scratch
FASTWAM_INIT=random
FASTWAM_NNODES=8
FASTWAM_GPUS_PER_NODE=8
```

注意：private overlay clone 需要 GitHub 私有仓库权限。
