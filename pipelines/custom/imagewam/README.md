# Custom WAM / ImageWAM

ImageWAM 是与 FastWAM 并列的 custom WAM 后端。它的官方项目把 image-editing foundation model 改造成 instruction-conditioned visual backbone，用于机器人 action prediction；官方代码包含 LIBERO、LIBERO-plus 和 RoboTwin 的训练/评测入口。

我们这里不 vendor 官方代码，只做项目级封装：

```text
upstreams/ImageWAM/        # 官方源码 checkout，本地 ignored
data/fastwam/libero-fastwam/
models/imagewam/
runs/experiments/custom/imagewam_flux2_4b_libero_pilot/
```

## 为什么先放 custom，而不是 LeRobot

ImageWAM 当前更像一个完整研究代码库：

- 有自己的 `uv` 环境和 dependency 组合；
- 有 FLUX.2 / OmniGen2 / Ovis-U1 多个模型变体；
- LIBERO / RoboTwin 数据准备和评测入口独立；
- 训练脚本是官方 repo 自己的 shell wrappers。

所以它更适合先作为 custom backend 接入。LeRobot 主线仍负责 official LeRobot policy 的训练和推理。

## 推荐第一阶段目标

第一阶段只做 FLUX.2 4B + LIBERO：

```text
backend = imagewam
variant = flux2_4b
task_type = libero
dataset = data/fastwam/libero-fastwam
release_ckpt = models/imagewam/flux2_klein_4b_libero
```

原因：

- 官方 README 推荐从 FLUX.2 ImageWAM 开始；
- LIBERO 数据可复用我们已经准备的 FastWAM release 数据；
- 4B 比 9B 更适合先做 A100 八卡 smoke/pilot；
- RoboTwin 先作为第二阶段，因为评测环境和资产准备更重。

## 准备上游源码

```bash
make prepare-imagewam-upstream
```

默认 clone 到：

```text
upstreams/ImageWAM/
```

## 下载 release checkpoint

默认下载 FLUX.2 4B LIBERO release：

```bash
make download-imagewam-artifacts
```

默认路径：

```text
models/imagewam/flux2_klein_4b_libero/
├── model.pt
├── dataset_stats.json
└── train_config.yaml
```

注意：release checkpoint 主要用于评测或作为续训/对照起点。要跑官方训练入口，还需要 FLUX.2 base model、AE 权重和 FLUX2 源码：

```bash
make download-imagewam-flux2-base
```

默认路径：

```text
models/imagewam/flux2/
├── FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors
└── FLUX.2-dev/ae.safetensors
```

FLUX.2 部分仓库可能需要 Hugging Face access approval。如果要连 9B base 一起下：

```bash
IMAGEWAM_DOWNLOAD_9B=true make download-imagewam-flux2-base
```

如果要下载 9B：

```bash
IMAGEWAM_POLICY_REPO_ID=yuyangalin/ImageWAM-FLUX.2-9B-LIBERO \
IMAGEWAM_POLICY_LOCAL_DIR="$PWD/models/imagewam/flux2_klein_9b_libero" \
make download-imagewam-artifacts
```

## 训练 / 评测入口

先跑 metadata smoke，确认路径、源码、权重和 CUDA 可见性：

```bash
IMAGEWAM_MODE=metadata-smoke IMAGEWAM_REQUIRE_CUDA=0 \
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```

真正训练建议在 `upstreams/ImageWAM/` 准备好官方环境后运行：

```bash
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate imagewam

IMAGEWAM_MODE=pilot \
IMAGEWAM_VARIANT=flux2_4b \
IMAGEWAM_FLUX2_VARIANT=4b \
IMAGEWAM_TASK_TYPE=libero \
IMAGEWAM_PRECOMPUTE_QWEN3_CACHE=true \
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```

当前 wrapper 会优先调用官方 FLUX.2 训练入口：

```text
upstreams/ImageWAM/scripts/flux2/run_train_flux2_klein_imagewam.sh
```

评测入口后续按官方 `scripts/flux2/run_eval_flux2_libero.sh` 接入；在没有完成仿真环境安装前，本项目只记录 checkpoint/stats/训练日志证据，不声明 LIBERO/RoboTwin 成功率。

## 和 FastWAM 的关系

ImageWAM 可以复用 FastWAM release 数据，但不是 FastWAM overlay 的子目录。两者并列：

```text
custom/
├── fastwam/
└── imagewam/
```

这对后续很关键：我们可以把 InternVLA-A、LingBot-VLA、GR00T-style policy wrapper 等继续作为新的 custom backend 加进来，而不把所有东西塞进 FastWAM。
