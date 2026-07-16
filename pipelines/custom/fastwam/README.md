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
