# ImageWAM Integration Plan

ImageWAM 是 custom WAM 线路里的第二个后端，和 FastWAM 并列，不作为 `custom_fastwam` 的子功能。

## 项目定位

我们对 ImageWAM 的短期定位是：

```text
official ImageWAM repo
  -> repo-local assets
  -> LIBERO train/eval wrapper
  -> logs / manifests / evidence
```

它补齐的是“视觉世界模型 / 图像编辑模型作为 action backbone”的方向。它不改变 LeRobot 主线：LeRobot 仍然负责 official policy 的数据读取、训练、推理复刻。

## 结构

```text
pipelines/custom_wam/imagewam/README.md
configs/imagewam/libero_train_eval.sh
scripts/imagewam/
├── prepare_imagewam_upstream.sh
├── download_artifacts.sh
├── run_train_eval.sh
└── slurm_libero_pilot.sbatch
models/imagewam/
upstreams/ImageWAM/
runs/imagewam/
```

## 第一阶段范围

优先做：

- FLUX.2 4B LIBERO release checkpoint；
- LIBERO metadata smoke；
- A100 单机八卡 pilot training wrapper；
- 训练日志、checkpoint 路径、manifest 证据归档。

暂缓：

- RoboTwin 完整评测；
- LIBERO success rate 宣称；
- 真机评测；
- 把 ImageWAM 改造成 LeRobot policy class。

## 常用命令

```bash
make prepare-imagewam-upstream
make download-imagewam-artifacts
make download-imagewam-flux2-base
make imagewam-train-smoke
```

pilot：

```bash
IMAGEWAM_MODE=pilot \
IMAGEWAM_VARIANT=flux2_4b \
IMAGEWAM_TASK_TYPE=libero \
IMAGEWAM_PRECOMPUTE_QWEN3_CACHE=true \
bash scripts/imagewam/run_train_eval.sh
```

SLURM：

```bash
sbatch scripts/imagewam/slurm_libero_pilot.sbatch
```

## 资产约定

| 类型 | 默认路径 |
|---|---|
| ImageWAM official repo | `upstreams/ImageWAM/` |
| FLUX.2 4B LIBERO ckpt | `models/imagewam/flux2_klein_4b_libero/` |
| FLUX.2 9B LIBERO ckpt | `models/imagewam/flux2_klein_9b_libero/` |
| FLUX.2 base / AE | `models/imagewam/flux2/` |
| LIBERO 数据 | `data/fastwam/libero-fastwam/` |
| 运行日志 | `runs/imagewam/` |
| manifest | `runs/artifact_manifests/imagewam_*.json` |

## 验收标准

M1：

- `make imagewam-check-scripts` 通过；
- `make prepare-imagewam-upstream` 能 clone 官方代码；
- `make download-imagewam-artifacts` 能下载 release checkpoint；
- `make download-imagewam-flux2-base` 能下载 FLUX.2 4B base 和 AE；
- `make imagewam-train-smoke` 能产出 `backend_manifest.json`。

M2：

- 在 A100 节点完成 `IMAGEWAM_MODE=pilot`；
- `runs/imagewam/<run>/train_stdout.log` 能解析出训练 step / loss；
- checkpoint 和 config 产物路径进入 manifest。

M3：

- 接入官方 LIBERO eval wrapper；
- 记录 success rate / task split / seed；
- 与 RoboDojo/RoboTwin 评测规划合并到统一 evidence schema。
