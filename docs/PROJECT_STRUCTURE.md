# Project Structure

项目现在按“两个 pipeline + 一个 core + 全局资产池”理解。第二条 pipeline 已从单一 FastWAM 扩展为 Custom WAM 后端族。

```text
.
├── pipelines/
│   ├── lerobot/          # 第一主线：LeRobot data→train→infer
│   ├── custom_wam/       # 第二主线：自拟/custom WAM 后端族
│   │   ├── fastwam/
│   │   └── imagewam/
│   └── custom_fastwam/   # 历史兼容入口
├── configs/
│   ├── lerobot/          # LeRobot 主线配置
│   ├── fastwam/          # Custom WAM / FastWAM 配置
│   ├── imagewam/         # Custom WAM / ImageWAM 配置
│   ├── runs/             # household/mock demo 运行配置
│   └── profiles/         # smoke/dev/release profile
├── scripts/
│   ├── lerobot/          # LeRobot 下载、训练、推理、报告脚本
│   ├── fastwam/          # FastWAM 权重、overlay、训练报告脚本
│   ├── imagewam/         # ImageWAM 上游源码、权重、训练 wrapper
│   └── reference/        # 外部参考项目 pin/fetch
├── src/embodied_demo/    # 项目 core：schema、CLI、mock runner、report
├── tasks/                # household/mock 任务定义
├── demo_chains/          # evidence/report 链路定义
├── docs/                 # 文档
├── references/           # 上游 pin、模型 registry
└── asset/local dirs      # data/models/runs/hf_cache/upstreams 等
```

## 两条 pipeline

两条 pipeline 都不拥有数据或权重。它们只从根目录资产池选择输入：

```text
data/      # dataset pool
models/    # model/checkpoint pool
hf_cache/  # HF/Torch cache
```

### 1. LeRobot pipeline

入口：[`pipelines/lerobot/README.md`](../pipelines/lerobot/README.md)

目标：

```text
LeRobot dataset
  -> official policy training/loading
  -> offline inference
  -> evidence/report
```

当前默认：

```text
dataset = lerobot/pusht
policy  = ACT
```

对应文件：

```text
configs/lerobot/
scripts/lerobot/
runs/lerobot/
data/lerobot/
models/lerobot/        # pretrained policy / stable local checkpoint
```

### 2. Custom WAM pipeline

入口：[`pipelines/custom_wam/README.md`](../pipelines/custom_wam/README.md)

目标：

```text
custom model / custom backend
  -> backend-specific release / training wrapper
  -> train/eval evidence
  -> report into demo pipeline
```

当前后端：

#### FastWAM

```text
official base = yuantianyuan01/FastWAM
private overlay = D-URing/fastwam-realrobot-pipeline
release ckpt = yuanty/fastwam
release data = yuanty/LIBERO-fastwam
```

对应文件：

```text
configs/fastwam/
scripts/fastwam/
runs/fastwam/
data/fastwam/
models/fastwam_release/
upstreams/FastWAM-realrobot/
```

#### ImageWAM

```text
official base = yuyangalin/ImageWAM
default variant = FLUX.2 4B LIBERO
release ckpt = yuyangalin/ImageWAM-FLUX.2-4B-LIBERO
```

对应文件：

```text
configs/imagewam/
scripts/imagewam/
pipelines/custom_wam/imagewam/
runs/imagewam/
models/imagewam/
upstreams/ImageWAM/
```

## Core 不是训练环境

`src/embodied_demo/` 是项目自己的稳定 core：

- task/config schema；
- mock rollout；
- evidence/report；
- FastWAM report adapter；
- CLI。

它不应该安装 CUDA、LeRobot、FastWAM、Isaac 或真机 SDK。集群上对应 `embodied-core` conda env。

## Household/mock demo 的位置

Household/mock demo 不是当前训练主线，但仍保留为展示层：

```text
tasks/
configs/runs/
scenes/mock/
src/embodied_demo/demo_runner.py
```

它的作用是后续把 LeRobot/FastWAM 产物接到家庭任务展示，而不是证明模型能力。

## 文档分层

优先读：

```text
README.md
docs/README.md
pipelines/lerobot/README.md
pipelines/custom_wam/README.md
docs/STORAGE_AND_ARTIFACTS.md
```

长文档保留历史、规划和细节，不作为第一入口。
