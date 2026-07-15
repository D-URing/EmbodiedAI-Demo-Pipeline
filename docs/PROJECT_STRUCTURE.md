# Project Structure

项目现在按“两个 pipeline + 一个 core + 一组本地资产目录”理解。

```text
.
├── pipelines/
│   ├── lerobot/          # 第一主线：LeRobot data→train→infer
│   └── custom_fastwam/   # 第二主线：自拟/custom backend，以 FastWAM 为例
├── configs/
│   ├── lerobot/          # LeRobot 主线配置
│   ├── fastwam/          # Custom/FastWAM 配置
│   ├── runs/             # household/mock demo 运行配置
│   └── profiles/         # smoke/dev/release profile
├── scripts/
│   ├── lerobot/          # LeRobot 下载、训练、推理、报告脚本
│   ├── fastwam/          # FastWAM 权重、overlay、训练报告脚本
│   └── reference/        # 外部参考项目 pin/fetch
├── src/embodied_demo/    # 项目 core：schema、CLI、mock runner、report
├── tasks/                # household/mock 任务定义
├── demo_chains/          # evidence/report 链路定义
├── docs/                 # 文档
├── references/           # 上游 pin、模型 registry
└── ignored local dirs    # data/models/runs/hf_cache/upstreams 等
```

## 两条 pipeline

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
models/lerobot/        # 后续 checkpoint/pretrained policy
```

### 2. Custom/FastWAM pipeline

入口：[`pipelines/custom_fastwam/README.md`](../pipelines/custom_fastwam/README.md)

目标：

```text
custom model / custom backend
  -> FastWAM release / realrobot overlay
  -> train/eval evidence
  -> report into demo pipeline
```

当前默认：

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
pipelines/custom_fastwam/README.md
docs/STORAGE_AND_ARTIFACTS.md
```

长文档保留历史、规划和细节，不作为第一入口。

