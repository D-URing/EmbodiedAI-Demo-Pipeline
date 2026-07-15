# Architecture

> 状态：当前架构说明<br>
> 日期：2026-07-15<br>
> 关联：[`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)、[`TRAINING_AND_INFERENCE.md`](TRAINING_AND_INFERENCE.md)

本仓库现在只维护训练、推理和证据归档基座，不再维护本地符号 rollout。

## 核心数据流

```text
Open dataset / local dataset
  -> LeRobot or custom backend training
  -> checkpoint / policy loading
  -> offline inference or train-log parsing
  -> evidence JSON + report
```

## 分层职责

| 层 | 职责 | 当前位置 |
|---|---|---|
| Pipeline | LeRobot 与 custom WAM 两条主线 | `pipelines/` |
| Experiment | 多次训练/推理实验的启动入口 | `experiments/` |
| Config | backend 默认参数 | `configs/` |
| Script | 下载、转换、训练、推理、解析 | `scripts/` |
| Evidence | schema、report、handoff | `src/embodied_demo/` |
| Asset registry | 数据、权重、cache、上游 pin | `references/`、`data/`、`models/`、`hf_cache/` |

## Python core

```text
src/embodied_demo/
├── cli.py              # embodied-demo report/export entry
├── config.py           # YAML compose + dump helpers
├── fastwam_report.py   # FastWAM training evidence importer/report
└── schemas/            # observation/action/evaluation/training evidence contracts
```

`src/embodied_demo/` 不安装 CUDA、LeRobot、FastWAM、Isaac 或真机 SDK。它只处理轻量合同和报告。

## LeRobot path

```text
data/lerobot/
models/lerobot/
configs/lerobot/
scripts/lerobot/
experiments/lerobot/
runs/experiments/lerobot/
```

目标是复刻 LeRobot 的 data → train/load → inference 路径。当前已覆盖 ACT、Diffusion、SmolVLA 和 FastWAM/LIBERO 推理入口。

## Custom WAM path

```text
data/custom/
models/custom/
configs/fastwam/
configs/imagewam/
scripts/fastwam/
scripts/imagewam/
experiments/custom/
runs/experiments/custom/
```

目标是保留自建模型或外部项目 overlay 的训练/评测入口。当前 custom 后端包括 FastWAM 和 ImageWAM。

## 边界

- 不维护 CPU toy trainer；
- 不维护本地符号 rollout；
- 不声明仿真或真机 closed-loop 成功；
- Make 只负责下载、环境、转换和检查；
- 训练/推理统一从 `experiments/` 启动。
