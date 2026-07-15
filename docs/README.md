# Documentation Index

如果你刚接手项目，按下面顺序读。本文档目录只保留当前可维护的公开项目文档。

## 必须读

| 文档 | 用途 |
|---|---|
| [`../README.md`](../README.md) | 项目现在是什么，怎么快速跑 |
| [`BOOTSTRAP.md`](BOOTSTRAP.md) | 从新 checkout 到可用工作区：目录、环境、数据、模型、cache |
| [`TRAINING_AND_INFERENCE.md`](TRAINING_AND_INFERENCE.md) | 训练、推理、下载、结果路径和排障的唯一主入口 |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | 仓库结构，LeRobot / Custom 两条线怎么分 |
| [`STORAGE_AND_ARTIFACTS.md`](STORAGE_AND_ARTIFACTS.md) | 数据、权重、cache、runs 分别放哪里 |
| [`../pipelines/lerobot/README.md`](../pipelines/lerobot/README.md) | LeRobot 主线说明和入口索引 |
| [`../experiments/README.md`](../experiments/README.md) | 训练/推理实验从哪里启动，结果怎么存 |
| [`../pipelines/custom/README.md`](../pipelines/custom/README.md) | Custom WAM 主线怎么跑 |

## 需要细节时读

| 文档 | 用途 |
|---|---|
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | macOS / Linux / SCUT Miniconda 环境细节 |
| [`MODEL_ARTIFACTS.md`](MODEL_ARTIFACTS.md) | 模型、数据、checkpoint 规范 |
| [`OPEN_DATA_AND_EVAL_PLAN.md`](OPEN_DATA_AND_EVAL_PLAN.md) | 开源数据下载分层与评测路线 |
| [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md) | pipeline 分层和代码结构 |
| [`FASTWAM_REALROBOT_INTEGRATION.md`](FASTWAM_REALROBOT_INTEGRATION.md) | FastWAM realrobot overlay 细节 |
| [`IMAGEWAM_INTEGRATION.md`](IMAGEWAM_INTEGRATION.md) | ImageWAM 后端接入规划和命令 |

## 当前最重要的事实

- 当前训练/推理命令以 [`TRAINING_AND_INFERENCE.md`](TRAINING_AND_INFERENCE.md) 为准；
- 已在 SCUT `gpu11` 跑通 LeRobot ACT/PushT 真实 GPU training smoke，并观察到 2-step loss 下降；
- 已在 SCUT `gpu11` 跑通 LeRobot FastWAM/LIBERO CUDA inference smoke，输出 action evidence；
- LeRobot 数据、policy、FastWAM v3 转换和 Wan/T5 base cache 均已项目内落盘；
- Custom FastWAM realrobot 入口已准备，真实训练仍依赖私有 overlay 权限；
- ImageWAM 已加入 Custom WAM 结构，默认目标是 FLUX.2 4B + LIBERO；
- `data/`、`models/`、`hf_cache/`、`runs/`、`upstreams/` 都是 ignored 本地/集群目录，不进 Git。
