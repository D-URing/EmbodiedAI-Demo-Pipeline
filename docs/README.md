# Documentation Index

如果你刚接手项目，按下面顺序读，不要从所有文档里随机找。

## 第一层：必须读

| 文档 | 用途 |
|---|---|
| [`../README.md`](../README.md) | 项目现在是什么，怎么快速跑 |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | 仓库结构，LeRobot / Custom 两条线怎么分 |
| [`STORAGE_AND_ARTIFACTS.md`](STORAGE_AND_ARTIFACTS.md) | 数据、权重、cache、runs 分别放哪里 |
| [`../pipelines/lerobot/README.md`](../pipelines/lerobot/README.md) | LeRobot 主线怎么跑 |
| [`LEROBOT_MULTI_MODEL_PLAN.md`](LEROBOT_MULTI_MODEL_PLAN.md) | LeRobot 多模型训练计划和集群命令 |
| [`OPEN_DATA_AND_EVAL_PLAN.md`](OPEN_DATA_AND_EVAL_PLAN.md) | 开源数据下载分层与评测路线 |
| [`../pipelines/custom_fastwam/README.md`](../pipelines/custom_fastwam/README.md) | FastWAM/custom 主线怎么跑 |

## 第二层：需要细节时读

| 文档 | 用途 |
|---|---|
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | macOS / Linux / SCUT Miniconda 环境细节 |
| [`MODEL_ARTIFACTS.md`](MODEL_ARTIFACTS.md) | 模型、数据、checkpoint 规范 |
| [`CLUSTER_ARTIFACTS_RUNBOOK.md`](CLUSTER_ARTIFACTS_RUNBOOK.md) | 集群下载和排障长说明 |
| [`LEROBOT_FIRST_PIPELINE.md`](LEROBOT_FIRST_PIPELINE.md) | LeRobot-first 设计说明 |
| [`LEROBOT_REPLICATION.md`](LEROBOT_REPLICATION.md) | LeRobot GPU 复刻细节 |
| [`FASTWAM_REALROBOT_INTEGRATION.md`](FASTWAM_REALROBOT_INTEGRATION.md) | FastWAM realrobot overlay 细节 |

## 第三层：规划和历史

| 文档 | 用途 |
|---|---|
| [`00_PROJECT_OVERVIEW.md`](00_PROJECT_OVERVIEW.md) | 项目背景和汇报口径 |
| [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md) | 早期架构设计 |
| [`MASTER_PLAN.md`](MASTER_PLAN.md) | 大规划、任务库、资源映射 |
| [`DEMO_COVERAGE_ROADMAP.md`](DEMO_COVERAGE_ROADMAP.md) | demo 覆盖路线图 |
| [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) | 阶段性实现状态 |
| [`REFERENCE_BASELINE.md`](REFERENCE_BASELINE.md) | 外部基准选择 |
| [`adr/`](adr/) | 架构决策记录 |

## 当前最重要的事实

- 第一阶段主线已升级为 LeRobot 多模型训练：ACT、Diffusion、SmolVLA；
- 已在 SCUT `gpu11` 跑通真实 GPU training smoke，并观察到 2-step loss 下降；
- 立即下载的数据/权重包括 PushT、SVLA SO100 pick-place、Diffusion PushT policy、SmolVLA base；
- FastWAM release 权重和 LIBERO 数据已经下载，但 FastWAM private overlay 还受 GitHub 私有仓库权限阻塞；
- `data/`、`models/`、`hf_cache/`、`runs/`、`upstreams/` 都是 ignored 本地/集群目录，不进 Git。
