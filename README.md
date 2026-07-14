# EmbodiedAI Demo Pipeline

面向家庭与生活服务场景的具身智能 Demo 工程基座。项目采用 **contract-first、headless-first、evaluation-first、backend-switchable** 的路线：先稳定任务、观测、动作、运行和评测契约，再逐步接入 mock、离线回放、NVIDIA 仿真集群、VLA 和真实机器人。

当前不以训练大模型、同时适配多个仿真器、搭建复杂可视化或立即接真机为目标。第一主线是 **LeRobot-first data-to-inference pipeline**：从 LeRobot 数据读取，到 policy 训练/加载，再到离线推理和证据报告。当前 LeRobot demo 默认模型是 **ACT on PushT**。第一次看项目建议先读 [`docs/00_PROJECT_OVERVIEW.md`](docs/00_PROJECT_OVERVIEW.md)、[`docs/LEROBOT_FIRST_PIPELINE.md`](docs/LEROBOT_FIRST_PIPELINE.md)、[`docs/MODEL_ARTIFACTS.md`](docs/MODEL_ARTIFACTS.md)、[`docs/CLUSTER_ARTIFACTS_RUNBOOK.md`](docs/CLUSTER_ARTIFACTS_RUNBOOK.md) 和 [`docs/01_ARCHITECTURE.md`](docs/01_ARCHITECTURE.md)。

## 如何理解这个项目

项目现在按“一条主干 + 两个扩展层”理解：

| 层级 | 回答的问题 | 当前状态 |
|---|---|---|
| LeRobot 主干 | 数据读取到 policy 推理是否打通？ | ACT/PushT 已有 data/train/infer/report 入口 |
| Custom backend 扩展 | 私有 FastWAM、自建模型和特殊 recipe 如何保留？ | FastWAM overlay 已接入 |
| Household 应用层 | 家庭任务如何展示、评测和后续接仿真/真机？ | 4 个 R1 mock demo |

## 当前状态

- 版本：M1 Core Contract `v0.1.0`
- Python：3.11+
- 已实现：严格 schema、YAML 显式组合、四项任务定义、运行配置、CLI 校验/dry-run、JSON Schema 导出、单元测试
- 已预留：mock/replay/sim/real 模式，local/Slurm launcher，inproc/WebSocket policy transport，CPU/GPU 资源声明
- 已固化：LeRobot-first 作为第一 demo 管线基准；FastWAM 是 LeRobot-native policy 路径和 custom overlay 双路径；RoboDojo 作为后续外部仿真评测目标
- 当前可运行：`embodied-demo run --config configs/runs/tabletop_sorting_mock.yaml` 可生成第一版 mock demo artifacts
- 当前 LeRobot 复刻：默认模型是 ACT/PushT；`make lerobot-data-smoke`、`make lerobot-train-smoke`、`make lerobot-infer-smoke` 已有入口；默认不下载大文件，真实运行需要集群/缓存里的 dataset 和 checkpoint
- 当前 FastWAM 集成：官方 LeRobot-native FastWAM 作为优先 policy 路径；私有 FastWAM overlay 作为自建模型/真机数据扩展路径
- 当前 Demo Chain：`embodied-demo report-fastwam --run-dir <runs/fastwam/...>` 可把 FastWAM 训练产物归一化成 demo 交付报告
- 当前规划格局：LeRobot data-to-inference 是主线；household mock 是应用层；私有 FastWAM overlay 是 custom backend 扩展
- 下一里程碑：实现 `lerobot-data-smoke`、`lerobot-infer-smoke` 和 LeRobot/FastWAM demo-chain report
- 暂缓：Viewer、真实 simulator adapter、重量级模型、大数据下载、多节点运行、真机闭环

## 快速开始

推荐使用仓库提供的受约束环境：

```bash
make setup
make doctor
```

`make setup` 创建 `.venv` 并使用 `requirements/constraints-py311.txt` 中经过验证的版本；不要在这个 core 环境里直接安装 CUDA、Isaac、VLA 或真机 SDK。macOS、Linux、离线节点和 NVIDIA 集群的准备方式见[环境配置指南](docs/ENVIRONMENT.md)。

验证四项任务和运行配置：

```bash
embodied-demo list-tasks
embodied-demo validate --config configs/runs/tabletop_sorting_mock.yaml
embodied-demo validate --config configs/runs/towel_folding_mock.yaml
embodied-demo validate --config configs/runs/kitchen_counter_sorting_mock.yaml
embodied-demo validate --config configs/runs/drawer_pick_place_mock.yaml
```

展开完整配置但不执行 rollout：

```bash
embodied-demo dry-run \
  --config configs/runs/tabletop_sorting_mock.yaml \
  --output runs/tabletop_sorting/resolved.yaml
```

运行第一版可交付 mock demo：

```bash
embodied-demo run --config configs/runs/tabletop_sorting_mock.yaml
embodied-demo run --config configs/runs/towel_folding_mock.yaml
```

运行完成后会在 `runs/<run_name>/<run_id>/` 下生成 `manifest.yaml`、`events.jsonl`、`result.json`、`metrics.json` 和 `report.md`。也可以直接执行：

```bash
make demo
```

扩展覆盖 demo 包含厨房台面整理和抽屉取放：

```bash
make demo-extended
```

在 CUDA 集群上运行真实 LeRobot 训练 smoke，观察 loss 是否下降：

```bash
bash scripts/lerobot/install_lerobot_cluster.sh
make download-lerobot-artifacts
make lerobot-train-smoke
```

该入口不会 fallback 到 CPU，也不调用本仓库的 toy trainer。它会调用官方 `lerobot-train`，默认复刻 LeRobot 的 `lerobot/pusht` + `act` 训练路径，并在 `runs/lerobot/...` 下保存 stdout、loss summary、LeRobot 输出目录和 checkpoint。详细说明见 [`docs/LEROBOT_REPLICATION.md`](docs/LEROBOT_REPLICATION.md)。

在已有 LeRobot dataset 缓存和本地 checkpoint 的环境里，跑 data-to-inference smoke：

```bash
make lerobot-data-smoke
LEROBOT_POLICY_PATH=/path/to/local/checkpoint make lerobot-infer-smoke
```

默认 `LEROBOT_ALLOW_DOWNLOAD=0`，不会下载大文件；如确实要在集群上下载，需显式设置 `LEROBOT_ALLOW_DOWNLOAD=1`。

公开开源数据和权重下载已经封装为 Make target：

```bash
# 下载 LeRobot PushT dataset；可选下载 LeRobot policy repo
make download-lerobot-artifacts

# 下载 FastWAM release 权重和 stats
make download-fastwam-artifacts
```

集群路径、Hugging Face cache、policy checkpoint 和 FastWAM release 的完整命令见 [`docs/CLUSTER_ARTIFACTS_RUNBOOK.md`](docs/CLUSTER_ARTIFACTS_RUNBOOK.md)。

在 CUDA 集群上接入你已有的 FastWAM 真机训练/评测 pipeline：

```bash
bash scripts/fastwam/prepare_fastwam_overlay.sh
FASTWAM_MODE=pilot FASTWAM_RECIPE=joint_base \
bash scripts/fastwam/run_realrobot_train_eval.sh
```

该入口会把官方 FastWAM 与私有 realrobot overlay 组合为外部 backend，并在 `runs/fastwam/...` 下保存 stdout、`loss_summary.json` 和 FastWAM 原生 checkpoint 路径。详细说明见 [`docs/FASTWAM_REALROBOT_INTEGRATION.md`](docs/FASTWAM_REALROBOT_INTEGRATION.md)。

把 FastWAM pilot 结果整理成第一版 demo chain 交付报告：

```bash
embodied-demo report-fastwam --run-dir runs/fastwam/<run_name>/<run_id>
```

或使用 Makefile：

```bash
FASTWAM_RUN_DIR=runs/fastwam/<run_name>/<run_id> make demo-chain-fastwam
```

产物会写到 `runs/demo_chains/fastwam_realrobot_v0/<run_id>/`，包括 `training_evidence.json`、`checkpoint_summary.json`、`report.md` 和 `handoff.md`。

运行测试和导出公共 schema：

```bash
pytest
embodied-demo export-schema --output-dir build/schemas
```

也可以使用 `make test`、`make validate`、`make dry-run` 和 `make schemas`。

复核外部复刻基准时，可以只拉取固定 commit 到本机 cache：

```bash
make reference-fetch
```

该命令不会安装 XPolicyLab 依赖、下载数据或启动仿真器，只准备上游源码锚点。

## 目录结构

```text
.
├── configs/
│   ├── base.yaml                 # 本地、headless、低成本默认值
│   ├── profiles/                 # smoke / dev / release，不允许混报
│   └── runs/                     # 可直接校验和展开的运行入口
├── demo_chains/                  # 可交付 demo/evidence 链路定义
├── docs/
│   ├── ENVIRONMENT.md            # macOS/Linux/NVIDIA 集群环境配置
│   ├── 00_PROJECT_OVERVIEW.md    # 新同事/汇报入口
│   ├── 01_ARCHITECTURE.md        # Pipeline 分层与代码结构
│   ├── CLUSTER_ARTIFACTS_RUNBOOK.md # 集群下载开源数据/模型和 smoke 验证
│   ├── LEROBOT_FIRST_PIPELINE.md # LeRobot-first 主线
│   ├── MODEL_ARTIFACTS.md        # 模型/数据/权重下载与存放规范
│   ├── DEMO_COVERAGE_ROADMAP.md  # demo 覆盖矩阵与 readiness 分级
│   ├── FASTWAM_REALROBOT_INTEGRATION.md
│   └── MASTER_PLAN.md            # 项目范围、架构、资源映射与路线图
├── requirements/                 # 经过验收的 Python 版本约束
├── references/                   # 上游复刻基准和引用 pin
│   └── model_registry.yaml       # 模型路径、状态和 artifact 约定
├── scripts/fastwam/              # FastWAM 外部 backend 准备、启动和日志解析
├── scenes/mock/                  # 轻量场景描述；不声称物理真实性
├── src/embodied_demo/
│   ├── environments/             # mock/replay/sim/real backend 实现
│   ├── policies/                 # scripted/learned/VLA policy adapter
│   ├── rollout/                  # 执行循环与 artifact 生成
│   ├── schemas/                  # Task/Observation/Action/Run/Evaluation 契约
│   ├── config.py                 # YAML 组合、校验和 resolved config
│   ├── demo_runner.py            # 兼容入口，转发到 rollout.mock_runner
│   ├── fastwam_report.py         # FastWAM 训练产物转 demo evidence/report
│   ├── registry.py               # 任务注册表加载
│   └── cli.py                    # validate/list-tasks/dry-run/run/report/export-schema
├── tasks/
│   ├── registry.yaml
│   ├── drawer_pick_place_v1/
│   ├── kitchen_counter_sorting_v1/
│   ├── tabletop_sorting_v1/
│   └── towel_folding_v1/
└── tests/                        # schema、配置组合、CLI 回归测试
```

## 配置约定

`configs/runs/*.yaml` 是运行入口，可通过 `extends` 组合公共默认值和评测档位。合并规则保持简单且显式：mapping 递归合并，scalar 和 list 由后出现的配置整体覆盖。所有公共 schema 均拒绝未知字段，拼写错误不会被静默忽略。

任务文件明确区分：

- policy 可见、evaluator-only、debug-only 和 restricted 观测；
- 当前 supported backend 与仅在路线图中的 planned backend；
- mock 的真实性等级和已知局限；
- 真机执行授权与安全标签；
- RoboDojo 风格的能力维度、阶段谓词和总计 100 分的 partial progress。

`dry-run` 产物会保存所有合并后的运行参数、完整 TaskSpec 和来源文件，可作为后续实验 manifest 的输入。

## 边界

本仓库只服务于 Demo 项目规划和工程落地。论文、模型与开源生态研究笔记继续由同级的 `EmbodiedAI-Research/` 知识库维护；这里只保留会影响接口、实现与验收的工程结论。

当前四个 household mock 任务仍标记为 `experimental`：它们的契约和配置可运行，但只有在 deterministic runner、evaluator、golden artifacts 和跨机器回归都落地后，才会升级为 `supported`。
