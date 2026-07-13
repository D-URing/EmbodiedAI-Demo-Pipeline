# EmbodiedAI Demo Pipeline

面向家庭与生活服务场景的具身智能 Demo 工程基座。项目采用 **contract-first、headless-first、evaluation-first、backend-switchable** 的路线：先稳定任务、观测、动作、运行和评测契约，再逐步接入 mock、离线回放、NVIDIA 仿真集群、VLA 和真实机器人。

当前不以训练大模型、同时适配多个仿真器、搭建复杂可视化或立即接真机为目标。完整规划与优先级见 [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md)，实际落地状态见 [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md)，本地与集群环境配置见 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)，复刻基准决策见 [`docs/REFERENCE_BASELINE.md`](docs/REFERENCE_BASELINE.md)。

## 当前状态

- 版本：M1 Core Contract `v0.1.0`
- Python：3.11+
- 已实现：严格 schema、YAML 显式组合、两项任务定义、运行配置、CLI 校验/dry-run、JSON Schema 导出、单元测试
- 已预留：mock/replay/sim/real 模式，local/Slurm launcher，inproc/WebSocket policy transport，CPU/GPU 资源声明
- 已固化：XPolicyLab `demo_policy`/debug flow 作为复刻基准，RoboDojo 作为后续外部仿真评测目标，LeRobot 作为后续数据/训练格式参考
- 当前可运行：`embodied-demo run --config configs/runs/tabletop_sorting_mock.yaml` 可生成第一版 mock demo artifacts
- 当前 LeRobot 复刻：`make lerobot-train-smoke` 在 CUDA 集群上调用真实 `lerobot-train` 训练 ACT/PushT smoke
- 下一里程碑：M2 Evaluation Core 完整化与 M3 deterministic mock demo 扩展
- 暂缓：Viewer、真实 simulator adapter、重量级模型、大数据下载、多节点运行、真机闭环

## 快速开始

推荐使用仓库提供的受约束环境：

```bash
make setup
make doctor
```

`make setup` 创建 `.venv` 并使用 `requirements/constraints-py311.txt` 中经过验证的版本；不要在这个 core 环境里直接安装 CUDA、Isaac、VLA 或真机 SDK。macOS、Linux、离线节点和 NVIDIA 集群的准备方式见[环境配置指南](docs/ENVIRONMENT.md)。

验证两项任务和运行配置：

```bash
embodied-demo list-tasks
embodied-demo validate --config configs/runs/tabletop_sorting_mock.yaml
embodied-demo validate --config configs/runs/towel_folding_mock.yaml
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

在 CUDA 集群上运行真实 LeRobot 训练 smoke，观察 loss 是否下降：

```bash
bash scripts/lerobot/install_lerobot_cluster.sh
make lerobot-train-smoke
```

该入口不会 fallback 到 CPU，也不调用本仓库的 toy trainer。它会调用官方 `lerobot-train`，默认复刻 LeRobot 的 `lerobot/pusht` + `act` 训练路径，并在 `runs/lerobot/...` 下保存 stdout、loss summary、LeRobot 输出目录和 checkpoint。详细说明见 [`docs/LEROBOT_REPLICATION.md`](docs/LEROBOT_REPLICATION.md)。

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
├── docs/
│   ├── ENVIRONMENT.md            # macOS/Linux/NVIDIA 集群环境配置
│   └── MASTER_PLAN.md            # 项目范围、架构、资源映射与路线图
├── requirements/                 # 经过验收的 Python 版本约束
├── references/                   # 上游复刻基准和引用 pin
├── scenes/mock/                  # 轻量场景描述；不声称物理真实性
├── src/embodied_demo/
│   ├── schemas/                  # Task/Observation/Action/Run/Evaluation 契约
│   ├── config.py                 # YAML 组合、校验和 resolved config
│   ├── demo_runner.py            # 第一版 deterministic mock demo runner
│   ├── registry.py               # 任务注册表加载
│   └── cli.py                    # validate/list-tasks/dry-run/run/export-schema
├── tasks/
│   ├── registry.yaml
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

当前两个任务仍标记为 `experimental`：它们的契约和配置可运行，但只有在 deterministic runner、evaluator、golden artifacts 和跨机器回归都落地后，才会升级为 `supported`。
