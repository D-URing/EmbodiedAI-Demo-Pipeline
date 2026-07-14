# LeRobot-First Demo Pipeline

> 状态：主线规划 v0.1<br>
> 日期：2026-07-14<br>
> 关联：[`adr/0003-lerobot-first-fastwam-pipeline.md`](adr/0003-lerobot-first-fastwam-pipeline.md)、[`LEROBOT_REPLICATION.md`](LEROBOT_REPLICATION.md)、[`FASTWAM_REALROBOT_INTEGRATION.md`](FASTWAM_REALROBOT_INTEGRATION.md)

## 1. 新主线

当前项目主线调整为 **LeRobot-first**：

```text
LeRobot Dataset
  -> dataset inspection
  -> policy training or checkpoint loading
  -> offline policy inference
  -> action/evidence/report
```

这条主线回答：

- 数据能不能按 LeRobot 格式读取？
- policy 能不能用 LeRobot 的配置/接口训练或加载？
- 推理能不能从 observation/batch 产生 action？
- loss、checkpoint、action shape、latency、设备和版本能不能被记录成可交付报告？

## 2. FastWAM 的新定位

FastWAM 不再被描述成 LeRobot 之外的一条完全独立主线，而是拆成两个层级：

| 路径 | 定位 | 用途 |
|---|---|---|
| LeRobot-native FastWAM | 官方优先路径 | 使用 LeRobot 的 `policy.type=fastwam`、LeRobotDataset、LeRobot policy API 跑 data-to-inference |
| Custom FastWAM overlay | 内部扩展路径 | 使用 `D-URing/fastwam-realrobot-pipeline` 支持私有真机数据、7D/10D recipe、集群训练和未来自建模型 |

这意味着：短期 demo 交付应该优先证明 LeRobot-native data-to-inference；内部 overlay 继续保留，作为后续真实数据、私有模型和新模型研发的扩展层。

## 3. 管线分层

```text
Source Data Layer
  LeRobotDataset / local LeRobot v3 data / Hugging Face dataset repo

Policy Layer
  ACT / Diffusion Policy / FastWAM / future custom policy

Execution Layer
  dataset inspection / train smoke / checkpoint load / offline inference

Evidence Layer
  dataset_profile.json / training_evidence.json / inference_evidence.json / report.md

Application Layer
  household tasks / replay evaluation / RoboDojo-RoboCasa-RoboTwin / real shadow
```

现在最重要的是 Execution Layer 和 Evidence Layer，而不是继续增加很多 household mock tasks。

## 4. 第一版 demo-chain

建议新增链路：

```text
demo_chains/lerobot_fastwam_data_to_inference_v0.yaml
```

阶段定义：

| 阶段 | 目标 | 产物 |
|---|---|---|
| dataset_inspection | 读取 LeRobot dataset，记录 features、fps、shape、样例 batch | `dataset_profile.json` |
| train_or_load_policy | 训练 smoke 或加载 checkpoint | `training_evidence.json` / `checkpoint_summary.json` |
| offline_inference | 对 dataset sample 或 mock observation 做 policy inference | `inference_evidence.json` |
| evidence_report | 汇总版本、数据、模型、action shape、latency、边界 | `report.md`、`handoff.md` |

第一版可以先使用官方轻量数据集和小模型路径，等集群可用后再切到 FastWAM。

## 5. 推荐实现顺序

### Step 1：dataset smoke

新增：

```bash
make lerobot-data-smoke
```

目标：

- 加载一个 LeRobot dataset；
- 输出 feature names、observation keys、action shape、fps、episode/task metadata；
- 不训练、不推理，只证明数据读通。

建议脚本：

```text
scripts/lerobot/inspect_dataset.py
```

### Step 2：training/load smoke

现有：

```bash
make lerobot-train-smoke
```

需要扩展：

- 默认仍可跑 ACT/PushT；
- 增加 FastWAM policy config 分支；
- 输出统一 `training_evidence.json`；
- 记录 checkpoint 路径和 LeRobot output dir。

### Step 3：offline inference smoke

新增：

```bash
make lerobot-infer-smoke
```

目标：

- 加载 LeRobot checkpoint 或预训练 policy；
- 从 dataset sample 构造输入；
- 调用 LeRobot policy API；
- 输出 action tensor / action chunk；
- 记录 shape、dtype、device、latency、batch size。

建议脚本：

```text
scripts/lerobot/run_policy_inference_smoke.py
```

### Step 4：demo-chain report

新增：

```bash
make demo-chain-lerobot-fastwam
```

目标：

- 把 dataset、training/loading、inference 三段结果汇总；
- 形成可交付报告；
- 明确标注这是 offline data-to-inference，不是真机闭环。

## 6. 自建模型管线是否还需要？

需要，而且很重要。

LeRobot-first 解决的是第一阶段 demo 管线的公共基准：数据格式、训练入口、policy API、推理路径。自建模型管线解决的是未来研发问题：

- 新 policy 还没有进入 LeRobot；
- 私有 real-robot 数据需要特殊 sampler、normalizer 或 action head；
- FastWAM overlay 有 7D/10D、V6 decision、codebook、DAgger 等内部 recipe；
- 新模型需要自定义训练 loop、分布式策略或额外模态；
- 后续真机/仿真评测需要本项目自己的 evidence/report contract。

因此正确结构不是二选一，而是：

```text
LeRobot-compatible path first
Custom model/backend path retained
Shared evidence/report contract above both paths
```

## 7. 当前已有工作的重新定位

| 已有工作 | 新定位 |
|---|---|
| `make lerobot-train-smoke` | LeRobot-first 主线的 training smoke |
| FastWAM realrobot overlay | custom FastWAM / future self-built model extension |
| `embodied-demo report-fastwam` | training evidence importer，可复用到 LeRobot-native FastWAM |
| 四个 household mock demo | application/evaluation layer，用于后续任务展示和报告，不是主线第一验收 |
| `DEMO_COVERAGE_ROADMAP.md` | 任务库路线，不再抢 data-to-inference 主线 |

## 8. 当前不做

- 不继续优先堆新的 household mock task；
- 不把 offline inference 说成 closed-loop success；
- 不把私有 FastWAM overlay 删除或降级为无用资产；
- 不把 CUDA/LeRobot/FastWAM 依赖塞进 core `.venv`；
- 不等待真机就地验证第一条 data-to-inference 链路。
