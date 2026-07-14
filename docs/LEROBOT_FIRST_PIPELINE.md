# LeRobot-First Demo Pipeline

> 状态：主线规划 v0.2；data/inference/report scaffold 已落地<br>
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

当前已经封装的第一条 LeRobot-native 模型 demo 是：

```text
dataset.repo_id = lerobot/pusht
policy.type     = act
policy class    = lerobot.policies.act.modeling_act.ACTPolicy
```

也就是 **ACT on PushT**。这条链路用于验证 LeRobot 标准 data-to-inference 结构；FastWAM 是下一条重点 LeRobot-native policy path。

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

已新增：

```bash
make download-lerobot-artifacts
make lerobot-data-smoke
```

目标：

- 加载一个 LeRobot dataset；
- 输出 feature names、observation keys、action shape、fps、episode/task metadata；
- 不训练、不推理，只证明数据读通。

建议脚本：

```text
scripts/lerobot/inspect_dataset.py
scripts/lerobot/run_dataset_smoke.sh
```

默认 `LEROBOT_ALLOW_DOWNLOAD=0`，脚本会设置 Hugging Face offline 环境变量；如果本地没有 dataset 缓存或 `LEROBOT_DATASET_ROOT`，会明确失败，不会偷偷下载大文件。需要在集群上下载公开 dataset 时，先运行 `make download-lerobot-artifacts`；完整下载流程见 [`CLUSTER_ARTIFACTS_RUNBOOK.md`](CLUSTER_ARTIFACTS_RUNBOOK.md)。

### Step 2：training/load smoke

现有：

```bash
make lerobot-train-smoke
```

已具备：

- 默认仍可跑 ACT/PushT；
- 记录 checkpoint 路径和 LeRobot output dir。

仍待扩展：

- 增加 LeRobot-native FastWAM policy config 分支；
- 输出统一 `training_evidence.json`。

### Step 3：offline inference smoke

已新增：

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
scripts/lerobot/run_inference_smoke.sh
```

默认要求 `LEROBOT_POLICY_PATH` 指向本地 checkpoint/pretrained 目录，不会下载权重。默认 `LEROBOT_INFERENCE_DEVICE=cuda`，如果没有 CUDA 会明确失败。

### Step 4：demo-chain report

已新增本地报告生成入口：

```bash
make demo-chain-lerobot-fastwam
```

目标：

- 把 dataset、training/loading、inference 三段结果汇总；
- 形成可交付报告；
- 明确标注这是 offline data-to-inference，不是真机闭环。

最小用法：

```bash
LEROBOT_DATASET_PROFILE=runs/lerobot_native/<run>/dataset_profile.json \
LEROBOT_INFERENCE_EVIDENCE=runs/lerobot_native/<run>/inference_evidence.json \
make demo-chain-lerobot-fastwam
```

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
| `make lerobot-data-smoke` | LeRobot-first 主线的 dataset inspection |
| `make lerobot-train-smoke` | LeRobot-first 主线的 training smoke |
| `make lerobot-infer-smoke` | LeRobot-first 主线的 offline inference smoke |
| `make demo-chain-lerobot-fastwam` | dataset/inference/training evidence 的报告入口 |
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

## 9. 模型和权重管理

模型、dataset、checkpoint 下载和存放规范见 [`MODEL_ARTIFACTS.md`](MODEL_ARTIFACTS.md)。集群下载 runbook 见 [`CLUSTER_ARTIFACTS_RUNBOOK.md`](CLUSTER_ARTIFACTS_RUNBOOK.md)。当前模型 registry 见 [`references/model_registry.yaml`](../references/model_registry.yaml)。

关键原则：

- 当前 LeRobot demo 是 ACT/PushT；
- FastWAM LeRobot-native 是下一条重点 policy；
- FastWAM realrobot overlay 是 custom finetuning backend，不是从零自拟模型；
- 大文件只放项目内 ignored 目录或显式外部共享盘，不提交到仓库。
