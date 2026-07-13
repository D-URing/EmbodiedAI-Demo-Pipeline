# 项目总览：EmbodiedAI Demo Evidence Pipeline

> 状态：当前团队入口<br>
> 日期：2026-07-14<br>
> 适合读者：第一次接触仓库的同事、需要快速汇报当前进展的人

## 1. 一句话

本项目不是单个机器人视频 demo，也不是从零训练一个大模型，而是一套 **具身智能 demo evidence pipeline**：

> 用统一任务定义、运行入口、日志产物、评测协议和训练证据链，把家庭服务 mock demo、真实 CUDA 训练、后续仿真/真机接入逐步打通。

## 2. 为什么会看起来有几条线

因为项目同时服务三个问题，而这三个问题不能混报：

| 问题 | 证据层 | 当前状态 |
|---|---|---|
| 家庭服务 demo pipeline 能不能跑？ | R1 Household Mock Rollout | 已有 4 个可运行 demo |
| 项目是不是真能训练模型、看 loss？ | R2 Training Evidence | 已接 LeRobot 与 FastWAM 入口 |
| 模型在仿真/真机里是否真的完成任务？ | R4/R6 Capability Evidence | 后续接 RoboDojo/RoboCasa/RoboTwin/real |

所以 FastWAM 和厨房整理不是并列的“两个 demo”。更准确的说法是：

- 厨房整理、叠毛巾、抽屉取放属于 **家庭任务 demo 线**；
- FastWAM / LeRobot 属于 **真实训练证据线**；
- RoboDojo / RoboCasa / RoboTwin / 真机属于 **后续能力评测线**。

## 3. 当前已经有什么

### R1：可运行 household mock demo

一条命令运行四个家庭任务：

```bash
make demo-extended
```

当前任务：

| 任务 | 场景 | 证明什么 |
|---|---|---|
| `tabletop_sorting_v1` | 桌面整理 | 多物体分类归位、stage progress、artifact 链路 |
| `towel_folding_v1` | 毛巾折叠 | 柔性物体语义、折叠阶段、对齐评测形状 |
| `kitchen_counter_sorting_v1` | 厨房台面整理 | 厨房语义、类别到区域、做菜前置整理 |
| `drawer_pick_place_v1` | 抽屉取放 | 关节物体状态机、打开后取放、长时序 |

这些 demo 证明工程链路可运行，不证明真实物理能力。

### R2：真实训练证据

训练入口是 CUDA-only，不提供 CPU toy fallback。

| 后端 | 作用 | 入口 |
|---|---|---|
| LeRobot | 复刻轻量训练 smoke，验证官方训练链路 | `make lerobot-train-smoke` |
| FastWAM | 接入已有 realrobot pipeline，产出 loss、checkpoint、handoff | `make fastwam-train-smoke` / `FASTWAM_MODE=pilot ...` |

FastWAM 的正确使用方式是：在 NVIDIA 集群跑 pilot，拿真实日志生成报告：

```bash
embodied-demo report-fastwam --run-dir runs/fastwam/<run_name>/<run_id>
```

这回答“loss 有没有下降”，不回答“厨房任务成功率是多少”。

## 4. 项目主轴

项目可以按五层理解：

```text
Task Layer        定义要做什么：任务、场景、阶段、成功条件
Runtime Layer     定义怎么跑：runner、policy、environment、config
Evidence Layer    定义留下什么证据：events、metrics、result、report、manifest
Backend Layer     定义接哪些外部生态：LeRobot、FastWAM、RoboDojo、RoboCasa...
Roadmap Layer     定义下一步扩哪里：R0-R6 readiness、任务覆盖、集群路线
```

最重要的边界：

- Task 不直接依赖模型或仿真器；
- Policy 不直接 import simulator；
- Environment 不直接 import 模型；
- Training evidence importer 只读取训练产物，不控制机器人；
- Viewer 永远是 artifacts 的消费者，不进入控制闭环。

## 5. 仓库怎么读

建议新同事按这个顺序读：

1. 本文件：项目是什么。
2. [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md)：代码和数据流怎么分层。
3. [`DEMO_COVERAGE_ROADMAP.md`](DEMO_COVERAGE_ROADMAP.md)：任务覆盖和 readiness。
4. [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md)：当前哪些已实现。
5. [`FASTWAM_REALROBOT_INTEGRATION.md`](FASTWAM_REALROBOT_INTEGRATION.md)：如何在集群跑 FastWAM。

## 6. 对外汇报版本

可以这样讲：

> 我们搭的是一个具身智能 demo evidence pipeline。当前已经有四个家庭服务 R1 mock demo，能统一跑任务、日志、评测和报告；同时接了 LeRobot/FastWAM 作为 R2 真实 CUDA 训练证据链，可以在集群上看 loss、checkpoint 和 handoff。下一阶段会把任务库继续扩到衣物/清洁，并把同一套任务逐步接 replay、RoboDojo/RoboCasa/RoboTwin 仿真和真机 shadow。

## 7. 下一步

短期最该做三件事：

1. 为 `make demo-extended` 增加统一 summary report；
2. 补一个衣物或清洁 R1 mock demo，例如 `laundry_sorting_v1` 或 `trash_sorting_v1`；
3. 在 NVIDIA 集群跑 FastWAM `pilot`，生成真实 loss 下降报告。
