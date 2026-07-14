# 项目总览：EmbodiedAI Demo Evidence Pipeline

> 状态：当前团队入口<br>
> 日期：2026-07-14<br>
> 适合读者：第一次接触仓库的同事、需要快速汇报当前进展的人

## 1. 一句话

本项目不是单个机器人视频 demo，也不是从零训练一个大模型，而是一套以 **LeRobot 为主参考** 的具身智能 demo evidence pipeline：

> 先按 LeRobot 跑通数据读取、policy 训练/加载、离线推理和证据报告；再把家庭服务任务、自建模型、FastWAM 私有 overlay、后续仿真/真机接入到同一套 evidence contract。

## 2. 为什么会看起来有几条线

因为项目同时服务三个层级，而这三个层级不能混报：

| 层级 | 回答的问题 | 当前状态 |
|---|---|---|
| LeRobot 主干 | 从数据读取到 policy 推理是否打通？ | 训练 smoke 已有，data/inference smoke 下一步 |
| Custom backend 扩展 | 私有 FastWAM、自建模型和特殊 recipe 怎么保留？ | FastWAM overlay 已有 |
| Household 应用层 | 家庭任务如何展示、评测和后续接仿真/真机？ | 已有 4 个 R1 mock demo |

所以 FastWAM、LeRobot 和厨房整理不是并列的“几个 demo”。更准确的说法是：

- LeRobot 是第一阶段主干；
- FastWAM 是 LeRobot-native policy 路径，同时也有私有 custom overlay；
- 厨房整理、叠毛巾、抽屉取放属于 household 应用/评测层；
- RoboDojo / RoboCasa / RoboTwin / 真机属于后续 capability evidence。

## 3. 当前已经有什么

### 主线：LeRobot data-to-inference

目标链路：

```text
LeRobotDataset
  -> dataset inspection
  -> train or load policy
  -> offline inference
  -> evidence report
```

当前已有 `make lerobot-train-smoke`，下一步补 `make lerobot-data-smoke` 和 `make lerobot-infer-smoke`。

### 应用层：可运行 household mock demo

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

### 扩展层：FastWAM 和自建模型

FastWAM 有两个定位：

| 路径 | 作用 |
|---|---|---|
| LeRobot-native FastWAM | 官方优先路径，作为 LeRobot policy 类型接入 data-to-inference 主线 |
| Custom FastWAM overlay | 内部扩展路径，支持私有真机数据、7D/10D recipe、集群训练和未来自建模型 |

私有 overlay 的正确使用方式是：在 NVIDIA 集群跑 pilot，拿真实日志生成报告：

```bash
embodied-demo report-fastwam --run-dir runs/fastwam/<run_name>/<run_id>
```

这回答“内部 FastWAM overlay 的 loss 有没有下降”，不回答“厨房任务成功率是多少”。

## 4. 项目主轴

项目可以按五层理解：

```text
Task Layer        定义要做什么：任务、场景、阶段、成功条件
Runtime Layer     定义怎么跑：runner、policy、environment、config
Evidence Layer    定义留下什么证据：events、metrics、result、report、manifest
Backend Layer     定义 LeRobot-native、custom backend、RoboDojo/RoboCasa 等接入
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
5. [`LEROBOT_FIRST_PIPELINE.md`](LEROBOT_FIRST_PIPELINE.md)：LeRobot data-to-inference 主线。
6. [`FASTWAM_REALROBOT_INTEGRATION.md`](FASTWAM_REALROBOT_INTEGRATION.md)：如何保留和运行内部 FastWAM overlay。

## 6. 对外汇报版本

可以这样讲：

> 我们搭的是一个 LeRobot-first 的具身智能 demo evidence pipeline。第一阶段主线是用 LeRobot 跑通 dataset read、policy train/load、offline inference 和 report；FastWAM 作为 LeRobot-native policy 优先接入，同时保留私有 FastWAM overlay 以支持真机数据和未来自建模型。家庭服务任务现在作为应用层，有四个 R1 mock demo；后续再逐步接 replay、RoboDojo/RoboCasa/RoboTwin 仿真和真机 shadow。

## 7. 下一步

短期最该做三件事：

1. 实现 `make lerobot-data-smoke`，证明 LeRobot dataset 读取和 batch shape；
2. 实现 `make lerobot-infer-smoke`，证明 policy 加载和 action 输出；
3. 把 LeRobot-native FastWAM 和 custom overlay 两条路径统一进 evidence report。
