# 复刻基准决策

> 状态：Accepted<br>
> 日期：2026-07-13<br>
> 关联文件：`references/upstreams.yaml`、`references/xpolicylab_baseline.yaml`、`docs/adr/0001-reference-baseline.md`、`docs/adr/0002-fastwam-evidence-chain.md`、`docs/adr/0003-lerobot-first-fastwam-pipeline.md`、`docs/LEROBOT_FIRST_PIPELINE.md`

## 1. 更新后的结论

我们现在把 **LeRobot-first data-to-inference** 作为第一 demo 管线基准：数据读取、训练/加载 policy、离线推理、证据报告都优先对齐 LeRobot。**FastWAM** 作为 LeRobot-native policy 路径优先进入这条主线；`D-URing/fastwam-realrobot-pipeline` 作为 custom overlay 保留，用于私有真机数据、集群 recipe 和未来自建模型。

XPolicyLab 的 `demo_policy` + `debug` evaluation flow 仍作为 policy/environment 解耦思想参考，RoboDojo 仍作为后续 NVIDIA/Isaac 仿真与外部评测基准，但它们不再是第一 demo 管线的主轴。

这不是选择“第一个要打榜的模型”，而是选择“第一个要对齐的工程边界”：

- policy 和 environment 依赖隔离；
- reset / update observation / get action 生命周期；
- action chunk 与 batch 能力声明；
- debug 模式先检查 shape、action key、序列化和退出；
- 未来可以平滑升级到 WebSocket、split process、NVIDIA cluster。

第一阶段不再以继续扩 household mock 为主，而是先补齐 LeRobot data-to-inference：dataset smoke、train/load smoke、offline inference smoke 和 report。XPolicyLab 是接口边界参考，不会变成 core 依赖。

FastWAM 的角色也被拆清楚：LeRobot-native FastWAM 是主线 policy path；私有 overlay 是自建/扩展 path。本仓库只包装入口、配置和产物，不复制 FastWAM 或私有 overlay 代码。

## 2. 为什么选它

XPolicyLab 对我们当前阶段最有价值的不是 policy zoo，而是它把模型运行时和环境运行时切开了。这个边界正好对应我们未来在 NVIDIA 集群上的部署形态：policy 可能在一个容器/节点/GPU 上，simulator 或 robot client 在另一个环境里。

`demo_policy` 也足够轻：它不加载 checkpoint、不训练模型，而是返回正确 action key 和维度的零动作，用来检查 policy server 与 environment client 的连线。这和我们 M2/M3 要做的“先把全流程打通”高度一致。

RoboDojo 暂时不作为本地第一依赖，因为官方环境需要 Linux、NVIDIA GPU、Isaac Sim/Isaac Lab 和较大缓存；这不适合卡在 Mac 本地开发的起点。但它适合成为 M5/M6 之后的外部验收目标，尤其是 `fold_clothes`、`organize_table`、`classify_objects` 等任务能映射到我们的两个 MVP。

LeRobot 现在升为主框架复刻对象。原因是需求已经明确为“从数据读取到推理全流程走通”，而 LeRobot 正好提供 dataset、policy training、checkpoint loading 和 inference/deployment 的公共接口。

FastWAM real-robot overlay 已经有真机数据读取、7D/10D 配置、DeepSpeed/Accelerate 训练、offline probe 和八卡 smoke 记录，因此适合作为 custom backend 和未来自建模型路线。它依赖 NVIDIA/CUDA 环境，不进入 core `.venv`。

## 3. 复刻范围

首轮复刻只对齐接口，不复制目录结构。

| 层级 | 复刻内容 | 当前做法 |
|---|---|---|
| 生命周期 | `reset`、`update_obs`、`get_action`、batch 变体 | 在本仓库定义 `PolicyAdapter`/runner 合同 |
| 运行边界 | policy 与 environment 解耦 | M2/M3 用 `inproc`，M5 后接 WebSocket |
| 调试模式 | 不依赖仿真器检查 IO 和动作 | 本地 mock/debug smoke |
| 评测思想 | smoke/single/benchmark、标准/随机、能力维度 | 已进入 Task/Evaluation schema |
| 任务映射 | `fold_clothes`、`organize_table`、`classify_objects` | 映射到 towel/tabletop MVP |

不复刻：

- XPolicyLab 的完整 policy zoo；
- RoboDojo simulator 本体；
- 上游数据下载脚本和权重；
- 上游目录结构；
- 上游分数作为本项目阶段性成绩。

## 4. 阶段路线

| 阶段 | 目标 | 产物 |
|---|---|---|
| R0 | 冻结上游引用和复刻范围 | `references/upstreams.yaml`、本文件、ADR |
| R1 | LeRobot dataset smoke | dataset profile、feature/action shape、metadata |
| R2 | LeRobot train/load smoke | LeRobot output、loss/checkpoint summary |
| R3 | LeRobot offline inference smoke | action output、shape、latency、policy metadata |
| R4 | LeRobot-native FastWAM path | `policy.type=fastwam` 训练/加载/推理证据 |
| R5 | Custom FastWAM overlay path | 私有真机数据、7D/10D recipe、cluster pilot |
| R6 | NVIDIA 集群接 RoboDojo/RoboCasa/RoboTwin | 外部仿真能力评测 |

下一步优先实现 LeRobot data smoke 和 inference smoke；household mock demo 暂时作为应用层保留，不再继续优先堆任务数量。

## 5. 任务映射

| 本项目任务 | RoboDojo 参考任务 | 用途 |
|---|---|---|
| `tabletop_sorting_v1` | `organize_table`、`classify_objects` | 桌面整理、分类归位、长时序和开放语言 |
| `towel_folding_v1` | `fold_clothes`、`fold_clothes_random` | 柔性物体、对齐、精密阶段评测 |

这些映射不是一开始就要求跑通 RoboDojo，而是保证我们现在写的 task/evaluator/action 合同以后能接到外部任务上。

## 6. 近期工程影响

M2/M3 需要额外满足：

- scripted policy 也必须走 `reset -> update_observation -> get_action`，不能直接操控 mock 环境；
- batch 只能通过 capability declaration 打开；
- action 输出先过 schema，再进入 backend；
- 每次 episode 都要记录 policy lifecycle 事件和 action validation 结果；
- artifact manifest 里预留上游 reference baseline 字段，方便未来对比 XPolicyLab/RoboDojo。

这会让早期代码多一点边界感，但以后接 OpenPI、GR00T、LingBot-VLA 或 RoboDojo 时会少很多返工。
