# 复刻基准决策

> 状态：Accepted<br>
> 日期：2026-07-13<br>
> 关联文件：`references/upstreams.yaml`、`references/xpolicylab_baseline.yaml`、`docs/adr/0001-reference-baseline.md`

## 1. 更新后的结论

我们现在以 **XPolicyLab 的 `demo_policy` + `debug` evaluation flow** 作为第一复刻基准，以 **RoboDojo** 作为后续 NVIDIA/Isaac 仿真与外部评测基准，以 **LeRobot** 作为后续数据/训练格式参考。

这不是选择“第一个要打榜的模型”，而是选择“第一个要对齐的工程边界”：

- policy 和 environment 依赖隔离；
- reset / update observation / get action 生命周期；
- action chunk 与 batch 能力声明；
- debug 模式先检查 shape、action key、序列化和退出；
- 未来可以平滑升级到 WebSocket、split process、NVIDIA cluster。

第一阶段仍然保持本仓库自己的 `inproc + mock_2d + deterministic evaluator` 路线。XPolicyLab 是复刻锚点，不会变成 core 依赖。

## 2. 为什么选它

XPolicyLab 对我们当前阶段最有价值的不是 policy zoo，而是它把模型运行时和环境运行时切开了。这个边界正好对应我们未来在 NVIDIA 集群上的部署形态：policy 可能在一个容器/节点/GPU 上，simulator 或 robot client 在另一个环境里。

`demo_policy` 也足够轻：它不加载 checkpoint、不训练模型，而是返回正确 action key 和维度的零动作，用来检查 policy server 与 environment client 的连线。这和我们 M2/M3 要做的“先把全流程打通”高度一致。

RoboDojo 暂时不作为本地第一依赖，因为官方环境需要 Linux、NVIDIA GPU、Isaac Sim/Isaac Lab 和较大缓存；这不适合卡在 Mac 本地开发的起点。但它适合成为 M5/M6 之后的外部验收目标，尤其是 `fold_clothes`、`organize_table`、`classify_objects` 等任务能映射到我们的两个 MVP。

LeRobot 暂时不作为主框架复刻对象，因为我们现在还没进入训练/数据转换阶段。但它适合在 M4 作为 replay、converter 和轻量 IL policy 的格式参考。

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
| R1 | 本地实现 XPolicyLab 风格 policy lifecycle | `PolicyAdapter`、contract tests |
| R2 | M2 logger/evaluator 与 lifecycle 绑定 | episode artifacts、manifest、golden fixtures |
| R3 | 两个 MVP mock demo 走同一循环 | scripted policy、mock backend、CLI run/score/report |
| R4 | 增加 WebSocket transport loopback | 本地 split-process smoke |
| R5 | NVIDIA 集群接 RoboDojo smoke | 官方 demo/debug、一个任务映射、产物对比 |

R0 不改变 M2/M3 的优先级。下一步仍是 Evaluation Core 和 deterministic mock demo，只是 policy/runner 的实现要主动贴近这份基准。

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
