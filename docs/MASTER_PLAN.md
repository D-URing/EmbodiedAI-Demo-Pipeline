# 具身智能 Demo 场景与开源全流程打通：工程总规划

> 文档状态：Planning Baseline v0.2<br>
> 基线日期：2026-07-13<br>
> 最近更新：2026-07-14<br>
> 适用阶段：项目立项至首个仿真/模型闭环<br>
> 维护原则：本文件是项目级稳定参考；具体实现变化通过 ADR、配置版本和变更记录更新，不静默改写关键决策。

## 0. 执行摘要

本项目要建设的不是单个机器人 Demo，也不是从零训练一个通用大模型，而是一套可扩展、可复现、可切换执行后端的具身智能 Demo Pipeline：

```text
任务定义
  -> 观测契约
  -> 策略适配器
  -> 动作契约
  -> mock / replay / sim / real 执行
  -> 训练/离线证据导入
  -> 轨迹与运行产物
  -> 分层评测
  -> 报告与后期可视化
```

项目处于初步起步阶段，未来将在 NVIDIA 集群上开展训练、推理和仿真。因此当前架构遵循以下判断：

1. **接口契约先于模型选择。** 模型、机器人和仿真器都会变化，核心 schema 与运行协议必须先稳定。
2. **无界面运行先于可视化。** CLI、配置、产物和评测必须能在无桌面的集群节点运行；Viewer 是产物消费者，不进入控制闭环。
3. **评测先于大规模训练。** 没有稳定的任务、种子、指标、日志和复现协议，大规模训练结果不可比较。
4. **单机可靠先于多机吞吐。** 先确保 `num_envs=1`、单任务、单 seed 可复现，再增加并行环境和集群分片。
5. **通过开关表达可变能力。** mock、回放、仿真、真机、不同 policy transport、不同 observation 字段和不同 evaluator profile 均由配置选择。
6. **不把泛用性写成口号。** 当前只保证接口层泛用；任务资产、动作空间、相机布置、模型权重和真机控制仍然是 embodiment-specific。

2026-07-13 的补充决策：本项目以 XPolicyLab 的 `demo_policy` 与 `debug` evaluation flow 作为第一复刻基准，以 RoboDojo 作为后续 NVIDIA/Isaac 外部评测目标，以 LeRobot 作为后续数据和轻量训练格式参考。该决策只影响接口和验收锚点，不把 XPolicyLab、RoboDojo、Isaac 或 LeRobot 变成 core 依赖。详见 [`REFERENCE_BASELINE.md`](REFERENCE_BASELINE.md) 与 [`adr/0001-reference-baseline.md`](adr/0001-reference-baseline.md)。

2026-07-14 的补充决策：需求已收敛为 **LeRobot-first demo pipeline**。第一主线不再是继续扩 household mock task，而是用 LeRobot 跑通 dataset read → policy train/load → offline inference → evidence report。FastWAM 作为 LeRobot-native policy path 优先接入，同时保留私有 FastWAM overlay 作为 custom backend 和未来自建模型路线。详见 [`LEROBOT_FIRST_PIPELINE.md`](LEROBOT_FIRST_PIPELINE.md) 与 [`adr/0003-lerobot-first-fastwam-pipeline.md`](adr/0003-lerobot-first-fastwam-pipeline.md)。

第一阶段的两个纵向 Demo 为：

- `tabletop_sorting_v1`：桌面/厨房物品分类归位，验证刚性多物体、语言条件、阶段进度和抓放链路。
- `towel_folding_v1`：叠毛巾，验证双臂语义、柔性物体状态和形变评测链路。

第一阶段最重要的交付不是视频页面，而是：两个任务能够通过同一个 runner 在 mock/replay 后端执行，产生可复现 episode、部分进度分数、成功率、失败原因和完整 manifest。

LeRobot-first 调整后，第一阶段交付拆成一条主线和两个扩展层：

- **LeRobot 主线**：dataset inspection、train/load smoke、offline inference、evidence report。
- **Custom backend 扩展**：保留 FastWAM 私有 overlay 和未来自建模型路径。
- **Household 应用层**：四个 R1 mock demo 用于任务展示和后续 replay/sim/real 映射，不再抢第一主线。

---

## 1. 项目目标与边界

### 1.1 工程目标

构建一套团队可共同维护的具身智能 Demo 基础设施，使以下对象彼此解耦：

- 任务语义与成功条件；
- 场景、机器人和相机配置；
- 模型原生输入输出；
- 统一 observation/action；
- 执行后端；
- 评测协议；
- 日志、视频、指标和报告。

最终使“新增任务、接新模型、换仿真器、换机器人”成为新增配置和 adapter，而不是复制一套工程。

### 1.2 分阶段目标

| 阶段 | 目标 | 完成标志 |
|---|---|---|
| Phase 0：规划与契约 | 固化工程边界、目录、schema、评测分层与优先级 | 本文通过评审，关键未决项被记录 |
| Phase 1：Headless Core | 打通配置校验、runner、mock、logger、evaluator | 两个 MVP 可由 CLI 运行且可复现 |
| Phase 2：LeRobot Data Smoke | 按 LeRobot 原生格式读取 dataset 和 batch | 输出 dataset_profile、features、shape、fps、metadata |
| Phase 3：LeRobot Train/Load | 使用 LeRobot 训练或加载 policy，FastWAM 是优先 policy path | 输出 loss/checkpoint 或 checkpoint_summary |
| Phase 4：LeRobot Inference Smoke | 对 dataset sample 做离线 policy inference | 输出 action shape、latency、device、policy metadata |
| Phase 5：任务覆盖扩展 | 扩充 R0/R1 家庭任务库，沉淀通用 mock primitives | 厨房、抽屉、衣物/清洁至少新增 2–3 个可校验任务规格 |
| Phase 6：NVIDIA 集群准备 | 容器化、作业 manifest、资源声明、任务分片、断点续跑 | 同一命令可通过 local/cluster launcher 提交 |
| Phase 7：首个仿真闭环 | 接 RoboDojo、RoboCasa 或 RoboTwin 中的一个 | 至少一个任务完成 smoke、dev、release 三档评测 |
| Phase 8：自建模型扩展 | 接入一个非 LeRobot-native custom policy/backend | 与 LeRobot-native path 共享 evidence/report |
| Phase 9：真机准备 | shadow mode、安全过滤、标定、遥操作数据、真机 adapter | 预测动作可审计，未授权时不得发送到硬件 |
| Phase 10：产品化展示 | viewer、运行对比、关键帧、失败分析 | UI 只读消费 artifacts，不改变评测结果 |

### 1.3 明确非目标

当前阶段不承诺：

- 从零训练 π、GR00T 或同规模基础模型；
- 在未知机器人上零样本直接执行开源 VLA；
- 一开始同时兼容全部仿真器、模型和数据格式；
- 使用 mock 成绩代替仿真或真机能力；
- 以单个演示视频作为任务成功证据；
- 在尚未确定硬件前实现 ROS/真机控制主干；
- 在评测协议稳定前开发复杂前端或 leaderboard；
- 通过一个统一动作空间掩盖不同 embodiment 的物理差异。

---

## 2. 决策优先级

### 2.1 重要性排序

| 优先级 | 工作 | 原因 | 当前阶段 |
|---|---|---|---|
| P0 | task/observation/action/run schema | 所有后端和模型共享的基础契约 | 立即 |
| P0 | 确定性 runner、日志和 artifact manifest | 没有可复现产物就无法调试和评测 | 立即 |
| P0 | smoke test、partial progress、success/failure | 防止只看“能启动”或单次成功视频 | 立即 |
| P0 | LeRobot dataset/inference smoke | 当前第一主线是 data-to-inference，不是继续堆 mock task | 立即 |
| P0 | training evidence importer | 回答真实训练、loss、checkpoint 与复现实验问题 | 已启动 |
| P0 | demo readiness 标注 | 防止 mock、训练、仿真和真机结果混报 | 已启动 |
| P1 | mock 与 replay 后端 | 无 GPU、无仿真、无真机时仍可开发核心 | 立即 |
| P1 | policy adapter debug contract | 提前发现图像、状态、动作 shape 与坐标系问题 | 立即 |
| P1 | 可扩展任务覆盖矩阵 | 让厨房、衣物、清洁、抽屉、灵巧操作都有排期 | LeRobot 主线后 |
| P1 | 容器、配置覆盖、集群作业 manifest | 为 NVIDIA 集群迁移避免重写入口 | 核心稳定后 |
| P2 | 首个仿真后端 | 提供物理闭环和并行评测 | 接口稳定后 |
| P2 | 首个开源 VLA/IL policy | 验证 adapter 与真实推理资源 | 仿真 smoke 后 |
| P3 | 真机 shadow mode 与安全控制 | 需要目标机器人和控制栈信息 | 硬件明确后 |
| P4 | Viewer 和对外展示页面 | 依赖稳定 artifacts，不影响核心正确性 | 明显后置 |

### 2.2 暂缓技术

以下技术在需求未出现前不进入核心依赖：

- Ray、Kubeflow、Airflow 等工作流平台；
- Kubernetes 专属 CRD；
- ROS 2 作为通用进程总线；
- 在线数据库和消息队列；
- learned reward 作为唯一成功判据；
- 大型 Web 前端；
- 自建模型注册平台。

它们可以通过后续 launcher、artifact store、transport 或 viewer adapter 接入。

---

## 3. 总体架构

### 3.1 逻辑分层

```text
Task Plane
  task schema / stage graph / success predicates / eval profile

Policy Plane
  policy adapter / model runtime / normalization / action chunk

Training Evidence Plane
  training backend wrapper / loss parser / checkpoint summary / handoff report

Environment Plane
  mock / replay / simulator / real robot client

Execution Plane
  rollout runner / transport / safety filter / timeout / retry

Evidence Plane
  event log / trajectory / video / metrics / manifest / report

Presentation Plane
  CLI summary / static report / viewer / comparison page
```

依赖方向必须保持自上而下的契约关系：

- Viewer 依赖 artifacts，runner 不依赖 viewer。
- Evaluator 可以读取 evaluator-only 真值，policy 不得读取。
- Policy adapter 不直接 import 某个仿真器。
- Training evidence importer 不直接控制环境或机器人，只读取训练产物并归一化报告。
- Environment backend 不直接 import 某个模型。
- 真机驱动只存在于 real backend，不能进入 core。

### 3.2 关键开关

建议统一使用一个 resolved run config 驱动所有运行：

```yaml
runtime:
  mode: mock              # mock | replay | sim | real
  launcher: local         # local | slurm | kubernetes_future
  seed: 0
  deterministic: true

policy:
  name: scripted_sorting
  transport: inproc       # inproc | websocket | grpc_future
  action_type: semantic   # semantic | joint | ee_delta | hand
  device: auto

environment:
  backend: mock_2d        # mock_2d | dataset_replay | robodojo | robocasa | robotwin | real
  num_envs: 1
  headless: true

evaluation:
  profile: dev            # smoke | dev | release | external
  evaluator: predicate
  save_video: false

features:
  depth: false
  wrist_cameras: false
  batch_policy: false
  domain_randomization: false
  viewer_export: false
```

原则：

- 开关必须改变依赖注入，不允许在业务逻辑里散布大量 `if backend == ...`。
- resolved config 必须随每个 run 保存。
- 未实现的开关必须启动失败并给出明确错误，不能静默回退。
- `real` 模式必须显式授权，默认禁止。

### 3.3 推荐目录基线

以下是进入代码阶段后的目标结构，不要求 Phase 0 一次性全部创建：

```text
EmbodiedAI-Demo-Pipeline/
├── README.md
├── pyproject.toml
├── docs/
│   ├── MASTER_PLAN.md
│   ├── TASK_AUTHORING.md
│   ├── POLICY_ADAPTER.md
│   ├── EVALUATION_PROTOCOL.md
│   ├── CLUSTER_RUNBOOK.md
│   └── adr/
├── configs/
│   ├── base.yaml
│   ├── profiles/{smoke,dev,release}.yaml
│   ├── runtime/{local,slurm}.yaml
│   ├── policies/
│   └── environments/
├── tasks/
│   ├── registry.yaml
│   ├── tabletop_sorting_v1/
│   └── towel_folding_v1/
├── src/embodied_demo/
│   ├── core/
│   ├── observations/
│   ├── actions/
│   ├── policies/
│   ├── environments/
│   ├── rollout/
│   ├── evaluation/
│   ├── artifacts/
│   └── launchers/
├── integrations/
│   ├── xpolicylab/
│   ├── lerobot/
│   ├── openpi/
│   ├── groot/
│   ├── robodojo/
│   ├── robocasa/
│   └── robotwin/
├── containers/
│   ├── core.Dockerfile
│   ├── policy.Dockerfile
│   └── simulator.Dockerfile
├── scripts/
├── tests/
├── sample_data/
└── runs/                 # gitignored，本地符号链接或挂载点
```

集成代码放在 `integrations/`，避免把第三方项目的依赖和版本约束污染 core。

---

## 4. 核心数据契约

### 4.1 TaskSpec

每个任务必须定义：

- 稳定 task id 与 version；
- 类别、能力维度和难度；
- 自然语言指令及其变体；
- 必需物体、初始状态约束和目标关系；
- observation 需求；
- action 能力需求；
- 阶段图；
- 成功、失败和终止条件；
- 支持的 backend；
- evaluator profile；
- mock 真实性等级；
- 安全标签。

示例：

```yaml
id: tabletop_sorting_v1
version: 1.0.0
category: household_organization
difficulty: L2

capabilities:
  primary: [long_horizon]
  secondary: [generalization, open_instruction]

instruction:
  canonical: 将杯子放入托盘，调料瓶放入收纳盒，垃圾放入垃圾桶
  variants:
    - 整理桌面上的物品

scene:
  scene_id: tabletop_mock_v1
  required_objects: [cup, bottle, trash, tray, storage_box, trash_bin]

observation:
  required: [instruction, cam_head_rgb, timestamp]
  optional: [robot_state]
  evaluator_only: [object_poses, target_regions]

action:
  supported: [semantic, ee_delta]

stages:
  - perceive
  - select_object
  - grasp
  - transport
  - place
  - verify

termination:
  max_steps: 120
  success: all_targets_satisfied
  failure: [timeout, object_out_of_bounds, unsafe_action]

backends:
  supported: [mock_2d, dataset_replay]
  planned: [robocasa, robodojo, real]
```

### 4.2 Observation

```text
Observation
├── schema_version
├── episode_id
├── step_id
├── timestamp
├── instruction
├── vision/
│   ├── cam_head/color
│   ├── cam_left_wrist/color?
│   ├── cam_right_wrist/color?
│   ├── depth?
│   ├── intrinsics?
│   └── extrinsics?
├── state/
│   ├── joint_state?
│   ├── ee_pose?
│   ├── gripper_state?
│   ├── hand_state?
│   └── mobile_base?
├── task_context/
└── metadata/
```

数据可见性分为：

- `policy_visible`：模型实际可以获得；
- `evaluator_only`：仿真真值、目标区域、接触状态等；
- `debug_only`：只用于开发定位，不进入正式评分；
- `restricted`：可能包含人脸、隐私或真机敏感信息。

任何正式评测都必须记录 policy-visible 字段列表，防止 observation 泄漏。

### 4.3 ActionChunk

```text
ActionChunk
├── schema_version
├── representation       semantic | joint | ee_delta | hand
├── frame                world | base | camera | ee
├── control_frequency_hz
├── horizon
├── actions[]
├── valid_mask?
└── metadata/
    ├── policy_name
    ├── policy_version
    ├── inference_latency_ms
    ├── confidence?
    └── debug?
```

动作转换必须显式处理：

- 坐标系；
- 位置/旋转表示；
- 绝对/增量；
- 归一化/反归一化；
- 单位；
- 控制频率；
- action chunk 的重叠、截断和重规划；
- 左右臂、夹爪和灵巧手字段映射。

### 4.4 PolicyContract

参考 XPolicyLab 的小接口思想，核心生命周期定义为：

```python
class PolicyAdapter:
    def reset(self, context) -> None: ...
    def update_observation(self, observation) -> None: ...
    def get_action(self) -> ActionChunk: ...

    # 可选；只有通过 capability declaration 后 runner 才调用
    def update_observation_batch(self, observations) -> None: ...
    def get_action_batch(self, env_indices=None) -> list[ActionChunk]: ...
```

需要额外声明：

- `supported_observations`；
- `supported_action_types`；
- `supports_batch`；
- `stateful`；
- `required_history`；
- `runtime_requirements`；
- `normalizer_id`；
- `checkpoint_digest`。

XPolicyLab 当前将策略依赖和环境依赖分离，以 WebSocket 连接 policy server 与环境 client，并支持 `debug/sim/real` 环境模式。这一边界适合未来 NVIDIA 集群，但本项目第一阶段仍保留 `inproc`，避免 mock 开发被网络栈拖慢。参见 [XPolicyLab 官方文档](https://robodojo-benchmark.com/doc/usage/xpolicylab/)。

### 4.5 EpisodeArtifact

每次运行必须生成：

```text
runs/<run_id>/
├── manifest.yaml
├── resolved_config.yaml
├── task_snapshot.yaml
├── policy_snapshot.yaml
├── environment_snapshot.yaml
├── events.jsonl
├── trajectory.parquet
├── metrics.json
├── result.json
├── videos/              # 可选
├── keyframes/           # 可选
├── stdout.log
└── system.json
```

`manifest.yaml` 至少记录：

- run id、时间、提交者；
- Git commit 和 dirty 状态；
- 容器 image digest；
- task/policy/environment/eval profile 版本；
- seed；
- CUDA、驱动、GPU、PyTorch 版本；
- checkpoint digest；
- 数据集版本；
- 启用的 feature flags；
- 父 run 或恢复点；
- 最终状态：completed/failed/cancelled。

---

## 5. Rollout 与执行后端

### 5.1 统一循环

```text
load and validate config
-> materialize task and environment
-> reset policy
-> reset environment with seed/layout
-> observe
-> policy update/get_action
-> validate and safety-filter action
-> environment step/action-chunk execution
-> log events and evaluator state
-> terminate on success/failure/timeout
-> finalize artifacts
-> aggregate metrics
```

### 5.2 后端等级

| 后端 | 用途 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| `mock` | 契约、阶段图、日志和确定性 evaluator | 工程链路正确 | 模型能力、物理可行性 |
| `replay` | 数据解码、时序、离线 policy/evaluator | 数据与接口正确 | 闭环控制能力 |
| `sim` | 物理交互、随机化、并行评测 | 仿真分布内控制表现 | 真机可靠性 |
| `real_shadow` | 真机观测下预测但不执行 | 输入适配和动作审计 | 控制成功 |
| `real` | 真实闭环 | 指定平台和协议下的真实表现 | 泛化到其他平台/家庭 |

所有报告必须显示 backend 等级，禁止把不同等级的分数直接合并排名。

### 5.3 失败类型

统一失败 taxonomy：

- `config_error`
- `dependency_error`
- `policy_startup_error`
- `observation_schema_error`
- `action_schema_error`
- `transport_error`
- `timeout`
- `invalid_action`
- `unsafe_action`
- `task_failure`
- `environment_instability`
- `resource_exhausted`
- `manual_abort`

系统错误和任务失败必须分开统计；模型不能因为模拟器崩溃被记为任务失败，反之亦然。

---

## 6. 参考 RoboDojo 的评测体系

### 6.1 吸收哪些设计

RoboDojo 当前提供 42 个基础仿真任务和 18 个真机任务，仿真侧分成 Generalization、Memory、Precision、Long-Horizon、Open 五个能力维度；同时通过 XPolicyLab 将 policy 与环境拆开。[RoboDojo 官方说明](https://robodojo-benchmark.com/doc/)

本项目吸收以下原则：

1. **能力维度分开报告**，不能只看总平均。
2. **部分进度和完整成功分开**，长时序任务即使失败也要保留诊断信息。
3. **标准场景与随机场景成对比较**，记录 randomization drop。
4. **按维度平衡聚合**，避免任务数量多的类别主导总分。
5. **固定 seed/layout 和任务版本**，正式比较不能临时手摆场景。
6. **先 smoke，再单任务，再 sweep**，新 adapter 不直接运行全量任务。
7. **policy server 与 environment client 解耦**，支持模型和仿真使用不同依赖或不同 GPU。
8. **正式结果包含可审计 artifacts**，不以口头成绩或精选视频为准。

RoboDojo 当前正式协议中，42 个基础任务按五个维度组织，Generalization 另有 12 个 random 版本；完整 sweep 使用多个 seed，任务使用原生 episode 数。其官方工具也明确区分 smoke、single-task、benchmark 和 summarize。[RoboDojo Quick Evaluation](https://robodojo-benchmark.com/doc/usage/quite-evaluation/)

### 6.2 本项目的能力维度

首版采用 RoboDojo 五维，并增加工程部署维度：

| 维度 | 核心问题 | 典型扰动/任务 | 首版指标 |
|---|---|---|---|
| Generalization | 场景变化后还能否完成 | 颜色、位置、背景、光照、杂物、语言改写 | standard/random success、score drop |
| Memory | 任务是否需要保存历史状态 | 遮挡、先看后取、顺序模仿 | memory checkpoint accuracy、success |
| Precision | 是否有精确位姿或接触控制 | 插入、对齐、旋转、折叠角点 | pose error、alignment、contact violations |
| Long-Horizon | 多阶段任务能否持续推进 | 分类多个物体、开柜取物、做菜流程 | stage completion、full success、retries |
| Open | 未见组合或自然语言能否落到动作 | 语言分类、图像目标、技能重组 | held-out instruction success、grounding error |
| Stability | 执行动作是否稳定 | 抖动、振荡、重复动作、停滞 | action jerk、oscillation、no-op ratio |
| Safety | 是否违反明确安全边界 | 越界、碰撞、超速、过力 | safety violations、abort rate |
| Efficiency | 资源和时间成本是否可接受 | 推理、通信、仿真吞吐 | latency、Hz、GPU memory、episodes/hour |

前五维用于能力诊断；后三维用于部署诊断。总榜不把安全违规用平均分冲淡：正式 release profile 中出现严重安全违规时，结果标记为 invalid/unsafe。

### 6.3 评测层级

| Level | 名称 | 环境 | 目的 | 是否计入能力成绩 |
|---|---|---|---|---|
| E0 | Schema | 无 | 配置、类型、字段、shape 校验 | 否 |
| E1 | Wiring Smoke | debug/mock | 启动、reset、序列化、动作键、退出 | 否 |
| E2 | Deterministic Mock | mock | stage/evaluator/logger 的确定性回归 | 只计工程回归 |
| E3 | Offline Replay | replay | 数据、时间同步、离线指标和 adapter | 单独报告 |
| E4 | Simulation | sim | 物理闭环与能力维度 | 是 |
| E5 | Real Shadow | real observation | 真机输入与动作审计 | 否 |
| E6 | Real Closed-loop | real | 真实平台性能 | 是，单独榜 |
| E7 | External Benchmark | RoboDojo 等 | 外部可比性 | 是，引用冻结日期 |

### 6.4 运行档位

| Profile | Episode/seed 建议 | Seeds | 视频 | 用途 |
|---|---:|---:|---|---|
| `smoke` | 1 | 1 | 失败时 | PR、安装和 adapter 检查 |
| `dev` | 5 | 1–3 | 失败及样例 | 日常开发对比 |
| `release` | 25/50 或任务原生值 | 3 | 抽样+失败 | 里程碑和内部基线 |
| `external` | 外部协议规定 | 外部规定 | 按外部要求 | RoboDojo 等正式评测 |

首版不需要机械地复刻 RoboDojo 全量次数；需要复制的是“档位明确且不可混报”的原则。

### 6.5 分数体系

每个 episode 至少产生：

```text
episode_success       bool
progress_score        0..100
completed_stages      int
total_stages          int
failure_type          enum | null
termination_reason    enum
episode_steps         int
wall_time_s           float
policy_latency_ms     statistics
safety_violations     list
```

任务聚合：

```text
task_success_rate = successful episodes / valid episodes
task_progress_score = mean(progress_score over valid episodes)
task_stability = report std and confidence interval
```

维度聚合：

```text
dimension_score = mean(task scores within dimension)
overall_capability_score = mean(five primary dimension scores)
```

执行维度不直接混入能力平均，而是作为独立约束和附表。

### 6.6 Partial progress 设计

进度分数必须来自任务定义的阶段和谓词，不由视频是否“看起来不错”决定。例如桌面整理：

| 阶段 | 分值 |
|---|---:|
| 正确选择一个目标物 | 10 |
| 稳定抓取 | 15 |
| 运输中未掉落 | 10 |
| 放入正确区域 | 20 |
| 每个额外正确归位物体 | 15 |
| 全部目标满足并结束 | 补足至 100 |

叠毛巾：

| 阶段 | 分值 |
|---|---:|
| 检测/选择正确角点 | 10 |
| 完成第一次对折 | 25 |
| 第一次折叠对齐达标 | 15 |
| 完成第二次对折 | 25 |
| 最终 IoU/角点/面积达标 | 25 |

每项分值和阈值写入任务版本；任务版本改变时不得与旧版本直接合并。

### 6.7 Generalization paired protocol

每个进入 Generalization 的任务至少定义：

- `standard`：固定资产集合和有限布局；
- `random`：只改变被声明的因素；
- `ood`：保留到 release/external，不用于调参。

记录：

```text
absolute_drop = standard_score - random_score
relative_drop = absolute_drop / max(standard_score, epsilon)
```

随机化因素必须逐项登记，避免同时改变物体、相机、光照和语言后无法定位失败来源。

### 6.8 评测治理

正式基线必须冻结：

- task version；
- scene/layout set；
- observation fields；
- action space；
- evaluator implementation commit；
- profile 和 seed；
- checkpoint；
- normalizer；
- 容器镜像；
- policy/environment transport 版本。

结果状态分为：

- `unverified`：本地开发结果；
- `reproducible_internal`：由团队另一成员复跑；
- `verified_internal`：固定容器和 release profile；
- `verified_external`：通过 RoboDojo 等外部协议。

截至 2026-07-13，RoboDojo 的远程 policy evaluation/submission 页面仍标记为后续开放，因此近期只做本地接入准备和官方仓库 smoke，不能把远程评测列为当前里程碑阻塞项。[RoboDojo Eval](https://robodojo-benchmark.com/eval)

---

## 7. 任务库规划

### 7.1 Readiness 分级

后续所有 demo 任务都按 readiness 标注：

| 等级 | 名称 | 含义 |
|---|---|---|
| R0 | Task Spec | 任务语义、物体、阶段、成功条件明确 |
| R1 | Mock Rollout | 同一 runner/logger/evaluator/report 可复现运行 |
| R2 | Training Evidence | 真实训练入口、loss、checkpoint 和日志可追溯 |
| R3 | Offline Action / Replay | 数据、动作、时间同步和离线 evaluator 对齐 |
| R4 | Simulation | 仿真闭环和能力维度评测 |
| R5 | Real Shadow | 真机观测下预测但不执行，可做动作审计 |
| R6 | Real Closed-loop | 指定平台真实闭环 |

详细任务覆盖矩阵维护在 [`DEMO_COVERAGE_ROADMAP.md`](DEMO_COVERAGE_ROADMAP.md)。主规划只保留优先级和原则。

### 7.2 场景覆盖与近期优先级

| 类别 | 当前/近期任务 | 后续扩展 | 主要能力 | 当前最高 readiness |
|---|---|---|---|---|
| 厨房 | `tabletop_sorting_v1`、`kitchen_counter_sorting_v1` | 简化食物组装、倒取、柜内/抽屉操作 | 抓放、分类、长时序 | R1 |
| 衣物 | `towel_folding_v1`、`laundry_sorting_v1` | T 恤展平、折叠、悬挂 | 双臂、柔性物体、重抓 | R1 / R0 |
| 桌面 | `tabletop_sorting_v1` | 按语言规则整理、杂乱桌面恢复 | 多物体、开放指令 | R1 |
| 清洁 | `trash_sorting_v1` | 擦拭、扫入簸箕、工具使用 | 分类、覆盖、接触 | R0 |
| 抽屉/柜门 | `drawer_pick_place_v1` | 开柜取放、柜内整理 | 关节物体、接触、长时序 | R1 |
| 找物/递送 | `find_and_deliver_v1` | 多房间寻找和递送 | 记忆、导航、移动操作 | R0 |
| 灵巧手 | `open_bottle_or_screw_cap_v1`、`clip_or_pinching_v1` | 插拔、工具、小物体操作 | 精密、接触、手指协调 | R0 |
| 真实训练证据 | `fastwam_package_sorting_v0` | FastWAM offline action check、LeRobot 训练复刻 | CUDA 训练、loss、checkpoint | R2 |

近期扩展不追求“多而全”。本轮已补两个能扩大能力空间但工程成本可控的 R1 任务：

1. `kitchen_counter_sorting_v1`：厨房语义，复用桌面整理链路。
2. `drawer_pick_place_v1`：加入关节物体和状态机。

下一步再从 `laundry_sorting_v1` 或 `trash_sorting_v1` 中选择一个，扩到衣物/清洁，但避开完整柔性物理或复杂工具接触。

### 7.3 首批任务的选择

#### `tabletop_sorting_v1`

入选原因：

- 对 mock、replay、RoboCasa、RoboDojo 和真机都容易定义；
- 成功谓词清晰；
- 可渐进加入语言、干扰物、随机化和错误恢复；
- 最适合验证 runner 和 evaluator。

首版能力标签：Long-Horizon 为主，Generalization/Open 为辅。

#### `towel_folding_v1`

入选原因：

- 能验证柔性物体和双臂语义；
- 矩形毛巾比任意衣物更易度量；
- 可从 2D polygon mock 逐步升级到仿真和真机；
- RoboDojo 任务目录也包含 `fold_clothes` 标准/随机评测，可作为后续外部任务映射参考。

首版能力标签：Precision/Long-Horizon 为主，Generalization 为辅。

#### `fastwam_package_sorting_v0`

入选原因：

- 不是家庭任务能力证明，而是真实训练链路证明；
- 可以在 NVIDIA/CUDA 环境中产出 loss summary、checkpoint 路径和 handoff；
- 与 mock demo 共用 evidence/report 思想，形成“可交付证据链”；
- 能回答团队当前最现实的问题：是否有可训练模型，loss 是否正常下降。

首版能力标签：Training Evidence，不进入家庭任务能力榜。

### 7.4 任务进入主库的门槛

一个任务只有满足以下条件才从 `experimental` 升为 `supported`：

- task schema 通过校验；
- 有 canonical instruction 和至少一个语言变体；
- 有稳定成功/失败谓词；
- 有至少一个可运行 backend；
- 有 smoke profile；
- 有确定性 evaluator 单元测试；
- 有最小样例 artifact；
- 写明 mock/sim/real 能力边界；
- 写明安全和许可风险。

---

## 8. 开源资源接入路线

完整的任务到开源生态映射见 [`DEMO_COVERAGE_ROADMAP.md`](DEMO_COVERAGE_ROADMAP.md)。本节只定义工程接入顺序。

### 8.1 数据与通用接口

| 资源 | 用途 | 接入方式 | 优先级 |
|---|---|---|---|
| LeRobot | 轨迹格式、数据工具、基础 policy/robot interface | 独立 converter/adapter | P1 |
| FastWAM realrobot pipeline | 真实 CUDA 训练、loss 和 checkpoint 证据 | 外部 backend wrapper/importer | P0 |
| DROID | 桌面和多场景真实轨迹 | 先小样本 replay，再考虑训练 | P2 |
| BridgeData | 多环境抓放参考 | 小样本 replay | P2 |
| Open-X | RLDS 与跨 embodiment 元数据 | converter，不作为 core 格式 | P3 |
| AgiBotWorld | 双臂、衣物和长时序任务 | 下载脚本与 LeRobot 转换 | P3 |
| Ego4D/EPIC-KITCHENS | 人类任务步骤和视觉语义 | 任务设计参考，不当作机器人 action | P3 |

### 8.2 模型

候选顺序不是模型强弱排名，而是工程接入顺序：

1. `scripted_policy`：验证完整链路。
2. `replay_policy`：验证 action 解码和 episode 对齐。
3. LeRobot 中较轻的 IL policy：验证训练/推理闭环。
4. FastWAM：当前作为 R2 真实训练证据后端，先验证 loss、checkpoint 和 handoff，不直接纳入家庭任务能力榜。
5. OpenPI、GR00T、LingBot-VLA 中选择一个：验证 VLA adapter 和独立 policy runtime。
6. InternVLA、DiT4DiT 等：在接口稳定后作为比较分支。

同一时间只推进一个重量级 VLA 接入；每个 adapter 必须先通过 E0/E1，再进入仿真。

### 8.3 仿真与评测

| 候选 | 最匹配方向 | 优点 | 风险 |
|---|---|---|---|
| RoboDojo | 综合能力诊断、NVIDIA/Isaac、多策略比较 | 评测维度和 XPolicyLab 完整 | 新项目、依赖较重、远程评测未开放 |
| RoboCasa | 厨房与家庭场景 | 任务资产丰富、家庭语义强 | 与 Isaac/NVIDIA 集群不是同一栈 |
| RoboTwin | 双臂任务与数据生成 | 双臂、任务丰富 | 任务和依赖需要单独适配 |
| SIMPLER | 已有真实策略的仿真评测 | 强调 sim-real 相关性 | 不是通用家庭场景平台 |

建议：

- **工程复刻基准对齐 XPolicyLab `demo_policy`/debug flow。** 这是当前即可执行的方向，先复刻 lifecycle、debug smoke、action key、batch capability 和 policy/environment 边界。
- **外部仿真评测目标对齐 RoboDojo。** 在 NVIDIA/Isaac 环境具备后，优先用 `fold_clothes`、`organize_table`、`classify_objects` 映射两个 MVP。
- **首个物理仿真后端在硬件和场景优先级确认后再选。** 若 NVIDIA/Isaac 集群优先，则 RoboDojo 优先；若厨房场景资产优先，则 RoboCasa 可能更快。
- 不强行让多个 simulator 共享内部对象模型；它们只需要满足 EnvironmentContract 和 artifact contract。

---

## 9. NVIDIA 集群落地规划

### 9.1 当前假设

已知：

- 未来主要计算环境是 NVIDIA GPU 集群；
- 当前节点型号、CUDA/驱动版本、调度器、共享存储和容器运行时尚未确认；
- 本地开发环境可能不是 Linux/NVIDIA。

因此只冻结集群无关契约，不冻结 Slurm/Kubernetes 等具体设施。

### 9.2 进程与节点拆分

推荐保留三种部署形态：

```text
A. local in-process
   runner + mock/replay + policy

B. single-node split-process
   policy server (GPU 0) <-> env client/simulator (GPU 1)

C. multi-node split
   policy node(s) <-> environment worker node(s) <-> artifact storage
```

RoboDojo/XPolicyLab 已验证 policy server 与 Isaac Sim client 分离的工作方式，并支持单机或跨机器 WebSocket 通信；我们将其作为外部兼容目标，而不是复制其所有启动脚本。[XPolicyLab Deployment Flow](https://robodojo-benchmark.com/doc/usage/xpolicylab/)

### 9.3 容器策略

至少拆成：

- `core`：schema、runner、logger、evaluator、CLI；
- `policy-<name>`：模型依赖、权重加载和服务端；
- `sim-<backend>`：Isaac/RoboDojo、RoboCasa 或 RoboTwin 环境。

原因：VLA、Isaac Sim 和数据工具的 CUDA/Python 版本经常冲突，不能强求单一万能环境。

容器要求：

- 固定 base image 和 digest；
- 记录 CUDA runtime 与最低驱动要求；
- 权重和数据不 bake 进镜像；
- 支持只读代码挂载；
- 缓存目录显式挂载；
- 无网络模式下能使用已缓存资源；
- `headless=true` 为默认；
- health check 能验证模型加载与协议握手。

### 9.4 Launcher 抽象

```python
class Launcher:
    def submit(self, run_spec) -> JobHandle: ...
    def status(self, job) -> JobStatus: ...
    def cancel(self, job) -> None: ...
    def collect(self, job) -> ArtifactRef: ...
```

首版实现：

- `LocalLauncher`
- `DryRunLauncher`

集群信息确认后实现：

- `SlurmLauncher` 或实际调度器 adapter

不要在业务代码中调用 `sbatch`、硬编码分区或写死 GPU id。

### 9.5 资源声明

RunSpec 需要声明而不是猜测：

```yaml
resources:
  policy:
    gpus: 1
    gpu_memory_gb_min: 24
    cpus: 8
    ram_gb: 32
  environment:
    gpus: 1
    cpus: 16
    ram_gb: 64
  walltime: 02:00:00
  placement: split_allowed
```

运行时记录实际分配资源和峰值：GPU memory、CPU memory、利用率、推理 Hz、仿真 steps/s。

### 9.6 并行策略

递进顺序：

1. 单环境、单 task、单 seed；
2. 单进程多环境；
3. 多 task 分片；
4. 多 seed 分片；
5. policy batching；
6. 多节点。

RoboDojo 的 Isaac Sim 并行环境通过 `num_envs` 和 `env_spacing` 配置，官方默认 `num_envs=1` 以优先保证可靠性，并建议根据显存、渲染和场景复杂度逐步增加。这一原则直接采用。[RoboDojo Parallel Environments](https://robodojo-benchmark.com/doc/sim-tasks/parallel-environments/)

### 9.7 存储与恢复

区分：

- Git：代码和小配置；
- artifact store：运行产物、报告、视频；
- dataset store：只读数据集；
- checkpoint store：权重和 normalizer；
- local scratch：仿真缓存、临时帧、解码缓存。

每个分片独立写目录，聚合器只读合并，避免多进程竞争写同一 JSON。

必须支持：

- job 失败后保留已完成 episode；
- `--resume` 只补缺失 episode；
- manifest 记录恢复来源；
- 相同 task/seed/episode id 不重复计分；
- 聚合前校验 schema 和版本一致。

---

## 10. 可视化的后置策略

### 10.1 当前只做什么

Phase 1 只提供：

- CLI 运行摘要；
- `result.json` 和 `metrics.json`；
- 可选 MP4；
- Markdown/HTML 静态报告生成器可以后补。

### 10.2 什么时候开始 Viewer

只有满足以下条件后进入 Viewer 开发：

- artifact schema 稳定；
- 两个任务都有可复现 run；
- evaluator 输出稳定；
- 至少两个 policy/backend 可比较；
- 团队已明确主要使用场景是内部调试还是对外展示。

### 10.3 Viewer 边界

Viewer 只读：

- 不在页面中重新计算正式分数；
- 不通过 UI 修改历史 resolved config；
- 不直接控制真机；
- 可发起新 run 时也只生成 RunSpec 并交给 launcher；
- 页面显示 mock/replay/sim/real 标签和验证等级。

技术栈暂不决定。若只做内部工作台，可优先 Streamlit/Gradio；若形成长期产品，再考虑独立 API + React。

---

## 11. 测试与质量门禁

### 11.1 测试层级

- 单元测试：schema、坐标转换、谓词、分数、manifest。
- 合约测试：每个 policy/environment adapter 运行统一 test suite。
- Golden test：固定 mock 输入产生固定事件和分数。
- Replay test：样例 episode 解码、时间戳和 action 对齐。
- Smoke test：进程启动、协议握手、一个 episode、正确退出和产物落盘。
- Integration test：真实 simulator 或外部 policy，仅在对应环境运行。

### 11.2 PR 门禁

P0/P1 阶段建议：

- format/lint；
- schema validation；
- unit tests；
- deterministic mock smoke；
- 不下载大权重；
- 不启动 Isaac；
- 检查样例 artifact schema。

GPU/simulator 测试放入 nightly 或手动 release pipeline。

### 11.3 Definition of Done

任何功能不得以“代码已写完”为完成标准，至少要求：

- 有配置入口；
- 有失败信息；
- 有测试；
- 有文档；
- 有产物示例；
- 有能力边界；
- 不破坏 headless 运行；
- 不默认启用高成本或危险路径。

---

## 12. 里程碑与验收门

### M0：规划基线

交付：

- 本文；
- 项目 README；
- 未决技术问题清单。

验收：团队确认范围、优先级和两个 MVP。

### M1：Core Contract

状态：**已完成（2026-07-13）**。实现与验收证据见 [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md)。

交付：

- Python package；
- TaskSpec/Observation/ActionChunk/RunSpec；
- 配置解析与 schema 校验；
- CLI dry-run。

验收：错误配置失败明确；resolved config 可落盘。

### M2：Evaluation Core

前置决策：M2 的 policy lifecycle、action validation 和 batch capability 需要贴合 ADR-0001 中的 XPolicyLab 基准，但仍使用本地 `inproc` 与 mock 后端。

状态：**当前 demo runner 已具备最小闭环；通用 evaluator 抽象仍需继续收敛。**

交付：

- episode logger；
- stage predicates；
- partial progress 与 success；
- smoke/dev/release profile；
- manifest 和聚合器。

验收：golden mock 结果在不同机器上一致。

### M3：两个 Mock Demo

状态：**已完成第一版（2026-07-13）**，可通过 CLI 生成 artifacts 和 report。

交付：

- `tabletop_sorting_v1`；
- `towel_folding_v1`；
- scripted policies；
- mock backends；
- 样例 artifacts。

验收：一条 CLI 命令完成运行、评分和报告；注入失败可被正确归类。

### M4：真实训练证据链

状态：**已完成第一版入口（2026-07-13 / 2026-07-14）**。LeRobot 和 FastWAM 均为 CUDA-only 外部训练入口，不提供 CPU toy fallback。

交付：

- LeRobot ACT/PushT training smoke；
- FastWAM realrobot train/eval backend wrapper；
- loss parser、checkpoint summary、training evidence schema；
- `demo_chains/fastwam_realrobot_v0.yaml`；
- `embodied-demo report-fastwam` handoff 报告。

验收：在 NVIDIA/CUDA 环境中真实调用上游训练入口，保存 stdout、loss summary、checkpoint 路径和 handoff；`pilot` 模式用于观察 loss 下降。

### M5：Demo Coverage Expansion

状态：**已启动（2026-07-14）**。`kitchen_counter_sorting_v1` 和 `drawer_pick_place_v1` 已进入任务库并具备 R1 mock run；通用 mock primitives 和衣物/清洁任务仍待继续。

交付：

- `DEMO_COVERAGE_ROADMAP.md`；
- `kitchen_counter_sorting_v1` TaskSpec；
- `drawer_pick_place_v1` TaskSpec；
- `laundry_sorting_v1` 或 `trash_sorting_v1` TaskSpec；
- 通用 mock primitives：object-in-region、category routing、articulated state、stage predicates。

验收：至少 2 个新增 R0/R1 任务通过 `embodied-demo validate`，其中 1 个可由 mock runner 产生 artifacts。

### M6：Replay 与数据样例

交付：

- replay backend；
- LeRobot 风格样例；
- 数据来源/许可 manifest；
- converter contract。

验收：mock 与 replay 使用同一 evaluator/report 接口。

### M7：NVIDIA Cluster Ready

交付：

- core/policy 容器；
- Local/DryRun launcher；
- 集群 RunSpec；
- 缓存、存储和恢复规范；
- 实际调度器 adapter 的接口占位。

验收：容器内 headless smoke；集群命令可 dry-run 展开。

### M8：首个仿真后端

交付：

- 一个 simulator adapter；
- 单环境 smoke；
- 逐步并行配置；
- standard/random 小规模评测。

验收：至少一个任务 E4 通过，能报告能力、稳定性和效率。

### M9：首个学习策略

交付：

- 一个真实 policy adapter；
- 独立模型运行环境；
- debug/inproc 或 server-client；
- checkpoint/normalizer digest。

验收：与 scripted baseline 在同一任务/seed/profile 下比较。

### M10：Viewer

交付：

- run browser；
- episode 回放；
- stage/失败/指标展示；
- policy/backend 对比。

验收：Viewer 关闭时核心功能和结果完全不受影响。

### M11：真机预备与闭环

交付：

- real adapter；
- shadow mode；
- safety filter；
- 标定与 reset procedure；
- 录像和人工复核协议。

验收：先完成 E5，再经单独授权进入 E6。

---

## 13. 风险与缓解

| 风险 | 表现 | 缓解 |
|---|---|---|
| 过早绑定模型 | core 出现 π/GR00T 专属字段 | adapter 边界、contract test |
| 过早绑定仿真器 | task 直接 import Isaac/MuJoCo | EnvironmentContract、integrations 隔离 |
| “开关”失控 | 大量组合无人测试 | capability declaration、支持矩阵、少量官方 profile |
| 评测泄漏 | policy 读到目标位姿/仿真真值 | 字段可见性与正式 manifest |
| 只看成功率 | 长时序失败无法定位 | progress/stage/failure taxonomy |
| 分数不可复现 | 手摆场景、seed/版本缺失 | layout set、manifest、release profile |
| 集群成本失控 | 一上来全任务、多 seed、大视频 | smoke/dev/release、默认不保存视频 |
| CUDA 依赖冲突 | 模型和仿真不能共环境 | policy/simulator 分容器和跨进程协议 |
| 并行不稳定 | 多环境显存溢出或结果漂移 | `num_envs=1` 起步、测量后扩容 |
| mock 被误解 | mock 视频被当成模型能力 | backend 水印、分榜、明确 evidence level |
| 可视化抢占核心时间 | 页面漂亮但结果不可信 | artifact-first，Viewer 设为 P4 |
| 真机安全 | 抖动、越界、碰撞、过力 | shadow、限速、限位、急停、人工授权 |
| 外部项目变化快 | API、任务数、版本发生变化 | integration pin、冻结日期、adapter tests |

---

## 14. 未决技术问题与建议默认值

这些问题不阻塞规划，但会阻塞正式代码实现中的相应部分。

| 问题 | 建议默认值 | 决定时点 |
|---|---|---|
| Python 版本 | 3.11，除非首个仿真器要求其他版本 | M1 前 |
| Schema 工具 | Pydantic + JSON Schema | M1 前 |
| CLI | Typer 或 argparse；优先轻量 | M1 前 |
| 配置组合 | 先 YAML + 显式 merge；复杂后再引入 Hydra | M1 前 |
| 核心轨迹 | Parquet + JSONL + MP4 | M2 前 |
| 首个仿真器 | NVIDIA/Isaac 优先时选 RoboDojo；厨房资产优先时选 RoboCasa | M6 前 |
| 首个复刻基准 | XPolicyLab `demo_policy` + debug evaluation flow；RoboDojo 作为后续外部评测目标 | 已决定，见 ADR-0001 |
| 首个 VLA | OpenPI/GR00T/LingBot 三选一 | M7 前 |
| Policy transport | 本地 inproc；分进程 WebSocket | M5 前 |
| 集群调度器 | 待集群确认；优先写 adapter | M5 前 |
| 容器运行时 | Docker 开发；集群按实际支持转 Apptainer/其他 | M5 前 |
| Artifact storage | 本地目录接口，后接 NFS/S3 兼容存储 | M5 前 |
| Viewer | 暂缓；内部工具优先 Streamlit/Gradio | M8 前 |
| 真机中间件 | 硬件明确后决定 ROS 2/厂商 SDK/自定义 client | M9 前 |

### 需要团队补充的集群信息

- GPU 型号、单节点 GPU 数和显存；
- 驱动/CUDA 范围；
- 调度器；
- 容器运行时；
- 共享存储和配额；
- 计算节点是否可访问公网；
- 是否允许跨节点端口通信；
- 日志和实验追踪现有设施；
- 安全/许可证约束。

---

## 15. 近期执行清单

在不等待仿真器、模型和集群信息的情况下，下一轮可以安全开始：

1. 实现 episode artifacts、progress/success 和 failure taxonomy。
2. 建立 smoke/dev/release profile 的真实运行产物。
3. 实现贴合 XPolicyLab lifecycle 的 `PolicyAdapter` debug contract。
4. 实现 deterministic mock runner。
5. 实现两个 MVP 的 scripted policy 和 mock backend。
6. 为 batch capability、action validation 和 reference baseline manifest 建立 contract tests。
7. 在上述内容通过后，再确认首个 simulator 和首个重量级 policy。

暂不开始：

- Viewer；
- 大数据下载；
- 大模型训练；
- RoboDojo 全量 benchmark；
- 多节点调度；
- 真机控制。

---

## 16. 外部参考

本规划主要借鉴以下一手工程资料：

- [RoboDojo 官方项目与任务维度](https://robodojo-benchmark.com/doc/)
- [RoboDojo 使用边界：simulator side 与 XPolicyLab policy side](https://robodojo-benchmark.com/doc/usage/)
- [XPolicyLab policy adapter、数据契约与 server-client 部署](https://robodojo-benchmark.com/doc/usage/xpolicylab/)
- [RoboDojo Quick Evaluation：smoke、single-task、benchmark、seed 与聚合](https://robodojo-benchmark.com/doc/usage/quite-evaluation/)
- [RoboDojo 环境配置拆分](https://robodojo-benchmark.com/doc/usage/configurations/)
- [RoboDojo Isaac Sim 并行环境原则](https://robodojo-benchmark.com/doc/sim-tasks/parallel-environments/)
- [RoboDojo 论文](https://arxiv.org/abs/2607.04434)
- [XPolicyLab GitHub 仓库](https://github.com/XPolicyLab/XPolicyLab)
- [RoboDojo GitHub 仓库](https://github.com/RoboDojo-Benchmark/RoboDojo)
- [LeRobot GitHub 仓库](https://github.com/huggingface/lerobot)

注意：RoboDojo 于 2026-07 发布，仍处于快速变化期。本项目引用其评测思想和接口边界，但第三方集成时必须固定 commit、记录冻结日期并运行本项目自己的 contract tests。

---

## 17. 变更记录

### v0.1 — 2026-07-13

- 建立项目规划基线。
- 将评测提升到 P0/P1，参考 RoboDojo 五维能力体系。
- 增加 Stability、Safety、Efficiency 三个部署诊断维度。
- 明确 mock/replay/sim/real 分级，不混合排名。
- 面向 NVIDIA 集群设计 inproc/split-process/multi-node 三种形态。
- 将 Viewer 后置到 artifacts 和 evaluator 稳定之后。
- 选择桌面/厨房归位和叠毛巾作为两个纵向 MVP。

### v0.2 — 2026-07-13

- M1 Core Contract 完成工程落地并通过 14 项自动测试。
- 固化 Python 3.11、Pydantic v2、argparse 和显式 YAML merge。
- 增加 RoboDojo 风格的 EpisodeResult、失败分类、评测 manifest 与 JSON Schema 导出。
- 两个 MVP 任务进入 `experimental` 注册表，等待 M2/M3 运行闭环后升级。

### v0.3 — 2026-07-13

- 增加 macOS、Linux 与 NVIDIA 集群环境配置指南。
- 增加 Python 3.11 已验证 constraints 和 `make doctor` 自检入口。
- 明确 core、policy、simulator、real robot 必须隔离环境维护。
- 记录 macOS 系统代理不会自动传递给 shell 的处理方式。

### v0.4 — 2026-07-13

- 接受 ADR-0001：以 XPolicyLab `demo_policy` + debug flow 作为第一复刻基准。
- 将 RoboDojo 明确为后续 NVIDIA/Isaac 外部仿真评测目标。
- 将 LeRobot 明确为后续 replay、converter 与轻量训练格式参考。
- 冻结上游引用到 `references/upstreams.yaml`，避免文档决策漂移。

### v0.5 — 2026-07-14

- 接受 ADR-0002：FastWAM realrobot pipeline 作为第一条真实 CUDA 训练证据链。
- 将项目格局更新为任务/工程链路、真实训练证据、后续仿真/真机能力三层证据。
- 新增 demo readiness 分级 R0–R6，避免 mock、loss、仿真和真机结果混报。
- 新增 [`DEMO_COVERAGE_ROADMAP.md`](DEMO_COVERAGE_ROADMAP.md)，扩展厨房、衣物、桌面、清洁、抽屉、递送和灵巧手任务覆盖。
- `kitchen_counter_sorting_v1` 与 `drawer_pick_place_v1` 已进入 R1 mock demo，可通过 CLI 运行并生成 artifacts。
- 将下一阶段重点调整为：补衣物/清洁 R0/R1 家庭任务规格、沉淀通用 mock primitives、在 NVIDIA 集群复跑 FastWAM pilot 并输出 loss 下降证据。

### v0.6 — 2026-07-14

- 新增 [`00_PROJECT_OVERVIEW.md`](00_PROJECT_OVERVIEW.md) 和 [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md)，作为项目入口和架构入口。
- 将 `demo_runner.py` 拆分为 `policies/`、`environments/`、`rollout/` 三层，并保留兼容入口。
- 明确 household mock rollout 与 FastWAM training evidence 是两条证据线，不再在代码和文档入口中混讲。

### v0.7 — 2026-07-14

- 接受 ADR-0003：项目第一主线调整为 LeRobot-first data-to-inference pipeline。
- FastWAM 改为双路径定位：LeRobot-native policy path 优先，私有 overlay 作为 custom backend / 自建模型扩展。
- 新增 [`LEROBOT_FIRST_PIPELINE.md`](LEROBOT_FIRST_PIPELINE.md) 和 `demo_chains/lerobot_fastwam_data_to_inference_v0.yaml`。
- 下一步优先实现 `lerobot-data-smoke`、`lerobot-infer-smoke` 和 LeRobot/FastWAM evidence report。
