# 工程落地状态

本文记录“规划中的设计”与“仓库中已验证实现”的边界，避免把接口占位误认为已接入能力。

## 2026-07-13：M1 Core Contract

状态：已实现并通过本地验收，可冻结为 `v0.1.0` 基线。

### 已落地

| 规划项 | 实现位置 | 当前能力 |
|---|---|---|
| Python package | `src/embodied_demo/` | Python 3.11+、可编辑安装、CLI entrypoint |
| TaskSpec | `schemas/task.py` | 任务版本、能力、观测可见性、动作、阶段、backend、mock 与安全边界 |
| Observation | `schemas/io.py` | episode/step/timestamp、视觉、状态、上下文、metadata |
| ActionChunk | `schemas/io.py` | action representation、frame、频率、horizon、valid mask |
| RunSpec | `schemas/run.py` | mode、launcher、policy transport、backend、profile、feature flags、资源声明 |
| Evaluation contracts | `schemas/evaluation.py` | episode 结果、失败分类、partial progress、聚合、验证状态与审计 manifest |
| YAML 组合 | `config.py` | 相对路径 `extends`、递归 mapping merge、循环检测、来源记录 |
| 任务注册表 | `tasks/registry.yaml` | 校验任务 id/path/status 一致性 |
| 两项 MVP TaskSpec | `tasks/*/task.yaml` | 桌面分类归位、矩形毛巾两次对折 |
| CLI | `cli.py` | `validate`、`list-tasks`、`dry-run`、`run`、`report-fastwam`、`export-schema` |
| 自动测试 | `tests/` | schema、错误配置、组合循环、CLI 和 resolved config 落盘 |
| 环境基线 | `docs/ENVIRONMENT.md` | 本地/集群分层、精确 constraints、doctor 自检与 NVIDIA 接入边界 |

### 已冻结的 M1 技术默认值

| 决策 | 当前默认值 | 未来切换方式 |
|---|---|---|
| Python | 3.11 | 仿真依赖要求变化时通过独立环境调整，不降低 core 合同兼容性 |
| Schema | Pydantic v2，未知字段报错 | 使用 `export-schema` 给非 Python 组件生成 JSON Schema |
| CLI | 标准库 argparse | 保持命令名和退出码，内部可替换实现 |
| 配置 | YAML + 显式 `extends` | 复杂 sweep 出现后再评估 Hydra，不让其进入 core schema |
| 默认执行形态 | local + inproc + headless + CPU | RunSpec 开关切到 WebSocket、Slurm、GPU、sim 或 real |
| 任务状态 | experimental | M3 golden mock 与跨机器回归通过后升级 supported |

### RoboDojo 评测思想的代码映射

- 五个能力维度与 Stability/Safety/Efficiency 均为枚举，不依赖自由文本拼写。
- smoke/dev/release/external profile 有独立 episode 和 seed 配置。
- task progress 由带权阶段谓词组成，权重必须严格合计 100。
- `EpisodeResult` 区分 success、progress、validity、failure type 和 termination reason。
- `EvaluationManifest` 预留 task version、evaluator commit、checkpoint、normalizer、容器和 transport 版本。
- 当前只借鉴评测与 policy/environment 解耦原则；没有宣称已运行 RoboDojo simulator 或官方 benchmark。

### 验收命令

```bash
make setup
make test
make validate
make dry-run
```

验收标准：错误字段和组合循环返回明确错误；两个任务及运行配置全部通过；resolved config 可持久化并包含所有来源、RunSpec 和 TaskSpec。

本次验收结果：Python `3.11.15`；`14 passed`；两个 run config 均返回 `VALID`；dry-run 产物包含 4 个来源；M1 当时成功导出 7 份 JSON Schema。FastWAM evidence 接入后当前导出 8 份 JSON Schema。

环境复现入口为 `make setup && make doctor`；core 环境不包含 CUDA、Isaac、VLA 或真机 SDK。

## 2026-07-13：Reference Baseline Decision

状态：已接受并固化为 R0 基线。

### 已落地

| 规划项 | 实现位置 | 当前能力 |
|---|---|---|
| 上游 pin | `references/upstreams.yaml` | 固定 XPolicyLab、RoboDojo、LeRobot 的 commit、许可、冻结日期和引用范围 |
| 复刻基准 | `references/xpolicylab_baseline.yaml` | 明确 XPolicyLab `demo_policy`/debug flow 的生命周期、transport 和验收映射 |
| 决策文档 | `docs/REFERENCE_BASELINE.md` | 用中文说明为什么选 XPolicyLab、RoboDojo 和 LeRobot，以及当前不做什么 |
| ADR | `docs/adr/0001-reference-baseline.md` | 记录 accepted decision、影响和非目标 |
| 上游源码锚点工具 | `make reference-fetch` | 将 XPolicyLab 固定 commit 拉到用户 cache；不安装依赖、不下载数据、不启动仿真 |

### 决策影响

- M2/M3 仍然优先做本仓库的 logger、evaluator、runner 和 deterministic mock demo。
- `PolicyAdapter` 的生命周期需要贴合 XPolicyLab：`reset`、`update_obs`、`get_action`、batch 变体和 capability declaration。
- WebSocket 是 M5/M6 的兼容目标；第一阶段继续使用 `inproc`，避免网络栈影响 mock 开发。
- RoboDojo 进入后续外部仿真评测目标，重点映射 `fold_clothes`、`organize_table`、`classify_objects`。
- LeRobot 进入 M4 之后的数据、replay、converter 和轻量训练格式参考。

### 明确未实现

- 没有 rollout loop、scripted policy 或 deterministic mock backend；这些属于 M2/M3。
- scene YAML 目前是轻量输入样例，尚未建立跨 backend 的 SceneSpec 公共合同。
- launcher、remote transport、simulator、VLA、NVIDIA GPU 和真机仅有声明式开关，没有 adapter 实现。
- 没有 episode artifact writer、JSONL logger、predicate evaluator 或聚合执行器。
- 没有 Viewer；评测产物和回放合同稳定后再进入 M8。

## 下一步：M2 Evaluation Core

执行优先级保持为：

1. 定义 episode artifact 目录、run id 与原子写入规则。
2. 实现 logger、manifest writer 和系统失败/任务失败分流。
3. 实现贴合 XPolicyLab lifecycle 的本地 `PolicyAdapter` contract tests。
4. 实现 stage predicate、partial progress、success evaluator 与 task aggregate。
5. 为结果建立 golden fixtures，验证 seed 固定时跨运行一致。
6. 之后进入 M3，实现两个 scripted policy 和 deterministic mock backend。

在确认 NVIDIA 集群的 GPU、CUDA、调度器、容器和共享存储前，不安装本机 CUDA/Isaac 依赖，也不把 Slurm 或容器实现绑定到某一种集群假设。

## 2026-07-13：First Runnable Mock Demo

状态：已实现最小可交付闭环。

### 已落地

| 规划项 | 实现位置 | 当前能力 |
|---|---|---|
| CLI run | `embodied-demo run --config ...` | 单命令执行 deterministic mock rollout |
| Scripted policy | `src/embodied_demo/demo_runner.py` | 贴合 `reset -> update_observation -> get_action` 生命周期 |
| Mock backend | `src/embodied_demo/demo_runner.py` | 支持 `tabletop_sorting_v1` 和 `towel_folding_v1` 的符号/运动学状态推进 |
| Artifact writer | `runs/<run>/<episode>/` | 输出 `manifest.yaml`、`events.jsonl`、`result.json`、`metrics.json`、`report.md` |
| 快速入口 | `make demo` / `make demo-extended` | `demo` 运行两个 MVP；`demo-extended` 运行四个 R1 household mock demo |

### 边界

- 当前 demo 证明工程链路可运行，不代表仿真或真机能力。
- 当前 evaluator 仍是任务专用阶段谓词，已覆盖四个 R1 mock demo；M2 仍需抽象为通用 evaluator。
- 当前 artifact 已足够汇报和调试，但还不是最终 release profile 聚合格式。

## 2026-07-13：LeRobot GPU Training Replication

### 已落地

| 规划项 | 实现位置 | 当前能力 |
|---|---|---|
| 安装脚本 | `scripts/lerobot/install_lerobot_cluster.sh` | 安装 Python 3.12 环境中的 CUDA PyTorch 与 pinned LeRobot 源码 |
| 训练脚本 | `scripts/lerobot/run_pusht_act_gpu_smoke.sh` | 检查 CUDA 后调用官方 `lerobot-train` |
| 训练配置 | `configs/lerobot/pusht_act_gpu_smoke.sh` | 默认 `lerobot/pusht`、`policy.type=act`、`policy.device=cuda` |
| 日志解析 | `scripts/lerobot/parse_train_log.py` | 从 `lerobot-train` stdout 中提取 loss summary |
| Slurm 样例 | `scripts/lerobot/slurm_pusht_act_gpu_smoke.sbatch` | 给未知集群一个可改的提交模板 |
| 快速入口 | `make lerobot-train-smoke` | 在已安装 LeRobot 的 CUDA 节点上启动真实训练 smoke |

### 当前结论

这版不再使用 CPU toy trainer。loss 是否下降由集群上的真实 `lerobot-train` 日志和 `loss_summary.json` 证明。

### 边界

- 本地 macOS core 环境不安装 LeRobot、PyTorch CUDA 或仿真依赖。
- `make lerobot-train-smoke` 必须在有 CUDA 的 LeRobot 环境中运行；没有 GPU 会失败。
- 默认任务是官方轻量 `lerobot/pusht` + ACT smoke，用于验证训练链路和 loss 下降，不代表家庭任务最终模型能力。

## 2026-07-13：FastWAM Real-Robot Backend Integration

状态：已完成第一版薄集成，可交给 NVIDIA 集群运行。

### 已落地

| 规划项 | 实现位置 | 当前能力 |
|---|---|---|
| backend 配置 | `configs/fastwam/realrobot_train_eval.sh` | pin 官方 FastWAM 与私有 realrobot overlay，声明模型/数据/运行开关 |
| overlay 准备 | `scripts/fastwam/prepare_fastwam_overlay.sh` | clone 官方 FastWAM，clone 私有 overlay，rsync 覆盖，排除权重/数据/runs |
| 训练入口 | `scripts/fastwam/run_realrobot_train_eval.sh` | CUDA-only 调用 FastWAM `scripts/train_zero1.sh`，支持 smoke/pilot/full |
| recipe 映射 | `scripts/fastwam/run_realrobot_train_eval.sh` | `joint_base`、`pose_base`、`v6_*` 变体映射到 FastWAM task |
| 日志解析 | `scripts/fastwam/parse_train_log.py` | 解析真实 FastWAM stdout，输出 loss、子 loss、checkpoint summary |
| Slurm 样例 | `scripts/fastwam/slurm_realrobot_pilot.sbatch` | 给未知 NVIDIA 集群一个可改模板 |
| 文档 | `docs/FASTWAM_REALROBOT_INTEGRATION.md` | 说明环境、路径、运行命令、产物和 RoboDojo-style 分层评测 |
| 上游 pin | `references/upstreams.yaml` | 固定官方 FastWAM 与私有 overlay commit |
| 回归测试 | `tests/test_fastwam_scripts.py` | 验证 parser、CUDA-only 契约和 overlay 准备契约 |
| demo chain 配置 | `demo_chains/fastwam_realrobot_v0.yaml` | 定义任务校验、mock、FastWAM pilot、报告生成四段 evidence 链 |
| evidence schema | `src/embodied_demo/schemas/training.py` | 标准化 loss、step、checkpoint、backend refs 和 validation status |
| 报告生成 | `embodied-demo report-fastwam` | 将 FastWAM run 转成 `training_evidence.json`、`report.md`、`handoff.md` |

### 当前结论

FastWAM 现在是本项目的 R2 Training Evidence 外部后端。第一阶段用它补上“真实可训练模型 + loss 曲线 + checkpoint”的交付证据；本仓库继续负责 contract、任务库、mock demo 和统一 artifact。

### 边界

- 本仓库不 vendor FastWAM 或私有 overlay 代码，不存权重、数据和 runs。
- `make fastwam-train-smoke` 必须在已准备好的 FastWAM CUDA 环境中运行；没有 GPU 会失败。
- smoke 只验证真实前反传和 checkpoint；要证明 loss 下降，应跑 `FASTWAM_MODE=pilot`。
- `embodied-demo report-fastwam` 是报告/importer，不会重新训练或验证 checkpoint 文件实际存在。
- 真机闭环和 RoboDojo/RoboTwin 仿真评测仍是后续阶段，当前只落地训练/离线评测入口。

## 2026-07-14：Planning Update After FastWAM Evidence Chain

状态：已更新项目格局和 demo 覆盖路线。

### 已落地

| 规划项 | 实现位置 | 当前能力 |
|---|---|---|
| 三层证据链决策 | `docs/adr/0002-fastwam-evidence-chain.md` | 区分任务/工程链路、真实训练证据、后续仿真/真机能力 |
| Demo readiness 分级 | `docs/DEMO_COVERAGE_ROADMAP.md` | 用 R0–R6 标注 Task Spec、Mock、Training、Replay、Sim、Real Shadow、Real |
| 扩展任务覆盖矩阵 | `docs/DEMO_COVERAGE_ROADMAP.md` | 覆盖厨房、衣物、桌面、清洁、抽屉/柜门、找物递送、灵巧手 |
| 厨房台面整理 R1 demo | `tasks/kitchen_counter_sorting_v1/`、`configs/runs/kitchen_counter_sorting_mock.yaml` | 可通过现有 mock runner 生成 artifacts 和 report |
| 抽屉取放 R1 demo | `tasks/drawer_pick_place_v1/`、`configs/runs/drawer_pick_place_mock.yaml` | 可通过抽屉状态机 mock 生成 artifacts 和 report |
| 主规划同步 | `docs/MASTER_PLAN.md` | 阶段目标、优先级、架构、任务库、里程碑与变更记录已调整 |

### 当前结论

FastWAM 接入改变的是项目构建格局：本项目可以同时推进“可运行家庭 mock demo”和“真实训练 loss 证据”，但两者必须分别汇报。当前已从两个 mock MVP 扩到四个可运行 R1 mock demo。下一阶段不应该马上铺复杂 Viewer，也不应该马上承诺真机家庭任务，而应补齐衣物/清洁方向的 R0/R1 任务，并在 NVIDIA 集群上跑出 FastWAM `pilot` 的真实 loss 下降证据。

### 下一步建议

1. 从 `laundry_sorting_v1` / `trash_sorting_v1` 中选一个进入任务库，补齐衣物/清洁方向覆盖。
2. 抽象通用 mock primitives：object-in-region、category routing、drawer/articulated state、stage predicates。
3. 为 `make demo-extended` 补统一 summary，让四个 R1 household demo 的结果可以一页汇报。
4. 在 NVIDIA 集群运行 FastWAM `pilot`，把真实日志交给 `embodied-demo report-fastwam` 生成 handoff。
