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
| CLI | `cli.py` | `validate`、`list-tasks`、`dry-run`、`export-schema` |
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

本次验收结果：Python `3.11.15`；`14 passed`；两个 run config 均返回 `VALID`；dry-run 产物包含 4 个来源；成功导出 7 份 JSON Schema。

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
| 快速入口 | `make demo` | 连续运行两个 MVP mock demo |

### 边界

- 当前 demo 证明工程链路可运行，不代表仿真或真机能力。
- 当前 evaluator 只覆盖两个 MVP 的内置阶段谓词；M2 仍需抽象为通用 evaluator。
- 当前 artifact 已足够汇报和调试，但还不是最终 release profile 聚合格式。

## 2026-07-13：First Trainable Demo

状态：已实现最小训练闭环。

### 已落地

| 规划项 | 实现位置 | 当前能力 |
|---|---|---|
| CLI train | `embodied-demo train-demo --config ...` | 单命令训练一个轻量行为克隆 policy |
| Dataset artifact | `dataset.jsonl` | 保存 mock observation-feature/action-label 样本 |
| Train log | `train_log.jsonl` | 逐 epoch 输出 `train_loss` |
| Checkpoint | `checkpoint.json` | 保存 softmax BC 分类器权重和 action vocabulary |
| Metrics/report | `metrics.json`、`report.md` | 记录 initial/final loss、下降比例和边界说明 |
| 快速入口 | `make train-demo` | 连续训练两个 MVP 的最小 BC demo |

### 当前结论

这版已经可以回答“有没有 loss 正常下降”：有，训练 demo 会输出 `loss_decreased=true`、初始 loss、最终 loss 和下降比例。

### 边界

- 当前模型是纯 Python softmax behavior cloning demo，不依赖 PyTorch/CUDA。
- 当前 dataset 是从 mock/scripted expert 生成的小样本，不是 LeRobot 真数据，也不是仿真/真机采集数据。
- 当前目标是证明训练管线、日志、checkpoint 和报告闭环，不是追求模型能力。
