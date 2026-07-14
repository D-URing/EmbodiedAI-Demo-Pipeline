# 工程架构：Pipeline 分层与代码结构

> 状态：当前架构说明<br>
> 日期：2026-07-14<br>
> 关联：[`00_PROJECT_OVERVIEW.md`](00_PROJECT_OVERVIEW.md)、[`MASTER_PLAN.md`](MASTER_PLAN.md)

## 1. 核心数据流

```text
RunConfig
  -> TaskSpec + Scene
  -> PolicyAdapter
  -> EnvironmentBackend
  -> Rollout Runner
  -> Events / Result / Metrics / Manifest / Report
```

LeRobot-first 训练/推理链路是当前主干：

```text
LeRobotDataset
  -> LeRobot policy train/load
  -> offline inference
  -> TrainingEvidence + InferenceEvidence
  -> Report / Handoff
```

私有 FastWAM overlay 是 custom backend extension，不进入 core `.venv`，也不替代 LeRobot-native 主线。household mock demo 走 rollout，用于应用层任务展示和后续评测，不作为 LeRobot data-to-inference 的第一验收。

## 2. 分层职责

| 层 | 职责 | 当前位置 |
|---|---|---|
| Task Layer | 任务语义、物体、阶段、成功/失败条件 | `tasks/`、`scenes/` |
| Config Layer | local/headless/profile/run 配置组合 | `configs/`、`embodied_demo.config` |
| Schema Layer | Task/Run/Observation/Action/Evaluation/TrainingEvidence 合同 | `src/embodied_demo/schemas/` |
| Policy Layer | policy lifecycle 与动作生成 | `src/embodied_demo/policies/` |
| Environment Layer | mock/replay/sim/real 后端状态推进 | `src/embodied_demo/environments/` |
| Rollout Layer | reset/observe/action/step/log/finalize 主循环 | `src/embodied_demo/rollout/` |
| Evidence Layer | result、metrics、manifest、report、handoff | `rollout/` 与 `fastwam_report.py` |
| Integration Layer | LeRobot-native、custom backend、外部评测生态接入 | `scripts/lerobot/`、`scripts/fastwam/`、`references/` |

## 3. 当前代码结构

```text
src/embodied_demo/
├── cli.py                         # embodied-demo 命令入口
├── config.py                      # YAML extends、resolved run、task/registry loading
├── registry.py                    # 任务注册表遍历
├── demo_runner.py                 # 兼容入口；转发到 rollout.mock_runner
├── fastwam_report.py              # FastWAM training evidence importer/report
├── schemas/                       # 公共数据合同
├── policies/
│   └── scripted.py                # R1 mock demo 的 scripted policy
├── environments/
│   └── mock.py                    # R1 mock environment 状态推进
└── rollout/
    └── mock_runner.py             # deterministic mock rollout + artifacts
```

`demo_runner.py` 保留是为了兼容旧 import；新增代码应优先放到对应分层里。

## 4. Household mock demo 的执行路径

以 `kitchen_counter_sorting_v1` 为例：

```text
configs/runs/kitchen_counter_sorting_mock.yaml
  -> tasks/kitchen_counter_sorting_v1/task.yaml
  -> scenes/mock/kitchen_counter_sorting_v1/scene.yaml
  -> ScriptedPolicy
  -> MockEnvironment
  -> run_mock_demo
  -> runs/kitchen_counter_sorting_mock/<episode>/
```

输出目录包含：

```text
manifest.yaml
resolved_config.yaml
task_snapshot.yaml
events.jsonl
result.json
metrics.json
report.md
```

这些产物用于证明 pipeline 和 evaluator wiring，不用于声明真机成功率。

## 5. LeRobot / FastWAM training evidence 的执行路径

LeRobot-native 路径是第一优先级：

```text
scripts/lerobot/inspect_dataset.py              # planned
  -> scripts/lerobot/run_*_train*.sh
  -> scripts/lerobot/run_policy_inference_smoke.py   # planned
  -> embodied-demo report-lerobot                    # planned
```

私有 FastWAM overlay 是 custom extension：

```text
scripts/fastwam/prepare_fastwam_overlay.sh
  -> scripts/fastwam/run_realrobot_train_eval.sh
  -> scripts/fastwam/parse_train_log.py
  -> embodied-demo report-fastwam
  -> training_evidence.json / report.md / handoff.md
```

这两个路径共同用于证明：

- 是否调用真实 LeRobot/FastWAM CUDA 训练或 checkpoint 加载；
- loss 是否下降；
- policy inference 是否产生 action；
- checkpoint、dataset、policy type 和版本是否记录；
- 使用的是 LeRobot-native FastWAM 还是 custom FastWAM overlay。

它不证明 household task 的 closed-loop capability。

## 6. 新增任务时放哪里

新增一个 R1 household demo，至少需要：

```text
tasks/<task_id>/task.yaml
scenes/mock/<task_id>/scene.yaml
configs/runs/<task_id>_mock.yaml
src/embodied_demo/policies/scripted.py          # 加 scripted action plan
src/embodied_demo/environments/mock.py          # 加 mock state transition
tasks/registry.yaml
tests/test_demo_runner.py
```

如果一个任务只是规划，还没有 mock rollout，只放入 `DEMO_COVERAGE_ROADMAP.md`，不要急着进 `tasks/registry.yaml`。

## 7. 后续优化方向

当前拆分已经把“一个大 runner 文件”变成了三块，但还不是最终形态。下一步可继续优化：

1. 将 `MockEnvironment` 内部按 task family 拆成 tabletop/kitchen/drawer/towel primitives；
2. 抽象 `object-in-region`、`category routing`、`articulated state`、`fold state`；
3. 把 report writer 从 rollout 中独立到 `evidence/`；
4. 增加 demo pack summary，把多个 R1 demo 结果合并成一个交付页面；
5. 增加 replay/offline action runner，与 mock runner 并列。
