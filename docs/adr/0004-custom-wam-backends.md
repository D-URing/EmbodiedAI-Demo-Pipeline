# ADR 0004: Custom WAM Backends

状态：accepted

日期：2026-07-15

## 背景

项目原先只有 `pipelines/custom_fastwam/`，适合解释 FastWAM overlay，但随着 ImageWAM 进入范围，继续把所有 custom 模型塞进 FastWAM 目录会让结构失真。

## 决策

新增 canonical 结构：

```text
pipelines/custom/
├── fastwam/
└── imagewam/
```

保留 `pipelines/custom_fastwam/` 和 `pipelines/custom_wam/` 作为兼容入口，避免已有文档和命令断掉。

## 影响

- LeRobot 主线不变；
- FastWAM 仍可通过已有 `configs/fastwam` 和 `scripts/fastwam` 运行；
- ImageWAM 拥有独立 `configs/imagewam` 和 `scripts/imagewam`；
- 后续 InternVLA-A、LingBot-VLA、GR00T-style wrapper 可以继续作为 `custom/<backend>` 接入。

## 非目标

- 不 vendor 上游源码；
- 不把 ImageWAM 强行包装成 LeRobot policy；
- 不在没有仿真环境的情况下宣称 LIBERO/RoboTwin success rate。
