# Pipelines

本目录是工程主线入口。项目现在明确分成两条线，不再把所有东西混成一个“demo pipeline”：

| Pipeline | 目标 | 当前验收 |
|---|---|---|
| [`lerobot/`](lerobot/) | 复刻 LeRobot 的数据读取 → 训练 → 推理链路 | ACT/PushT 已在 SCUT `gpu11` 跑通 GPU 训练 smoke |
| [`custom_fastwam/`](custom_fastwam/) | 保留自拟/自建模型接口，以 FastWAM realrobot overlay 为第一个例子 | release 权重和 LIBERO 数据已准备；私有 overlay 需要远端 GitHub 权限 |

约定：

- `pipelines/` 放“怎么跑、跑什么、产物怎么看”；
- `scripts/` 放实际执行脚本；
- `configs/` 放参数；
- `docs/` 放背景、结构、存储和长说明；
- `data/`、`models/`、`hf_cache/`、`runs/`、`upstreams/` 都是本地/集群 ignored 目录，不提交 Git。

