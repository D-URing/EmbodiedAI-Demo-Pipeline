# Pipelines

本目录是工程主线入口。项目现在明确分成两条线，不再把所有东西混成一个“demo pipeline”：

| Pipeline | 目标 | 当前验收 |
|---|---|---|
| [`lerobot/`](lerobot/) | 复刻 LeRobot 的数据读取 → 训练 → 推理链路 | ACT/PushT 已在 SCUT `gpu11` 跑通 GPU 训练 smoke |
| [`custom/`](custom/) | 保留自拟/自建模型接口，FastWAM 和 ImageWAM 并列 | FastWAM release/LIBERO 已准备；ImageWAM FLUX.2 4B/LIBERO 接入为可下载后端 |

约定：

- `pipelines/` 放“怎么跑、跑什么、产物怎么看”；
- `experiments/` 放训练/推理启动入口；
- `scripts/` 放可复用执行器；
- `configs/` 放底层默认参数；
- `docs/` 放背景、结构、存储和长说明；
- `data/` 和 `models/` 是根目录全局资产池，各 pipeline 自行选择需要的 dataset/model；
- `hf_cache/`、`runs/`、`upstreams/` 是本地/集群 ignored 目录，不提交 Git。

新后端统一放入 `custom/<backend>/`。不要再新增兼容型 pipeline 目录。
