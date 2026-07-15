# Data Asset Pool

`data/` 是仓库级数据资产池。它在仓库根目录下统一管理大数据，但按 pipeline 路线分区存放。

关键规则：同源但格式不同、生命周期不同的数据，不跨 pipeline 共享同一个目录。宁可多放一份副本，也要保证 LeRobot 和 custom 后端的输入路径稳定、可复现。

```text
data/
├── lerobot/
│   ├── pusht/                         # LeRobot ACT/PushT demo dataset
│   ├── svla_so100_pickplace/          # LeRobot SmolVLA training dataset
│   └── libero-fastwam/
│       ├── v2.1/                      # copied raw FastWAM LIBERO release, for conversion/reference
│       └── v3/                        # converted LeRobot-current format, used by LeRobot loaders
├── custom/
│   ├── fastwam/
│   │   └── libero-fastwam/            # FastWAM native/custom route release data
│   └── imagewam/
│       └── robotwin2.0/               # ImageWAM optional RoboTwin data
├── internet/
├── human/
├── perception/
├── vla/
└── simulation/
```

规则：

- 数据集不提交 Git；
- 只提交这个 README 说明目录职责；
- 新数据集优先放到 `data/<route_or_ecosystem>/<dataset_name>/`；
- pipeline 文档应写清楚它依赖哪个数据目录；
- 如果同一个上游数据要服务不同 pipeline，按路线拆目录，例如 `data/lerobot/...` 和 `data/custom/...`。

当前 SCUT 已准备：

```text
data/lerobot/pusht
data/lerobot/svla_so100_pickplace
data/lerobot/libero-fastwam/v2.1
data/lerobot/libero-fastwam/v3              # 转换目标，可能为空
data/custom/fastwam/libero-fastwam
data/internet/rovid-20k-10s
data/human/xperience-10m-sample
```
