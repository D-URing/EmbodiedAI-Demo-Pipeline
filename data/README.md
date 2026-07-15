# Data Asset Pool

`data/` 是仓库级数据资产池，不属于某一条 pipeline 私有。

各 pipeline 只引用这里的数据：

```text
data/
├── lerobot/
│   └── pusht/                         # LeRobot ACT/PushT demo dataset
└── fastwam/
    └── libero-fastwam/                # FastWAM LIBERO release data, LeRobot v2.1
```

规则：

- 数据集不提交 Git；
- 只提交这个 README 说明目录职责；
- 新数据集优先放到 `data/<ecosystem>/<dataset_name>/`；
- pipeline 文档应写清楚它依赖哪个数据目录；
- 如果同一个数据要被多个 pipeline 使用，不要复制，引用同一个目录。

当前 SCUT 已准备：

```text
data/lerobot/pusht
data/fastwam/libero-fastwam
```

