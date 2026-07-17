# LeRobot/pi05 cluster120 two-node probe

这个实验用于验证 LeRobot/pi05 在 `cluster_120` 上的两节点 SSH 分布式训练链路。

日常启动：

```bash
cd /mnt/pfs/qahi3i/dingxibo/EmbodiedAI-Demo-Pipeline
python experiments/lerobot/pi05_cluster120_2node_probe/run.py
```

只看将要执行的底层命令：

```bash
python experiments/lerobot/pi05_cluster120_2node_probe/run.py --dry-run
```

这个入口仍然遵守项目统一约定：实验目录内维护 `config.yaml + run.py`。
