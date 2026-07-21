# pi05_baige_pytorchjob_probe

百舸 PyTorchJob 原生启动的 LeRobot / pi05 / SO100 训练测速实验。

核心原则：Master 和 Worker 执行同一条 command，由百舸注入的 `RANK`、`WORLD_SIZE`、`NPROC_PER_NODE` 自动决定节点角色，不走 ssh。

推荐 job command：

```bash
cd /mnt/cluster/dingxibo/EmbodiedAI-Demo-Pipeline
source .venv_lerobot/bin/activate
python experiments/lerobot/pi05_baige_pytorchjob_probe/run.py
```

训练前先确认 NCCL 多卡 smoke 已通过；如果 NCCL 单机 2 卡都失败，不要启动本实验。
