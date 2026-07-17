# FastWAM two-node smoke

这个实验用于验证 custom/FastWAM 路线的多节点启动链路，不用于正式长实验。

默认设置：

- `backend: fastwam`
- `fastwam.mode: smoke`
- `mode.smoke.max_steps: 1`
- `mode.smoke.batch_size: 1`
- `fastwam.text_embeddings.precompute: false`

`precompute: false` 的原因是：在部分集群里，每个节点的项目路径前缀不同，即使底层存储相同，rank0 写出的 marker 文件也不一定能被 rank1 用同一路径看到。正式长实验前，应先确保两边已有：

```bash
upstreams/FastWAM-realrobot/data/text_embeds_cache/libero/*.pt
```

cluster_120 实测启动：

```bash
cd /mnt/pfs/qahi3i/dingxibo/EmbodiedAI-Demo-Pipeline
/opt/conda/envs/fastwam-sm120/bin/python scripts/distributed/ssh_launch.py \
  --config experiments/custom/fastwam_realrobot_2node_smoke/config.yaml \
  --profile configs/distributed/cluster120_2node.yaml \
  --run-id fastwam_smoke_$(date +%Y%m%d_%H%M%S)
```

成功标志：

```text
FASTWAM_TRAIN_COMPLETE
DISTRIBUTED_LAUNCH_COMPLETE
```
