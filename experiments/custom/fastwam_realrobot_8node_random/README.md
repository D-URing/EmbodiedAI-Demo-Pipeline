# FastWAM realrobot 8-node random-init training

用途：在 8 机 × 8 卡上启动 custom FastWAM realrobot 训练，并显式使用随机初始化。

默认配置：

```text
FASTWAM_MODE=pilot
FASTWAM_RECIPE=v6_scratch
FASTWAM_INIT=random
FASTWAM_NNODES=8
FASTWAM_GPUS_PER_NODE=8
```

`FASTWAM_INIT=random` 会传入：

```text
resume=null
model.skip_dit_load_from_pretrain=true
model.action_dit_pretrained_path=null
```

因此不会加载 LIBERO release checkpoint，也不会加载 ActionDiT backbone。Wan/T5/VAE 这类运行时组件仍按上游 FastWAM 代码路径从 cache/model base 查找。

启动：

```bash
sbatch experiments/custom/fastwam_realrobot_8node_random/slurm.sbatch
```

如果调度系统没有 Slurm，可以在每个节点分别设置：

```bash
export FASTWAM_NNODES=8
export FASTWAM_NODE_RANK=<0-7>
export FASTWAM_MASTER_ADDR=<rank0-host-or-ip>
export FASTWAM_MASTER_PORT=29500
export FASTWAM_GPUS_PER_NODE=8
export FASTWAM_RUN_ID=<shared-run-id>

bash experiments/custom/fastwam_realrobot_8node_random/launch.sh
```

结果：

```text
runs/experiments/custom/fastwam_realrobot_8node_random/<run_id>/
upstreams/FastWAM-realrobot/runs/<task>/<run_id>/
```
