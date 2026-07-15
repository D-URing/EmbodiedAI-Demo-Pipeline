# LeRobot / SmolVLA SO100 8-GPU long run

用途：A100 单机八卡长期实验入口。支持通过环境变量扩展到多机：

依赖：

```text
data/lerobot/svla_so100_pickplace/
models/lerobot/smolvla/smolvla_base/
```

准备：

```bash
make download-lerobot-svla-so100-pickplace-dataset
make download-lerobot-smolvla-base-policy
```

```bash
export LEROBOT_NUM_MACHINES=2
export LEROBOT_MACHINE_RANK=0
export LEROBOT_MAIN_PROCESS_IP=<rank0-ip>
```

启动：

```bash
bash experiments/lerobot/smolvla_so100_8gpu_long/launch.sh
```

常用覆盖：

```bash
export LEROBOT_NUM_PROCESSES=8
export LEROBOT_BATCH_SIZE=8
export LEROBOT_STEPS=20000
export LEROBOT_SAVE_FREQ=1000
```

SLURM：

```bash
sbatch experiments/lerobot/smolvla_so100_8gpu_long/slurm.sbatch
```

输出：

```text
runs/experiments/lerobot/smolvla_so100_8gpu_long/<run_id>/
```
