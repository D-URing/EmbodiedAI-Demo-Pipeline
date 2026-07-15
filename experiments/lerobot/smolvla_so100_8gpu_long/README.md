# LeRobot / SmolVLA SO100 8-GPU long run

用途：A100 单机八卡长期实验入口。支持通过环境变量扩展到多机：

```bash
export LEROBOT_NUM_MACHINES=2
export LEROBOT_MACHINE_RANK=0
export LEROBOT_MAIN_PROCESS_IP=<rank0-ip>
```

启动：

```bash
bash experiments/lerobot/smolvla_so100_8gpu_long/launch.sh
```

SLURM：

```bash
sbatch experiments/lerobot/smolvla_so100_8gpu_long/slurm.sbatch
```
