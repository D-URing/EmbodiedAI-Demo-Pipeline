# 多机分布式训练启动说明

当前项目里，LeRobot/pi05 和 custom/FastWAM 都已经有单机多卡入口；多机自动化由统一 SSH launcher 负责：

```bash
python scripts/distributed/ssh_launch.py \
  --config <experiment config.yaml> \
  --profile <cluster profile.yaml>
```

这个 launcher 的目标是：在 trainer0/rank0 节点上执行一次，自动 SSH 到所有节点，给每台机器分配 rank，并启动已有后端训练脚本。

## 已打通和未打通的边界

已支持：

- 根据 profile 自动分配 `rank / nnodes / master_addr / master_port`；
- LeRobot/pi05：覆盖 `LEROBOT_NUM_MACHINES / LEROBOT_MACHINE_RANK / LEROBOT_MAIN_PROCESS_IP`；
- FastWAM：覆盖 `FASTWAM_NNODES / FASTWAM_NODE_RANK / FASTWAM_MASTER_ADDR`；
- 统一注入 `run_id`，避免多机输出目录分裂；
- 每个 rank 生成独立 shell config；
- 每个 rank 保存 launcher 日志；
- `--dry-run` 预览所有远端命令。

仍需按集群实际情况确认：

- trainer0 到所有训练节点免密 SSH；
- 所有节点看到同一个项目路径、数据路径、模型路径；
- 所有节点有相同 conda 环境名；
- 多节点之间 master port 未被占用；
- 多节点训练前，先用单节点 profile 做 smoke。

已在 `cluster_120` 实测：

- LeRobot/pi05：2 节点 × 8 卡，2 step smoke 成功，loss `0.347 -> 0.141`；
- custom/FastWAM：2 节点 × 8 卡，1 step smoke 成功，loss `2.3717`，约 `0.21 step/s`、`3.37 samples/s`；
- node1 需要可用 CUDA Toolkit / `nvcc`，当前使用 `/usr/local/cuda`；
- FastWAM LIBERO 数据链接必须是项目内相对链接：`upstreams/FastWAM-realrobot/data/libero_mujoco3.3.2 -> ../../../data/custom/fastwam/libero-fastwam`。

## Profile 结构

profile 示例在：

```bash
configs/distributed/scut_gpu11_single.yaml
configs/distributed/cluster120_single.yaml
configs/distributed/cluster120_2node.yaml
configs/distributed/template_2node_ssh.yaml
```

核心字段：

```yaml
paths:
  repo_root: /shared/path/EmbodiedAI-Demo-Pipeline

environment:
  conda_init: /path/to/conda/etc/profile.d/conda.sh
  backend_conda_envs:
    lerobot: lerobot-cu126
    fastwam: fastwam

distributed:
  master_addr: trainer0
  master_port: 29505
  backend_master_ports:
    lerobot: 29505
    fastwam: 29500
  gpus_per_node: 8

nodes:
  - host: trainer0
    local: true
    gpus: 8
  - host: trainer1
    gpus: 8
```

多机时，`master_addr` 不能用 `127.0.0.1`，必须是其它节点能访问到的 rank0 地址。

`local: true` 表示该 rank 在当前 trainer0 进程本地启动，不走 SSH。通常 rank0/trainer0 应该设置 `local: true`，其它节点走 SSH。

## pi05 / LeRobot

单节点 SCUT dry-run：

```bash
cd /mnt/gpu11_200T/dingxibo/EmbodiedAI-Demo-Pipeline
python scripts/distributed/ssh_launch.py \
  --config experiments/lerobot/pi05_so100_8gpu_probe/config.yaml \
  --profile configs/distributed/scut_gpu11_single.yaml \
  --dry-run
```

单节点 SCUT 正式跑：

```bash
python scripts/distributed/ssh_launch.py \
  --config experiments/lerobot/pi05_so100_8gpu_probe/config.yaml \
  --profile configs/distributed/scut_gpu11_single.yaml
```

多节点时复制 `template_2node_ssh.yaml`，设置：

```yaml
distributed:
  master_addr: trainer0
  master_port: 29505

nodes:
  - host: trainer0
    gpus: 8
  - host: trainer1
    gpus: 8
```

LeRobot 的有效 batch 约为：

```text
effective_batch = training.batch_size * distributed.num_processes * distributed.num_machines
```

但在 SSH launcher 下，`num_processes/num_machines` 会被 profile 自动覆盖。

`cluster_120` 两节点实测命令：

```bash
cd /mnt/pfs/qahi3i/dingxibo/EmbodiedAI-Demo-Pipeline
/opt/conda/envs/lerobot-sm120/bin/python scripts/distributed/ssh_launch.py \
  --config experiments/lerobot/pi05_so100_8gpu_probe/config.yaml \
  --profile configs/distributed/cluster120_2node.yaml \
  --run-id pi05_2node_$(date +%Y%m%d_%H%M%S)
```

## FastWAM

单节点 SCUT dry-run：

```bash
python scripts/distributed/ssh_launch.py \
  --config experiments/custom/fastwam_realrobot_single8_random/config.yaml \
  --profile configs/distributed/scut_gpu11_single.yaml \
  --dry-run
```

单节点 SCUT 正式跑：

```bash
python scripts/distributed/ssh_launch.py \
  --config experiments/custom/fastwam_realrobot_single8_random/config.yaml \
  --profile configs/distributed/scut_gpu11_single.yaml
```

多节点 FastWAM 使用同一个 profile 机制。launcher 会自动覆盖：

```text
FASTWAM_NNODES
FASTWAM_NODE_RANK
FASTWAM_GPUS_PER_NODE
FASTWAM_MASTER_ADDR
FASTWAM_MASTER_PORT
FASTWAM_RUN_ID
```

FastWAM rank0 会先预计算 text embeddings，其它 rank 会等待 marker 文件。若各节点 run 目录不是同一个共享路径，建议先离线准备好 `upstreams/FastWAM-realrobot/data/text_embeds_cache/libero/*.pt`，并在 smoke/测速配置里设置：

```yaml
fastwam:
  text_embeddings:
    precompute: false
```

LIBERO 数据不要使用节点专属绝对软链接。推荐由 `scripts/fastwam/prepare_fastwam_overlay.sh` 自动创建相对链接：

```bash
upstreams/FastWAM-realrobot/data/libero_mujoco3.3.2 \
  -> ../../../data/custom/fastwam/libero-fastwam
```

`cluster_120` 两节点实测命令：

```bash
cd /mnt/pfs/qahi3i/dingxibo/EmbodiedAI-Demo-Pipeline
/opt/conda/envs/fastwam-sm120/bin/python scripts/distributed/ssh_launch.py \
  --config experiments/custom/fastwam_realrobot_2node_smoke/config.yaml \
  --profile configs/distributed/cluster120_2node.yaml \
  --run-id fastwam_smoke_$(date +%Y%m%d_%H%M%S)
```

长实验再切回 `experiments/custom/fastwam_realrobot_single8_random/config.yaml` 或复制新实验目录调整 `mode.pilot/full`。

FastWAM runner 默认做了几件集群兼容处理：

- 自动推断 `CUDA_HOME`：优先 conda 内 `nvcc`，其次 `/usr/local/cuda/bin/nvcc`；
- 默认 `NCCL_DEBUG=WARN`，需要排查网络时再显式设置 `FASTWAM_NCCL_DEBUG=INFO`；
- 默认 `HYDRA_FULL_ERROR=1`，失败时保留完整栈。

## 日志位置

统一 launcher 日志：

```bash
runs/distributed/<backend>/<experiment>/<run_id>/launcher_manifest.json
runs/distributed/<backend>/<experiment>/<run_id>/launcher_logs/rankXX_<host>.log
```

后端训练日志仍在各自 backend run 目录：

```bash
runs/experiments/lerobot/<run_name>/<run_id>/
runs/experiments/custom/<run_name>/<run_id>/
```

## 推荐测速顺序

1. 单节点 dry-run；
2. 单节点 2-step smoke；
3. 单节点 200-step 测速；
4. 两节点 smoke；
5. 两节点 200-step；
6. 扩展到更多节点。

不要一开始直接上长实验。多节点第一次失败时，先看 `launcher_logs/rankXX_*.log`，再看 backend 的 `train_stdout*.log`。
