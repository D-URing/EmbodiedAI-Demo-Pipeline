# LeRobot GPU Replication

> 状态：cluster-ready scaffold；正在升级为 LeRobot-first data-to-inference 主线<br>
> 目标：在 CUDA 集群上复刻 LeRobot 的数据读取、训练/加载和推理链路，而不是本仓库的 CPU toy trainer。

## 结论

当前已落地的 LeRobot 复刻入口是训练 smoke：

```bash
bash scripts/lerobot/install_lerobot_cluster.sh
bash experiments/lerobot/pusht_act_smoke/launch.sh
```

它会调用官方 `lerobot-train`，默认训练：

```text
dataset.repo_id = lerobot/pusht
policy.type     = act
policy.device   = cuda
```

如果没有 CUDA，脚本会失败；不会 fallback 到 CPU。

新的主线目标见 [`LEROBOT_FIRST_PIPELINE.md`](LEROBOT_FIRST_PIPELINE.md)。下一步要补：

```bash
make lerobot-data-smoke
bash experiments/lerobot/diffusion_pusht_infer/launch.sh
python scripts/lerobot/generate_data_to_inference_report.py
```

FastWAM 在新主线中优先走 LeRobot-native policy path；私有 `fastwam-realrobot-pipeline` 继续作为 custom overlay 保留。

## 集群安装

官方 LeRobot 当前推荐 Python 3.12、`lerobot[training]`，Linux CUDA 场景需要选择合适的 PyTorch CUDA wheel。本项目默认使用 LeRobot pinned commit：

```text
e40b58a8dfa9e7b86918c374791599d070518d11
```

安装脚本默认行为：

- 检查 Python >= 3.12；
- 安装 CUDA PyTorch，默认 `cu128` index；
- clone Hugging Face LeRobot 到项目内 `upstreams/lerobot`；
- checkout pinned commit；
- 安装 `.[training,pusht]`；
- 检查 `lerobot-train` 可用。

```bash
bash scripts/lerobot/install_lerobot_cluster.sh
```

常用覆盖：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
bash scripts/lerobot/install_lerobot_cluster.sh
```

如果要让脚本创建 conda 环境：

```bash
CONDA_EXE=/mnt/gpu11_200T/dingxibo/miniconda3/bin/conda \
LEROBOT_CREATE_CONDA=1 LEROBOT_CONDA_ENV=lerobot \
bash scripts/lerobot/install_lerobot_cluster.sh
```

`CONDA_CHANNEL_ARGS` 默认使用清华 conda-forge 镜像：

```text
--override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
```

如集群有自己的 conda 镜像，可显式覆盖该变量。

## 训练 smoke

```bash
bash experiments/lerobot/pusht_act_smoke/launch.sh
```

SCUT `gpu11` 默认使用：

```text
dataset.video_backend = pyav
policy.repo_id        = local/pusht_act_gpu_smoke
policy.push_to_hub    = false
TORCH_HOME            = <repo>/hf_cache/torch
```

这样做是为了避免计算节点联网下载 ResNet18 backbone。当前 LeRobot v3 / FastWAM 视频读取已改为在 `lerobot` 环境固定 `ffmpeg=6.*`，避免旧 host glibc 下 `ffmpeg=8` 相关 native ABI 问题。

或显式指定配置：

```bash
bash scripts/lerobot/run_pusht_act_gpu_smoke.sh configs/lerobot/pusht_act_gpu_smoke.sh
```

默认输出：

```text
runs/lerobot/pusht_act_gpu_smoke/<run_id>/
├── command.txt
├── train_stdout.log
├── loss_summary.json
├── loss_report.md
└── lerobot_output/
```

`loss_summary.json` 由 `scripts/lerobot/parse_train_log.py` 从真实 `lerobot-train` stdout 中解析。它会记录：

- `initial_loss`
- `final_loss`
- `loss_drop_ratio`
- `loss_decreased`
- 解析到的全部 loss 值

## Slurm

未知集群的分区、镜像和模块系统差异很大，所以当前只给可改模板：

```bash
sbatch experiments/lerobot/pusht_act_smoke/launch.sh
```

如需覆盖训练步数：

```bash
LEROBOT_STEPS=2000 LEROBOT_BATCH_SIZE=16 \
sbatch experiments/lerobot/pusht_act_smoke/launch.sh
```

## 边界

- 这不是 CPU 伪实现；
- 这不是本仓库自写 softmax/BC toy trainer；
- 这不是家庭任务最终模型；
- 这是 LeRobot 官方数据格式、官方 policy 入口、官方训练 CLI 的 GPU smoke；
- 它只是 LeRobot-first 主线的一段，还需要 dataset inspection 与 offline inference smoke 才算 data-to-inference 闭环；
- 第一次成功标准是能下载/加载 LeRobot dataset、启动 ACT policy、记录真实 training loss、保存 LeRobot checkpoint。

## 官方依据

- LeRobot README: `lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_mobile_cabinet`
- LeRobot installation: Python 3.12、`lerobot[training]`、CUDA PyTorch wheel 说明
- LeRobot multi-GPU: `accelerate launch $(which lerobot-train) ...`
- LeRobot dataset format: MP4/images + Parquet state/action data
