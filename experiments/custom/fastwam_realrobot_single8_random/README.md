# FastWAM realrobot single-node 8-GPU random-init training

用途：在单机 8 卡上跑通 custom FastWAM realrobot 训练链路，默认随机初始化。

这个实验是日常调试入口；多机长期任务再使用 `fastwam_realrobot_8node_random`。

## 1. 准备环境

如果 `conda activate fastwam` 报 `EnvironmentNameNotFound`，先创建并安装 FastWAM 环境：

```bash
cd /mnt/gpu11_200T/dingxibo/EmbodiedAI-Demo-Pipeline

CONDA_EXE="$(command -v conda)" \
FASTWAM_CREATE_CONDA=1 \
FASTWAM_INSTALL=1 \
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

然后激活：

```bash
conda activate fastwam
```

## 2. 检查资产

```bash
make check-assets-custom-fastwam
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

## 3. 干跑配置

```bash
python experiments/custom/fastwam_realrobot_single8_random/run.py --dry-run
```

## 4. 启动训练

```bash
python experiments/custom/fastwam_realrobot_single8_random/run.py
```

默认配置在 `config.yaml`：

```text
nnodes=1
gpus_per_node=8
init=random
mode=pilot
recipe=v6_scratch
max_steps=20
batch_size=1
```

结果位置：

```text
runs/experiments/custom/fastwam_realrobot_single8_random/<run_id>/
upstreams/FastWAM-realrobot/runs/<task>/<run_id>/
```

看日志：

```bash
tail -f runs/experiments/custom/fastwam_realrobot_single8_random/<run_id>/train_stdout.log
```

看 loss 摘要：

```bash
cat runs/experiments/custom/fastwam_realrobot_single8_random/<run_id>/loss_summary.json
```
