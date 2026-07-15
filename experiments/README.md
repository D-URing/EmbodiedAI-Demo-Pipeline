# Experiments

这里放真正用于启动训练、推理、长期试验的入口。原则是：

> `make` 只做环境、下载、检查；训练/推理从 `experiments/<route>/<experiment>/launch.sh` 启动。

这样做的好处是每次实验都有自己的：

- `config.sh`：可复制、可 diff、可复现；
- `launch.sh`：单机启动入口；
- `slurm.sbatch`：可选，集群提交入口；
- 固定结果路径：`runs/experiments/<route>/<experiment>/<run_id>/...`。

## 当前实验入口

```text
experiments/
├── lerobot/
│   ├── pusht_act_smoke/
│   ├── pusht_diffusion_train/
│   ├── smolvla_so100_8gpu_long/
│   ├── diffusion_pusht_infer/
│   ├── smolvla_so100_infer/
│   └── fastwam_libero_infer/
└── custom/
    ├── fastwam_realrobot_smoke/
    └── imagewam_flux2_4b_libero_pilot/
```

## 使用方式

先准备环境和资产：

```bash
make download-lerobot-pusht-dataset
make download-lerobot-smolvla-base-policy
make download-fastwam-artifacts
make prepare-imagewam-upstream
make download-imagewam-artifacts
make download-imagewam-flux2-base
```

再启动实验：

```bash
bash experiments/lerobot/pusht_act_smoke/launch.sh
bash experiments/lerobot/pusht_diffusion_train/launch.sh
bash experiments/lerobot/smolvla_so100_8gpu_long/launch.sh
bash experiments/lerobot/diffusion_pusht_infer/launch.sh
bash experiments/custom/fastwam_realrobot_smoke/launch.sh
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```

## 结果路径约定

默认都落在：

```text
runs/experiments/<route>/<experiment>/<run_id>/
```

每个 run 至少应包含：

```text
backend_manifest.json
command.txt
train_stdout.log 或 inference_evidence.json
loss_summary.json   # 如果是训练且 parser 能解析
```

大模型权重、数据集和缓存不放在 `experiments/`，统一从根目录资产池读取：

```text
data/
models/
hf_cache/
upstreams/
```

## 新增实验模板

新增实验时复制一个目录，而不是新增 Make target：

```text
experiments/<route>/<experiment_name>/
├── README.md
├── config.sh
├── launch.sh
└── slurm.sbatch     # 可选
```
