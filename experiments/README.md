# Experiments

这里放真正用于启动训练、推理、长期试验的入口。

原则：

> `make` 只做环境、下载、转换和检查；训练/推理从 `experiments/<route>/<experiment>/launch.sh` 启动。

完整训练/推理说明见 [`../docs/TRAINING_AND_INFERENCE.md`](../docs/TRAINING_AND_INFERENCE.md)。

## 当前实验入口

```text
experiments/
├── lerobot/
│   ├── pusht_act_smoke/               # ACT / PushT training
│   ├── pusht_diffusion_train/         # Diffusion / PushT training
│   ├── smolvla_so100_8gpu_long/       # SmolVLA / SO100 8-GPU training
│   ├── diffusion_pusht_infer/         # Diffusion / PushT inference
│   ├── smolvla_so100_infer/           # SmolVLA / SO100 inference
│   └── fastwam_libero_infer/          # FastWAM / LIBERO inference
└── custom/
    ├── fastwam_realrobot_smoke/       # custom FastWAM realrobot smoke/pilot/full
    └── imagewam_flux2_4b_libero_pilot/# ImageWAM FLUX.2 4B LIBERO metadata/pilot
```

## 资产准备

LeRobot：

```bash
make download-lerobot-pusht-dataset
make download-lerobot-svla-so100-pickplace-dataset
make download-lerobot-diffusion-pusht-policy
make download-lerobot-smolvla-base-policy
make download-lerobot-fastwam-libero-policy
make download-lerobot-fastwam-libero-dataset
make convert-lerobot-fastwam-libero-v3
make download-lerobot-fastwam-base-cache
```

Custom：

```bash
make download-custom-fastwam-libero-dataset
make download-fastwam-artifacts
make prepare-imagewam-upstream
make download-imagewam-artifacts
make download-imagewam-flux2-base
```

## LeRobot 训练

ACT / PushT：

```bash
bash experiments/lerobot/pusht_act_smoke/launch.sh
```

Diffusion / PushT：

```bash
bash experiments/lerobot/pusht_diffusion_train/launch.sh
```

SmolVLA / SO100 单机八卡：

```bash
bash experiments/lerobot/smolvla_so100_8gpu_long/launch.sh
```

SmolVLA Slurm：

```bash
sbatch experiments/lerobot/smolvla_so100_8gpu_long/slurm.sbatch
```

## LeRobot 推理

Diffusion / PushT：

```bash
bash experiments/lerobot/diffusion_pusht_infer/launch.sh
```

SmolVLA / SO100：

```bash
bash experiments/lerobot/smolvla_so100_infer/launch.sh
```

FastWAM / LIBERO：

```bash
bash experiments/lerobot/fastwam_libero_infer/launch.sh
```

FastWAM/LIBERO 已验证样例：

```text
runs/experiments/lerobot/fastwam_libero_infer/20260715-210113/inference_evidence.json
policy_type=fastwam
device=cuda
action.shape=[1, 7]
```

## Custom WAM

FastWAM：

```bash
FASTWAM_MODE=offline-smoke bash experiments/custom/fastwam_realrobot_smoke/launch.sh
FASTWAM_MODE=smoke FASTWAM_RECIPE=joint_base bash experiments/custom/fastwam_realrobot_smoke/launch.sh
FASTWAM_MODE=pilot FASTWAM_RECIPE=joint_base bash experiments/custom/fastwam_realrobot_smoke/launch.sh
```

ImageWAM metadata smoke：

```bash
IMAGEWAM_MODE=metadata-smoke IMAGEWAM_REQUIRE_CUDA=0 \
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```

ImageWAM pilot：

```bash
IMAGEWAM_MODE=pilot IMAGEWAM_REQUIRE_CUDA=1 \
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```

## 结果路径

默认都落在：

```text
runs/experiments/<route>/<experiment>/<run_id>/
```

训练 run 通常包含：

```text
backend_manifest.json
command.txt
train_stdout.log
loss_summary.json
lerobot_output/ 或 native upstream output 指针
```

推理 run 通常包含：

```text
config.sh
inference_evidence.json
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
