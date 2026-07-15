# Configs

配置按 pipeline 分层：

```text
configs/
├── lerobot/     # LeRobot 主线，含 train profiles
├── fastwam/     # Custom/FastWAM 主线
├── runs/        # Household/mock demo
└── profiles/    # smoke/dev/release 通用 profile
```

## `configs/lerobot/`

用于官方 LeRobot 复刻链路：

- dataset inspection；
- ACT/PushT GPU training；
- Diffusion/PushT GPU training；
- SmolVLA/SO100 fine-tuning；
- SmolVLA/SO100 单机八卡长期训练；
- offline inference；
- report evidence。

训练 profile：

```text
configs/lerobot/train/
├── pusht_act.sh
├── pusht_diffusion.sh
├── svla_so100_smolvla.sh
├── svla_so100_smolvla_8gpu_long.sh
└── aloha_pi0fast_template.sh
```

推理 profile：

```text
configs/lerobot/infer/
├── pusht_diffusion.sh
├── svla_so100_smolvla.sh
└── fastwam_libero.sh
```

默认入口见 [`../pipelines/lerobot/README.md`](../pipelines/lerobot/README.md)。

## `configs/fastwam/`

用于 custom backend / FastWAM overlay：

- release checkpoint；
- realrobot overlay；
- smoke / pilot / full 训练配置。

默认入口见 [`../pipelines/custom_fastwam/README.md`](../pipelines/custom_fastwam/README.md)。

## `configs/runs/`

用于 household/mock demo，不代表真实模型能力：

- tabletop sorting；
- towel folding；
- kitchen counter sorting；
- drawer pick-place。

这层后续会承接 LeRobot/FastWAM 产物，用于展示和报告。
