# Configs

`configs/` 是 backend 默认参数库，不是日常实验启动目录。

使用规则：

- 只想跑实验：去 `experiments/<route>/<experiment>/`；
- 想改某个实验：优先改实验目录里的 `config.yaml` / `config.sh`；
- 想改所有同类实验的默认值：才改这里的 `configs/`；
- 不确定时不要直接改 base config，否则会影响多个实验。

```text
configs/
├── lerobot/     # LeRobot train/infer profiles
├── fastwam/     # Custom WAM / FastWAM defaults
└── imagewam/    # Custom WAM / ImageWAM defaults
```

## LeRobot

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

## FastWAM

`configs/fastwam/realrobot_train_eval.sh` 定义 custom FastWAM overlay 的默认路径、模式、初始化方式和训练规模。

最重要的选择是 `FASTWAM_INIT`：

- `random`：真实训练链路验证，不是 release 微调；
- `base`：不 resume release ckpt，但保留 base/pretrained 初始化；
- `release`：面向 checkpoint 微调，建议显式设置 `resume=...`。

关键开关：

```text
FASTWAM_MODE=smoke|pilot|full
FASTWAM_RECIPE=joint_base|pose_base|v6_clean|v6_scratch|...
FASTWAM_INIT=release|base|random
FASTWAM_NNODES=<nodes>
FASTWAM_NODE_RANK=<rank>
FASTWAM_GPUS_PER_NODE=<gpus>
```

## ImageWAM

`configs/imagewam/libero_train_eval.sh` 定义 ImageWAM FLUX.2 4B LIBERO pilot 的默认路径和运行参数。
