# Configs

`configs/` 是 backend 默认参数库，不是日常实验启动目录。多次训练/推理实验请复制或新增 `experiments/<route>/<experiment>/config.sh`。

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

关键开关：

```text
FASTWAM_MODE=smoke|pilot|full|offline-smoke
FASTWAM_RECIPE=joint_base|pose_base|v6_clean|v6_scratch|...
FASTWAM_INIT=release|base|random
FASTWAM_NNODES=<nodes>
FASTWAM_NODE_RANK=<rank>
FASTWAM_GPUS_PER_NODE=<gpus>
```

## ImageWAM

`configs/imagewam/libero_train_eval.sh` 定义 ImageWAM FLUX.2 4B LIBERO pilot 的默认路径和运行参数。
