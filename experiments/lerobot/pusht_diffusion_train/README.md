# LeRobot / PushT Diffusion train

用途：训练 LeRobot Diffusion Policy on PushT。

依赖：

```text
data/lerobot/pusht/
```

准备：

```bash
make download-lerobot-pusht-dataset
```

启动：

```bash
bash experiments/lerobot/pusht_diffusion_train/launch.sh
```

默认结果：

```text
runs/experiments/lerobot/pusht_diffusion_train/<run_id>/
```
