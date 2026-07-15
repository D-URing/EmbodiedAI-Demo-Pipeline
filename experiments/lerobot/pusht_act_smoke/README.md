# LeRobot / PushT ACT smoke

用途：快速验证 LeRobot 数据读取、ACT 训练、loss 日志和 checkpoint 输出。

依赖：

```text
data/lerobot/pusht/
hf_cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

准备：

```bash
make download-lerobot-pusht-dataset
```

启动：

```bash
bash experiments/lerobot/pusht_act_smoke/launch.sh
```

快速 2-step 检查：

```bash
LEROBOT_STEPS=2 LEROBOT_BATCH_SIZE=2 LEROBOT_NUM_WORKERS=0 \
LEROBOT_LOG_FREQ=1 LEROBOT_SAVE_FREQ=2 \
bash experiments/lerobot/pusht_act_smoke/launch.sh
```

默认结果：

```text
runs/experiments/lerobot/pusht_act_smoke/<run_id>/
```
