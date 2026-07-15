# LeRobot / Diffusion PushT inference

用途：加载已下载的 LeRobot Diffusion PushT policy，跑离线推理 smoke。

依赖：

```text
data/lerobot/pusht/
models/lerobot/diffusion/diffusion_pusht/
```

准备：

```bash
make download-lerobot-pusht-dataset
make download-lerobot-diffusion-pusht-policy
```

启动：

```bash
bash experiments/lerobot/diffusion_pusht_infer/launch.sh
```

输出：

```text
runs/experiments/lerobot/diffusion_pusht_infer/<run_id>/inference_evidence.json
```
