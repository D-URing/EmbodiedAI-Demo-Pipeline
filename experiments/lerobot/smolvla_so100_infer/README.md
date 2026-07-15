# LeRobot / SmolVLA SO100 inference

用途：加载本地 SmolVLA policy/base 或训练 checkpoint，跑离线推理 smoke。

依赖：

```text
data/lerobot/svla_so100_pickplace/
models/lerobot/smolvla/smolvla_base/
```

准备：

```bash
make download-lerobot-svla-so100-pickplace-dataset
make download-lerobot-smolvla-base-policy
```

启动：

```bash
bash experiments/lerobot/smolvla_so100_infer/launch.sh
```

输出：

```text
runs/experiments/lerobot/smolvla_so100_infer/<run_id>/inference_evidence.json
```
