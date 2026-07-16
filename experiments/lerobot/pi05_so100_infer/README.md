# LeRobot / pi05 SO100 inference

用途：加载本地 pi05 base 或训练 checkpoint，对 SO100 的一个 sample 做离线 action 预测，确认推理链路可用。

## 准备

```bash
make download-lerobot-svla-so100-pickplace-dataset
make download-lerobot-pi05-base-policy
make download-lerobot-pi05-runtime-cache
```

`download-lerobot-pi05-runtime-cache` 依赖 `google/paligemma-3b-pt-224` 访问权限；如果未授权，先在 Hugging Face 申请并登录集群侧 `hf auth login`。

## 运行

```bash
conda activate lerobot
python experiments/lerobot/pi05_so100_infer/run.py
```

测训练 checkpoint：

```bash
LEROBOT_POLICY_PATH="$PWD/runs/experiments/lerobot/pi05_so100_8gpu_probe/<run_id>/lerobot_output/checkpoints/<step>/pretrained_model" \
python experiments/lerobot/pi05_so100_infer/run.py
```

输出：

```text
runs/experiments/lerobot/pi05_so100_infer/<run_id>/inference_evidence.json
```
