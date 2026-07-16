# LeRobot / pi05 SO100 8-GPU probe

用途：在 LeRobot 官方架构下适配 pi05，先跑通 `SO100 dataset -> pi05 base -> accelerate train -> loss/speed report`。

## 资产准备

```bash
make download-lerobot-svla-so100-pickplace-dataset
make download-lerobot-pi05-base-policy
make download-lerobot-pi05-runtime-cache
make augment-lerobot-svla-so100-quantile-stats
```

默认路径：

```text
data/lerobot/svla_so100_pickplace/
models/lerobot/pi05/pi05_base/
hf_cache/hub/models--google--paligemma-3b-pt-224/
```

`download-lerobot-pi05-runtime-cache` 会下载 `google/paligemma-3b-pt-224` 的 tokenizer/config。该 Hugging Face repo 可能需要先申请访问并在集群上 `hf auth login` 或设置 `HF_TOKEN`。

也可以直接使用聚合入口：

```bash
make prepare-lerobot-pi05-so100-assets
```

`augment-lerobot-svla-so100-quantile-stats` 用于补齐 pi05 quantile normalization 所需的本地 `q01/q99` stats，只写项目内数据目录，不会上传 Hub。

## 单机八卡探针

```bash
conda activate lerobot
python experiments/lerobot/pi05_so100_8gpu_probe/run.py
```

等价入口：

```bash
bash experiments/lerobot/pi05_so100_8gpu_probe/launch.sh
```

默认是 200 steps、每卡 batch size 1，用于快速确认能否训练和测速。常用覆盖：

```bash
LEROBOT_STEPS=1000 \
LEROBOT_BATCH_SIZE=2 \
LEROBOT_POLICY_COMPILE_MODEL=false \
python experiments/lerobot/pi05_so100_8gpu_probe/run.py
```

输出：

```text
runs/experiments/lerobot/pi05_so100_8gpu_probe/<run_id>/
├── backend_manifest.json
├── command.txt
├── config.sh
├── loss_summary.json
├── speed_summary.json
└── train_stdout.log
```

`loss_summary.json` 看 loss 是否下降，`speed_summary.json` 看近似 step/s 和 sample/s。第一次开启 `torch.compile` 时会包含编译开销，正式测速建议第二次运行或关闭 compile 做对照。

已验证真实 8 卡 2-step：

```text
run_id=smoke2_quiet_20260716_202905
loss=0.347 -> 0.141
parsed_step_metrics.mean_samples_per_second=6.0
parsed_step_metrics.max_memory_gb=45.76
```
