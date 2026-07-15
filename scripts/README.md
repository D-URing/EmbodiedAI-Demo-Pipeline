# Scripts

脚本按 pipeline 分层。优先通过 `make` 或 `pipelines/*/README.md` 里的命令调用，不建议新同事直接猜脚本参数。

```text
scripts/
├── lerobot/     # LeRobot 主线脚本
├── fastwam/     # Custom/FastWAM 主线脚本
└── reference/   # 外部参考项目 fetch/pin
```

## LeRobot

```text
scripts/lerobot/
├── install_lerobot_cluster.sh
├── download_artifacts.sh
├── run_dataset_smoke.sh
├── run_pusht_act_gpu_smoke.sh
├── run_inference_smoke.sh
├── inspect_dataset.py
├── run_policy_inference_smoke.py
├── parse_train_log.py
└── generate_data_to_inference_report.py
```

主要 Make target：

```bash
make download-lerobot-artifacts
make lerobot-data-smoke
make lerobot-train-smoke
make lerobot-infer-smoke
make demo-chain-lerobot-fastwam
```

## FastWAM / custom

```text
scripts/fastwam/
├── download_release_artifacts.sh
├── prepare_fastwam_overlay.sh
├── run_realrobot_train_eval.sh
├── parse_train_log.py
└── slurm_realrobot_pilot.sbatch
```

主要 Make target：

```bash
make download-fastwam-artifacts
make fastwam-train-smoke
make demo-chain-fastwam
```

注意：FastWAM LIBERO 数据当前用 `/home/scut/hfd.sh yuanty/LIBERO-fastwam --dataset` 手动下载，尚未封装为 Make target。

