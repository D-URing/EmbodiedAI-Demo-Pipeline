# Scripts

脚本按 pipeline 分层，但这里不是日常实验入口。

约定：

- `make` 只做环境、下载、检查；
- `experiments/<route>/<experiment>/launch.sh` 负责启动训练/推理；
- `scripts/` 只放可复用执行器，不建议新同事直接猜脚本参数。

```text
scripts/
├── lerobot/     # LeRobot 主线脚本
├── fastwam/     # Custom WAM / FastWAM 后端脚本
├── imagewam/    # Custom WAM / ImageWAM 后端脚本
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

训练/推理入口见 [`../experiments/README.md`](../experiments/README.md)。

## FastWAM / custom

```text
scripts/fastwam/
├── download_release_artifacts.sh
├── prepare_fastwam_overlay.sh
├── run_realrobot_train_eval.sh
└── parse_train_log.py
```

训练入口见 [`../experiments/custom/fastwam_realrobot_smoke/`](../experiments/custom/fastwam_realrobot_smoke/)。

FastWAM LIBERO 数据已封装为 Make target，并按路线拆分：

```bash
make download-custom-fastwam-libero-dataset   # data/custom/fastwam/libero-fastwam
make download-lerobot-fastwam-libero-dataset  # data/lerobot/libero-fastwam/v2.1
```

LeRobot v3 转换入口：

```bash
make convert-lerobot-fastwam-libero-v3
```

## ImageWAM / custom

```text
scripts/imagewam/
├── prepare_imagewam_upstream.sh
├── download_artifacts.sh
└── run_train_eval.sh
```

下载 / upstream target：

```bash
make prepare-imagewam-upstream
make download-imagewam-artifacts
make download-imagewam-flux2-base
```

训练入口见 [`../experiments/custom/imagewam_flux2_4b_libero_pilot/`](../experiments/custom/imagewam_flux2_4b_libero_pilot/)。
