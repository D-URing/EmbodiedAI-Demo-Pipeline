# Custom / FastWAM realrobot smoke

用途：FastWAM custom backend 的 smoke/pilot/full 入口。默认是 smoke。

依赖：

```text
models/custom/fastwam/release/libero_uncond_2cam224.pt
models/custom/fastwam/release/libero_uncond_2cam224_dataset_stats.json
data/custom/fastwam/libero-fastwam/
upstreams/FastWAM-realrobot/        # smoke/pilot/full 需要，offline-smoke 不需要完整训练
```

准备：

```bash
make download-custom-fastwam-libero-dataset
make download-fastwam-artifacts
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

启动：

```bash
bash experiments/custom/fastwam_realrobot_smoke/launch.sh
```

常用模式：

```bash
FASTWAM_MODE=offline-smoke bash experiments/custom/fastwam_realrobot_smoke/launch.sh
FASTWAM_MODE=smoke FASTWAM_RECIPE=joint_base bash experiments/custom/fastwam_realrobot_smoke/launch.sh
FASTWAM_MODE=pilot bash experiments/custom/fastwam_realrobot_smoke/launch.sh
```

注意：realrobot overlay 是私有仓库，需要 GitHub 权限。
