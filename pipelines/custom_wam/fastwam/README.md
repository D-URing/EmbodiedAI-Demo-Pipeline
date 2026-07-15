# Custom WAM / FastWAM

这是 FastWAM 在 `custom_wam` 结构下的规范入口。

历史兼容入口仍保留在 [`../../custom_fastwam/`](../../custom_fastwam/)，但后续文档和新增后端都以 `custom_wam/<backend>` 为准。

## 当前定位

FastWAM 这条线不是“完全从零自研模型”，而是：

```text
FastWAM official release
  + private realrobot overlay
  + 项目级数据/模型/日志约定
  -> custom backend fine-tuning / evaluation evidence
```

## 关键路径

```text
configs/fastwam/realrobot_train_eval.sh
scripts/fastwam/download_release_artifacts.sh
scripts/fastwam/prepare_fastwam_overlay.sh
scripts/fastwam/run_realrobot_train_eval.sh
scripts/fastwam/slurm_realrobot_pilot.sbatch
pipelines/custom_fastwam/README.md
docs/FASTWAM_REALROBOT_INTEGRATION.md
```

## 常用命令

```bash
make download-fastwam-artifacts
make fastwam-train-smoke
```

在 SCUT 上如果要继续 realrobot overlay：

```bash
bash scripts/fastwam/prepare_fastwam_overlay.sh
FASTWAM_MODE=pilot FASTWAM_RECIPE=joint_base bash scripts/fastwam/run_realrobot_train_eval.sh
```

注意：private overlay clone 需要 GitHub 私有仓库权限。
