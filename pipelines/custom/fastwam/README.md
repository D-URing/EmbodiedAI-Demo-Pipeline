# Custom WAM / FastWAM

这是 FastWAM 在 `custom` 结构下的规范入口。

历史兼容入口仍保留在 [`../../custom_fastwam/`](../../custom_fastwam/)，但后续文档和新增后端都以 `custom/<backend>` 为准。

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
experiments/custom/fastwam_realrobot_smoke/
pipelines/custom_fastwam/README.md
docs/FASTWAM_REALROBOT_INTEGRATION.md
```

## 准备资产

```bash
make download-fastwam-artifacts
```

## 启动实验

训练/评测入口放在 `experiments/`，不要用 Makefile 启动：

```bash
bash scripts/fastwam/prepare_fastwam_overlay.sh
bash experiments/custom/fastwam_realrobot_smoke/launch.sh
```

切换 pilot：

```bash
FASTWAM_MODE=pilot FASTWAM_RECIPE=joint_base \
bash experiments/custom/fastwam_realrobot_smoke/launch.sh
```

注意：private overlay clone 需要 GitHub 私有仓库权限。
