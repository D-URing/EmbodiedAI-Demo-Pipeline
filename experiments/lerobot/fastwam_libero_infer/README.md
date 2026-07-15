# LeRobot / FastWAM LIBERO inference

用途：通过 LeRobot-compatible FastWAM policy 跑离线推理 smoke。

启动：

```bash
bash experiments/lerobot/fastwam_libero_infer/launch.sh
```

注意：LeRobot 路线使用自己的 LIBERO/FastWAM 数据目录：

```text
data/lerobot/libero-fastwam/v2.1/     # 原始 release 副本
data/lerobot/libero-fastwam/v3/       # 当前 LeRobot loader 的转换目标
```

当前 `configs/lerobot/infer/fastwam_libero.sh` 默认读取：

```text
data/lerobot/libero-fastwam/v3/libero_10_no_noops_lerobot/
```

不要让这个实验直接读写 `data/custom/fastwam/libero-fastwam/`；那是 custom/FastWAM pipeline 的数据。

如果 `v3` 目录还没有转换：

```bash
make convert-lerobot-fastwam-libero-v3
```
