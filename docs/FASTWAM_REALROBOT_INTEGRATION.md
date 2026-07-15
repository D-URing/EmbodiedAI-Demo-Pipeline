# FastWAM 真机训练/评测后端集成方案

本文把 `D-URing/fastwam-realrobot-pipeline` 纳入当前 demo pipeline，定位是：**custom FastWAM / realrobot overlay**。项目主线已经调整为 LeRobot-first；FastWAM 分成两条路径：

- **LeRobot-native FastWAM**：优先路径，通过 LeRobot 的 FastWAM policy 接入 data-to-inference 主线；
- **Custom FastWAM overlay**：本文描述的内部扩展路径，用于私有真机数据、7D/10D recipe、集群训练和未来自建模型。

本仓库不复制 FastWAM 大模型代码、不下载权重、不把 CUDA 依赖塞进 core 环境；它只负责稳定入口、配置开关、日志解析和结果归档。

## 1. 集成目标

本文这条 overlay 路径的第一阶段目标不是“训练最强家庭机器人模型”，而是保留一条真实可训练、可扩展的内部模型链路：

```text
demo pipeline config
  -> FastWAM overlay workspace
  -> CUDA/DeepSpeed/Accelerate training
  -> native FastWAM checkpoints
  -> runs/experiments/custom/fastwam_realrobot_smoke/* 统一归档
  -> loss_summary.json / loss_report.md 回答 loss 是否下降
```

这条链路满足三个约束：

- 不再使用 CPU toy trainer，也没有 CPU fallback；
- FastWAM 私有仓库作为 overlay，不 vendor 到本仓库；
- 第一版只做 headless 训练/离线评测，Viewer 和真机闭环靠后。

## 2. 双路径定位

当前推荐把 FastWAM 拆成两条路径：

| 路径 | 当前选择 | 作用 |
|---|---|---|
| LeRobot-native | LeRobot `policy.type=fastwam` | 第一 demo 主线，优先跑 dataset read → train/load → inference |
| Custom overlay | `D-URing/fastwam-realrobot-pipeline` | 私有 realrobot 数据、内部 recipe、未来自建模型 |

本文后续只描述 custom overlay：

| 层级 | 当前选择 | 作用 |
|---|---|---|
| 上游模型基座 | `yuantianyuan01/FastWAM` | 官方 Fast-WAM 训练/推理代码基座 |
| 内部 overlay | `D-URing/fastwam-realrobot-pipeline` | 真机 LeRobot v3 数据读取、7D/10D 配置、训练脚本、离线 probe |
| demo pipeline | 本仓库 `scripts/fastwam/*` | custom overlay 的环境准备、统一启动、日志解析、结果归档 |
| 后续评测思想 | RoboDojo-style staged evaluation | 分阶段、partial progress、失败分类，不急着接完整 simulator |

私有 overlay 已验证过真实八卡 smoke：`step=1/1 loss=1.4862 loss_action=1.1472 loss_video=0.3390`，并能落盘 weights/state checkpoint。这个事实让它适合作为 custom backend 和未来自建模型路线，而不是被 LeRobot-native 主线替代。

## 3. 环境准备

在 NVIDIA 集群的 FastWAM 环境里使用 Python 3.10。先准备 overlaid source：

```bash
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

这一步会：

- clone 官方 FastWAM 到 `$FASTWAM_WORKDIR`；
- clone 私有 overlay 到 `$FASTWAM_OVERLAY_DIR`；
- checkout 到 `configs/fastwam/realrobot_train_eval.sh` 中 pin 的 commit；
- 用 `rsync` 把 overlay 覆盖到官方 FastWAM 工作树；
- 排除 `runs/`、`data/`、`checkpoints/`、`evaluate_results/`，避免把大文件或产物带入源码。

如果同一个 `$FASTWAM_WORKDIR` 已经 overlay 过，工作树会是 dirty 状态。为了避免误删手工修改，脚本不会默认重置；确认它只是生成缓存后再运行：

```bash
FASTWAM_RESET_WORKDIR=1 bash scripts/fastwam/prepare_fastwam_overlay.sh
```

如果要同时安装 CUDA 环境，在已选好的集群节点上执行：

```bash
FASTWAM_CREATE_CONDA=1 FASTWAM_INSTALL=1 \
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

默认 PyTorch wheel 是 CUDA 12.8：

```bash
torch==2.7.1+cu128
torchvision==0.22.1+cu128
```

如果集群 driver/CUDA 策略不同，用 `FASTWAM_TORCH_INDEX_URL` 和团队镜像覆盖，不要改公共脚本。

## 4. 模型、数据和路径

默认路径全部落在项目内，集群上仍可通过环境变量覆盖：

| 变量 | 默认 | 说明 |
|---|---|---|
| `FASTWAM_WORKDIR` | `$PROJECT_ROOT/upstreams/FastWAM-realrobot` | overlay 后的可运行 FastWAM 树 |
| `FASTWAM_MODEL_BASE` | `$PROJECT_ROOT/models` | Wan/FastWAM 模型根目录 |
| `FASTWAM_RELEASE_CKPT` | `$FASTWAM_MODEL_BASE/custom/fastwam/release/libero_uncond_2cam224.pt` | FastWAM LIBERO release 权重 |
| `FASTWAM_RELEASE_DATASET_STATS` | `$FASTWAM_MODEL_BASE/custom/fastwam/release/libero_uncond_2cam224_dataset_stats.json` | release stats |
| `FASTWAM_PIN_STATS` | 空 | V6 多机 recipe 建议显式传，避免在线重扫 stats |

release 权重建议用本仓库封装入口准备；脚本会自动兼容旧版 `huggingface-cli` 和新版 `hf` CLI：

```bash
make download-fastwam-artifacts
```

ActionDiT backbone 如果缺失，建议生成到项目内 `checkpoints/fastwam/`：

```bash
cd "$FASTWAM_WORKDIR"
mkdir -p "$PROJECT_ROOT/checkpoints/fastwam"
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output "$PROJECT_ROOT/checkpoints/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
  --device cuda \
  --dtype bfloat16
```

## 5. 训练入口

最小 CUDA smoke：

```bash
FASTWAM_MODE=smoke FASTWAM_RECIPE=joint_base \
bash scripts/fastwam/run_realrobot_train_eval.sh
```

用于回答“loss 有没有下降”的 pilot：

```bash
FASTWAM_MODE=pilot FASTWAM_RECIPE=joint_base \
bash scripts/fastwam/run_realrobot_train_eval.sh
```

更接近当前真机路线的 V6 decision pilot：

```bash
FASTWAM_MODE=pilot \
FASTWAM_RECIPE=v6_decision \
FASTWAM_PIN_STATS=/root/paddlejob/share-storage/gpfs/system-public/dingxibo/Embodied_AI/data/package_scan_v6/meta/fastwam_v6_delta_stats.json \
bash scripts/fastwam/run_realrobot_train_eval.sh
```

Slurm 集群可从模板开始：

```bash
sbatch experiments/custom/fastwam_realrobot_smoke/slurm.sbatch
```

如果不是 Slurm，直接在调度器命令里执行 `scripts/fastwam/run_realrobot_train_eval.sh` 即可；不要把集群私有参数写进仓库。

## 6. Recipe 与任务映射

| `FASTWAM_RECIPE` | FastWAM task | 建议用途 |
|---|---|---|
| `joint_base` | `real_robot_joint_2cam224_1e-4` | 第一条低风险真机 7D smoke/pilot |
| `pose_base` | `real_robot_uncond_2cam224_1e-4` | 10D 末端位姿数据路线 |
| `v6_clean` | `real_robot_joint_2cam224_v6_clean` | canonical V6 recipe |
| `v6_decision` | `real_robot_joint_2cam224_v6_decision` | 当前推荐的决策帧加权路线 |
| `v6_codebook` | `real_robot_joint_2cam224_v6_codebook` | World Fast Codebook 可选扩展 |
| `v6_scratch` | `real_robot_joint_2cam224_v6_scratch` | 不继承 LIBERO 任务权重的对照 |
| `v6_discrim` | `real_robot_joint_2cam224_v6_2_discrim` | 判别性重采样对照 |
| `v6_dagger` | `real_robot_joint_2cam224_v6_dagger` | 漂移恢复增强 |
| `v6_robust` | `real_robot_joint_2cam224_v6_1_robust` | 鲁棒性变体 |

也可以直接覆盖：

```bash
FASTWAM_TASK_NAME=real_robot_joint_2cam224_v6_codebook \
FASTWAM_MODE=pilot \
bash experiments/custom/fastwam_realrobot_smoke/launch.sh
```

## 7. 产物结构

本仓库镜像产物：

```text
runs/experiments/custom/fastwam_realrobot_smoke/<run_id>/
├── backend_manifest.json
├── command.txt
├── fastwam_native_output_dir.txt
├── train_stdout.log
├── loss_summary.json
└── loss_report.md
```

FastWAM 原生产物仍在：

```text
$FASTWAM_WORKDIR/runs/<task_name>/<run_id>/
├── config.yaml
├── dataset_stats.json
├── checkpoints/
│   ├── weights/step_*.pt
│   └── state/step_*/
└── eval/
```

`loss_summary.json` 会记录：

- `initial_loss`、`final_loss`、`loss_drop_ratio`；
- `loss_decreased`；
- `loss_action`、`loss_video` 等子指标的首末值；
- final step / max steps；
- latest weights/state checkpoint。

这就是给同事回答“loss 是否正常下降”的第一证据。smoke 只有 1 step 时 `loss_decreased=unknown` 是正常的；要证明下降，跑 pilot 或更长。

## 8. 生成 Demo Chain 报告

FastWAM 训练完成后，用本仓库的统一 importer 把训练 run 变成交付 evidence：

```bash
embodied-demo report-fastwam --run-dir runs/experiments/custom/fastwam_realrobot_smoke/<run_id>
```

或者：

```bash
FASTWAM_RUN_DIR=runs/experiments/custom/fastwam_realrobot_smoke/<run_id> embodied-demo report-fastwam
```

默认输出：

```text
runs/demo_chains/fastwam_realrobot_v0/<run_id>/
├── chain_manifest.yaml
├── training_evidence.json
├── checkpoint_summary.json
├── report.md
└── handoff.md
```

其中 `training_evidence.json` 是给工程侧消费的稳定摘要，`report.md` 是给团队同步/交差的可读报告，`handoff.md` 写明最小复现步骤和边界。

## 9. 与 demo pipeline 的契约映射

| demo pipeline 概念 | FastWAM 当前对应 |
|---|---|
| experiment config | FastWAM Hydra task，例如 `real_robot_joint_2cam224_v6_decision` |
| dataset episodes | 当前不接仿真 scene；由真实数据 episode 和相机视角承载 |
| `observation` | LeRobot v3 风格样本：`video[3,9,224,448] / proprio / context` |
| `policy interface` | 训练时 `training_loss(sample)`；离线/部署时 `infer_action()` |
| `action output` | action horizon/chunk，7D joint 或 10D pose |
| `rollout` | 第一阶段是训练集/离线 probe；仿真 rollout 后续接 RoboDojo/RoboTwin |
| `logger` | FastWAM stdout + 本仓库 parser 归档 |
| `evaluator` | loss/ckpt smoke、offline action check；后续扩展 RoboDojo-style stage score |
| `viewer` | 暂缓；优先保证训练与评测产物稳定 |

## 10. RoboDojo-style 评测分层

这里不急着声称已经跑 RoboDojo simulator，而是借用它的工程评测思想，把验收拆成可逐步通过的阶段：

| 阶段 | 名称 | 通过标准 | 失败分类 |
|---|---|---|---|
| E0 | backend preflight | CUDA、模型、数据、task config 可见 | environment/config |
| E1 | data sample smoke | dataset 可实例化，shape 与 action dim 正确 | data/contract |
| E2 | train smoke | 真实前反传、loss 有数、checkpoint 落盘 | training/runtime |
| E3 | train pilot | 多个 log 点 loss 下降，无 NaN/inf | training/convergence |
| E4 | offline action check | `infer_action()` 对训练样本输出正常，L1 与既有基线可比 | policy/regression |
| E5 | sim benchmark | 映射 RoboDojo/RoboTwin/RoboCasa 任务，输出 success/progress | simulation |
| E6 | real robot gated eval | 安全栅栏下执行真机闭环 | safety/real_robot |

当前本仓库已落地 E2/E3 的启动和日志归档；E1/E4 可复用 overlay 里的 `preflight_*` 和 `offline_action_check.py`，后续再包装成统一 evaluator。

## 11. 当前边界

- 本仓库不包含官方 FastWAM、私有 overlay、权重、数据或 runs。
- `prepare_fastwam_overlay.sh` 需要 GitHub 私有仓库权限。
- `run_realrobot_train_eval.sh` 直接调用 `scripts/train_zero1.sh`，避免使用 overlay 中仍硬编码旧 GPFS 路径的 launcher。
- 真实 loss 下降只能在 CUDA 集群上证明；本地测试只验证 parser 和脚本契约。
- 多机/PaddleCloud 的完整自动探测逻辑仍保留在 overlay 的 `train_v6_multinode.sh` / `run_mpirun_v6.sh`，本仓库第一版只提供通用单节点/Slurm 入口和可覆盖开关。
