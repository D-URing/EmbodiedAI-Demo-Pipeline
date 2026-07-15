# LeRobot Multi-Model Training Plan

这份文档是接下来一个阶段的工程准绳：我们不再只做 demo，而是在 LeRobot 主线下真实跑通多个可训练模型，并且保留离线推理证据。

## 决策

第一轮 acceptance 不追求“最大模型全量训完”，而是要求：

1. 至少 2 个 LeRobot policy 能真实启动训练并看到 loss 日志；
2. 至少 1 个 VLA/foundation policy 能从本地开源权重 fine-tune；
3. 所有数据和权重都落到根目录资产池：

```text
data/
models/
hf_cache/
runs/
```

## 模型优先级

| 优先级 | Policy | 数据 | 目标 | 状态 |
|---|---|---|---|---|
| P0 | ACT | `lerobot/pusht` | 保底真实训练，快速看 loss | 已在 SCUT 验证 |
| P0 | Diffusion | `lerobot/pusht` | 第二条 IL 训练链路 | 已配置 |
| P1 | SmolVLA | `lerobot/svla_so100_pickplace` + `lerobot/smolvla_base` | A100 上的 VLA fine-tune | 已配置，待集群验证 |
| P1 | FastWAM | LIBERO/FastWAM 数据 + `lerobot/fastwam_libero_uncond_2cam224` | LeRobot-compatible world/action model 权重推理 | 权重已下载 |
| P2 | Pi0-FAST | `lerobot/aloha_sim_insertion_human` + base policy | 重 VLA 候选 | 模板已放入，不作为第一轮必跑 |
| P2 | GR00T N1.7 | DROID/LIBERO/SIMPLER 相关数据 | 大模型/评测候选 | 后续单独开任务 |

## 为什么先选这三个

- ACT：官方推荐入门 policy，训练快，已经证明能在我们的 A800/A100 类节点上跑出 loss 下降；
- Diffusion：同样是 LeRobot-native 的 imitation learning policy，可以验证第二套模型结构；
- SmolVLA：是真正的轻量 VLA/foundation model，官方文档说明可从 `lerobot/smolvla_base` fine-tune，20k steps 约单 A100 数小时量级；我们先用 2k steps 做工程验收。

## 集群命令

基础环境：

```bash
export BASE=/mnt/gpu11_200T/dingxibo
export PROJECT=$BASE/EmbodiedAI-Demo-Pipeline
export CONDA=$BASE/miniconda3/bin/conda

cd "$PROJECT"
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot

export PROJECT_ROOT="$PROJECT"
export EMBODIED_DATA_ROOT="$PROJECT/data"
export EMBODIED_MODEL_ROOT="$PROJECT/models"
export EMBODIED_RUN_ROOT="$PROJECT/runs"
export HF_HOME="$PROJECT/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PROJECT/hf_cache/torch"
export HF_ENDPOINT=https://hf-mirror.com
export HFD_BIN=/home/scut/hfd.sh
export HFD_TOOL=aria2c
export HFD_THREADS=10
export HFD_JOBS=4
```

下载第一轮资产：

```bash
make download-lerobot-pusht-dataset
make download-lerobot-svla-so100-pickplace-dataset
make download-lerobot-diffusion-pusht-policy
make download-lerobot-smolvla-base-policy
make download-lerobot-fastwam-libero-policy
```

训练 ACT：

```bash
export LEROBOT_STEPS=1000
export LEROBOT_BATCH_SIZE=8
make lerobot-train-act
```

训练 Diffusion：

```bash
export LEROBOT_STEPS=1000
export LEROBOT_BATCH_SIZE=8
make lerobot-train-diffusion
```

Fine-tune SmolVLA：

```bash
export LEROBOT_STEPS=2000
export LEROBOT_BATCH_SIZE=8
make lerobot-train-smolvla
```

如果显存压力偏大，优先降 batch size：

```bash
export LEROBOT_BATCH_SIZE=2
export LEROBOT_NUM_WORKERS=2
make lerobot-train-smolvla
```

## 推理

下载的开源 policy 推理：

```bash
export LEROBOT_DATASET_REPO_ID=lerobot/pusht
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/pusht"
export LEROBOT_POLICY_TYPE=diffusion
export LEROBOT_POLICY_CLASS=lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy
export LEROBOT_POLICY_PATH="$PROJECT/models/lerobot/diffusion/diffusion_pusht"
make lerobot-infer-smoke
```

训练产物推理时，把 `LEROBOT_POLICY_PATH` 指向对应 run 的 `lerobot_output/checkpoints/...` 或最终整理后的 `models/lerobot/<policy>/<name>`。

## 多卡和多节点

当前仓库先保证单机单卡/单机可见 GPU 的 LeRobot training command 可稳定运行。下一步再把同一组 profile 接入：

- `torchrun --nproc_per_node=N`；
- Slurm `sbatch`；
- 多节点 rendezvous 参数。

原则是 profile 不变，只替换 launcher。

## 验收标准

每次训练至少保留：

```text
runs/lerobot/<run_name>/<run_id>/
├── command.txt
├── train_stdout.log
├── loss_summary.json
└── lerobot_output/
```

`loss_summary.json` 应至少能回答：

- 是否出现 loss；
- first loss / last loss；
- loss 是否下降；
- 运行用了哪个 policy、dataset、steps、batch size。

## 参考来源

- LeRobot README / model zoo: https://github.com/huggingface/lerobot
- LeRobot ACT docs: https://huggingface.co/docs/lerobot/act
- LeRobot SmolVLA docs: https://huggingface.co/docs/lerobot/smolvla
- LeRobot Pi0 docs: https://huggingface.co/docs/lerobot/pi0
- LeRobot Pi0-FAST docs: https://huggingface.co/docs/lerobot/pi0fast
- LeRobot dataset v3 docs: https://huggingface.co/docs/lerobot/lerobot-dataset-v3
