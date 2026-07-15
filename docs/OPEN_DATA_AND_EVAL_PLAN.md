# Open Data and Evaluation Plan

这份文档负责回答：哪些开源数据现在值得下，哪些只适合先登记，评测体系先参考什么。

## 数据分层

### A. 立即用于训练/推理 smoke

| 数据 | 路径 | 用途 | 命令 |
|---|---|---|---|
| `lerobot/pusht` | `data/lerobot/pusht` | ACT / Diffusion 训练与推理 | `make download-lerobot-pusht-dataset` |
| `lerobot/svla_so100_pickplace` | `data/lerobot/svla_so100_pickplace` | SmolVLA fine-tune | `make download-lerobot-svla-so100-pickplace-dataset` |
| `yuanty/LIBERO-fastwam` | `data/fastwam/libero-fastwam` | FastWAM / ImageWAM custom WAM 路线 | 见 `pipelines/custom_wam/README.md` |

### B. 建议先下 sample / 子集

| 数据 | 路径 | 为什么 |
|---|---|---|
| `Perflow-Shuai/RoVid-20K-10s` | `data/internet/rovid-20k-10s` | RoVid-X 的实用子集，适合先验证视频/世界模型数据读取 |
| `ropedia-ai/xperience-10m-sample` | `data/human/xperience-10m-sample` | Xperience-10M sample 约 GB 级，适合先看人类 4D 数据结构 |

命令：

```bash
make download-data-rovid20k
make download-data-xperience10m-sample
```

### C. 高价值但不建议盲目全量下载

| 数据 | 用途 | 风险 |
|---|---|---|
| `DAGroup-PKU/RoVid-X` | 机器人视频/世界模型预训练 | 大于 TB 级 |
| `robbyant/mdm_depth` | RGB-D / depth 表征 | 多 TB 级 |
| `XDOF/ABC-130k` | 双臂遥操作 VLA 数据 | HF gated + TB 级 |
| `agibot-world/AgiBotWorld-Alpha/Beta` | 大规模真机/人形数据 | 大规模、多格式适配成本 |
| `InternRobotics/InternData-A1` | 仿真+真实混合 manipulation 数据 | gated/license + 大规模 |

对应命令已经准备好，但执行前要确认共享盘容量和访问权限：

```bash
make download-data-rovidx
make download-data-mdm-depth
make download-data-abc130k
make download-data-agibotworld-alpha
make download-data-interndata-a1
```

### D. 需要官方工具链的数据

| 数据 | 处理方式 | 项目定位 |
|---|---|---|
| Ego4D | 官方 CLI | 人类 egocentric activity prior |
| EPIC-KITCHENS / HD-EPIC | 官方下载脚本 | 厨房动作、物体、recipe prior |
| EgoVerse | 官方 repo + S3/R2 sync | human-to-robot transfer |
| Open-X-Embodiment | 官方 builders / dataset tools | 多机器人预训练参考 |
| DROID / BridgeData | 官方数据工具 | 真机 finetune reference |
| RoboCOIN | 官方 toolkit，LeRobot-compatible | 双臂多平台数据和 metadata 设计参考 |

## 当前评测策略

我们暂时不做真机评测，也不把仿真评测作为第一轮 blockers。当前只要求：

```text
dataset inspection
  -> training loss evidence
  -> offline inference evidence
  -> report / manifest
```

但评测设计要提前贴近 RoboDojo 的能力维度：

| RoboDojo 维度 | 我们当前怎么映射 |
|---|---|
| Generalization | 同 policy 换不同 dataset / task profile 后能否训练和推理 |
| Memory | 暂时记录长程任务需求，后续接 kitchen/household long-horizon |
| Precision | 用 PushT、LIBERO、drawer/pick-place 类任务作为候选 |
| Long-horizon | 后续接 RoboCasa365 / RoboTwin 2.0 |
| Open-vocabulary | SmolVLA / Pi0 系列 profile 保留语言入口 |

后续仿真优先级：

1. LeRobot eval + LIBERO：最贴近当前 FastWAM 数据和已有命令；
2. RoboCasa365：厨房任务丰富，适合做菜/整理厨房 demo 扩展；
3. RoboTwin 2.0：双臂任务、bimanual 能力；
4. SIMPLER：参考 sim-real correlation 思路；
5. RoboDojo：作为能力矩阵和 leaderboard 风格的总评测框架参考。

## 参考来源

- RoVid-X: https://huggingface.co/datasets/DAGroup-PKU/RoVid-X
- MDM Depth: https://huggingface.co/datasets/robbyant/mdm_depth
- Xperience-10M: https://huggingface.co/datasets/ropedia-ai/xperience-10m
- ABC-130k: https://huggingface.co/datasets/XDOF/ABC-130k
- AgiBotWorld: https://huggingface.co/agibot-world
- InternData-A1: https://huggingface.co/datasets/InternRobotics/InternData-A1
- Ego4D: https://ego4d-data.org/docs/start-here/
- EPIC-KITCHENS: https://epic-kitchens.github.io/
- EgoVerse: https://github.com/GaTech-RL2/EgoVerse
- RoboDojo: https://robodojo-benchmark.com
- RoboCasa: https://robocasa.ai/
- RoboTwin: https://robotwin-platform.github.io/
- SIMPLER: https://simpler-env.github.io/
