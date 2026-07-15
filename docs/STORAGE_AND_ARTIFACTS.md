# Storage and Artifacts

这份文档只回答一个问题：

> 项目里哪些目录是代码，哪些目录是本地/集群缓存，下载的数据和权重应该放哪里？

## 总原则

当前 SCUT 共享盘方案是：整个仓库放在共享盘，所有大文件也放在仓库目录下的 ignored 子目录里。

优点是路径稳定、团队容易复现；缺点是这些目录绝不能提交 Git。

最重要的约定：

> `data/` 和 `models/` 是仓库级资产池，不属于某个 pipeline。LeRobot、FastWAM、ImageWAM 和后续自拟模型都从这里选择自己需要的数据和权重。

## 代码目录 vs 本地目录

| 目录 | 提交 Git | 用途 |
|---|---:|---|
| `pipelines/` | 是 | LeRobot 与 Custom WAM 工程主线入口 |
| `experiments/` | 是 | 训练/推理实验启动入口和实验级配置 |
| `configs/` | 是 | 底层默认参数 |
| `scripts/` | 是 | 下载、训练、解析、报告执行器 |
| `src/` | 是 | 本项目 core Python 包 |
| `tasks/` | 是 | household/mock 任务定义 |
| `docs/` | 是 | 文档 |
| `references/` | 是 | 上游项目 pin 和 registry |
| `data/` | 只提交 `README.md` | 全局数据集资产池 |
| `models/` | 只提交 `README.md` | 全局模型/权重/checkpoint 资产池 |
| `checkpoints/` | 否 | 本项目或训练过程产生的 checkpoint |
| `runs/` | 否 | 日志、loss summary、report、manifest |
| `hf_cache/` | 否 | Hugging Face / Torch cache |
| `upstreams/` | 否 | clone 的 LeRobot / FastWAM 等上游源码 |
| `.external/` | 否 | 临时外部仓库参考，不作为项目依赖 |

这些 ignored 目录由 `.gitignore` 保护。

## 推荐环境变量

```bash
export PROJECT_ROOT="$PWD"
export EMBODIED_DATA_ROOT="$PROJECT_ROOT/data"
export EMBODIED_MODEL_ROOT="$PROJECT_ROOT/models"
export EMBODIED_RUN_ROOT="$PROJECT_ROOT/runs"
export EMBODIED_CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"
export HF_HOME="$PROJECT_ROOT/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PROJECT_ROOT/hf_cache/torch"
export PIP_CACHE_DIR="$PROJECT_ROOT/hf_cache/pip"
```

## 当前 SCUT 已知资产

| 资产 | 路径 | 说明 |
|---|---|---|
| LeRobot PushT | `data/lerobot/pusht` | ACT/PushT demo 数据 |
| LeRobot SVLA SO100 pick-place | `data/lerobot/svla_so100_pickplace` | SmolVLA fine-tune 数据 |
| ResNet18 backbone | `hf_cache/torch/hub/checkpoints/resnet18-f37072fd.pth` | ACT 默认视觉 backbone |
| LeRobot diffusion PushT policy | `models/lerobot/diffusion/diffusion_pusht` | 可选开源预训练 policy |
| LeRobot SmolVLA base | `models/lerobot/smolvla/smolvla_base` | SmolVLA fine-tune 起点 |
| LeRobot FastWAM LIBERO policy | `models/lerobot/fastwam/fastwam_libero_uncond_2cam224` | LeRobot-compatible FastWAM 权重 |
| FastWAM release ckpt | `models/custom/fastwam/release/libero_uncond_2cam224.pt` | 约 12G |
| FastWAM release stats | `models/custom/fastwam/release/libero_uncond_2cam224_dataset_stats.json` | stats / normalizer |
| FastWAM LIBERO 数据 | `data/custom/fastwam/libero-fastwam` | LeRobot v2.1，已解压 |
| ImageWAM FLUX.2 4B LIBERO | `models/custom/imagewam/flux2_klein_4b_libero` | ImageWAM release checkpoint，待集群下载 |
| ImageWAM FLUX.2 base / AE | `models/custom/imagewam/flux2` | 官方训练入口必需，部分 HF repo 可能 gated |
| ImageWAM upstream | `upstreams/ImageWAM` | 官方源码 checkout，ignored |

## Manifest

下载脚本或手动准备完成后，应在 `runs/artifact_manifests/` 留 manifest：

```text
runs/artifact_manifests/
├── lerobot_artifacts_manifest.json
├── lerobot_pusht_dataset_manifest.json
├── lerobot_svla_so100_pickplace_dataset_manifest.json
├── lerobot_diffusion_pusht_policy_manifest.json
├── lerobot_smolvla_base_policy_manifest.json
├── lerobot_fastwam_libero_policy_manifest.json
├── fastwam_release_artifacts_manifest.json
├── fastwam_libero_dataset_manifest.json
├── imagewam_upstream_manifest.json
└── imagewam_artifacts_manifest.json
```

Manifest 不是大文件，可以用于记录“当前共享盘上有什么”。如果 manifest 内容包含绝对路径，它只代表当前集群环境，不代表跨机器可复现路径。

## 不要混淆的几个概念

| 概念 | 放哪里 | 例子 |
|---|---|---|
| Dataset | `data/` | `data/lerobot/pusht` |
| Pretrained/release weight | `models/` | `models/lerobot/diffusion/diffusion_pusht`、`models/custom/fastwam/release/*.pt` |
| Torch 自动下载 backbone | `hf_cache/torch` | ResNet18 |
| 训练输出 | `runs/` | loss summary、stdout、LeRobot output |
| 实验启动配置 | `experiments/` | `experiments/lerobot/smolvla_so100_8gpu_long/config.sh` |
| 手工保留 checkpoint | `checkpoints/` 或 `models/custom/` | 后续稳定模型 |
| 上游源码 checkout | `upstreams/` | `upstreams/lerobot` |

## 清理原则

- 可以清理 `runs/` 里的旧 smoke，但保留关键成功 run 的 `loss_summary.json`；
- 不要删除 `data/` 和 `models/`，除非确认可重下；
- 不要把 `.venv` 放在共享盘项目内作为集群主环境，SCUT 走 Miniconda；
- 如果 NFS 出现空目录删不掉，先隔离成 `.trash`，不要让它阻塞主线。
