# 环境配置指南

本文是团队搭建本地开发环境和未来 NVIDIA 集群环境的稳定入口。原则是：**core、policy、simulator、real robot 分环境维护，通过已版本化 contract 通信**，不创建一个包含所有依赖的巨型环境。

## 1. 环境分层

| 环境 | 当前状态 | 主要内容 | 不应包含 |
|---|---|---|---|
| Core | 当前必需 | schema、配置、CLI、logger/evaluator、mock/replay | CUDA、Isaac、VLA 权重、厂商 SDK |
| Policy | M7 接入 | LeRobot/OpenPI/GR00T 等策略及其 PyTorch/CUDA | simulator 私有依赖 |
| Simulator | M6 接入 | RoboDojo/Isaac、RoboCasa 或其他首选后端 | 多个互相冲突的 simulator |
| Real robot | 硬件确定后 | ROS 2 或厂商 SDK、安全控制 | 训练和仿真工具链 |

本地调试时，轻量 policy 可以与 core 同进程；集群和重量级模型默认通过 WebSocket 等 transport 独立运行。跨环境只交换 Observation、ActionChunk、RunSpec 和 artifact contract。

## 2. 当前支持基线

| 项目 | 基线 |
|---|---|
| Python | 3.11；当前验收版本 3.11.15 |
| CPU 架构 | macOS arm64 已验收；Linux x86_64 作为集群目标 |
| 包管理 | 本地 core 默认 `venv` + pip；NVIDIA/SCUT 集群推荐 Miniconda |
| 依赖声明 | `pyproject.toml` |
| 已验证约束 | `requirements/constraints-py311.txt` |
| 默认运行 | local、CPU、headless、inproc、mock |

`pyproject.toml` 表达允许的依赖范围；constraints 文件记录最后一次完整验收的精确版本。升级 constraints 必须重新运行测试、配置校验和 dry-run。

## 3. macOS 开发环境

### 3.1 前置工具

```bash
xcode-select --install
brew install python@3.11 gh
gh auth login
```

如果已经安装，先检查：

```bash
brew --version
python3.11 --version
gh auth status
```

GitHub 只影响代码发布，不影响本地 pipeline 运行。

### 3.2 创建 core 环境

在仓库根目录执行：

```bash
make setup
make doctor
make test
make validate
```

常规开发需要激活环境时：

```bash
source .venv/bin/activate
```

删除或重建 `.venv` 不影响仓库数据；`runs/`、`.venv/`、本地密钥和缓存均不会进入 Git。

### 3.3 系统代理与 shell 代理

macOS 的“系统设置 → 网络 → 代理”不会保证把代理自动导出为 shell 环境变量。浏览器可以访问 GitHub、但 `git push` 直连 443 超时时，先比较：

```bash
scutil --proxy
env | grep -i proxy
```

如果本地代理监听 `127.0.0.1:7890`，可在个人 `~/.zprofile` 中加入以下可开关配置；端口应按个人代理软件调整，不要提交到项目文件：

```bash
proxy_on() {
  export HTTP_PROXY="http://127.0.0.1:7890"
  export HTTPS_PROXY="http://127.0.0.1:7890"
  export ALL_PROXY="socks5h://127.0.0.1:7890"
  export NO_PROXY="localhost,127.0.0.1,::1,*.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
  export http_proxy="$HTTP_PROXY"
  export https_proxy="$HTTPS_PROXY"
  export all_proxy="$ALL_PROXY"
  export no_proxy="$NO_PROXY"
}

proxy_off() {
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  unset http_proxy https_proxy all_proxy no_proxy
}

if nc -z 127.0.0.1 7890 2>/dev/null; then
  proxy_on
fi
```

保存后执行 `source ~/.zprofile`，再验证：

```bash
git ls-remote --heads origin
```

同时设置大写和小写变量是为了兼容不同 CLI。代理可能关闭时，优先使用上述可达性检测，不建议把带凭据的 proxy URL 写入 Git remote 或仓库 YAML。

## 4. Linux 工作站与 NVIDIA 集群

### 4.1 先收集集群事实

在安装 CUDA、PyTorch、Isaac 或容器前，先确认并记录：

- GPU 型号、单节点 GPU 数、显存和 MIG 策略；
- NVIDIA driver 与允许的 CUDA 范围；
- 调度器（Slurm/Kubernetes/其他）和容器运行时；
- 共享存储、scratch、配额和清理策略；
- 计算节点是否能访问 PyPI、GitHub、模型仓库；
- 是否允许节点间端口通信；
- 团队已有的日志、实验追踪与镜像仓库。

不要根据本机 macOS 环境推断集群 CUDA 版本，也不要在登录节点运行 simulator 或下载大模型。

### 4.2 创建集群 core 环境

优先使用集群提供的 Python 3.11 module；具体 module 名称由集群管理员确定：

```bash
module load <python-3.11-module>
python3.11 --version
VENV="$SCRATCH/venvs/embodied-core" make setup
VENV="$SCRATCH/venvs/embodied-core" make doctor
VENV="$SCRATCH/venvs/embodied-core" make test
```

如果集群不提供 module，推荐统一安装 Miniconda 到共享盘或个人目录；不要让每位成员选择不同的 CUDA/Python 组合。

#### SCUT/A100 共享盘 Miniconda 推荐做法

当前 SCUT 管理节点和 `gpu11` 已验证可使用共享盘 Miniconda。假设项目位于：

```bash
export BASE=/mnt/gpu11_200T/dingxibo
export PROJECT=$BASE/EmbodiedAI-Demo-Pipeline
export CONDA=$BASE/miniconda3/bin/conda
```

如尚未安装 Miniconda：

```bash
mkdir -p "$BASE/downloads"
wget -O "$BASE/downloads/Miniconda3-latest-Linux-x86_64.sh" \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash "$BASE/downloads/Miniconda3-latest-Linux-x86_64.sh" -b -p "$BASE/miniconda3"
"$CONDA" --version
```

创建 core 环境时使用清华 conda-forge 镜像，并避开 defaults channel 的交互式 ToS 问题：

```bash
"$CONDA" create -y -n embodied-core --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  python=3.11 pip setuptools wheel

cd "$PROJECT"
"$CONDA" run -n embodied-core python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --upgrade pip setuptools wheel
"$CONDA" run -n embodied-core python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -c requirements/constraints-py311.txt -e ".[dev]"
"$CONDA" run -n embodied-core python -m pytest
```

当前已验证结果：

```text
34 passed
```

需要交互式 shell 时：

```bash
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate embodied-core
```

如果之前误建了项目内 `.venv`，先不要在共享盘上反复 `pip install --upgrade pip`。安全处理方式是把它移出主路径：

```bash
cd "$PROJECT"
mv .venv ".venv.bad.$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true
```

若 NFS 返回“目录非空”导致 `rm -rf` 无法删除，但 `.venv` 主入口已经不存在，可以暂时保留 `.venv.bad.*.trash`；它不会影响 conda 环境。后续由有共享盘维护权限的一侧清理即可。

### 4.3 下载、缓存和产物位置

当前项目假设整个仓库目录放在团队共享盘上，因此默认把数据、模型权重、Hugging Face cache、上游源码和运行输出都放在项目内 ignored 目录：

```bash
export PROJECT_ROOT="$PWD"
export EMBODIED_DATA_ROOT="$PROJECT_ROOT/data"
export EMBODIED_MODEL_ROOT="$PROJECT_ROOT/models"
export EMBODIED_RUN_ROOT="$PROJECT_ROOT/runs"
export HF_HOME="$PROJECT_ROOT/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PROJECT_ROOT/hf_cache/torch"
export PIP_CACHE_DIR="$PROJECT_ROOT/hf_cache/pip"
```

对应目录 `data/`、`models/`、`checkpoints/`、`runs/`、`artifacts/`、`upstreams/` 和 `hf_cache/` 均已被 `.gitignore` 忽略。若某个集群必须使用独立 scratch/cache，再显式覆盖这些变量。

LeRobot ACT/PushT 训练还会使用 torchvision ResNet18 backbone。若计算节点不能联网，需要提前准备：

```bash
mkdir -p "$TORCH_HOME/hub/checkpoints"
wget -O "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth" \
  https://download.pytorch.org/models/resnet18-f37072fd.pth
```

### 4.4 无公网计算节点

当前 core 依赖很小，可在有网络的同架构节点准备 wheelhouse，再复制到受限环境：

```bash
python3.11 -m venv .wheel-builder
source .wheel-builder/bin/activate
python -m pip install --upgrade pip
python -m pip wheel -c requirements/constraints-py311.txt \
  --wheel-dir wheelhouse ".[dev]"
```

在离线节点中：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheelhouse \
  'embodied-ai-demo-pipeline[dev]'
.venv/bin/python -m pip check
```

wheelhouse 必须由与目标节点相同的操作系统和 CPU 架构构建。M5 会用容器和正式离线安装流程替代这套临时方式。

## 5. NVIDIA 与 simulator 接入边界

当前仓库不安装 `torch`、CUDA Toolkit、Isaac Sim、RoboDojo 或其他 simulator。确定首个后端后再新增独立 constraints/container，并记录：

```bash
nvidia-smi
python --version
python -m pip freeze
```

每个 simulator adapter 必须先通过 E0 schema 和 E1 wiring smoke，再进入 GPU sweep。不能因为 simulator 支持并行环境，就默认将 `num_envs` 调大；只有 policy 声明 `supports_batch: true` 后配置才允许多环境。

### 5.1 FastWAM real-robot policy/training 环境

FastWAM 属于独立 CUDA policy 环境，不进入 core `.venv`。推荐在 NVIDIA 集群创建 Python 3.10/Conda 环境：

```bash
FASTWAM_CREATE_CONDA=1 FASTWAM_INSTALL=1 \
CONDA_EXE=/mnt/gpu11_200T/dingxibo/miniconda3/bin/conda \
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

它会把官方 FastWAM 与私有 `D-URing/fastwam-realrobot-pipeline` overlay 到 `$FASTWAM_WORKDIR`，默认位于项目内 `upstreams/FastWAM-realrobot`，再按 CUDA 12.8 默认安装 PyTorch 与 editable FastWAM。若集群 CUDA wheel、模型目录或私有仓库 checkout 位置不同，通过以下变量覆盖：

```bash
export FASTWAM_WORKDIR="$PROJECT_ROOT/upstreams/FastWAM-realrobot"
export FASTWAM_MODEL_BASE="$PROJECT_ROOT/models"
export FASTWAM_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
```

训练入口明确禁止 CPU fallback：

```bash
FASTWAM_MODE=smoke FASTWAM_RECIPE=joint_base \
bash scripts/fastwam/run_realrobot_train_eval.sh
```

要观察 loss 是否下降，跑 pilot：

```bash
FASTWAM_MODE=pilot FASTWAM_RECIPE=joint_base \
bash scripts/fastwam/run_realrobot_train_eval.sh
```

结果会写到 `runs/experiments/custom/fastwam_realrobot_smoke/.../loss_summary.json`。FastWAM 原生 checkpoint 仍保留在 `$FASTWAM_WORKDIR/runs/<task>/<run_id>/`，本仓库只记录路径和摘要。

## 6. 配置开关

当前默认配置位于 `configs/base.yaml`：

```yaml
runtime:
  mode: mock
  launcher: local
policy:
  transport: inproc
  supports_batch: false
environment:
  num_envs: 1
  headless: true
features:
  enable_viewer: false
  enable_simulation: false
  enable_real_robot: false
```

切换环境时创建新的 `configs/runs/*.yaml` 并通过 `extends` 覆盖，不直接修改团队公共默认值。sim、real、remote policy 与 viewer 都必须显式开启；未实现的 adapter 即使配置可表达，也不代表运行能力已存在。

## 7. 密钥与本地变量

- 不把 GitHub token、模型 token、机器人凭据或集群密钥写入 YAML。
- `.env` 和 `.env.*` 已被忽略；如需共享变量格式，只提交去除秘密的 `.env.example`。
- 正式集群使用调度器 secret、容器 secret 或团队已有密钥系统。
- 日志和 artifact 不得记录 token、完整用户目录或受限数据路径。

## 8. 日常验证与故障定位

一条命令检查 Python、虚拟环境、依赖和两个 run config：

```bash
make doctor
```

完整开发验收：

```bash
make test
make validate
make dry-run
make schemas
```

常见问题：

| 现象 | 处理 |
|---|---|
| `python3.11 not found` | macOS 使用 Homebrew 安装；集群加载管理员提供的 module |
| `.venv` 不存在 | 运行 `make setup` |
| 依赖冲突 | 不在 core 环境安装 simulator/VLA；重建 `.venv` 后按 constraints 安装 |
| GitHub 登录失败 | `gh auth status`，必要时重新执行 `gh auth login` |
| 浏览器可访问但 `github.com:443` 超时 | 对比 `scutil --proxy` 与 `env | grep -i proxy`，让 shell 继承 HTTP/HTTPS proxy |
| 集群没有公网 | 使用同平台 wheelhouse，或等待 M5 提供正式容器 |
| 没有 `nvidia-smi` | core/mock 开发正常；只有 sim/policy GPU 环境才要求 NVIDIA runtime |

环境发生变化时，把 Python、依赖约束、容器镜像、driver/CUDA 和验证命令结果写入对应里程碑记录，避免“在某台机器上能跑”成为隐含条件。
