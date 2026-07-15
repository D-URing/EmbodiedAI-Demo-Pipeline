# Project Structure

项目按“两个 pipeline + 轻量 evidence core + 全局资产池”理解。

```text
.
├── pipelines/
│   ├── lerobot/          # LeRobot data→train/load→infer
│   └── custom/           # FastWAM / ImageWAM / future custom backends
├── experiments/          # 训练/推理启动入口
│   ├── lerobot/
│   └── custom/
├── configs/
│   ├── lerobot/          # LeRobot train/infer defaults
│   ├── fastwam/          # FastWAM defaults
│   └── imagewam/         # ImageWAM defaults
├── scripts/
│   ├── lerobot/          # LeRobot download/train/infer/report
│   ├── fastwam/          # FastWAM overlay/train/report
│   ├── imagewam/         # ImageWAM upstream/download/train wrapper
│   └── reference/
├── src/embodied_demo/    # schema、CLI、evidence report
├── demo_chains/          # report/evidence chain definition
├── docs/
└── references/
```

## Asset pool

Pipeline 不拥有数据或权重，只引用根目录资产池：

```text
data/
models/
checkpoints/
runs/
artifacts/
upstreams/
hf_cache/
```

详细规则见 [`STORAGE_AND_ARTIFACTS.md`](STORAGE_AND_ARTIFACTS.md)。

## LeRobot pipeline

入口：[`../pipelines/lerobot/README.md`](../pipelines/lerobot/README.md)

```text
LeRobot dataset
  -> official policy training/loading
  -> offline inference
  -> evidence/report
```

对应文件：

```text
configs/lerobot/
scripts/lerobot/
experiments/lerobot/
runs/experiments/lerobot/
data/lerobot/
models/lerobot/
```

## Custom WAM pipeline

入口：[`../pipelines/custom/README.md`](../pipelines/custom/README.md)

```text
custom model / custom backend
  -> backend-specific release / training wrapper
  -> train/eval evidence
  -> report
```

当前后端：

```text
configs/fastwam/
scripts/fastwam/
experiments/custom/fastwam_realrobot_single8_random/
experiments/custom/fastwam_realrobot_8node_random/
data/custom/fastwam/
models/custom/fastwam/

configs/imagewam/
scripts/imagewam/
experiments/custom/imagewam_flux2_4b_libero_pilot/
models/custom/imagewam/
```

## Experiments are launchers

训练和推理从 `experiments/` 启动，不从 Makefile 启动：

```text
experiments/<route>/<experiment>/
├── README.md
├── config.sh
├── launch.sh
└── slurm.sbatch
```

`configs/` 放底层默认参数，`scripts/` 放可复用执行器，`experiments/` 是多次试验时复制、改名、保存配置的地方。

## Core is lightweight

`src/embodied_demo/` 只包含：

- public schemas；
- `embodied-demo` CLI；
- FastWAM evidence report adapter；
- YAML helper。

它不安装 CUDA、LeRobot、FastWAM、Isaac 或真机 SDK。集群上对应 `embodied-core` conda env。
