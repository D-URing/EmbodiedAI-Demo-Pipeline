# Model Asset Pool

`models/` 是仓库级模型/权重资产池，不属于某一条 pipeline 私有。

各 pipeline 只引用这里的模型：

```text
models/
├── lerobot/
│   ├── act/
│   ├── diffusion/
│   └── fastwam/
├── fastwam_release/
└── custom/
```

规则：

- 权重、checkpoint、预训练 policy 不提交 Git；
- 只提交这个 README 说明目录职责；
- 开源预训练模型放在 `models/<ecosystem>/<policy_or_model>/<name>/`；
- release 权重可放在 `models/<project>_release/`；
- 本地训练后值得保留的模型可以从 `runs/` 复制到 `models/custom/` 或对应模型目录；
- pipeline 文档应写清楚它依赖哪个模型目录。

当前 SCUT 已准备：

```text
models/fastwam_release/libero_uncond_2cam224.pt
models/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

LeRobot policy 建议补充：

```text
models/lerobot/diffusion/diffusion_pusht/
```

下载命令：

```bash
make download-lerobot-diffusion-pusht-policy
```
