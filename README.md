# EmbodiedAI Demo Pipeline

面向家庭与生活服务场景的具身智能 Demo 工程规划仓库。

当前仓库处于规划基线阶段：先稳定任务、策略、执行、日志与评测契约，再逐步接入 mock、离线回放、NVIDIA 仿真集群和真实机器人。当前不以训练大模型、搭建复杂可视化或立即接真机为目标。

完整方案见：[`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md)。

## 当前状态

- 版本：Planning Baseline v0.1
- 日期：2026-07-13
- 已确定：contract-first、headless-first、evaluation-first、backend-switchable
- 尚未确定：首个仿真平台、首个 VLA、Viewer 技术栈、目标机器人平台、集群调度器
- 当前不包含：业务代码、模型权重、大型数据集、仿真资产和真机控制代码

## 项目边界

该目录只服务于 Demo 项目规划和后续工程落地。

论文、模型与开源生态的研究笔记仍由同级的 `EmbodiedAI-Research/` 知识库维护；本项目只引用经过筛选、能够影响工程决策的结论，不复制论文综述。
