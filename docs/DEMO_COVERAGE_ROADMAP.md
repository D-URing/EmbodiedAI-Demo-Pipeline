# Demo 覆盖路线图：从可交付证据链到家庭任务库

> 状态：v0.1<br>
> 日期：2026-07-14<br>
> 关联：[`MASTER_PLAN.md`](MASTER_PLAN.md)、[`FASTWAM_REALROBOT_INTEGRATION.md`](FASTWAM_REALROBOT_INTEGRATION.md)、[`adr/0002-fastwam-evidence-chain.md`](adr/0002-fastwam-evidence-chain.md)

## 1. 新格局

FastWAM real-robot pipeline 接入后，本项目不再只有“任务 mock demo”这一条线，而是形成三层证据链：

1. **任务与工程链路证据**：TaskSpec、mock/replay runner、logger、evaluator、report，证明 demo pipeline 可运行、可复现、可扩展。
2. **真实训练证据**：LeRobot / FastWAM CUDA 训练入口、loss summary、checkpoint 路径、训练报告，证明团队能在真实开源生态上跑训练，不是 CPU toy trainer。
3. **能力评测证据**：后续接 RoboDojo、RoboCasa、RoboTwin 或真机，证明策略在物理闭环里完成任务。

这三层不能混报。FastWAM 可以回答“有没有真实训练和 loss 下降”，但不能直接回答“厨房整理成功率是多少”。家庭任务 mock 可以回答“任务库、日志、评测和报告是否跑通”，但不能冒充模型能力。

## 2. Readiness 分级

每个 demo 任务都按以下等级标注，避免范围扩展后目标漂移。

| 等级 | 名称 | 可以证明 | 不能证明 | 当前作用 |
|---|---|---|---|---|
| R0 | Task Spec | 任务语义、物体、阶段、成功条件明确 | 运行链路 | 进入任务库前置 |
| R1 | Mock Rollout | runner/logger/evaluator/report 可复现 | 物理可行性、模型能力 | 第一阶段交付主力 |
| R2 | Training Evidence | 真实训练入口、loss、checkpoint 和日志可追溯 | 对应家庭任务成功 | 回答“是否可训练” |
| R3 | Offline Action / Replay | 数据解码、动作 shape、离线轨迹和 evaluator 对齐 | 闭环控制 | 接 LeRobot/DROID/BridgeData |
| R4 | Simulation | 仿真闭环、随机化、能力维度评测 | 真机表现 | NVIDIA 集群后重点 |
| R5 | Real Shadow | 真机观测下预测与安全审计 | 真机控制成功 | 真机前安全门 |
| R6 | Real Closed-loop | 指定平台上的真实闭环表现 | 跨平台泛化 | 长期目标 |

## 3. Demo 覆盖矩阵

| 优先级 | 任务名称 | 场景 | 目标 | 所需物体 | 机器人能力 | 难度 | 当前 readiness | 第一版可 mock | 后续适合仿真/真机 | 展示价值 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | `tabletop_sorting_v1` | 桌面/厨房台面 | 将杯子、瓶子、垃圾放到指定区域 | 杯子、瓶子、垃圾、托盘、收纳盒、垃圾桶 | 识别、分类、抓取、放置、长时序 | L2 | R1 已有 | 是 | RoboDojo `organize_table` / RoboCasa / 真机 | 最直观，适合快速汇报 |
| A | `towel_folding_v1` | 衣物整理 | 将矩形毛巾两次对折到目标形状 | 毛巾、桌面、角点/边界标注 | 双臂语义、柔性物体、对齐 | L3 | R1 已有 | 是 | RoboDojo `fold_clothes` / RoboTwin / 真机 | 有“具身味道”，展示差异化强 |
| A | `fastwam_package_sorting_v0` | WAM/package scan | 基于 FastWAM realrobot pipeline 跑真实训练、loss 和 checkpoint | FastWAM 数据、7D/10D action、checkpoint | 真实训练、动作预测、离线评测 | L3 | R2 已有入口 | 不按家庭语义 mock | FastWAM 外部 backend / 后续真实 WAM | 回答同事“loss 有没有正常下降” |
| A | `kitchen_counter_sorting_v1` | 厨房/备菜 | 将食材、餐具、调料按区域整理 | 蔬果、碗盘、调料瓶、砧板、托盘 | 分类、抓放、语言 grounding、阶段推进 | L2 | R1 已有 | 是 | RoboCasa / RoboDojo `classify_objects` | 厨房场景贴近最终愿景 |
| A | `drawer_pick_place_v1` | 抽屉取放 | 打开抽屉，取出目标物，放到桌面区域 | 抽屉、把手、目标物、桌面目标区 | 关节物体、接触、开合状态、取放 | L3 | R1 已有 | 是，状态机 | RoboCasa / RoboTwin / 真机 | 长时序和接触明显 |
| A | `laundry_sorting_v1` | 衣物整理 | 按颜色/类别把衣物放入不同篮子 | 毛巾、T 恤、袜子、衣篮 | 分类、抓取柔性物体、容器放置 | L2 | R0 建议新增 | 是 | RoboTwin / 真机 | 比完整叠衣更容易先做 |
| A | `trash_sorting_v1` | 家庭清洁 | 将桌面垃圾按可回收/厨余/其他投放 | 纸团、瓶子、果皮、垃圾桶 | 分类、抓放、避障、安全区 | L2 | R0 建议新增 | 是 | RoboDojo / RoboCasa / 真机 | 清洁服务语义明确 |
| B | `wipe_table_v1` | 家庭清洁 | 用抹布覆盖指定污渍区域并达到覆盖率 | 抹布、污渍区域、桌面 | 工具使用、接触、覆盖规划 | L3 | R0 | 可做 2D 覆盖 mock | RoboCasa / 真机 | 清洁任务观感好，但接触评测更难 |
| B | `meal_assembly_v1` | 做菜/简餐 | 按步骤把食材放入碗/盘中，形成简化沙拉/三明治 | 食材块、碗盘、夹具、砧板 | 顺序执行、分类、取放、容器操作 | L3 | R0 | 是，先用块状物 | RoboCasa / 真机 | “做菜”主题强，第一版可降维 |
| B | `find_and_deliver_v1` | 家庭递送/找物 | 在多个区域找到目标物并递送到用户指定位置 | 目标物、房间/区域拓扑、托盘 | 记忆、导航抽象、目标定位、递送 | L3-L4 | R0 | 是，拓扑图 mock | 后续移动操作/真机 | 家庭服务叙事完整 |
| B | `cabinet_pick_place_v1` | 厨房/柜门 | 打开柜门或柜格，取放杯碗 | 柜门、把手、杯碗、架子 | 关节物体、位姿约束、避碰 | L4 | R0 | 可做离散状态机 | RoboCasa / RoboTwin / 真机 | 比抽屉更接近厨房 |
| B | `tshirt_folding_v1` | 衣物整理 | 展平 T 恤并完成指定折叠 | T 恤、桌面、关键点 | 柔性物体、双臂、重抓、对齐 | L4 | R0 | 可用 polygon mock | RoboDojo / RoboTwin / 真机 | 叠衣服最终目标之一 |
| C | `pour_or_transfer_v1` | 厨房 | 将颗粒/液体从一个容器转移到另一个容器 | 杯、碗、颗粒或液体替代物 | 姿态控制、接触/流体近似、容器约束 | L4 | R0 | 可做符号 mock，但价值有限 | RoboCasa / 真机 | 做菜展示强，仿真/真机难度高 |
| C | `open_bottle_or_screw_cap_v1` | 灵巧手操作 | 拧开瓶盖或旋紧瓶盖 | 瓶子、瓶盖、夹爪/灵巧手 | 双手协调、扭矩、接触、精细操作 | L4-L5 | R0 | 可做状态机 mock | DexVerse / RoboDojo / 真机 | 灵巧操作代表性强 |
| C | `clip_or_pinching_v1` | 灵巧手操作 | 使用夹子夹住指定物体或布料 | 夹子、小物体、布料 | 多指协调、弹性件、精确接触 | L4 | R0 | 可做离散 mock | DexVerse / 真机 | 小而精，适合后续 dex demo |
| C | `tool_use_sweep_v1` | 清洁/工具 | 用刷子或刮板把碎屑扫入区域 | 刷子、碎屑、目标区域 | 工具使用、连续控制、覆盖 | L4 | R0 | 可做 2D 粒子 mock | RoboCasa / 真机 | 清洁具身感强 |
| D | `mobile_fetch_service_v1` | 多房间服务 | 根据用户指令跨区域找物、拿取、递送 | 移动底盘、机械臂、目标物、用户点位 | 导航、移动操作、记忆、人机交互 | L5 | R0 | 只能拓扑 mock | 真机优先，仿真需大工程 | 最接近最终家庭服务愿景 |

## 4. 第一阶段推荐扩展

第一阶段不要一次做十几个 runnable demo。当前已在两个 MVP 基础上补入厨房台面整理和抽屉取放两个 R1 mock demo；下一步只需再补一个衣物或清洁任务，避免范围失控。

| 推荐任务 | 为什么现在做 | 首版 mock 方法 | 升级路径 |
|---|---|---|---|
| `kitchen_counter_sorting_v1` | 与 `tabletop_sorting_v1` 共享大量 runner/evaluator，语义上扩到做菜/厨房 | 已落地：2D 桌面区域 + 多类物体 + 语言规则 | RoboCasa 厨房资产，后续 LeRobot/BridgeData 回放 |
| `drawer_pick_place_v1` | 增加关节物体和接触状态，扩展 demo 的结构复杂度 | 已落地：抽屉 `closed/open` 状态机 + handle action + 取放谓词 | RoboCasa/RoboTwin 仿真，再接真机 shadow |
| `laundry_sorting_v1` 或 `trash_sorting_v1` | 快速扩到衣物/清洁，不需要立刻解决完整柔性物理 | 分类归位 + 轻量形变/类别属性 | RoboTwin/DROID/真机小样本 |

FastWAM 继续作为 R2 训练证据链，不建议强行把它包装成“厨房任务”。它应该用于交付真实 loss 曲线、checkpoint 与外部训练 pipeline 复现能力。

## 5. 开源资源映射

| 资源 | 数据参考 | 模型参考 | 仿真/评测参考 | 任务设计参考 | 后续代码接入参考 | 当前优先级 |
|---|---|---|---|---|---|---|
| LeRobot | 轨迹格式、PushT/ALOHA 样例 | ACT、Diffusion Policy 等轻量训练 | 基础训练 smoke | 数据/模型最小闭环 | converter、训练脚本、policy adapter | P1 |
| OpenPI / π 系列 | 大规模机器人数据接口参考 | VLA policy runtime | 间接 | 泛化任务 | 独立 policy server | P3 |
| GR00T | 数据/embodiment 思路 | 通用机器人 foundation policy | NVIDIA 生态相关 | 多机器人愿景 | 后续独立 runtime | P3 |
| FastWAM | WAM realrobot 数据与动作 | 已作为真实训练后端 | 离线 probe / checkpoint evidence | 7D/10D 动作与真实链路 | 当前已接入外部 backend | P0 |
| DiT4DiT | 训练范式参考 | diffusion/transformer policy | 间接 | 长时序策略参考 | 后续 adapter | P3 |
| InternVLA-A | 多模态 VLA | VLA adapter 目标 | 间接 | 语言条件任务 | 独立推理服务 | P3 |
| LingBot-VLA | VLA/语言动作 | VLA adapter 目标 | 间接 | 家庭服务指令 | 独立推理服务 | P3 |
| DexVerse | 灵巧手数据/任务 | dex policy 参考 | dex 仿真/评测 | 旋盖、夹取、小物体 | dex adapter | P3 |
| RoboDojo | 任务与评测维度 | policy/env 解耦参考 | 外部仿真评测目标 | `fold_clothes`、`organize_table`、`classify_objects` | NVIDIA/Isaac smoke | P2 |
| RoboTwin | 双臂任务与数据 | 双臂 policy 参考 | 双臂仿真 | 叠衣、柜门、协同操作 | sim adapter | P2-P3 |
| RoboCasa | 厨房/家庭资产 | 间接 | 厨房仿真 | 台面、柜门、抽屉、做菜 | sim adapter | P2 |
| SIMPLER | 真实策略仿真评测 | sim-real 评测 | 仿真评估参考 | 抓放/移动操作 | evaluator 对齐参考 | P3 |
| DROID | 多场景真实轨迹 | imitation 数据 | replay/offline | 桌面/家庭操作 | dataset replay | P2 |
| Open-X-Embodiment | 跨 embodiment 数据 | 大规模训练参考 | 数据标准 | 泛化边界 | RLDS converter | P3 |
| BridgeData | 多环境抓放 | imitation 数据 | replay/offline | 桌面抓放 | dataset replay | P2 |
| AgiBotWorld | 双臂、长时序、家庭数据 | imitation/VLA 数据 | replay/offline | 衣物/整理/服务 | dataset converter | P3 |
| Ego4D | 人类第一视角视频 | 不直接作机器人 action | 间接 | 任务分解、长时序语义 | instruction/step mining | P3 |
| EPIC-KITCHENS | 厨房人类动作 | 不直接作机器人 action | 间接 | 做菜步骤和物体语义 | task authoring reference | P3 |
| EgoSteer | 第一视角/导航线索 | 视觉导航参考 | 间接 | 找物/递送 | later navigation adapter | P4 |
| CHORD / contact wrench | 接触/力信息 | contact-aware policy | 接触评测 | 擦拭、拧盖、插拔 | force/action schema | P4 |
| K0 叠衣服 | 衣物技能参考 | 折叠策略参考 | 间接 | towel/T-shirt folding | task/evaluator inspiration | P2-P3 |

## 6. 下一步落地顺序

1. **保持 LeRobot-first 主线**：先实现 dataset read、train/load、offline inference 和 report。
2. **保留 custom backend**：FastWAM 私有 overlay 继续作为自建模型/真机数据扩展路径。
3. **暂缓继续堆 household mock task**：`kitchen_counter_sorting_v1` 与 `drawer_pick_place_v1` 已进入 R1；`laundry_sorting_v1` 或 `trash_sorting_v1` 可等 LeRobot 主线跑通后再补。
4. **抽象通用 mock primitives**：object-in-region、container、drawer state、category routing、stage predicates。
5. **推进 replay/offline action**：优先接 LeRobot dataset/inference，再考虑 BridgeData/DROID。
6. **NVIDIA 集群后推进 R4**：RoboDojo/RoboCasa/RoboTwin 三选一，不同时硬上三个。

## 7. 当前不做

- 不把 FastWAM loss 下降说成家庭任务成功率。
- 不把 mock success rate 放到模型排行榜。
- 不为了覆盖更多场景而提前接复杂 Viewer。
- 不在 core `.venv` 里安装 CUDA、Isaac、VLA 或真机 SDK。
- 不把所有开源项目同时作为硬依赖；每个外部生态必须通过 adapter、脚本或独立环境接入。
