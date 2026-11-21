# OMOMO 抱箱：G1 与 R1 的 SuperMimic 基线

本流程使用 Isaac PhysX 和 `sub3_largebox_003`，从头训练纯机器人动作模仿策略。
箱子为有质量、受重力和碰撞影响的自由刚体，只在回合重置时设置初始姿态。
训练没有物体轨迹、接触、抓取、抬升或交互奖励；沿用身体和根部跟踪、动作变化与关节限位项。
策略输入是本体观测历史和未来机器人参考动作。世界模型只预测机器人状态，箱体状态未加入模型。
因此接触动力学对世界模型是部分可观测的；本基线的失败不能证明纯动作模仿永远无法抱箱。

## 代码入口与本地产物

仓库包含环境构建、转换、接触约束处理、训练、评测、导出、无动作部署检查和测试代码。
本文记录的 `data/`、`logs/`、模型包、视频、隔离依赖及冻结实验目录均为本地产物，
不随代码分发；复现时需自行准备相应数据和 R1 资源。

正式代码中的 `run_supermimic_r1_trajectory` 默认选择 BF16 导出，
恰好 10k 的检查点保存逻辑及完整源码归档已纳入训练入口。
需等待已有 Omni 队列再运行固定接触 032 时，可使用：

```bash
"$Train_CONDA_PREFIX/bin/python" -m holomotion.scripts.training.queue_r1_after_omni \
  --recipe /path/to/local-input-recipe.json --output /path/to/new-output --dry-run
# 核对 dry-run 后，移除 --dry-run 启动等待队列。
```

recipe 使用队列入口读取的 `manifest`、`robot_urdf`、`input_sha256`、
`contact_reference_audit`、`wait_for_batch`、`wait_for_service` 字段；
固定接触数据的 `validation.json` 与处理源码散列必须匹配。
本轮已启动的冻结队列保持原样，不因整理正式代码而替换运行源码。

## 已排队：032 B4 固定接触锚点重训（11 月 22 日）

用户要求在保留接触点的前提下重训 032，并明确选择等待现有 Omni 三条完成后再运行。
新任务独立从头训练 20k actor 更新，10k 自动评测、生成参考/策略对比视频；
仍使用官方落地 Stance 起步、左右 24 cm 训练箱和左右 30.5 cm 评测箱，
前后 24.5 cm、上下 27 cm、质量 0.1 kg 不变，没有增加交互奖励或物体观测。

新入口为 `holomotion.scripts.training.prepare_r1_fixed_contacts`。它读取原 B4 的
`inputs/fixed_palm_targets.npz`，恢复其声明的箱面锚点，并精确保留原始箱体局部
Y/Z 坐标、接触面、法向和有效接触时间。只随新宽度移动 X 接触面，
不按最近碰撞点重新选择锚点、不将双手改到箱面中心。
源 NPZ 中沿法向的 10 mm 虚拟预压先还原成其声明的表面锚点；本轮不额外增加预压。

为同时满足接触和运动学约束，允许骨盆平移及腿、腰、双臂调整，
同时用硬约束保持原始两脚完整位姿。骨盆最大平移 2.388 cm，
根部朝向、头关节、主任务箱体 XY/朝向保持原始轨迹；箱心高度按新尺寸保持箱底调整。
官方站姿首帧精确保留，贴脚间隙约为左 5.00、右 6.85 mm。
轨迹时间轴不变，完整参考仍为 377 帧、7.52 秒。

335 个有效手接触帧通过实际 float32 HDF5 reader 和训练 FK 复核：
最大接触点误差 3.53e-7 m，最大法向误差 0.0343°；
源脚部位姿在双精度 FK 中保持，最大位置误差约 4.74e-13 m。
最大直接关节速度为限速的 48.21%，没有关节越限或非手部穿箱。
24 项相关测试与 Ruff 检查通过；脚部和接触 Jacobian 另用独立有限差分检验。
这些误差只证明离线参考的约束传递，不能代表仿真策略或实机维持了同样接触。

- 选定数据：`data/omomo-r1/r1_b4_032_fixed_contacts_train240_eval305_20261122/prepared_final/`。
  `fixed_contacts_50hz.npz` 保存接触定义，`validation.json` 保存接触、脚部、限位及速度审计。
  `reference_preview.mp4` 仅为运动学参考；`prepared_v*` 是排查中间结果，不用于训练。
- 新输出：`logs/omomo-r1-b4-032-fixed-contacts-20261122/`。
  根 `status.json` 在等待期间显示 `waiting_for_existing_omni_batch`；
  开训后实际训练和 10k 视频在 `sub10_largebox_032/r1/` 下。
- 冻结源码：`not_for_commit/omomo-r1-b4-032-fixed-contacts-runtime-20261122/`，
  包含队列入口 `queue_r1_fixed032_20261122.py`、冻结 `input_recipe.json` 和源码散列。
  物理与学习核心沿用 Omni/B4 对照，只新增这次数据处理器与队列/输入归档。
- 服务：`holomotion-r1-b4-032-fixed-contacts-20261122.service`。
  它等待现有 Omni 三条全部结束，不中断当前任务；随后先执行 Isaac 物理预检，
  预检完成后才训练，并在最终评测后执行 BF16 ONNX 对照。

本轮尚未执行训练、完整片段物理评测或实机动作，不能预先声称消除了箱体倾斜。

## 11 月 22 日接触约束传递审计

当前 B4 与 Omni 批次都经过 `prepare_r1_stance_grip.close_arms` 后处理，不能将其称为
完整保留原始重定向接触锚点的轨迹。原 B4 包声明固定接触锚点，但此后处理没有读取
`fixed_palm_targets` 中的接触点与法向；它约束的是碰撞距离、腕部位置和软腕部姿态。
因此，物理策略回放中重建出的接触位置不能用于证明原始 B4 重定向的接触点有误。

032 已逐数组核验：主任务箱体 XY、箱体四元数、机器人根部 qpos 均与转换后的源轨迹
完全相同；贴脚初始化通过改变新增官方站姿的根部位置和朝向实现，没有单独平移主任务
箱体 XY。箱体局部 XYZ 从源包的 27×30.5×24.5 cm 改为训练的 24×24.5×27 cm，
不仅改变了左右宽度；为保持最低点，箱心初始抬高 1.25 cm，任务中 Z 调整约为
0.197–1.25 cm。评测再将左右宽度设为 30.5 cm。

物理箱体也没有锁在参考上：032 B4 原生评测在 2.5 s 时水平偏离参考约 1.06 cm，
3.8 s 时约 9.33 cm。原始接触锚点、后处理后的机器人/箱体几何与实际物理接触应分别
核验；目前尚未确定翻转的主因，不能直接认定原始接触点错误或夹持过紧。
数据处理审计保存在 B4 训练根目录
`evaluation_review_20261122/032-data-processing-audit.json`。

## 当前 R1：同三个 largebox 任务的 Omni 对照训练（11 月 22 日）

本批输入为 `data/omomo-r1/r1_largebox_batch_20261120/` 下
`sub10_largebox_032/omni`、`sub10_largebox_087/omni`、`sub10_largebox_089/omni`。
用户要求用 Omni 版本比较当前 B4 姿态，每条分别从头训练 **20k actor 更新**，
**10k 自动评测并生成参考/策略对比视频**，按 032 → 087 → 089 顺序使用单张 RTX 3090。
不从 B4 权重微调，以保持与上一批从头训练的对照条件一致。

沿用官方落地 Stance 实测首帧、0.5 秒站立、2 秒过渡及 0.5 秒末帧保持。
保留 Omni 主任务的根部、腿、腰、头，只调整双臂适配窄箱。
训练尺寸为左右 24、前后 24.5、上下 27 cm，评测为左右 30.5、前后 24.5、上下 27 cm，
质量均为 0.1 kg。仍是纯机器人动作模仿，没有物体观测或交互奖励。
实际训练 reader 独立复核脚尖与实物尺寸箱体间隙为左 5.00 mm、右 6.85 mm；
切换评测尺寸后，机器人未来参考特征逐元素不变。

三条源场景与实际训练 URDF 的运动学位置差均为 0，已通过命名关节、时间轴、
轴向、关节限位及速度检查。完整参考分别为 377、432、412 帧，最大直接关节速度为
各关节限速的 63.79%、69.73%、98.28%；这些是静态检查，不能据此推断抬箱效果。
19 项相关测试通过；参考预览仅为运动学回放。

- 数据：`data/omomo-r1/r1_omni_batch_train240_eval305_nearfeet_20261122/`。
  每条 `<task>/prepared_final/` 包含实际训练 HDF5、manifest、审计与 `reference_preview.mp4`。
- 训练输出：`logs/omomo-r1-omni-batch-train240-eval305-20261122/`。
  根 `status.json` 记录队列，每条 `<task>/r1/training/progress.json` 记录实际更新数。
  10k 视频输出为 `<task>/r1/preview_actor_010000/reference-vs-policy.mp4`。
- 冻结运行源码：`not_for_commit/omomo-r1-omni-batch-train240-eval305-runtime-20261122/`。
  `experiment.patch` 记录相对主代码的精确 10k 保存、源码归档和 BF16 导出入口改动，
  学习与物理实现沿用 B4 对照；原 B4 运行目录保持不变。
- 输入指纹与命令：`not_for_commit/omomo-r1-omni-batch-20261122-recipe.json`、
  `not_for_commit/omomo-r1-omni-batch-20261122-plan.json`。
- 后台服务：`holomotion-r1-omni-batch-train240-eval305-20261122.service`。
  本地时间 11 月 22 日 01:28 启动，032 的首次 Isaac 物理预检已产出完成报告，
  实际训练箱体尺寸、质量、PD、关节限位和几何审计符合配置。

训练使用 BF16，最终自动导出和物理对照也选择 CUDA BF16 ONNX，并保留严格数值检查。
尚未完成本批 10k/20k 评测，不预先声称 Omni 姿态或抱箱效果更好。
部署观测处理的端到端一致性仍需单独验证，本批不发送实机动作。
SDK 的有界关闭及其他既有限制同下文。

## B4 对照：三条独立训练，24 cm 窄箱与贴脚初始摆放

本批仅选择 `data/omomo-r1/r1_largebox_batch_20261120/` 下的
`sub10_largebox_032/b4`、`sub10_largebox_087/b4`、`sub10_largebox_089/b4`。
按用户确认分别从头训练三个策略，各 20k actor 更新，恰好 10k 保存检查点并评测、出视频。
单张 RTX 3090 按 032 → 087 → 089 排队运行。

方向顺序固定为左右、前后、上下：参考与训练为 **24×24.5×27 cm**，原生评测、
10k 预览、ONNX 评测和导出为 **30.5×24.5×27 cm**，质量均为 0.1 kg。
24 cm 是本批显式参数，上一条已完成实验的 24.45 cm 配置保持原样。

保留各 b4 主任务的根部、腿、腰、头关节数据，只对双臂做窄箱 IK；根据腕部实际
所在箱面选择闭合方向，不能假定所有轨迹的左手都在箱体局部 +X 面。
首帧使用官方落地 Stance 实测关节角，站姿面向箱体，并以实际脚部碰撞网格设置
贴脚位置。实机尺寸箱体距左右脚尖分别约 **5.00 mm、6.85 mm**；这个间隙由
网格支撑平面和箱面上的见证点计算，另以训练实际 reader 和原生 CCD 复核。
这次没有将上一条轨迹的世界坐标位移直接套到三个新片段。

数据位于 `data/omomo-r1/r1_b4_batch_train240_eval305_nearfeet_20261120/`，
每条的最终输入都在 `<task>/prepared_final/`：`reference_50hz.npz`、
`dataset/manifest.json`、`dataset/motions.h5`、`validation.json`，
`reference_preview.mp4` 仅为运动学参考。最终片段分别为 377、432、412 帧。
三个片段的最大直接关节速度分别为限速的 41.03%、39.69%、39.04%，没有关节越限。
实际 reader 切换至 30.5 cm 箱体后，机器人未来目标与初始箱体 XY 保持一致。
仍是纯机器人动作模仿，没有新增箱体、抓取、接触或抬升奖励。

训练输出为 `logs/omomo-r1-b4-batch-train240-eval305-20261120-v2/`，
服务为 `holomotion-r1-b4-batch-train240-eval305-20261120-v2.service`，
冻结源码位于 `not_for_commit/omomo-r1-b4-batch-train240-eval305-runtime-20261120-v2/`。
根目录 `status.json` 记录队列，各 `<task>/status.json`、
`<task>/r1/training/progress.json` 记录训练状态和 actor 更新数。
10k 视频位于 `<task>/r1/preview_actor_010000/reference-vs-policy.mp4`。

早期摆放使用旧 CCD 距离查询，网格复核发现实际间隙不符后已停止并标记为 superseded；
那些 `prepared_v*` 和首个无 `-v2` 后缀的批运行仅保留排查记录，不作为当前结果。
修正后的三条数据均通过静态检查；29 项相关测试通过。物理预检与训练效果分别记录，
参考视频和预检不代表已学会抱箱。未执行实机控制。

### 20k 结果与 11 月 22 日导出补验

三个策略均已完成 **20,000 actor 更新**，原生评测全部跑完整段且机器人未跌倒。
以下持续时间要求箱底连续超过地面 5 cm；评测物理箱体均为 30.5×24.5×27 cm、0.1 kg。

| 轨迹 | 原生箱底最高离地 | 原生持续离地 | 原生最大箱体倾角 | ONNX 箱底最高 / 持续离地 |
| --- | ---: | ---: | ---: | ---: |
| 032 | 20.23 cm | 1.92 s | 66.41° | 21.75 cm / 1.92 s |
| 087 | 56.92 cm | 3.18 s | 55.13° | 0.81 cm / 0 s |
| 089 | 57.54 cm | 3.12 s | 100.58° | 0.0012 cm / 0 s |

原生 087 相对较好但仍有明显倾斜；089 抱起后翻到侧面，结束倾角约 90°。
因此抬升判据通过不代表夹持稳定。032、089 的 10k 版本没有抱起；087 的 10k
版本虽然持续离地 3.44 秒，也有约 101° 的侧翻，不能仅按离地时长选择它。

原批处理显示 failed 的直接原因发生在导出评测：部署加载器仍硬编码旧训练宽度
24.45 cm，拒绝了本批实际的 24 cm。现已改为交叉核验导出清单、原生评测的尺寸与
来源检查点，同时保留实物尺寸和坐标轴校验。新增兼容/拒绝用例与相关测试共 17 项通过。
原冻结源码、训练检查点、ONNX 文件和参考数组均保持原样；修复只放在主代码及
`not_for_commit/omomo-r1-b4-onnx-recovery-runtime-20261122/`，有独立修改与文件散列记录。

补验后只有 **032 的 ONNX 仿真通过抬升判据**；087、089 的部署仿真没有抱起。
三条的 30 帧无动作推理均通过，但后两条 `exported_simulation_passed=false`，
不具备该运行时要求的实机激活条件，未执行实机控制。

精度差异已定位：原生评测使用 BF16，导出使用 FP32。087 在相同物理观测历史上，
BF16 输出与保存动作完全一致，BF16 与 ONNX 最大动作差为 0.03162575，
FP32 与 ONNX 最大差仅为 0.000005722。087 原生复跑的动作、机器人状态、箱体状态
均与第一次逐帧相同。现有 FP32 导出数值检查不能证明其接触行为复现 BF16 原生策略；
部署一致性仍需解决。

结果与证据位于训练根目录 `evaluation_review_20261122/summary.json`；
每条原生视频为 `<task>/r1/reference-vs-policy.mp4`，左侧窄箱参考、右侧实物尺寸物理结果。
ONNX 补验位于 `<task>/r1/onnx-evaluation-recovered-20261122/`；对应物理视频位于
`evaluation_review_20261122/onnx_<编号>/physical-rollout.mp4`。
原批处理错误记录保留，根 `status.json` 增加补验结果，未把部分部署失败改成全流程成功。

### 11 月 22 日 BF16 ONNX 对照

沿用上述三个 20k 检查点，不重新训练、不改变权重或物理箱体摆放。
新导出保留 FP32 归一化与 tanh，编码器和解码器使用真正的 CUDA BF16 GEMM。
ELU 用 FP32 运算并恢复 BF16 边界；本机穷举全部 65,280 个有限 BF16 输入，
该激活实现与 Torch CUDA BF16 ELU 没有输出差异。

| 轨迹 | 原生 BF16 连续离地 | FP32 ONNX 连续离地 | BF16 ONNX 连续离地 | BF16 ONNX 箱底最高 | BF16 ONNX 箱体表现 |
| --- | ---: | ---: | ---: | ---: | --- |
| 032 | 1.92 s | 1.92 s | 1.90 s | 20.83 cm | 放下后侧翻，结束约 90° |
| 087 | 3.18 s | 0 s | 3.20 s | 56.91 cm | 最大倾角约 55.08°，结束约 9.35° |
| 089 | 3.12 s | 0 s | 3.14 s | 61.47 cm | 搬运中翻至侧面，结束约 90° |

三条均跑完整段，机器人没有跌倒。BF16 恢复了 087、089 的抬箱效果，但不能据此
称为稳定夹持。它们是单次、无随机化的 Isaac 物理结果，未执行实机控制。

数值检查以保存的物理状态重建观测历史，给原生 CUDA BF16 与 ONNX 输入相同的
部署包参考特征；032/087/089 分别有 372/376、429/431、409/411 帧完全一致，
最大动作差分别为 0.00390625、0.00048828125、0.001953125，均超过原有 2e-4 门槛。
重建观测与原始运行并非逐帧位级相同：原生重建输出对保存动作的最大差分别为
0.0078125、0、0.0625，不能把这项检查称为原始输入完整回放。

本次 ONNX 物理对照继续使用原生训练侧观测和 GPU 生成的参考特征；部署入口则使用
NumPy 观测和包内缓存参考。两套参考通过浮点近似检查，尚未证明端到端部署行为一致。
三个包均保留 `numerical_precision_passed=false`，实机启动入口会在导入 SDK 前拒绝激活。
30 次无动作推理各自完成，不初始化硬件、不发送电机命令，也不代表目标机实时性通过。

本机 CUDA 12 运行时为隔离安装的 `onnxruntime-gpu==1.23.2`：
`not_for_commit/onnxruntime_gpu_1_23_2_verified/`；原训练覆盖层的 CPU ORT 保持不变。
运行需以该目录作为 `PYTHONPATH`，同时使用 `.venv-omomo` 解释器和可见的 BF16 CUDA GPU。
加载器核对实际执行 provider，拒绝未安装 CUDA 或初始化退回 CPU 的情况。
重新准备该可选运行时时，可安装到独立目录，不覆盖训练环境中的 CPU ORT：

```bash
ORT_EXTRA="$PWD/not_for_commit/onnxruntime-gpu-cu12"
"$Train_CONDA_PREFIX/bin/python" -m pip install --no-deps \
  --target "$ORT_EXTRA" 'onnxruntime-gpu==1.23.2'
export PYTHONPATH="$ORT_EXTRA${PYTHONPATH:+:$PYTHONPATH}"
```

源码入口为 `holomotion.scripts.training.export_supermimic_bf16`，参数为
`--training <训练目录> --output <新包目录>`；导出必须在与检查点一致的源码环境中运行。
本次冻结运行目录为 `not_for_commit/omomo-r1-bf16-cuda-runtime-20261122/`，
每条产物在 `<task>/r1/bundle-bf16-cuda-20261122/`，物理评测在
`<task>/r1/onnx-evaluation-bf16-cuda-20261122/`。
汇总为 `bf16_trial_20261122/summary.json`，实际物理状态视频为
`bf16_trial_20261122/bf16_<编号>/physical-rollout.mp4`。
原生检查点及原 FP32 包保持原样；CPU 上用 FP32 算子加 BF16 舍入的早期原型不作为成功结果。

## 已完成：sub3 窄箱训练与实机尺寸评测（训练宽度 24.45 cm）

以下方向均相对机器人，顺序固定为 **左右、前后、上下**：

| 阶段 | 左右 | 前后 | 高度 |
| --- | ---: | ---: | ---: |
| 重定向参考、训练碰撞箱 | 24.45 cm | 24.5 cm | 27 cm |
| 10k 预览、最终评测、ONNX 评测、实机配置 | 30.5 cm | 24.5 cm | 27 cm |

两套配置的质量均为 0.1 kg，只在左右宽度上不同。不是评测时把箱子立高，也不是
单独把世界 Z 方向转 90°。参考与训练用窄箱，评测/部署保持窄箱训练出来的机器人动作，
用实际较宽箱体检验夹持效果；不在评测或部署时把手臂轨迹重新张开。

唯一尺寸来源为 `holomotion/src/env/reference_tracking/r1_box_contract.py`，
manifest 明确记录 `box_axis_order=[left_right, front_back, up]`、训练 `box` 和
`evaluation_box`。`use_evaluation_box` 在原生、预览、ONNX 评测和导出时选择实机尺寸。
它不改变机器人未来目标、箱体初始 XY 或朝向；本片段两套箱体初始高度相同，中心 Z 也不变。
导出包同时记录 `training_box` 和实机 `box`，无动作部署检查输出这两套尺寸。

本次产物：

- 数据：`data/omomo-r1/r1_original_b4_train2445_eval305_20261120/`，其中
  `reference_50hz.npz`、`dataset/motions.h5` 为窄箱双臂 IK 与官方站姿过渡后的数据。
- 参考回放：该目录的 `reference_preview.mp4`，画面明确标出
  `LR=24.45 FB=24.5 H=27cm`，不是训练后策略结果。
- 训练：`logs/omomo-r1-train2445-eval305-20261120/`，由
  `holomotion-r1-train2445-eval305-20261120.service` 管理。
- 冻结源码：`not_for_commit/omomo-r1-train2445-eval305-runtime-20261120/`。

从头训练，目标 20k 策略更新，恰好 10k 自动出评测和视频。
这次所有对外评测使用 30.5 cm 的实机左右宽度，训练物理箱确实使用 24.45 cm。
参考/策略对比视频左侧画 24.45 cm 的窄箱参考，右侧画实际模拟的 30.5 cm 箱体。
27 项相关测试、实际训练 reader 与未来目标不变检查通过；Isaac 预检测得训练箱
尺寸为 `[0.2445, 0.245, 0.270] m`。20k 训练现已完成：原位实机尺寸箱体的原生
Isaac 评测箱底最高离地 49.10 cm、持续超过 5 cm 达 3.70 秒；ONNX 评测持续
3.68 秒，均完成 9.5 秒片段且机器人未跌倒。`r1/bundle/manifest.json` 的原生与
导出仿真标志均已通过；这仍不是实机测试结果。

沿用官方落地 Stance 实测首帧、0.5 秒站立、2 秒过渡、6.5 秒任务与 0.5 秒末帧保持，
共 476 帧。只重新计算手臂 IK，并按新箱体高度保持原箱底轨迹；主任务的根部、腿、腰、
头不改。数据检查没有关节越限、过渡段穿地或箱体重叠，最大直接关节速度为限速的 69.6%。
头部已纳入模仿跟踪，仍没有物体、接触或抬升奖励。

此前所有错误摆放训练、独立旋转/立高评测均已停止；旧文件和检查点保留。
旧 `omomo-r1-stance-grip-20261120` 最后报告 6,814 更新，保留 6,555 更新检查点，
不会作为当前训练继续。以下旧实验章节仅供追溯。

### 10k 检查点：向机器人移近箱体的评测

使用同一个 `checkpoint_actor_010000.pt`，仅在回合重置时，将实机尺寸箱体的
中心沿机器人初始朝向的反方向平移。机器人轨迹、策略、箱体朝向、高度、质量及尺寸不变，
没有在过程中推动或回放箱体。这是额外的摆放对比，不覆盖原始评测或导出验收结果。

| 箱体初始摆放 | 箱底最高离地 | 连续离地超过 5 cm | 全程最大倾角 | 7 秒前最大倾角 |
| --- | ---: | ---: | ---: | ---: |
| 原位 | 52.87 cm | 3.54 s | 90.56° | 17.96° |
| 向机器人移近 1 cm | 50.89 cm | 3.54 s | 33.82° | 33.82° |
| 向机器人移近 2 cm | 49.95 cm | 3.70 s | 111.75° | 31.16° |

三次都完成 475 步、9.5 秒，机器人没有跌倒。原位约 8.68 秒时箱子侧翻。
移近 1 cm 的片段没有侧翻，但搬运中倾斜增大，结束时仍有约 24.52° 倾角，
不能据此称为水平稳定夹持。移近 2 cm 的侧翻反而更明显，因此后续附加评测先用 1 cm。
这里都是单次、无随机化的 Isaac 物理结果，尚未进行实机测试。

视频左右均是保存的物理状态，左侧原位、右侧移近，使用共同相机和相同时间帧：

- `logs/omomo-r1-train2445-eval305-20261120/r1/preview_actor_010000_toward_robot_01cm/placement-comparison.mp4`
- 同目录的 `placement-comparison.json` 记录指标和初始中心坐标；`.png` 为关键帧。
- 2 cm 对照保存在相邻的 `preview_actor_010000_toward_robot_02cm/`。

入口为 `holomotion.scripts.evaluation.evaluate_r1_box_offset`，参数
`--checkpoint`、`--output`、`--toward-robot-m 0.01`。旧检查点必须在对应冻结源码目录
下运行；入口额外核对原训练源码、输入和检查点散列，并记录本次摆放入口的源码散列。
`compare_r1_box_placements` 在生成对比前验证相同检查点、实物尺寸、完整帧和 trace 散列。
两次评测均成功产出完整结果；SDK 关闭仍超时，由原监督器有界清理，详见
`evaluation/isaac_process.json`，不能算作 SDK 正常退出。

20k 原流程及 `holomotion-r1-box-offset-after-20k-20261120.service` 现均已完成。
同一 20k 检查点的 1 cm 位置对比保存在 `r1/eval_actor_020000_toward_robot_1cm/`，
箱底最高离地 49.17 cm，持续超过 5 cm 达 3.68 秒，最大倾角 43.77°，结束倾角
9.57°；完整片段无机器人跌倒、无箱体侧翻。状态记录在训练根目录的
`box_offset_after_training.json`。这是新增评测，不会修改训练或机器人部署包。

### 20k 检查点：箱体移至脚尖附近

按后续要求进一步向机器人移动箱体，用官方站姿首帧的实际脚部碰撞网格和实物尺寸
箱体计算间隙：相对原位共向机器人移动 **8.01816 cm**，即比 1 cm 对照再近
7.01816 cm。只做水平平移，尺寸、质量、方向、高度以及机器人参考动作不变。
最近的左脚/箱体表面间隙为 **5.00 mm**，右脚为 **16.64 mm**；初始无机器人与
箱体碰撞或穿透。两脚间隙不同来自原有站姿及箱体朝向，未额外转箱或移动机器人。

几何计划位于 `r1/near_feet_geometry/placement_plan.json`，同目录
`initial-placement.png` 为初始几何和侧面近照。对应完整物理评测位于
`r1/eval_actor_020000_near_feet_5mm/`：475 步全部完成，机器人未跌倒，箱底最高
离地 **49.36 cm**、持续超过 5 cm 达 **3.64 秒**；最大倾角 **49.71°**，结束
倾角 **9.41°**，片段中没有侧翻。靠近脚尖并没有消除搬运时的倾斜。
另外，初始贴脚不等于抓取前位置保持不动：从 0.02 秒至 2.5 秒的站姿到下蹲过渡中，
箱体又向前移动 4.21 cm、横向移动 1.58 cm，水平总位移约 4.50 cm；这来自记录的
实际物理状态，箱体没有被固定在新初始位置。

`placement-comparison.mp4` 左侧为同一 20k 策略的原位物理结果，右侧为脚尖附近
的物理结果。该位置是独立评测配置，不覆盖原位和 1 cm 对照，也不修改部署包验收标志。

## 历史实验：实测官方 Stance 起步与双臂内收（错误箱体摆放，已停止）

当前数据目录为 `data/omomo-r1/r1_original_b4_measured_stance_grip_20261120/`，
训练服务为 `holomotion-r1-stance-grip-20261120.service`，输出目录为
`logs/omomo-r1-stance-grip-20261120/`。从头训练，目标 20k，恰好 10k 自动评测与出视频。
旧 original b4 任务已在报告 14,934 更新时被替换；其 14,662 更新的完整检查点与
10k 物理评测和视频保留，旧任务没有完成 20k。按用户后续要求，旧 G1 长训也已停止，
最后报告 17,474 更新，保留 17,300 更新的检查点；目前只训练这个新版 R1。

站姿依据本地 HoloSoma：

- `runs/r1_stance_grounded_01.npz` 标注 `grounded_official_stance`，从 35 个 DDS 槽
  按名字映射 26 个关节，取末尾 10 秒、8,332 个样本的中位数。
  这与 HoloSoma 配置里的 nominal/default angles 不同。
- 使用 `tools/prepend_r1_measured_stance_transition.py` 的读取逻辑，保留实测
  IMU roll/pitch。整体 yaw 与 XY 按两脚位置对齐原动作，再用实际足部碰撞几何确定高度。
  不能复用下蹲首帧的骨盆位置：这会让直立后的膝盖和箱体重叠。
- 新片段为 0.5 秒 Stance、2 秒 quintic/SLERP 过渡、6.5 秒原任务和 0.5 秒末帧保持，
  共 476 帧、9.5 秒。过渡段不穿地，与箱体没有重叠。
  首帧关节角与捕获中位数完全一致；直接帧间关节速度不超过所用电机限速的 69.6%。

双臂调整依据实际 URDF 腕部碰撞网格。原搬运段左右腕部最大箱面间隙约 28.6 mm、12.5 mm；
离线 IK 只修改十个手臂关节，保持原任务的根部、腿、腰、头和箱体轨迹。
以两侧箱面为目标，逐渐建立 3 mm 的虚拟内收量，并对前臂向箱内的穿透和关节修改变化加约束。
调整后搬运段腕部的有符号间隙约为 `[-3.7, -1.2] mm`、`[-3.1, -2.9] mm`。
负值表示参考目标的微小预压，不是要求实际刚体穿透，也不是接触力闭环。
实际训练没有增加物体、接触或抬升奖励，头部仍属于身体模仿跟踪。

`reference_preview.mp4` 明确标为 **REFERENCE ONLY**，可核对站姿和动作，不代表策略成功。
`validation.json`、`grip_audit.npz` 记录运动学检查；`dataset/manifest.json` 是训练实际输入。
生成器：`holomotion.scripts.training.prepare_r1_stance_grip`，本次使用 `--preload-mm 3`。

抱稳仍未得到确认。隔离底座平衡的 MuJoCo 局部测试即使腕面贴合也出现滑落，
早期含 BADQACC/自动重置的探针已标为无效，未计作成功。
受控探针与训练的碰撞设置区别在 `probe_r1_grip.py` 和对应报告中明确记录；
最终以新策略在原生 Isaac 完整片段中的持续离地结果为准。

新版独立源码目录为 `not_for_commit/omomo-r1-stance-grip-runtime-20261120/`，
保留源码散列与 `experiment.patch`，没有修改 G1 长训所依赖的源码。
训练实际 reader、关节/地面/箱体检查、14 项相关测试与 Isaac 物理预检已通过。
本机 SDK 的关闭超时仍由原监督器有界清理；预检成功不等于 SDK 干净退出。

部署入口也同步调整为 HoloSoma 的 grounded-Stance 流程：官方 Stance 请求成功后才准备只读桥接，
等待落地确认，再从实测姿态接管。策略在第 0 帧持续闭环反馈，松开后重新按 A 才播放，
B 或 L1+R1 退出并进入原阻尼清理。末帧保持同样继续推理，避免冻结一次性目标。
缺少官方站姿数据来源的导出包会在初始化 SDK 前被拒绝。这里只完成代码与无硬件逻辑测试，
没有执行实机控制；实机仍需显式确认以及原生/ONNX 抬箱评测通过。

### 训练完成后：按用户纠正，用 27.5 cm 边替换掌面法向的 24.5 cm 边

**当前状态：用户再次否定了下述候选摆放。评测服务已停止，尚未执行物理评测，
`box_placement_comparison/status.json` 标为 `awaiting_placement_direction`。**
当前训练/原视频的名义边向为机器人左右 27 cm、前后 30.5 cm、上下 24.5 cm；
下述候选预览实际变成左右 24.5 cm、前后 30.5 cm、上下 27.5 cm，即竖高了箱体。
该候选不能视为用户已确认的评测方向。等待用户以机器人左右/前后/上下明确 27.5 cm 的方向，
再设置评测；训练中的箱体及动作不变。

用户明确指出 `reference_preview.mp4` 中所指的是掌面法向的 24.5 cm 边，
评测要换成 27.5 cm。之前依据双手连线推断 27 → 30.5 cm 的世界 Z 轴旋转方案错误，
对应后台服务已停止，尚未执行物理评测；其记录在 `box_yaw_comparison/status.json`
标为 `cancelled_wrong_axis`。不再执行 30.5 cm 的方案。

新的评测箱体轴向参数为 `XYZ=[24.5, 30.5, 27.5] cm`，替换训练中的
`XYZ=[27, 30.5, 24.5] cm`：交换原 X/Z 的摆放方向，27.5 cm 严格采用用户本次给定值。
机器人参考、奖励、策略权重和箱体质量均不改变。初始箱体中心 XY 不变，
Z 自动抬高约 1.5 cm，让较高的摆放仍保持原来约 1 mm 的箱底离地间隙，避免穿地。
它是独立的评测场景，不修改训练数据或当前训练，也不重放箱体轨迹。

后台服务 `holomotion-r1-box-placement-after-20k-20261120.service` 等待本次新版 R1
的 20k 最终检查点及原生完整片段评测，再执行单独的
`box_27p5cm_placement_intervention` 物理评测。结果保留源码、输入和共同检查点散列，
不覆盖原生验收或导出门槛。

产物目录为 `logs/omomo-r1-stance-grip-20261120/box_placement_comparison/`：

- `status.json`：等待、27.5 cm 摆放评测、渲染、完成或失败。
- `comparison.json`：两组完整片段结果、持续抬升时间和共同检查点散列。
- `comparison.mp4`：左侧原摆放，右侧 27.5 cm 摆放，都是实际物理状态。
  提前结束的一侧明确标记 `ENDED`，不把冻结画面计作执行。
- `original.mp4`、`eval_27p5cm.mp4`：各自实际执行的视频。

入口为 `holomotion.scripts.evaluation.posttrain_r1_box_placement`，
`evaluate_r1_box_placement` 在独立评测进程内配置箱体尺寸轴向和初始地面位置。
可视化读取评测实际使用的箱体参数，不会仍画成训练尺寸。
3 项测试覆盖 27.5 cm 目标轴向、箱底间隙保持、输入不可变性；
实际训练 reader 的机器人未来目标应逐值一致。动态抱箱结果待训练完成后确定。

## 前一版 R1：original b4（2026-11-20）

按用户指定，新的原始输入目录是
`data/omomo-r1/r1_26dof_largebox003_original_b4_20261120/`，
使用其中 `trajectory/r1_joint_trajectory_30hz.npz` 的命名关节、基座和箱体轨迹。
它与 `trajectory/sub3_largebox_003_r1_original_omni.npz` 的 qpos 完全一致。
原始数据保留；转换产物位于同级的
`r1_26dof_largebox003_original_b4_20261120_supermimic50/`。

- 原始 30 Hz、196 帧转换为 50 Hz、326 帧，两者首末帧间隔均为 6.5 秒。
  位置、关节角线性插值，四元数 SLERP；不增加站立前缀，不平滑、裁剪或变速。
- 基座和箱体共用同一个初始 XY/yaw 变换。源场景的箱体局部 XYZ 为
  `0.270 × 0.305 × 0.245 m`，为已确认箱体尺寸的轴排列，沿用源轨迹的摆放方向，
  质量 `0.1 kg`；初始箱底离地 1 mm。
- 38 个同名机器人部件的源场景和训练 URDF 位置完全相同，最大旋转误差约
  `7.3e-8 rad`。重采样后没有关节越限，数值关节速度最大为电机限速的 69.5%。
  源场景缺失的 `camera_link.STL` 只用于审计场景的可视化，省略其零密度、
  无碰撞视觉几何，保留其惯性与所有碰撞几何；不替换训练 URDF。
- 两个头部参考关节始终为零。旧 R1 跟踪列表漏掉了头部，旧 11,133 更新策略的
  第 3.6 秒头部俯仰/偏航约为 `-26.7°/-82.1°`。新运行将
  `head_pitch_link`、`head_yaw_link` 纳入原有身体位姿/速度跟踪，仍为纯动作模仿。
  实际训练 reader 和奖励梯度检查通过：人为转头会降低奖励，并对两个头关节产生梯度。

为保留仍在运行的 G1 的源码校验，新 R1 使用独立源码目录：
`not_for_commit/omomo-r1-original-b4-runtime-20261120/`。
其中 `experiment.patch` 列出头部跟踪与恰好 10k 检查点的改动，
`runtime_provenance.json` 记录源码散列。主目录中旧 G1 所依赖的源码没有修改。

转换入口：

```bash
"$PWD/.venv-omomo/bin/python" \
  -m holomotion.scripts.training.prepare_supermimic_r1_omni \
  --source data/omomo-r1/r1_26dof_largebox003_original_b4_20261120 \
  --output data/omomo-r1/r1_26dof_largebox003_original_b4_20261120_supermimic50 \
  --robot-urdf ../holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/models/r1/r1_26dof.urdf
```

输出目录必须为空。转换器校验源散列、命名数组、时间轴、四元数、关节限位、
URDF 运动学和箱体尺寸，再调用已有 OMOMO 训练数据准备器。
转换目录的 `dataset/manifest.json` 是新训练实际读取的 manifest。

可在已冻结的独立源码目录检查训练命令（实际运行须选新的空输出目录）：

```bash
OMOMO_PROJECT_ROOT="$PWD"
cd "$OMOMO_PROJECT_ROOT/not_for_commit/omomo-r1-original-b4-runtime-20261120"
"$OMOMO_PROJECT_ROOT/.venv-omomo/bin/python" \
  -m holomotion.scripts.training.run_supermimic_r1_trajectory \
  --manifest "$OMOMO_PROJECT_ROOT/data/omomo-r1/r1_26dof_largebox003_original_b4_20261120_supermimic50/dataset/manifest.json" \
  --robot-urdf "$OMOMO_PROJECT_ROOT/../holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/models/r1/r1_26dof.urdf" \
  --output "$OMOMO_PROJECT_ROOT/logs/omomo-r1-original-b4-20261120" \
  --dry-run
```

训练服务为 `holomotion-r1-original-b4-20261120.service`，
输出根目录为 `logs/omomo-r1-original-b4-20261120/`。
本次从头训练，不加载旧策略或优化器。目标仍为 20k 策略更新；
达到恰好 10k 时原子保存 `r1/training/checkpoint_actor_010000.pt`，
后台独立 Isaac 进程评测整个片段，主训练继续。
预览报告和视频位于 `r1/preview_actor_010000/`，
根目录的 `preview_status.json` 区分预览评测完成与失败。
20k 结束后继续完整片段评测、视频、ONNX 导出、ONNX 物理评测与无动作部署检查。
实机动作未授权、未执行；抬箱是否成功以物理评测中的持续离地结果为准。

## 20k 策略更新与 R1 复核

十分钟实验只用于验证链路，不能判断充分训练后能否抱箱。
正式长训配置为 `training/motion_tracking/omomo_box_20k`，终点是
`actor_updates=20000`，另有约 20,100 次世界模型拟合事件和 50,314 个向量仿真步。
每次策略更新的批量、24 步想象时域、微批量和奖励保持原配方。
24 小时只是单任务的异常耗时上限；任务若提前结束且不足 20k，监督器会记录失败。

```bash
"$PWD/.venv-omomo/bin/python" -m holomotion.scripts.training.run_omomo_20k \
  --output logs/omomo-box-20k-20261120-v2 --robots r1 g1 --dry-run
```

去掉 `--dry-run` 会开始训练，需使用新的空目录。监督器依次执行物理预检、20k 训练、
完整片段评测、视频生成；R1 还会执行 ONNX 导出、ONNX 物理评测和无动作检查。
每 15 分钟保存检查点，一项任务失败会保留错误记录并继续另一机器人。

本次实际任务由用户级服务 `holomotion-omomo-20k-20261120-v2.service` 管理，
输出根目录为 `logs/omomo-box-20k-20261120-v2/`，先训练 R1，再训练 G1。
根目录 `status.json` 区分排队、预检、学习、评测和失败；
各机器人 `training/progress.json` 中的 `actor_updates` 才是策略更新计数。
首次队列 `omomo-box-20k-20261120` 因预检报告缺少 `process_wall_s` 而未进入学习，
该启动器错误已修复并新增回归测试，原记录保留。

R1 的专项复核结果：

- 原始 `R1.urdf` 与 Unitree 官方提交 `ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee`
  完全一致；43 个本地网格均匹配该提交的 Git 对象散列。
- `r1_26dof.urdf` 与原始文件的机器人结构相同，只适配 MuJoCo 加载设置。
- 450 帧、26 关节顺序正确，40 个共同链接的位姿与源场景一致；关节未越限，
  数值速度峰值约为所用电机限速的 32.5%。
- HDF5 相对原数据的关节角误差小于 `3e-8 rad`，位置误差小于 `6e-8 m`。
- 在参考抬升段，左右实际腕部碰撞网格到箱面的最大间隙分别约 3.3 mm 和 8.7 mm。
  源场景另有重定向用手脚辅助球，此处也单独核查了官方真实腕部几何。

这些检查确认资产与导入一致，不保证轨迹的动力学可追踪性或物理抬箱成功。
复核产物在 `not_for_commit/r1-audit/`：`reproducible/report.json`、
`upstream_urdf.json`、`upstream_meshes.json` 和 `r1-reference-vs-policy.mp4`。
并排视频左侧是直接设定姿态的参考回放，右侧是短训策略的真实物理轨迹。
渲染已恢复 URDF 材质并隐藏重复碰撞面，不改变训练碰撞设置。

复核可由 `holomotion.scripts.evaluation.audit_omomo_reference` 重现，
它会检查源 NPZ、源 MJCF、URDF 和训练 manifest，并保存各输入散列。

## 本机实测

单张 RTX 3090，每个机器人从头训练约 600 秒，未添加交互奖励。
这是同一训练片段的基本复现，不是独立测试集或原论文成绩复现。

| 机器人 | 原生评测分数 | 实际执行 / 计划时长 | 跌倒终止 | 箱底最高离地 | 连续离地 5 cm | 抬箱成功 |
| --- | --- | --- | --- | --- | --- | --- |
| G1，原始箱体网格 | 0.7804 | 6.48 / 6.48 s | 否 | 3.83 cm | 0 s | 否 |
| R1，30.5×24.5×27 cm，0.1 kg | 0.6964 | 8.98 / 8.98 s | 否 | 2.89 cm | 0 s | 否 |

R1 的实际 ONNX 在 Isaac 中另行评测，分数 0.6839，执行完整片段，未抬箱。
20 组 FP32 PyTorch/ONNX 输出对比的最大绝对误差约 `3.10e-6`。
部署包的无动作检查完成 30 次有限推理，未初始化硬件接口、未发送实机指令。
这些结果不足以进入实机抬箱阶段，主动部署门槛保持关闭。

有效实验与产物：

- `logs/omomo-box/g1-mesh-baseline-10min/`：G1 检查点、原生评测和视频。
- `logs/omomo-box/r1-baseline-10min/`：R1 检查点、原生评测和视频。
- `logs/omomo-box/r1-onnx-evaluation/`：导出策略的真实物理评测。
- `logs/omomo-box/r1-bundle/`：R1 ONNX、参考特征、契约及无动作检查。
- `logs/omomo-box/baseline-results.json`、`baseline-results.png`：原始指标汇总和曲线。

旧的 `g1-baseline-10min` 使用包围盒长方体近似，首帧穿地约 5.6 cm，已标为无效；
不得用它的提前失败结果判断原始箱体上的策略表现。修正后的网格最低点首帧约为
地面下 3.1 mm，保留原始数据和几何原点，不平移箱子来掩盖错误。
本次新增工作流及相关回归测试共 65 项通过。

本机 SDK 在训练/评测成功写出产物并发送完成通知之后，关闭阶段仍超时。
监督进程等待后终止 SDK 子进程；对应 `isaac_process.json` 记录
`operation_succeeded=true`、`shutdown_timeout=true`、`exit_code=-9`。
这是产物已完成后的有界清理，不是 SDK 正常退出；此环境的干净关闭问题尚未解决。

## 环境

新增环境入口允许选择解释器，不改变默认训练环境：

```bash
export HOLOMOTION_TRAIN_PREFIX="$PWD/.venv-omomo"
source train.env
"$Train_CONDA_PREFIX/bin/python" -c 'import sys; print(sys.executable)'
```

有现成 Isaac 环境时，用它创建隔离的本地覆盖层：

```bash
<isaac-env>/bin/python environments/setup_omomo.py
```

此覆盖层复用基环境和 Isaac 自带的 Warp，不修改基环境。机器需已安装可用的
Isaac Sim 和 Isaac Lab。独立安装完整 SDK 见 [环境安装](environment_setup.md)。
本次本机使用 Isaac Sim 5.1.0.0、Isaac Lab 源码包 0.47.2、Torch 2.7.0+cu128、
模拟器自带 Warp 1.8.2，以及本地 CPython 3.11 MuJoCo 3.11.0；这与主教程的
Sim 5.0 / Lab 2.2 版本组合不同。运行版本、输入和源码指纹写在每次实验的
`training/specification.json`，本机环境盘点保存在 `not_for_commit/omomo-environment.json`。
复用的辅助库依赖尚未完全协调，`pip check` 的未解决项见
`not_for_commit/omomo-pip-check.txt`；不声称依赖全局无冲突。

## 数据和机器人资源

通过环境变量指定本地 HoloSoma，默认是相邻的 `../holosoma`：

```bash
export HOLOSOMA_ROOT="$PWD/../holosoma"
PY="$PWD/.venv-omomo/bin/python"

"$PY" -m holomotion.scripts.training.prepare_supermimic_omomo \
  --source "$HOLOSOMA_ROOT/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz" \
  --output data/omomo-g1-mesh --robot g1 \
  --box-size .47115421 .45873013 .40789548 --box-mass .1 \
  --box-mesh "$HOLOSOMA_ROOT/src/holosoma/holosoma/data/scene_objects/boxes/largebox.obj"

"$PY" -m holomotion.scripts.training.prepare_supermimic_omomo \
  --source "$HOLOSOMA_ROOT/exp/retargeting/r1_sub3_largebox_003_realbox_20261001/tasks/sub3_largebox_003/semantic_b4_cap10_standing_transition/sub3_largebox_003_semantic_b4_cap10_stand05_blend20_mj_w_obj.npz" \
  --output data/omomo-r1 --robot r1 \
  --box-size .305 .245 .270 --box-mass .1
```

输入是已有机器人重定向结果，不重新估计人体动作。转换程序验证关节名称、50 Hz、
WXYZ 源四元数和完整时间轴，并对机器人和物体共同移除初始 XY 和偏航。
训练机器人根部四元数保存为标准 XYZW；物体参考保存为 WXYZ。
数据、SDK、网格和生成的模型包均保留在本地，不加入 Git。

G1 片段 325 帧。使用 HoloSoma 的半球手碰撞 URDF，以及原始 `largebox.obj`
扫描网格的凸包碰撞。保留网格坐标和原点；尺寸参数仅记录包围盒元数据。
不能用相同包围盒长方体替代这个倾斜扫描箱体：该近似会在首帧穿地约 5.6 cm。
初始化最低点低于地面 1 cm 的任务现在会直接拒绝启动。
R1 片段 450 帧，包括原数据的站立过渡，使用 26 关节模型（包含头部两关节）。
R1 尺寸按用户确认采用 XYZ = 30.5×24.5×27 cm、质量 0.1 kg。
R1 使用自己的关节顺序、PD、力矩和速度上限，不能使用 G1 权重或 G1 电机映射。

默认资源路径由 `run_omomo_box.sh` 解析，可用 `ROBOT_URDF` 覆盖。
实际使用的 URDF 与全部引用网格均进入训练指纹。

## 预检和训练

```bash
bash holomotion/scripts/training/run_omomo_box.sh g1 preflight
bash holomotion/scripts/training/run_omomo_box.sh r1 preflight

# 每个任务默认 30 分钟；MAX_HOURS=.166667 表示约 10 分钟训练预算。
MAX_HOURS=.166667 RUN_OUTPUT=logs/omomo-box/g1-mesh-baseline-10min \
  bash holomotion/scripts/training/run_omomo_box.sh g1 train
MAX_HOURS=.166667 RUN_OUTPUT=logs/omomo-box/r1-baseline-10min \
  bash holomotion/scripts/training/run_omomo_box.sh r1 train
```

每次使用新输出目录。脚本支持追加原生训练 CLI 参数，例如 `--no-compile`。
默认 128 个环境、500 步回放、24 步想象时域、世界模型批量 256、
策略逻辑批量 512、微批量 128，学习率 `4e-4`，随机种子 42。
编译和准备耗时计入训练预算；训练后的独立评测另计。
Hydra 对应配置为 `training/motion_tracking/omomo_box`，需设置数据 manifest 和机器人 URDF。

箱体任务从片段首帧重置，避免随机后段重置把箱子初始化在空中。
实际物理审计校验关节顺序、PD、力矩/速度限制、总质量、限位及前向运动学。
`preflight` 成功只表示执行链路可用，不表示学会动作。

## 评测与视频

最终评测使用独立 Isaac 进程，从首帧开始运行完整计划时间；跌倒后的尾部计入归一化分母。
`training/evaluation_final.json` 包含逐片段动作分数、跌倒、执行长度和 `box_lift`：

- `peak_bottom_clearance_m`：考虑旋转后，长方体最低角点或原始网格最低顶点距离地面的最大高度。
- `longest_lift_seconds`：箱底至少离地 5 cm 的最长连续时间。
- `lifted`：连续抬升至少 0.5 秒，且完整片段无机器人失败。

这只是物理抬升判定；同时保留视频以判断是否确实由双臂抱起。
原始箱体和机器人状态保存在 `training/evaluation/rollout_final.pt`。
可以渲染这些真实 Isaac 评测状态，不重放物体参考轨迹：

```bash
"$PY" -m holomotion.scripts.evaluation.render_omomo_rollout \
  --training logs/omomo-box/r1-baseline-10min/training \
  --output logs/omomo-box/r1-baseline-10min/evaluation.mp4
```

## R1 导出和无动作部署检查

```bash
"$PY" -m holomotion.scripts.training.export_supermimic \
  --checkpoint logs/omomo-box/r1-baseline-10min/training/checkpoint_final.pt \
  --output logs/omomo-box/r1-bundle

"$PY" -m holomotion.scripts.evaluation.evaluate_supermimic_onnx \
  --training logs/omomo-box/r1-baseline-10min/training \
  --bundle logs/omomo-box/r1-bundle \
  --output logs/omomo-box/r1-onnx-evaluation

"$PY" -m holomotion.src.deployment.supermimic_r1 \
  --bundle logs/omomo-box/r1-bundle --check
```

模型包包含 ONNX、参考特征、根部旋转、关节/PD/动作缩放约定、检查点来源及 SHA256。
导出验证原始训练源码快照、输入和当前策略/参考重建实现；训练后的物理场景修复
不会覆盖旧实验的源码记录。已完成 R1 实验的原始源码在其 `training/source_snapshot/`。
导出用 20 组不同参考相位和扰动观测验证 PyTorch 与 ONNX 的数值一致性。
`--check` 只加载包、检查维度/哈希并执行 30 次推理，不导入或构造机器人硬件接口。
检查不等于策略通过仿真，也不等于通过实机验证。
导出会绑定同一检查点的完整评测结果，并另外在 Isaac 中运行实际 ONNX 策略。
未同时通过原生策略和 ONNX 完整片段抬箱评测的模型可以执行
`--check`，但主动启动入口会在导入硬件 SDK 前拒绝运行。

实机代码复用 HoloSoma 已有 R1 C++ DDS/PD bridge，其稀疏 35 电机槽到 26 关节的
映射由已有 SDK 负责。需要在目标机准备 HoloMotion 源码、完整模型包、
HoloSoma inference 包及已编译 bridge。与旧 HoloSoma 139 维 actor 的 ONNX 不兼容。

在模型通过仿真、操作员准备好支撑及急停，并确认下列动作命令后，才运行：

```bash
python -m holomotion.src.deployment.supermimic_r1 \
  --bundle /path/to/r1-bundle --holosoma-root /path/to/holosoma \
  --interface <已确认的机器人网卡> --enable-robot
```

主动入口要求数据首帧为已记录的官方落地 Stance，并已通过原生及 ONNX 物理抬箱检查。
先请求官方 Stance，等待手柄 A 的落地确认后接管；保持第 0 帧的实时策略反馈，
松开并再次按 A 后播放动作。B、L1+R1 或 Select 停止并进入阻尼；异常、非有限值、
目标越限或策略错过 20 ms 时限同样停止。末帧继续策略闭环保持。
这些控制保持和阻尼逻辑未在本次任务中执行实机验证。
新策略的实机执行必须另行立即确认，本次代码验证不发送实机动作。
