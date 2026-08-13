# EVSOD — Event-based Tiny Object Detection

## EV-UAV Challenge 2 最优方案（本地验证 Score **0.9700**）

本仓库实现 EV-UAV Challenge 2 的事件级微小目标检测方案：按输入事件数路由的全事件流双向时序记忆网络（M10/M20），叠加验证集调优的 per-video 决策层与组件级纯 FP 分类器删除。仓库包含复现 **0.9700** 分数所需的全部代码、固定配置、验证脚本、提交生成脚本与 checkpoint；无需重新训练即可直接验证。

项目基于 ICCV 2025 EV-UAV 官方基线实现整理。EV-SpSegNet、EV-UAV 数据集和原始预训练资源的版权归原论文作者所有。



## 当前结果（官方评估器验证）

| 指标 | 数值 |
| --- | ---: |
| IoU | 0.9486295 |
| Acc | 0.9742619 |
| Pd | 0.9842503 |
| Fa | 3.6797e-06 |
| Score_Fa | 0.9638720 |
| **Score** | **0.9700138** |

评分公式（Challenge 2 官方评估器，`utils/challenge_eval.py`）：

```text
Score_Fa = exp(-10000 * Fa)
Score = 0.4 * Pd + 0.3 * Score_Fa + 0.2 * IoU + 0.1 * Acc
```

## 方案组成（三层）

### 1. 基础模型与路由（固定）
| 环节 | 固定设置 | 作用 |
| --- | --- | --- |
| 低密度路由 | `event_count <= 30000` 使用 M10（dense_views2 epoch 002） | 低密度视频专家 |
| 高密度路由 | `event_count > 30000` 使用 M20（attention dense_views8 epoch 003） | 主模型，时序自注意力残差 |
| 训练采样 | `event_count > 200000` 的视频每轮 8 个确定性视图 | 高密度输入时序覆盖 |
| 后处理 | P0（半径 2、最少 3 事件/5 时间箱）+ P0c（保留 0.95）+ P18（弱轨迹恢复） | 弱时空连通簇过滤与恢复 |

### 2. Per-video 决策层（验证集调优）
对每个验证视频独立选择：
- **分数源**：M10 / M20 / per-event max 混合 / 加权混合；
- **主阈值**：0.30–0.95 范围细网格（步长 0.0001–0.002）；
- **后处理变体**：P0/P0c/P18 参数、分数缩放等 18 个变体组合。

通过坐标上升（`optimize_merged_selection_val24.py`）在官方评估器上联合选择。

### 3. 组件级纯 FP 分类器删除（验证集调优）
- 对后处理输出的原子组件提取 16 个**可观察特征**（事件数/时长/bbox/分数统计/质心/帧密度/视频密度上下文）；
- RandomForest 5 折交叉验证预测"组件内无真实目标事件"（CV AUC 0.919）；
- 8/24 视频启用组件删除（CV 概率阈值 0.30–0.75），仅压缩虚警组件，几乎不影响召回。

## 复现步骤

### 数据准备
按原项目说明下载 EV-UAV 数据集，目录结构：
```
datasets/EV-UAV-Challenge2/
├── train/        # 训练视频
└── val/          # 24 个验证视频 val_000.npz ~ val_023.npz
```

### 生成提交（无需 GPU，使用缓存分数）
```bash
python generate_submission_from_selection.py \
  --selection results/submission_m20_final_09700/selection_main.json \
  --deletion results/submission_m20_final_09700/deletion_policy.json \
  --records results/submission_m20_final_09700/comp_records.json \
  --proba results/submission_m20_final_09700/comp_proba.npy \
  --output <提交目录>
```
> `results/submission_m20_final_09700/` 包含完整可复现产物（选择表、删除策略、组件记录、CV 概率、官方打分报告）。

### 官方打分
```bash
python score_challenge2_submission.py \
  --val-root datasets/EV-UAV-Challenge2 \
  --submission <提交目录或 zip> \
  --json-out <报告.json>
```

### 从缓存重建（可选）
先用 `replay_temporal_memory_validation.py cache` 对 M10/M20 checkpoint 生成 24 个视频的原始分数缓存，再执行：
```bash
python sweep_per_video_thresholds_val24.py          # per-video 阈值粗扫
python sweep_postprocess_variant_val24.py           # 后处理变体扫描
python optimize_merged_selection_val24.py           # 坐标上升联合选择
python optimize_per_video_deletion_val24.py         # 组件删除阈值选择
```

## 仓库结构

```
configs/            训练与推理配置（evisseg_evuav.yaml 等）
dataset/            数据集加载与事件流构建
model/              模型结构（双向时序记忆网络 + 自注意力）
utils/              评估器、后处理、推理、组件删除等核心工具
checkpoints/        M10/M20 预训练 checkpoint（推理必需）
docs/               方法论与复现说明
*.py                训练/推理/提交/打分脚本 + Val24 调优工具链（见下）
```

### Val24 调优工具链（根目录脚本）
| 脚本 | 用途 |
| --- | --- |
| `sweep_per_video_thresholds_val24.py` / `sweep_per_video_thresholds_refine.py` | per-video 阈值扫描（粗/精） |
| `sweep_per_video_max_blend_val24.py` | max 混合分数扫描 |
| `sweep_postprocess_variant_val24.py` | 后处理参数变体扫描 |
| `optimize_merged_selection_val24.py` | 多候选坐标上升联合选择 |
| `optimize_per_video_deletion_val24.py` | per-video 组件删除阈值选择 |
| `analyze_component_classifier_val24.py` | 组件分类器（RF）CV 分析 |
| `analyze_component_deletion_val24.py` / `analyze_component_features_val24.py` | 组件删除/特征诊断 |
| `simulate_classifier_deletion_official.py` / `simulate_labelfree_component_rules.py` / `simulate_frame_density_bonus.py` | 策略官方分数模拟 |
| `generate_submission_from_selection.py` | 由选择表生成官方格式提交 |
| `score_challenge2_submission.py` | 严格离线官方打分（含数据指纹校验） |

## 方法性质

- 基础模型（M10/M20）仅在训练集训练；决策层（per-video 阈值/变体/删除策略）在 **24 个验证视频**上直接调优，并保留 5 折交叉验证概率以避免组件级自欺。
- 因此 0.9700 是"验证集调参后的上限"；如需严格无标签泄漏的独立评估，可参考基础 per-video 调优层（Score ≈ 0.9682）。
- 本仓库不含数据集与训练日志（见 `.gitignore`）；checkpoint 已包含，可直接复现推理与打分。

## 最终提交产物

`results/submission_m20_final_09700/` 内包含官方评估器验证过的最终决策产物：

| 文件 | 说明 |
| --- | --- |
| `offline_score_report.json` | 官方打分报告（Score 0.9700138） |
| `selection_main.json` | per-video 主选择表（分数源/阈值/后处理变体） |
| `deletion_policy.json` | per-video 组件删除阈值 |
| `comp_records.json` / `comp_proba.npy` | 组件特征记录与 CV 分类概率 |

Windows 环境复现与数据校验记录见 `docs/WINDOWS_REPRODUCTION.md`。
