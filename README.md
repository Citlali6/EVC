# EVC

## 基于 EV-SpSegNet 的事件相机微小目标检测复现与改进

EVC 是一个用于复现、评估和改进 **EV-SpSegNet** 的研究仓库，面向 EV-UAV 事件相机无人机微小目标检测基准。

> 本仓库基于 ICCV 2025 论文 *Event-based Tiny Object Detection: A Benchmark Dataset and Baseline* 的官方实现整理。EV-SpSegNet、EV-UAV 数据集和预训练权重的原始贡献均属于原论文作者；本仓库不将基线方法或数据集作为新的方法或数据集主张。

---

## 操作导航

| 目标 | 使用的脚本或配置 | 是否写入预测文本 |
| --- | --- | --- |
| 测试官方预训练权重 | `test.py` + `evisseg_evuav.yaml` | 否 |
| 检查训练流程是否可运行 | `train.py` + `evisseg_evuav_smoke.yaml` | 否 |
| 在 4GB 显存设置下训练 | `train.py` + `evisseg_evuav_4gb.yaml` | 否 |
| 从头训练完整基线 | `train.py` + `evisseg_evuav_scratch.yaml` | 否 |
| 在赛道二验证集上计算指标和总分 | `test2.py` + `evisseg_evuav_challenge2.yaml` | 否 |
| 生成赛道二提交文件 | `submit_challenge2.py` + `evisseg_evuav_challenge2.yaml` | 是，生成 `val-pred-txt/*.txt` |

---

## 当前可复现最优提交候选

> **适用范围：** 以下结果来自赛道二官方验证集 `val/`，用于选择当前提交候选；未知测试集上的实际成绩仍需以官方评测为准。所有路径均为当前 WSL 工作区中的实际权重路径。

截至当前实验，完整命令可复现的最高总分为 **`0.8618022242`**。该方案不是单次训练得到的单模型，而是由两份同架构 EV-SpSegNet 权重进行加权集成，再对原始与水平镜像事件流的概率做平均，随后按视频事件数选择决策阈值，并在二值化后应用时空簇过滤和高置信恢复。

| 可复现的赛道二验证结果 | 数值 |
| --- | ---: |
| IoU | 0.7831628323 |
| Acc | 0.8029493690 |
| Pd | 0.8357832843 |
| Fa | 3.1967536389e-06 |
| Score_Fa | 0.9685380238 |
| Score | **0.8618022242** |

### 方案组成

| 环节 | 已启用的方法与参数 | 作用 |
| --- | --- | --- |
| 基础网络 | EV-SpSegNet（原始稀疏时空分割网络） | 产生每个事件为目标的概率。 |
| 主模型 | **P1b 目标保持采样**；`max_events_num=100000`；训练 `100` epoch；Adam，初始学习率 `0.001`，余弦退火至 `1e-5`；不启用 HNM。权重为 `e1v2_p1b_cosine100/.../best_score_seed37.pt`。 | 在事件预算受限时优先保留目标事件，背景事件随机补足；这是当前分数最高的单模型来源。 |
| 次模型 | **P1b 目标保持采样 + P15 训练期水平翻转增强**；训练 `100` epoch；每个训练样本以概率 `0.5` 做标签保持的水平翻转。权重为 `p15_flip_p1b_cosine100/.../best_score_seed37.pt`。 | 训练期几何扰动使它与主模型产生互补的召回错误；它只作为 E1 的次模型。 |
| E1 推理集成 | 主模型概率权重 `0.895`，次模型概率权重 `0.105`。 | 在二值化前融合两份预测，降低单模型的偏差。 |
| P6 密度自适应阈值 | `event_count_cutoff=100000`；事件数 `<=100000` 使用 `0.45`，事件数 `>100000` 使用 `0.92`。 | 按视频事件量选择二值化阈值；不使用标签，不改变训练过程。 |
| P0 后处理 | 时空连通簇过滤：`spatial_radius=2`、`temporal_bin_size=50`、`temporal_radius_bins=1`、`min_cluster_events=3`、`min_duration_bins=1`。 | 删除小型孤立正预测簇，主要用于抑制虚警并改善 `Score_Fa`。 |
| P0c 高置信恢复 | 启用；`retain_min_score=0.975`。 | P0 原本会删除但最高置信度不低于阈值的簇会被保留，减少对高置信小目标的误删。 |
| P8 随机分块推理 | 启用；仅事件数 `>100000` 的视频分块；`chunk_size=100000`；`random_seeds=[37,73,101]`。 | 每个随机划分覆盖全部事件一次，将三次划分的事件分数平均，降低密集视频因输入预算截断带来的波动。 |
| P14 水平翻转 TTA | 启用；原始与水平镜像输入的概率权重均为 `0.5`。 | 只镜像可观测事件坐标与归一化 x 特征，在 P6 和 P0 前平均分数；不读取标签。 |

`TEST.prediction_threshold=0.900` 保留在命令中作为 P6 关闭时的回退阈值；在当前 `P6` 启用的候选中，实际二值化阈值由上表的 `0.45/0.92` 策略决定。未在上表列出的模块不属于这一已确认方案。

### 最佳方案的实现顺序

训练阶段和推理阶段使用的是两套互补机制，不能把推理后处理参数加入训练命令来理解：

1. **训练主模型**：P1b 目标保持采样在 100000 事件预算内优先保留目标事件，再用背景事件补足输入；主模型不启用 HNM。
2. **训练次模型**：使用同样的 P1b 采样，并以 `0.5` 概率做 P15 标签保持的水平翻转增强，得到与主模型互补的次模型。
3. **逐事件模型预测**：对每个视频的原始流或镜像流，P8 只在事件数超过 `100000` 时把完整事件流做确定性随机分块；每个分块都恢复到原事件顺序，并对 `[37,73,101]` 三次划分的分数求平均。低密度视频直接完整前向。
4. **E1 概率融合**：在每个完整前向或分块前向内部，主模型和 P15 次模型分别输出事件分数，再按 `0.895/0.105` 加权；融合发生在二值化之前。
5. **P14 水平翻转 TTA**：对同一视频的原始流和水平镜像流各执行一次上述推理，将两份分数按 `0.5/0.5` 平均，并映射回原事件顺序。P14 不读取标签。
6. **P6 密度阈值**：根据原视频事件数选择二值化阈值，`<=100000` 用 `0.45`，`>100000` 用 `0.92`；`0.900` 只作为关闭 P6 时的回退值。
7. **P0/P0c 后处理**：在二值化候选上按 `r=2、时间箱=50、时间邻域=1` 建立时空簇，删除事件数少于 `3` 的簇（`min_duration_bins=1`）；若被删除簇中最高分达到 `0.975`，由 P0c 恢复。
8. **提交导出**：保留原始事件的 `x y t p`，只把最终二值预测写入最后一列 `label`；`test2.py` 和 `submit_challenge2.py` 必须使用完全相同的流程。

### 复现前提

1. 使用本仓库当前完整工作树、赛道二数据包 `dataset/训练集、验证集/`、Python 3.9、PyTorch 1.9.1 + CUDA 11.1、已编译的 `HAIS_OP`。仅拿官方原始仓库或只复制 README 不能复现本方案，因为 P1b、P15、E1、P0/P0c、P6、P8、P14 均为本仓库新增模块。
2. 训练和验证均固定 `TRAIN.seed=37`。`train.py` 会设置 Python、NumPy、PyTorch/CUDA 种子和确定性选项；不同 CUDA、PyTorch、spconv 或 HAIS_OP 构建版本仍可能造成数值差异。
3. 训练使用 `train/`，评分和提交使用同一份 `val/`。验证集应显示 `24` 个视频；不要修改 `pd_detT=50`、`correct_thresh=0.0001` 或评分代码。
4. 需要精确复测本节的 `0.8618022242` 时，应使用下方给出的两份已保存权重。重新训练可复现该训练方案，但每次都会生成新的时间戳运行目录，应重新以自己的两个 `best_score_seed37.pt` 完整验证。

每个新 WSL 终端先执行：

```bash
conda activate EV39
export PROJECT_DIR=/mnt/d/AI/ESOD/EV-UAV-main
export PYTHONPATH=$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "$PROJECT_DIR"
python -c "import torch; import spconv.pytorch; import HAIS_OP; print('CUDA:', torch.cuda.is_available(), 'HAIS_OP: ok')"
```

### 步骤 1：从头训练主模型

主模型只启用 P1b 目标保持采样，训练 100 个 epoch。训练期的 P0 为历史权重选择流程的一部分，使用 `spatial_radius=1`；它不参与反向传播。P0c、P6、P8、P14 只在最终推理启用，不能加入此训练命令。

```bash
PRIMARY_ROOT="$PROJECT_DIR/log/e1v2_p1b_cosine100_4gb_seed37"

python train.py --config configs/evisseg_evuav_4gb.yaml \
  --set TRAIN.seed=37 \
  SAMPLING.target_preserving_enabled=true \
  TRAIN.epochs=100 \
  TRAIN.lr=0.001 \
  TRAIN.scheduler=cosine \
  TRAIN.scheduler_min_lr=0.00001 \
  TRAIN.validation_start_epoch=30 \
  TRAIN.checkpoint_interval=0 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_spatial_radius=1 \
  POSTPROCESS.p0_temporal_bin_size=50 \
  POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1 \
  TRAIN.model_save_root="$PRIMARY_ROOT"
```

训练结束时记录控制台打印的 `best Score checkpoint:`。原始可复现运行的主权重为：

```text
log/e1v2_p1b_cosine100_4gb_seed37/runs/20260724-182346_seed37_pid544/best_score_seed37.pt
```

该运行的训练期最佳验证记录为 `Score=0.8138135298`。这是选择主权重的中间结果，不是最终 E1/P0c/P6/P8/P14 分数。

### 步骤 2：从头训练次模型

次模型在相同 P1b 采样基础上增加 P15 训练期水平翻转增强，训练 100 个 epoch。翻转同时变换事件坐标和归一化 x 特征，标签保持不变；推理时仍使用 P14 的原始/镜像概率平均。该模型只作为 E1 的次模型。

```bash
SECONDARY_ROOT="$PROJECT_DIR/log/p15_flip_p1b_cosine100_4gb_seed37"

python train.py --config configs/evisseg_evuav_4gb.yaml \
  --set TRAIN.seed=37 \
  SAMPLING.target_preserving_enabled=true \
  SAMPLING.p15_horizontal_flip_augmentation_enabled=true \
  SAMPLING.p15_horizontal_flip_augmentation_probability=0.5 \
  TRAIN.epochs=100 \
  TRAIN.lr=0.001 \
  TRAIN.scheduler=cosine \
  TRAIN.scheduler_min_lr=0.00001 \
  TRAIN.validation_start_epoch=30 \
  TRAIN.checkpoint_interval=0 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_spatial_radius=1 \
  POSTPROCESS.p0_temporal_bin_size=50 \
  POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1 \
  TRAIN.model_save_root="$SECONDARY_ROOT"
```

同样记录 `best Score checkpoint:`。原始可复现运行的次权重为：

```text
log/p15_flip_p1b_cosine100_4gb_seed37/runs/20260726-114604_seed37_pid5970/best_score_seed37.pt
```

该运行的训练期最佳验证记录为 `Score=0.7837092019`。最终得分依赖下一步的概率融合和推理后处理，不应把 `last_seed37.pt`、`best_loss_seed37.pt` 或 `best_iou_seed37.pt` 替换为这里的 `best_score_seed37.pt`。

### 步骤 3：用固定权重复测 `0.8618022242`

```bash
PRIMARY=/mnt/d/AI/ESOD/EV-UAV-main/log/e1v2_p1b_cosine100_4gb_seed37/runs/20260724-182346_seed37_pid544/best_score_seed37.pt
SECONDARY=/mnt/d/AI/ESOD/EV-UAV-main/log/p15_flip_p1b_cosine100_4gb_seed37/runs/20260726-114604_seed37_pid5970/best_score_seed37.pt

sha256sum "$PRIMARY" "$SECONDARY"
# expected primary:   8ae3687bcc1e508df8cd1dc4bef1fdf4f08354c4197bd4f5378c6a48b35afdab
# expected secondary: 37649c4017a73cef3ac0f9f01e8c0e2db0cf6d0a23e4c22e116b87783b89f6d9

python test2.py --config configs/evisseg_evuav_challenge2.yaml \
  --set TEST.model_path="$PRIMARY" \
  TEST.prediction_threshold=0.900 \
  STRUCTURE.p16_global_patch_attention_enabled=false \
  ENSEMBLE.ensemble_enabled=true \
  ENSEMBLE.ensemble_secondary_model_path="$SECONDARY" \
  ENSEMBLE.ensemble_primary_weight=0.895 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 \
  POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.975 \
  POSTPROCESS.p6_density_threshold_enabled=true \
  POSTPROCESS.p6_event_count_cutoff=100000 \
  POSTPROCESS.p6_low_density_threshold=0.45 \
  POSTPROCESS.p6_high_density_threshold=0.92 \
  INFERENCE_CHUNK.p8_enabled=true \
  INFERENCE_CHUNK.p8_event_count_cutoff=100000 \
  INFERENCE_CHUNK.p8_chunk_size=100000 \
  INFERENCE_CHUNK.p8_random_seeds='[37,73,101]' \
  INFERENCE_TTA.p14_horizontal_flip_enabled=true \
  INFERENCE_TTA.p14_horizontal_flip_original_weight=0.5
```

预期输出中的指标应为：

```text
IoU:      0.7831628323
Acc:      0.8029493690
Pd:       0.8357832843
Fa:       3.1967536389e-06
Score_Fa: 0.9685380238
Score:    0.8618022242
```

日志还应显示 P6 使用 `0.450: 22 videos, 0.920: 2 videos`、P14 的原始/镜像权重均为 `0.500`，以及 P8 对 2 个高密度视频执行 42 次分块前向推理（原始和镜像各一次）。若权重哈希或上述指标不一致，不要直接导出提交文件，先检查工作树、数据集路径和全部命令覆盖项。

使用自己重新训练的模型时，只需将 `PRIMARY` 和 `SECONDARY` 改为步骤 1、步骤 2 控制台打印的两个 `best_score_seed37.pt` 绝对路径，再执行同一验证命令。

### 步骤 4：生成提交文本

生成提交文本时必须复用完全相同的模型、权重比例、阈值、P0/P0c、P6、P8 和 P14 参数。使用一个新的输出目录，避免与旧提交文本混用：

```bash
OUTPUT_DIR="$PROJECT_DIR/log/challenge2/val-pred-txt-score8618"

python submit_challenge2.py --config configs/evisseg_evuav_challenge2.yaml \
  --set TEST.model_path="$PRIMARY" \
  TEST.challenge_output_dir="$OUTPUT_DIR" \
  TEST.prediction_threshold=0.900 \
  STRUCTURE.p16_global_patch_attention_enabled=false \
  ENSEMBLE.ensemble_enabled=true \
  ENSEMBLE.ensemble_secondary_model_path="$SECONDARY" \
  ENSEMBLE.ensemble_primary_weight=0.895 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 \
  POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.975 \
  POSTPROCESS.p6_density_threshold_enabled=true \
  POSTPROCESS.p6_event_count_cutoff=100000 \
  POSTPROCESS.p6_low_density_threshold=0.45 \
  POSTPROCESS.p6_high_density_threshold=0.92 \
  INFERENCE_CHUNK.p8_enabled=true \
  INFERENCE_CHUNK.p8_event_count_cutoff=100000 \
  INFERENCE_CHUNK.p8_chunk_size=100000 \
  INFERENCE_CHUNK.p8_random_seeds='[37,73,101]' \
  INFERENCE_TTA.p14_horizontal_flip_enabled=true \
  INFERENCE_TTA.p14_horizontal_flip_original_weight=0.5

cd "$OUTPUT_DIR"
zip -j ../e1_p1b_p15_p14_0861802.zip val_*.txt
```

### Frozen candidate status

The `0.8618022242` candidate is frozen for submission. Its reproducible
configuration is the E1 ensemble plus P0/P0c, P6, P8 and P14 described above.
The following experimental branches are intentionally disabled and must not be
added to the submission command:

- `STRUCTURE.p16_global_patch_attention_enabled=false`: the corrected global
  attention screen reached `Score=0.8424685722` after 10 epochs and is not a
  candidate.
- `LOSS.p17_positive_ranking_enabled=false`: implemented as an optional
  training loss, but not validated as an improvement and therefore not part of
  the frozen model.
- P9 density dual-view sampling and P16 target-context sampling: not part of
  the frozen model. P9 was already tested; target-context sampling is useful
  only while the 100000-event budget is active and is unnecessary for the later
  full-stream `max_events_num=700000` run.

The current candidate was trained with `max_events_num=100000`. The full
`700000`-event configuration is a separate future capacity run and must not be
mixed with these checkpoint paths. Re-evaluate P0/P6/P8 thresholds after using
the full-stream checkpoints.

本节顶部的方案是当前唯一冻结的提交方案；后文的早期 P0、HNM-E1、P2a、P3-Lite
等分数只用于说明实验演变和负结果，不能替换这里的权重或命令参数。提交前应以本节
步骤 3 的验证命令复测通过，再执行步骤 4 导出文本。

---

## 目录结构

```text
EVC/
|-- configs/
|   |-- evisseg_evuav.yaml              # 官方预训练权重测试配置
|   |-- evisseg_evuav_4gb.yaml          # 4GB 显存训练配置
|   |-- evisseg_evuav_smoke.yaml        # 1 epoch 烟雾测试配置
|   |-- evisseg_evuav_scratch.yaml      # 完整事件数的从头训练配置
|   `-- evisseg_evuav_challenge2.yaml   # 赛道二验证与评分配置
|-- dataset/
|   |-- EV-UAV-dataset/                 # 原始本地数据集，不提交到 Git
|   `-- 训练集、验证集/                  # 赛道二官方数据包，不提交到 Git
|-- lib/hais_ops/                        # 自定义 CUDA 扩展 HAIS_OP
|-- model/                               # EV-SpSegNet 网络实现
|-- utils/                               # STC Loss、P0/P1/P2 优化与评估工具
|-- tests/test_postprocess.py            # P0 时空簇过滤单元测试
|-- train.py                             # 训练入口
|-- test.py                              # 原始测试集评估入口
|-- test2.py                             # 赛道二验证集评分入口
|-- submit_challenge2.py                 # 赛道二提交文本生成入口
`-- log/                                 # 本地权重、预测和实验结果，不提交到 Git
```

`.gitignore` 会排除数据集、权重、日志、MLflow 记录和 CUDA 编译产物。不要执行会把这些本地文件加入版本库的强制提交操作。

---

## 方法摘要

EV-SpSegNet 将事件相机微小目标检测建模为稀疏点云分割问题。运动目标在时空事件点云中通常形成连续轨迹，而背景噪声更常表现为孤立、弱相关的事件。

基线由以下部分组成：

- **GDSCA**：分组空洞稀疏卷积，用于提取多尺度时空特征。
- **Sp-SE**：稀疏特征融合模块。
- **Patch Attention**：用于体素下采样和全局上下文建模。
- **STC Loss**：时空相关损失，保留具有连续性的目标事件并抑制孤立背景事件。

![EV-SpSegNet 网络结构](imgs/framework.png)

EV-UAV 基准包含 147 段带事件级标注的序列。原论文报告的无人机目标平均尺寸约为 6.8 x 5.4 像素，属于极小目标检测场景。

---

## 环境配置

### 已验证的软件版本

| 组件 | 版本 |
| --- | --- |
| WSL | Ubuntu + NVIDIA GPU 支持 |
| Python | 3.9 |
| PyTorch | 1.9.1 + CUDA 11.1 (`cu111`) |
| torchvision | 0.10.1 + CUDA 11.1 (`cu111`) |
| 编译 HAIS_OP 使用的 CUDA Toolkit | CUDA 11.x，仅从源码编译时需要 |
| NumPy | `< 2` |

以下命令以当前项目路径为例：

```bash
export PROJECT_DIR=/mnt/d/AI/ESOD/EV-UAV-main
```

开始前确认 WSL 能识别显卡：

```bash
nvidia-smi
```

### 1. 下载 CUDA 版 PyTorch wheel

为避免 conda 自动安装 CPU 版 PyTorch，使用 Python 3.9 对应的 CUDA 11.1 wheel。下载中断后重复执行相同命令即可续传。

```bash
mkdir -p "$HOME/.cache/evuav-wheels"
cd "$HOME/.cache/evuav-wheels"

curl -fL -C - --retry 20 --retry-all-errors --connect-timeout 20 \
  -o torch-1.9.1+cu111-cp39-cp39-linux_x86_64.whl \
  "https://mirrors.aliyun.com/pytorch-wheels/cu111/torch-1.9.1%2Bcu111-cp39-cp39-linux_x86_64.whl"

curl -fL -C - --retry 20 --retry-all-errors --connect-timeout 20 \
  -o torchvision-0.10.1+cu111-cp39-cp39-linux_x86_64.whl \
  "https://mirrors.aliyun.com/pytorch-wheels/cu111/torchvision-0.10.1%2Bcu111-cp39-cp39-linux_x86_64.whl"
```

若 `curl` 长时间保持 `0 B/s`，通常是镜像暂时无响应。不要安装未下载完整的 `.whl` 文件。

### 2. 创建 conda 环境

```bash
conda create -n EV39 python=3.9 pip -y
conda activate EV39
```

如需完全重建同名环境：

```bash
conda env remove -n EV39 -y
```

### 3. 安装 Python 依赖

以下命令使用阿里云 PyPI 镜像安装除 PyTorch 外的依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ \
  numpy==1.23.5 pyyaml==6.0.2 tqdm==4.66.5 pandas==2.0.3 \
  opencv-python==4.8.1.78 mlflow==2.17.2 spconv-cu111 \
  typing-extensions==4.12.2 pillow==10.4.0
```

### 4. 安装 CUDA 版 PyTorch

```bash
python -m pip install --no-deps \
  "$HOME/.cache/evuav-wheels/torch-1.9.1+cu111-cp39-cp39-linux_x86_64.whl" \
  "$HOME/.cache/evuav-wheels/torchvision-0.10.1+cu111-cp39-cp39-linux_x86_64.whl"
```

验证 GPU PyTorch：

```bash
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

输出中必须包含 `cuda: 11.1` 和 `available: True`。

### 5. 配置 HAIS_OP

`HAIS_OP` 是项目使用的自定义 CUDA 扩展。`lib/hais_ops/build/` 是本地编译产物，被 `.gitignore` 排除，因此 GitHub 新克隆的仓库不包含预编译二进制文件。

已有 Python 3.9 预编译产物时，在每个新 WSL 终端执行：

```bash
conda activate EV39
export PROJECT_DIR=/mnt/d/AI/ESOD/EV-UAV-main
export PYTHONPATH=$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd $PROJECT_DIR
python -c "import torch; import spconv.pytorch; import HAIS_OP; print('HAIS_OP ok')"
```

全新克隆且没有预编译产物时，需要完整 CUDA Toolkit、兼容的 C++ 编译器及 `libsparsehash-dev`，随后从源码编译：

```bash
sudo apt update
sudo apt install -y build-essential libsparsehash-dev ninja-build

cd $PROJECT_DIR/lib/hais_ops
export CUDA_HOME=/path/to/your/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/usr/include:$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
python setup.py build_ext develop
python -c "import HAIS_OP; print('HAIS_OP ok')"
```

`nvcc --version` 必须能正常输出版本。若 CUDA 报错 `g++` 版本过高，请安装与当前 CUDA Toolkit 兼容的 GCC/G++，并通过 `CC`、`CXX`、`CUDAHOSTCXX` 显式指定。

---

## 数据集与权重

数据集、预训练权重、训练日志和实验结果均不会提交到本仓库，请通过原项目官方链接下载：

- EV-UAV 数据集：[百度网盘](https://pan.baidu.com/s/15pAlu3KP1uXych-c3SC5qA?pwd=sbr2)（提取码：`sbr2`）或 [Google Drive](https://drive.google.com/drive/folders/1VIkBFx5Po0KPIFBYOL_appLVie5wgdyi?usp=drive_link)
- EV-SpSegNet 预训练权重：[百度网盘](https://pan.baidu.com/s/1e6a_Ool5WZ3cBMPvoJvWbg?pwd=ztp4)（提取码：`ztp4`）或 [Google Drive](https://drive.google.com/file/d/1nNZsckiN0qp2oo1uX40tU6oz3mUcrSHq/view?usp=drive_link)

基础数据集目录应包含：

```text
dataset/EV-UAV-dataset/
|-- train/
|-- val/
`-- test/
```

赛道二官方数据包应包含：

```text
dataset/训练集、验证集/
|-- train/
|-- val/
`-- val_Challenge2.py
```

运行前，请确认 YAML 中的 `DATA.root`、`TRAIN.model_save_root` 和 `TEST.model_path` 都是本机的 WSL 路径。

---

## 基线训练与测试

### 官方预训练权重测试

```bash
python test.py --config configs/evisseg_evuav.yaml
```

本地一次已跑通的参考结果：

```text
iou: 0.5843424201011658
seg_acc: 0.6784908771514893
pd: 0.7846212700841622
fa: 8.493834145404406e-06
```

### 训练烟雾测试

该配置以 `max_events_num: 100000` 训练 1 个 epoch，只用于检查数据读取、CUDA 扩展、前向反向传播、权重保存和评估流程，不代表最终性能。

```bash
python train.py --config configs/evisseg_evuav_smoke.yaml
python test.py --config configs/evisseg_evuav_smoke.yaml
```

### 4GB 显存训练

```bash
python train.py --config configs/evisseg_evuav_4gb.yaml
python test.py --config configs/evisseg_evuav_4gb.yaml
```

每次执行 `train.py` 都会自动新建一个独立运行目录，不覆盖旧训练结果：

```text
log/baseline_4gb_seed37/runs/
`-- 20260723-143000_seed37_pid1234/
    |-- config.yaml
    |-- best_loss_seed37.pt
    |-- best_iou_seed37.pt       # 仅在第 40 个 epoch 后产生
    |-- best_score_seed37.pt     # 仅在启用 ROC 评估时产生
    |-- last_seed37.pt
    `-- run_summary.json
```

当 `TEST.roc: True` 时，训练从第 40 个 epoch 起会计算完整的 `IoU/Acc/Pd/Fa/Score`，并保存 `best_score_seed37.pt`。常规 50 epoch 比赛训练优先使用它；`best_iou_seed37.pt` 保留用于分析，1 epoch 烟雾测试仍使用 `best_loss_seed37.pt`。评估或提交前，将对应绝对 WSL 路径填入所用 YAML 的 `TEST.model_path`，例如：

```yaml
model_path: /mnt/d/AI/ESOD/EV-UAV-main/log/baseline_4gb_seed37/runs/20260723-143000_seed37_pid1234/best_score_seed37.pt
```

该配置在训练时最多随机采样 `100000` 个事件。降低 `max_events_num` 会改变训练点云密度和目标轨迹完整性，因而会影响 `IoU`、`Acc`、`Pd`、`Fa` 和比赛得分。它适合小显存开发和方案筛选，但不能与完整事件数基线直接等价比较。

### 从头训练完整基线

```bash
python train.py --config configs/evisseg_evuav_scratch.yaml
python test.py --config configs/evisseg_evuav_scratch.yaml
```

完整配置使用 `max_events_num: 700000`，显存需求较高。

---

## 赛道二验证与评分

### `test.py`、`test2.py` 与官方提交脚本的区别

| 脚本 | 使用的数据划分 | 作用 | 输出 |
| --- | --- | --- | --- |
| `test.py` | `test/` | 原始项目的测试集评估 | IoU、seg_acc、Pd、Fa |
| `test2.py` | 赛道二数据包中的 `val/` | 本地验证模型并计算赛道二总分 | IoU、Acc、Pd、Fa、Score_Fa、Score |
| `submit_challenge2.py` | 赛道二数据包中的 `val/` | 按官方格式生成比赛提交文件 | 每个视频一个 `val_xxx.txt` |

`test2.py` 不训练模型、不读取 `test/`、也不写提交文本。它加载 YAML 的 `TEST.model_path`，对验证集推理并使用真实标注计算本地指标。

### 运行赛道二验证

默认配置 [configs/evisseg_evuav_challenge2.yaml](configs/evisseg_evuav_challenge2.yaml) 指向：

- 验证集：`dataset/训练集、验证集/val/`
- 权重：`log/baseline_4gb_seed37/best_iou_seed37.pt`

### 赛道二运行前必须确认的配置

| 文件 | 项目 | 需要改成什么 | 是否每次实验都需要改 |
| --- | --- | --- | --- |
| `configs/evisseg_evuav_challenge2.yaml` | `DATA.root` | 赛道二数据包的父目录，例如 `/mnt/d/AI/ESOD/EV-UAV-main/dataset/训练集、验证集`。该目录下必须有 `val/`。 | 仅路径变化时修改。 |
| `configs/evisseg_evuav_challenge2.yaml` | `TEST.model_path` | 当前要评估或提交的权重绝对 WSL 路径；现有基线通常为 `best_iou_seed37.pt`，完成比赛得分验证改造后应比较并优先选择 `best_score_seed37.pt`。`test2.py` 和 `submit_challenge2.py` 都读取此项。 | 每次更换模型权重时修改。 |
| `configs/evisseg_evuav_challenge2.yaml` | `TEST.eval`、`TEST.roc` | 必须保持为 `True`，否则无法计算全部四项指标和总分。 | 不要修改。 |
| `configs/evisseg_evuav_challenge2.yaml` | `TEST.pd_detT`、`TEST.correct_thresh` | 保持比赛/基线评估使用的 `50` 与 `0.0001`。 | 不要为了提高本地分数而随意修改。 |
| `configs/evisseg_evuav_challenge2.yaml` | `TEST.prediction_threshold` | 导出 `txt` 前将概率转为二值标签的决策阈值。官方接收的是二值化后的 `txt`，因此它是合法的推理超参数；`test2.py` 与 `submit_challenge2.py` 必须使用同一个、经验证集搜索确定的值。 | 更换模型、集成或后处理后重新搜索；不要改动 `pd_detT`、`correct_thresh` 或评分公式。 |
| `configs/evisseg_evuav_challenge2.yaml` | `TEST.challenge_output_dir` | 运行 `submit_challenge2.py` 后保存 `val_xxx.txt` 的目录。 | 只在需要更改提交文件位置时修改。 |
| `configs/evisseg_evuav_challenge2.yaml` | `POSTPROCESS.p0_enabled` | P0 时空虚警簇过滤开关，默认 `false`。 | 仅在 P0 消融实验或确认后的融合实验中启用。 |
| `configs/evisseg_evuav_challenge2.yaml` | `ENSEMBLE.*` | E1 加权集成开关、第二权重路径和主权重比例；默认关闭。 | 仅在不重训的双模型融合实验中修改。 |
| 所有 `evisseg_evuav*.yaml` | `FUSION.p3_lite_*` | P3-Lite 事件帧与点云融合开关、时间片数和 2D 分支宽度；默认关闭。 | 仅在 P3-Lite 训练、验证和提交时保持一致地启用。 |
| `test2.py`、`submit_challenge2.py` | 权重、阈值、输出路径和 P0 设置 | 无需修改代码，统一从 YAML 与可选的 `--set` 覆盖读取。 | 不需要修改。 |

`TRAIN` 段的 `epochs`、`lr`、`max_events_num` 在 `test2.py` 推理时不会重新训练模型；它们保留在 YAML 中是为了记录该权重对应的训练设置。

```bash
conda activate EV39
export PROJECT_DIR=/mnt/d/AI/ESOD/EV-UAV-main
export PYTHONPATH=$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd $PROJECT_DIR
python test2.py --config configs/evisseg_evuav_challenge2.yaml
```

运行开始时，脚本会打印实际的验证集路径、视频数量、权重路径及预测阈值。若这些信息不符合预期，应停止并修正 YAML，而不是继续比较分数。

### 决策阈值搜索

官方提交文件中的最后一列是二值 `label`，而不是模型概率。因此 `TEST.prediction_threshold` 是合法的推理超参数，不是官方固定评测条件。它必须与模型、E1 融合比例和 P0 参数作为一个整体在验证集上搜索，并在生成提交文本时原样复用。

`sweep_thresholds.py` 会只运行一次模型前向推理，将验证集分数缓存到 CPU 内存，再逐个重算 P0、IoU、Acc、Pd、Fa 和总分。相比多次运行 `test2.py`，它不会重复加载模型或重复执行 GPU 推理。

未启用 P0c、P6、P8 的基础 E1 配置使用 100 epoch P1b 主模型和历史 HNM 次模型，在固定 `0.9` 阈值下的已知 Score 为 `0.8195608435`。先执行以下粗搜索：

```bash
PRIMARY=/mnt/d/AI/ESOD/EV-UAV-main/log/e1v2_p1b_cosine100_4gb_seed37/runs/20260724-182346_seed37_pid544/best_score_seed37.pt
SECONDARY=/mnt/d/AI/ESOD/EV-UAV-main/log/p1b_hnm_p0_4gb_seed37/runs/80score_seed37_pid1044/best_score_seed37.pt

python sweep_thresholds.py --config configs/evisseg_evuav_challenge2.yaml \
  --set TEST.model_path="$PRIMARY" \
  ENSEMBLE.ensemble_enabled=true \
  ENSEMBLE.ensemble_secondary_model_path="$SECONDARY" \
  ENSEMBLE.ensemble_primary_weight=0.895 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1
```

脚本会按 Score 排序并打印 `best threshold`。若最佳值位于相邻两个候选之间，再固定其余参数、只在该小区间细搜，例如：

```bash
python sweep_thresholds.py --config configs/evisseg_evuav_challenge2.yaml \
  --set TEST.model_path="$PRIMARY" \
  ENSEMBLE.ensemble_enabled=true \
  ENSEMBLE.ensemble_secondary_model_path="$SECONDARY" \
  ENSEMBLE.ensemble_primary_weight=0.895 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1 \
  SWEEP.thresholds='[0.875, 0.880, 0.885, 0.890, 0.895, 0.900, 0.905]'
```

最终验证和导出必须使用同一个结果，例如最佳阈值为 `0.885`：

```bash
BEST_THRESHOLD=0.885
python test2.py --config configs/evisseg_evuav_challenge2.yaml \
  --set TEST.model_path="$PRIMARY" \
  TEST.prediction_threshold="$BEST_THRESHOLD" \
  ENSEMBLE.ensemble_enabled=true \
  ENSEMBLE.ensemble_secondary_model_path="$SECONDARY" \
  ENSEMBLE.ensemble_primary_weight=0.895 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1
```

### 可选优化模块与命令覆盖

所有优化模块在 YAML 中默认关闭，基线行为不会被改变。项目支持通用的命令行覆盖格式：

```bash
--set SECTION.KEY=VALUE
```

同一个命令可设置多个值，例如：

```bash
--set POSTPROCESS.p0_enabled=true POSTPROCESS.p0_min_cluster_events=3
```

该覆盖机制会校验 section 和 key 是否存在；拼写错误会立即报错。训练时使用的覆盖值会写入该次运行目录的 `config.yaml` 和 `run_summary.json`，便于后续消融与融合实验追溯。

P0 属于推理后处理，不参与反向传播。训练时启用 P0 不会改变梯度或 loss，但会应用于第 40 个 epoch 之后的验证推理，因此 `best_score_seed37.pt` 与最终提交流程一致。`test2.py` 和生成提交文本时必须使用同一个开关。

### P0：时空虚警簇过滤

P0 使用当前配置的决策阈值识别候选正事件。它只会把被判定为过小簇的正事件分数置为 `0`，不会重排事件，也不会修改保留事件的原始置信度。阈值变化会改变进入 P0 的候选簇，因此阈值与 P0 参数必须作为一组在验证集上评估。

默认参数如下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `p0_enabled` | `false` | 是否启用 P0。 |
| `p0_spatial_radius` | `1` | 同一时间邻域内，`x/y` 相差不超过 1 的事件可连接。 |
| `p0_temporal_bin_size` | `50` | 时间分箱宽度，与官方 `pd_detT` 一致。 |
| `p0_temporal_radius_bins` | `1` | 相邻时间箱的簇可连接。 |
| `p0_min_cluster_events` | `2` | 初始实验仅过滤单事件簇。 |
| `p0_min_duration_bins` | `1` | 初始实验不额外要求持续时间。 |

基线评估：

```bash
python test2.py --config configs/evisseg_evuav_challenge2.yaml
```

启用 P0 的第一轮评估：

```bash
python test2.py --config configs/evisseg_evuav_challenge2.yaml \
  --set POSTPROCESS.p0_enabled=true
```

P0 参数必须随候选模型单独验证，不能把某个模型上的最优参数直接迁移到另一个模型。早期 4GB 基线使用默认 `P0(2/1)` 时，`Score` 从 `0.6927527893` 提升至 `0.7099144857`；50 epoch P1b 单模型在 `P0(3/1)` 下达到 `0.7888286899`。早期 E1 融合比较过 `P0(2/1)`、`P0(4/1)`、`P0(3/2)`，其中 `P0(3/1)` 的结果为 `0.8054258892`。100 epoch 主模型 + HNM 次模型的基础 E1 配置沿用 `P0(3/1)`，总分为 `0.8195608435`；当前可复现候选在此基础上增加了 P0c、P6 和 P8，见[当前可复现最优提交候选](#当前可复现最优提交候选)。其中 `3/1` 表示 `p0_min_cluster_events=3`、`p0_min_duration_bins=1`；更换模型、融合比例或阈值后仍应重新验证。

训练时按最终 P0 流程选择 `best_score`：

```bash
python train.py --config configs/evisseg_evuav_4gb.yaml \
  --set POSTPROCESS.p0_enabled=true
```

生成提交文本时必须使用与本地验证完全相同的开关：

```bash
python submit_challenge2.py --config configs/evisseg_evuav_challenge2.yaml \
  --set POSTPROCESS.p0_enabled=true
```

### P1：背景难例抑制损失

P1 仅在训练阶段生效。它保留原始 STC Loss，并从每个训练批次的背景事件中选出预测分数最高的部分，附加一个目标为 `0` 的 BCE 项。这样会直接压低最可能形成虚警的背景预测，但不修改官方阈值 `0.9`、验证逻辑或提交文件格式。

为避免训练早期过度压制前景，P1 默认前 10 个 epoch 只使用原始 STC Loss；之后才开始加入背景难例项。默认参数如下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `p1_hard_negative_enabled` | `false` | 是否启用 P1。 |
| `p1_hard_negative_weight` | `0.02` | 背景难例 BCE 在总损失中的权重。 |
| `p1_hard_negative_ratio` | `0.01` | 每批背景事件中按预测分数选取的最高比例，即最高 1%。 |
| `p1_hard_negative_warmup_epochs` | `10` | 仅使用原始 STC Loss 的预热 epoch 数。 |

P1 首轮消融应关闭 P0，并为结果使用独立目录：

```bash
python train.py --config configs/evisseg_evuav_4gb.yaml \
  --set LOSS.p1_hard_negative_enabled=true \
  TRAIN.model_save_root=/mnt/d/AI/ESOD/EV-UAV-main/log/p1_hnm_4gb_seed37
```

训练日志会额外记录 `p1_hard_negative_loss` 和 `p1_hard_negative_count`。评估时不需要再次设置 P1，只需把本次运行生成的 `best_score_seed37.pt` 传给 `test2.py`：

```bash
python test2.py --config configs/evisseg_evuav_challenge2.yaml \
  --set TEST.model_path=/实际路径/best_score_seed37.pt
```

只有 P1 单独提高 `Score` 后，才与当前固定参数的 P0 叠加。后续 P2、P3 模块也遵循“YAML 默认关闭、`--set` 显式启用、独立运行快照记录”的同一规范。

### P1b：目标保持训练采样

当前 4GB 配置将每段训练样本限制为 `100000` 个事件。对 99 段训练数据的统计表明，其中 15 段超过该上限；均匀随机采样预计只能保留 `80.69%` 的正事件，而每段正事件最多只有 `15399` 个，均可完整放入 10 万事件预算。

P1b 因此只在训练阶段、且事件数达到上限时保留全部正标注事件，再从背景事件中随机补足剩余预算。验证、测试和提交从不使用标签采样，也不改变事件总预算、官方阈值或评测实现。它的目标是避免极小目标的稀疏轨迹在训练输入中被随机打断，优先提升 `Pd`、`IoU` 和 `Acc`。

首轮 P1b 消融保持 P0 和 P1 背景难例损失关闭：

```bash
python train.py --config configs/evisseg_evuav_4gb.yaml \
  --set SAMPLING.target_preserving_enabled=true \
  TRAIN.model_save_root=/mnt/d/AI/ESOD/EV-UAV-main/log/p1b_target_sampling_4gb_seed37
```

训练启动时应显示：

```text
training event sampling: target-preserving
P1 background hard-negative loss: disabled
validation P0 cluster filter: disabled
```

完成后，先以 P0 关闭的方式评估 `best_score_seed37.pt`，只与纯基线的 `Score` 比较。P1b 独立有效后，再依次测试 `P1b + P0`，最后才考虑叠加背景难例损失。

### 已完成训练结论

P1b 目标保持采样 + P0(3/1) 在首轮 50 epoch 训练中达到 `0.7888286899`，是确认 P1b 有效的关键结果，但已不是当前最好的单模型。相同采样策略在 100 epoch、余弦学习率调度下的主模型达到 `0.8138151868`；其后与 HNM 次模型集成，形成基础 E1 配置的 `0.8195608435`。当前可复现提交候选详见[当前可复现最优提交候选](#当前可复现最优提交候选)。

HNM 按首轮参数（`weight=0.02`、`ratio=0.01`）与 P1b、P0(3/1) 叠加后，作为单模型会明显压低 Pd 和 Acc，单模型总分为 `0.7677599395`。不过它与 P1b 主模型在 Pd、IoU 与 Fa 上具有互补性，因此保留为 E1 的次模型，而不单独作为提交模型。以下是早期 50 epoch 结果：

| 方案 | IoU | Acc | Pd | Fa | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1b 单模型 + P0(3/1) | 0.5067 | 0.7831 | 0.8097 | `5.03e-6` | 0.7888286899 |
| P1b + HNM 融合，主权重 0.895，P0(3/1) | 0.6408 | 0.7643 | 0.7814 | `3.99e-6` | 0.8054258892 |

### E1：历史基础集成对照

E1 是一个默认关闭的推理模块，不改变训练、标签或评分公式。它将 `TEST.model_path` 指定的主模型与 `ENSEMBLE.ensemble_secondary_model_path` 指定的次模型逐事件做概率加权：

```text
融合分数 = 主模型分数 x ensemble_primary_weight
         + 次模型分数 x (1 - ensemble_primary_weight)
```

融合分数随后使用配置中的决策阈值，再执行 P0。因此验证和生成提交文本的逻辑完全一致。下面的命令记录的是**历史基础 E1 对照**：主模型为 100 epoch P1b 权重，次模型为历史 HNM 权重；在固定阈值 `0.9`、主权重 `0.895` 和 P0(3/1) 下，Score 为 `0.8195608435`。它不是当前 `0.8618022242` 冻结候选；当前候选的次模型是 P15 权重，并额外启用 P0c、P6、P8 和 P14，必须使用本 README 顶部步骤 3/4 的完整命令。主/次权重、阈值和 P0 参数应在同一验证流程中有限搜索，不应无边界地反复拟合该验证集。

```bash
PRIMARY=/mnt/d/AI/ESOD/EV-UAV-main/log/e1v2_p1b_cosine100_4gb_seed37/runs/20260724-182346_seed37_pid544/best_score_seed37.pt
SECONDARY=/mnt/d/AI/ESOD/EV-UAV-main/log/p1b_hnm_p0_4gb_seed37/runs/80score_seed37_pid1044/best_score_seed37.pt

python test2.py --config configs/evisseg_evuav_challenge2.yaml \
  --set TEST.model_path="$PRIMARY" \
  TEST.prediction_threshold=0.900 \
  ENSEMBLE.ensemble_enabled=true \
  ENSEMBLE.ensemble_secondary_model_path="$SECONDARY" \
  ENSEMBLE.ensemble_primary_weight=0.895 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1
```

E1 已在当前验证集上超过单模型方案。生成提交文本时必须使用**完全相同**的主/次权重、融合比例和 P0 参数：

```bash
python submit_challenge2.py --config configs/evisseg_evuav_challenge2.yaml \
  --set TEST.model_path="$PRIMARY" \
  TEST.prediction_threshold=0.900 \
  ENSEMBLE.ensemble_enabled=true \
  ENSEMBLE.ensemble_secondary_model_path="$SECONDARY" \
  ENSEMBLE.ensemble_primary_weight=0.895 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1
```

### P2a：稀疏目标 STC 补偿

P2a 是已完成首轮验证的训练损失实验，直接针对小目标、快速目标可能在固定 STC 邻域中局部支持不足的问题。原始 STC 对正样本的损失权重为 w_stc；当某个真实目标事件的预测邻域很弱时，该权重会很低，导致该正样本几乎没有梯度。

启用 P2a 后，仅将正样本权重改为：

```text
w_pos = max(w_stc, p2_positive_stc_floor)
w_neg = 1 - w_stc
```

因此它只补偿低支持的真实目标事件，不改变背景项、模型结构、事件预算、官方阈值、验证脚本或提交文本。默认关闭，关闭时与现有 STC Loss 完全一致。首轮使用 floor=0.35，以控制召回提升与 Fa 上升之间的风险。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `p2_positive_stc_floor_enabled` | `false` | 是否启用 P2a；默认不改变基线。 |
| `p2_positive_stc_floor` | `0.35` | 低支持正样本的最小 STC 损失权重，范围为 `[0, 1]`。 |

P2a 仅在训练阶段生效。训练日志会记录 `p2_boosted_positive_count`，用于确认每个批次实际被补偿的低支持正样本数。首轮只与已经验证有效的 P1b 和 P0(3/1) 组合，不叠加 P1 背景难例损失、P3-Lite 或新的后处理：

```bash
python train.py --config configs/evisseg_evuav_4gb.yaml \
  --set SAMPLING.target_preserving_enabled=true \
  LOSS.p2_positive_stc_floor_enabled=true \
  LOSS.p2_positive_stc_floor=0.35 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1 \
  TRAIN.model_save_root=/mnt/d/AI/ESOD/EV-UAV-main/log/p2a_positive_stc_floor_4gb_seed37
```

完成后，用生成运行目录内的 `best_score_seed37.pt` 通过普通 `test2.py` 验证即可。P2a 是训练期损失，验证命令不需要再设置任何 P2 参数。首轮淘汰标准是：若单模型 P1b + P2a + P0(3/1) 的 Score 未明确高于 `0.7888286899`，或 Pd/IoU 没有改善，则不继续调 floor，转向密度条件化 GDSCA。

### P4：目标时间帧检测辅助损失

P4 是当前优先尝试的结构性训练改进。官方 `Pd` 按“目标 ID x 50 时间单位”的目标帧统计：只要该目标帧中足够比例的正事件被输出为前景，就记为一次命中。训练集统计表明，官方 `correct_thresh=0.0001` 下每个目标帧最多只需一个事件越过决策阈值；原始 STC Loss 是逐事件损失，不能直接表达这个评分目标。

P4 在保留原始 STC Loss 的前提下，对每个真实目标帧取最高预测分数，并对低于当前 `TEST.prediction_threshold` 的部分施加 hinge 损失。它不修改官方评分、验证逻辑或提交格式；时间恰好落在 `0/50/100...` 边界的事件会与官方 `Pd` 实现一致地忽略。默认关闭。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `p4_target_frame_enabled` | `false` | 是否启用 P4；关闭时训练与既有 STC Loss 完全一致。 |
| `p4_target_frame_weight` | `0.05` | P4 辅助损失权重。 |
| `p4_target_frame_warmup_epochs` | `10` | 前 10 个 epoch 仅训练原始 STC Loss，之后启用 P4。 |

先在 `EV39` 中完成 P4 单元测试，再执行训练。P4 首轮只叠加已验证有效的 P1b 和 P0(3/1)，不与 HNM、P2a、P3-Lite 或额外后处理同时训练：

```bash
python -m unittest discover -s tests -p 'test_target_frame_loss.py'

BEST_THRESHOLD=0.900  # 先替换为 sweep_thresholds.py 搜到的最佳值
python train.py --config configs/evisseg_evuav_4gb.yaml \
  --set SAMPLING.target_preserving_enabled=true \
  LOSS.p4_target_frame_enabled=true \
  LOSS.p4_target_frame_weight=0.05 \
  TEST.prediction_threshold="$BEST_THRESHOLD" \
  TRAIN.epochs=100 \
  TRAIN.scheduler=cosine \
  TRAIN.scheduler_min_lr=0.00001 \
  TRAIN.validation_start_epoch=30 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1 \
  TRAIN.model_save_root=/mnt/d/AI/ESOD/EV-UAV-main/log/p4_target_frame_cosine100_4gb_seed37
```

P4 是否保留只看完整验证 Score。它首先应超过同样阈值下 P1b 单模型的 Score；只有单模型有明确正收益，才与冻结 HNM 次模型做 E1 复验。

### P3-Lite：多时间片事件帧与点云融合

P3-Lite 是已经完成首轮完整消融的表征升级模块，目标是直接提高 `Pd`、`IoU` 和 `Acc`，而不是继续微调 Fa。现有 3D 稀疏点云分支保持不变；新增的轻量 2D 分支将完整事件流划分为 4 个时间片，并分别累计正、负极性事件，形成 8 通道事件帧。两层窄通道 CNN 提取每个像素的局部上下文后，按稀疏体素的 `x/y` 坐标采样并门控残差融合到 3D 网络首层。

事件帧只由坐标、时间和极性构造，不读取标签。训练集在 3D 分支进行 10 万事件采样前先构造帧，因此 2D 分支不会因采样截断完整时空观测。默认配置如下：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `p3_lite_enabled` | `false` | 是否启用 P3-Lite；默认不改变基线。 |
| `p3_lite_temporal_bins` | `4` | 时间片数量，事件帧通道数为 `2 x temporal_bins`。 |
| `p3_lite_width` | `8` | 2D 编码器通道宽度，针对 4GB 显存设计。 |

P3-Lite 已通过 1 epoch 烟雾训练、Challenge 2 推理和旧 E1 回归验证。首轮完整消融只叠加已验证的目标保持采样与 P0(3/1)，不叠加 HNM：

首轮完整训练已在 50 个 epoch 后结束，最佳验证结果出现在 epoch 41：IoU=0.433076、Acc=0.786294、Pd=0.772575、Fa=4.162594e-06、Score=0.762043。该分数低于 P1b + P0(3/1) 单模型的 0.7888286899，因此当前这版 P3-Lite 不进入提交候选，也不再与 HNM 或 E1 继续叠加。代码保留为默认关闭的负结果消融，未来只有重新设计融合位置或事件帧表征后才重新启动 P3。

```bash
python train.py --config configs/evisseg_evuav_4gb.yaml \
  --set SAMPLING.target_preserving_enabled=true \
  FUSION.p3_lite_enabled=true \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1 \
  TRAIN.model_save_root=/mnt/d/AI/ESOD/EV-UAV-main/log/p3_lite_p1b_4gb_seed37
```

训练结束后，先用同一架构配置验证 P3-Lite 单模型：

```bash
RUN=/mnt/d/AI/ESOD/EV-UAV-main/log/p3_lite_p1b_4gb_seed37/runs/实际运行目录
python test2.py --config configs/evisseg_evuav_challenge2.yaml \
  --set FUSION.p3_lite_enabled=true \
  TEST.model_path="$RUN/best_score_seed37.pt" \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1
```

P3-Lite 权重不能直接与旧 P1b/HNM 权重做 E1 融合，因为模型参数结构不同。由于首轮单模型无效，当前不训练同架构的互补次模型。

### 得分细则

赛道二按以下公式计算。提交文本中的 `label` 已是二值标签，模型概率到二值标签的决策阈值由参赛者在验证集上选择：

```text
IoU      = |GT ∩ Pred| / |GT ∪ Pred|
Acc      = TP_target / (TP_target + FN_target)
Pd       = TD / AT
Fa       = FD / (N_frame x H x W)

Score_Fa = exp(-10000 x Fa)
Score    = 0.4 x Pd + 0.3 x Score_Fa + 0.2 x IoU + 0.1 x Acc
```

| 指标 | 含义 | 对总分的影响 |
| --- | --- | --- |
| `IoU` | 前景事件预测与真实前景事件的交并比 | 权重 `0.2` |
| `Acc` | 真实目标事件被正确预测为前景的比例 | 权重 `0.1` |
| `Pd` | 被成功检测的目标数占所有目标数的比例 | 权重 `0.4` |
| `Fa` | 每帧、每像素归一化的虚警目标数 | 经指数惩罚后权重 `0.3` |
| `Score_Fa` | `exp(-10000 x Fa)` | `Fa` 越小越接近 `1` |

`Fa` 的指数惩罚很敏感。例如 `Fa` 每增加 `1e-5`，`Score_Fa` 会额外乘以约 `exp(-0.1)`，约为 `0.905`。因此比赛优化不能只看 IoU，也必须控制虚警率。

`test2.py` 使用项目原有 `utils/eval.py` 中的 `Pd/Fa` 实现，分辨率为 `346 x 260`，时间帧宽度和目标判定阈值由 YAML 的 `pd_detT` 与 `correct_thresh` 控制。

### 生成赛道二提交文本

`submit_challenge2.py` 按官方 `val_Challenge2.py` 的输出格式生成提交文件。它不会训练模型，并且已纳入仓库，不依赖数据包内被忽略的脚本。

先在 `configs/evisseg_evuav_challenge2.yaml` 中确认 `TEST.model_path`、`TEST.prediction_threshold` 和 `TEST.challenge_output_dir`。不需要修改 Python 脚本顶部的路径。

然后运行：

```bash
python submit_challenge2.py --config configs/evisseg_evuav_challenge2.yaml
```

提交目录为：

```text
log/challenge2/val-pred-txt/
`-- val_xxx.txt
```

不要将 `test2.py` 的本地验证总分与 `submit_challenge2.py` 混淆：前者用于评估，后者用于导出提交文本。

---

## 改进方向

### 固定评测条件

官方评分公式、`pd_detT: 50` 与 `correct_thresh: 0.0001` 不是优化变量。相反，`TEST.prediction_threshold` 是将模型概率写入二值 `txt` 标签前的合法决策阈值，应针对每个固定的模型、集成和后处理组合在验证集上有限搜索，并在提交时原样复用。

按当前已经得到的两组结果计算，官方预训练权重总分约为 `0.7741`，100000 事件的 4GB 基线约为 `0.6109`。两者约 `0.1633` 的差距中，`Fa` 项的贡献差距约为 `0.1274`，占总差距约 `78%`。这不能证明某个方案必然有效，但说明第一轮应优先抑制虚警，而不是先扩大网络。

### 可行性结论

| 优先级 | 所属方向 | 具体改进点 | 可行性判断 | 推荐的第一版 | 暂不采用的做法 |
| --- | --- | --- | --- | --- | --- |
| P4 | 损失函数 / 评分对齐 | 目标时间帧检测辅助损失 | **当前优先实验。** 原始 STC 是逐事件损失，而 `Pd` 只要求每个目标时间帧至少出现足量前景事件。训练数据中该条件等价于至少一个事件跨过决策阈值。 | 以 P1b 为基础，在 warmup 后对每个目标帧最高分加入 hinge 损失；阈值与当前验证最优提交阈值保持一致。 | 首轮不叠加 HNM、P2a、P3-Lite 或其他新损失。 |
| P2a | 损失函数 | 稀疏目标 STC 补偿 | **首轮未作为当前主线。** 固定 STC 邻域可能漏掉小目标和快速短轨迹，但现有结果没有证明 floor=0.35 能稳定提高 Score。 | 保留实现和实验记录；仅在 P4 结论明确后再考虑独立重启。 | 不同时叠加 HNM、Focal、Tversky 或额外后处理。 |
| P0 | 后处理 | 时空虚警簇过滤 | **最值得先试。** `Fa` 按虚警连通区域计数并指数惩罚，时空孤立预测与失分直接相关。 | 在当前决策阈值后，只删除事件数过少的时空孤立簇；验证有效后，再单独加入最短持续时间条件。 | 第一版不补全断轨迹，因为新增前景可能提高 `Fa`；不同时调整多个过滤参数。 |
| P1 | 损失函数 | 背景难例抑制 | **当前首轮配方无效。** `weight=0.02`、`ratio=0.01` 与 P1b、P0 叠加后损失了过多 Pd/Acc。 | 暂停同配方重训；若后续重启，只测试更弱的高置信背景约束。 | 不直接把现有 HNM 权重作为最终模型，也不叠加 Focal、Tversky 或类别平衡。 |
| P1b | 训练采样策略 | 目标保持采样 | **已验证有效，当前主方案。** 15 段训练视频超过 10 万事件上限，均匀采样预计丢失约 `19.31%` 正事件。 | 仅在训练集保留所有可容纳的正事件，背景事件随机补足，保持总预算不变；与 P0(3/1) 组合。 | 验证和测试阶段禁止标签采样；时间分层背景采样留待独立消融后再加入。 |
| E1 | 推理集成 | 主模型与互补次模型加权 | **已验证有效。** 历史 HNM 次模型构成基础 E1 对照；当前冻结候选改用 P15 次模型，并叠加 P0c、P6、P8、P14，详见 README 顶部。 | 在 P0 前平均概率；固定模型和后处理后有限搜索主权重与决策阈值。 | 不按单项指标选权重，不在未知测试集上试错。 |
| P2b | 损失函数 | 多尺度或自适应 STC 邻域 | **P2a 无效后再考虑。** 固定 `k=3, t=5` 对不同速度和事件密度的目标适应性有限，但额外稀疏卷积会增加显存与变量数。 | 先完成 P2a；只有其结果明确后，才在少量离散邻域之间进行轻量门控。 | 不直接预测任意大小，也不与新的背景难例损失同时首次引入。 |
| P2 | 特征提取 | 密度条件化 GDSCA | **可以尝试，但预期成本高于前述方向。** 当前四个分支的空洞率固定为 `[1, 2, 3, 4]`。 | 保留现有卷积分支，只根据局部事件密度学习轻量分支权重。 | 暂不实现自由偏移的可变形稀疏卷积，其自定义算子、显存和兼容风险较高。 |
| P3-Lite | 多表示融合 | 点云与多时间片事件帧融合 | **首轮完整消融无效。** Score=0.762043，低于 P1b + P0(3/1) 单模型。 | 保留默认关闭实现和负结果记录，不作为当前提交候选。 | 不再为当前结构叠加 HNM 或 E1；不把失败分支当作互补模型。 |

### 实验前提

`train.py` 从第 40 个 epoch 起会计算验证集 `Pd/Fa/IoU/Acc/Score`，并保存 `best_score`。这只是正确选择权重的实验基础，不算新的算法贡献，也不能和某个改进方向混为一次实验。

后处理需要由 `test2.py` 和 `submit_challenge2.py` 调用同一个实现，并使用相同的决策阈值，从而保证本地验证结果与最终提交文本一致。

### 推荐实施顺序

1. **评测准备**：先固定模型、集成和后处理，使用 `sweep_thresholds.py` 搜索决策阈值，再按完整 `Score` 保存和比较 `best_score` 权重。
2. **P0 后处理实验**：只增加“时空簇最小事件数”过滤，与无后处理基线比较；通过后再单独测试持续时间过滤。
3. **P1b 训练采样实验**：已验证有效；当前以 P1b + P0(3/1) 作为冻结主方案。
4. **E1 推理集成实验**：已验证有效；基础对照使用主权重 `0.895`、次权重 `0.105` 和 P0(3/1)。当前最终提交候选还应使用顶部冻结方案中的 P15 次模型、P0c、P6、P8 和 P14。
5. **P4 目标时间帧检测损失**：当前优先训练实验；只与 P1b、P0(3/1) 组合，先比较单模型 Score、Pd、IoU 和 Acc。
6. **P2a 稀疏目标 STC 补偿**：仅在 P4 结论明确后再考虑独立重启，不与 P4 首轮叠加。
7. **密度条件化 GDSCA**：仅在损失方向结论明确后实施；它是下一项核心特征提取候选，每次只改变分支权重逻辑。
8. **P1 背景难例损失修订 / P3-Lite 重设计**：当前首轮配方或结构无效，只有提出可验证的新假设后才重新启动。

每个方向必须先单独与原始基线比较。只有单项在 `Score` 上稳定提升，才能加入当前最优组合；叠加后必须重新训练和评估，不能假设两个单项收益可以直接相加。

---

## 实验规范

1. 先冻结一个可复现基线；每个消融实验新增独立 YAML，并且一次只改变一个变量。
2. 不要先把多个改进全部叠加。单项改进只有在独立实验中通过后，才能加入当前最优组合；每次叠加后重新评估，因为不同改进可能互相抵消。
3. 每次实验记录配置、随机种子、Git commit、事件上限、显存占用、训练时间、推理时间，以及 `IoU/Acc/Pd/Fa/Score_Fa/Score` 全部指标。
4. 100000 事件和固定 seed 只用于第一轮快速筛选。候选方案至少使用多个随机种子复核，并最终在目标事件预算或完整 700000 事件配置下重新训练验证。
5. 选择比赛模型时按验证 `Score` 保存和比较 `best_score`，同时保留 `best_iou`；不要按训练 loss 或单独的 IoU 决定最终提交模型。
6. `TEST.prediction_threshold` 是合法推理超参数，但每次仅在模型、集成和后处理固定后有限搜索；调后处理时固定模型、损失和采样，调模型时关闭后处理或固定为已经确认的版本。
7. 不要将 1 epoch 烟雾测试结果视为最终模型性能，也不要把在完整验证集上反复调参得到的最高分当作对未知测试集的可靠估计。

---

## 引用与致谢

使用 EV-UAV 数据集或 EV-SpSegNet 时，请引用原始论文：

```bibtex
@misc{chen2025eventbasedtinyobjectdetection,
  title={Event-based Tiny Object Detection: A Benchmark Dataset and Baseline},
  author={Nuo Chen and Chao Xiao and Yimian Dai and Shiman He and Miao Li and Wei An},
  year={2025},
  eprint={2506.23575},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.23575}
}
```

原始实现基于 [HAIS](https://github.com/hustvl/HAIS) 和 [spconv](https://github.com/traveller59/spconv)。使用本仓库时，请同时遵守原项目、数据集、预训练权重、HAIS 和 spconv 的许可证与使用条款。
