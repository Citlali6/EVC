# EVm4+m5 - Bidirectional Temporal Memory for Event-based Tiny Object Detection

基于 ICCV 2025 [*Event-based Tiny Object Detection*](https://arxiv.org/abs/2506.23575)
官方 P23 全事件流时序帧方案的改进版。

在 EV-UAV Challenge 2 的 `val/` 验证集（24 个视频）上，当前最高本地结果为
**Score 0.95959**。最终方案以 M13 高密度连续序列重采样模型为主模型，在低密度输入上
回退至 M10，并使用固定的 P0/P0c 与低密度 P18 轨迹恢复。所有路由条件只依赖输入的
事件数，不读取验证标签、视频名称或目标 ID。

## 结果总览

| 模型 | 阈值 | Score | Pd | IoU | Acc | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P23 baseline（官方复现） | 0.600 | 0.93820 | 0.94939 | 0.90670 | 0.95210 | 6.22e-06 |
| M4 + DACC | 0.700 | 0.94822 | 0.96073 | 0.91887 | 0.96127 | 5.47e-06 |
| M4 + DACC + M5 | 0.700 | 0.94965 | 0.97291 | 0.91711 | 0.96736 | 6.78e-06 |
| M4+DACC+M5 + 调优 P0（扫描最优） | 0.700 | 0.95444 | 0.96661 | 0.92852 | 0.96568 | 4.94e-06 |
| M6 短程微调 + 低密度 P18 | 0.700 | 0.95780 | 0.97648 | 0.93092 | 0.97721 | 5.73e-06 |
| M9 普通短程微调 + 低密度 P18 | 0.700 | 0.95793 | 0.97433 | 0.93313 | 0.97630 | 5.24e-06 |
| M10 高密度重采样（4 views）+ P18 | 0.700 | 0.95844 | 0.97690 | 0.93291 | 0.97709 | 5.70e-06 |
| M11 高密度重采样（6 views）+ P18 | 0.700 | 0.95875 | 0.97438 | 0.93475 | 0.97454 | 5.27e-06 |
| M11 + M10 低密度路由 + P18 | 0.700 | 0.95929 | 0.97648 | 0.93476 | 0.97551 | 5.41e-06 |
| M13 高密度重采样（8 views）+ P18 | 0.700 | 0.95891 | 0.97417 | 0.93591 | 0.97638 | 5.33e-06 |
| **M13 + M10 低密度路由 + P18（当前最佳）** | **0.700** | **0.95959** | **0.97648** | **0.93610** | **0.97745** | **5.47e-06** |

评分使用官方 Challenge 2 公式：

```text
Score_Fa = exp(-10000 * Fa)
Score = 0.4 * Pd + 0.3 * Score_Fa + 0.2 * IoU + 0.1 * Acc
```

> 本文全部为本地 `val/` 结果。未知官方测试集的实际得分仍应以排行榜为准。

## 当前最佳：M13 高密度视图重采样 + M10 低密度路由 + P18

### 方法概述

1. **M4 双向时序记忆**：在 P23 U-Net bottleneck 上运行正向和反向 ConvGRU，聚合跨时间箱
   的目标证据。残差投影零初始化，因此能安全继承 P23/M5 权重。
2. **DACC 密度自适应通道校准**：从输入事件计数估计全局密度，对 decoder 特征做通道级
   校准，减轻低、高密度视频的分布差异。
3. **M5 轨迹外推损失**：仅训练期按 `target_id` 拟合已观测轨迹，对未观测时间箱的预测
   施加弱正约束；推理不读取标签或轨迹。
4. **M13 高密度视图重采样**：99 个训练视频中，15 个 `event_count > 200000` 的视频每轮
   从 2 个连续序列视图增至 8 个（倍率 4）；每轮训练序列数由 198 增至 288。该改动只影响
   训练采样，推理端没有使用训练视频名称或标签。
5. **M10 低密度路由**：推理时 `event_count <= 30000` 使用 M10 第 2 轮权重，其他视频使用
   M13 第 4 轮权重。这个分界与既有低密度 P18 的适用范围一致，且只从未标注输入本身计算。
6. **P0/P0c + 低密度 P18**：P0 过滤短暂的孤立时空噪声簇；P0c 恢复高置信小簇；P18 仅在
   `1 < event_count <= 30000` 时从有种子支持的短轨迹中恢复一个弱事件。

### M13 Checkpoint 选择

M13 训练的四个 checkpoint 均按同一固定推理协议完整验证。训练 loss 最低不等于 Challenge 2
最高分，最终必须使用第 4 轮权重：

| checkpoint | 单模型 Score |
| --- | ---: |
| `epoch_001_seed42.pt` | 0.9550648532 |
| `epoch_002_seed42.pt` | 0.9578435518 |
| `epoch_003_seed42.pt` | 0.9584171887 |
| `epoch_004_seed42.pt` | **0.9589086755** |
| `epoch_004_seed42.pt` + M10 低密度路由 | **0.9595899017** |

M12 的两阶段低学习率续训最高为 `0.9576690693`，M10/M11 等权分数融合为
`0.9585171106`，均未采用。M13 已超过此前最佳，本轮不再继续 M14。

### 已固定结果与权重

```text
primary checkpoint (event_count > 30000)
  checkpoints/m13_dense_views4_epoch_004_seed42.pt

secondary checkpoint (event_count <= 30000)
  checkpoints/m10_dense_views2_epoch_002_seed42.pt

local val metrics
  Score = 0.9595899017
  Pd    = 0.9764804704
  IoU   = 0.9360964894
  Acc   = 0.9774524570
  Fa    = 5.4691447798e-06

submission archive
  log/challenge2/m13e4_m10low30000_p18low30000_f055.zip
```

提交 ZIP 是一次本地运行产物，因此不随仓库发布；按下文命令可重新生成。

### 免训练评估（复现 0.9595899017）

```bash
conda activate EV39
export PROJECT_DIR=/absolute/path/to/EVC
export DATA_ROOT=/absolute/path/to/dataset/训练集、验证集
export CUDA_HOME=/usr/local/cuda-11.1  # 改为本机 CUDA Toolkit 的实际路径
export PATH="$CUDA_HOME/bin:$PATH"
cd "$PROJECT_DIR/lib/hais_ops"
python setup.py build_ext --inplace
export PYTHONPATH="$PROJECT_DIR/lib/hais_ops:$PROJECT_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
cd "$PROJECT_DIR"

M13_CKPT="$PROJECT_DIR/checkpoints/m13_dense_views4_epoch_004_seed42.pt"
M10_CKPT="$PROJECT_DIR/checkpoints/m10_dense_views2_epoch_002_seed42.pt"

python test2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.eval=true TEST.roc=true TEST.prediction_threshold=0.70 \
  TEMPORAL_FRAME.temporal_frame_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_model_path="$M13_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_model_path="$M10_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000 \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0 \
  TEMPORAL_MEMORY.temporal_memory_inference_batch_size=8 \
  POSTPROCESS.p0_enabled=true POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 POSTPROCESS.p0_min_duration_bins=5 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.92 POSTPROCESS.p0b_enabled=false \
  POSTPROCESS.p18_score_track_recovery_enabled=true \
  POSTPROCESS.p18_event_count_cutoff=1 POSTPROCESS.p18_max_event_count=30000 \
  POSTPROCESS.p18_candidate_floor=0.55 POSTPROCESS.p18_spatial_radius=2 \
  POSTPROCESS.p18_temporal_bin_size=50 POSTPROCESS.p18_max_link_distance=6.0 \
  POSTPROCESS.p18_max_gap_bins=1 POSTPROCESS.p18_min_track_bins=2 \
  POSTPROCESS.p18_restore_mode=best \
  POSTPROCESS.p6_density_threshold_enabled=false
```

### 生成提交 TXT 与 ZIP

使用上节相同的 checkpoint 和参数，仅改提交入口和输出目录：

```bash
OUTPUT_DIR="$PROJECT_DIR/log/challenge2/m13e4_m10low30000_p18low30000_f055"

python submit_challenge2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" TEST.challenge_output_dir="$OUTPUT_DIR" \
  TEST.prediction_threshold=0.70 \
  TEMPORAL_FRAME.temporal_frame_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_model_path="$M13_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_model_path="$M10_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000 \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0 \
  TEMPORAL_MEMORY.temporal_memory_inference_batch_size=8 \
  POSTPROCESS.p0_enabled=true POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 POSTPROCESS.p0_min_duration_bins=5 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.92 POSTPROCESS.p0b_enabled=false \
  POSTPROCESS.p18_score_track_recovery_enabled=true \
  POSTPROCESS.p18_event_count_cutoff=1 POSTPROCESS.p18_max_event_count=30000 \
  POSTPROCESS.p18_candidate_floor=0.55 POSTPROCESS.p18_spatial_radius=2 \
  POSTPROCESS.p18_temporal_bin_size=50 POSTPROCESS.p18_max_link_distance=6.0 \
  POSTPROCESS.p18_max_gap_bins=1 POSTPROCESS.p18_min_track_bins=2 \
  POSTPROCESS.p18_restore_mode=best \
  POSTPROCESS.p6_density_threshold_enabled=false

(cd "$OUTPUT_DIR" && python -m zipfile -c ../m13e4_m10low30000_p18low30000_f055.zip val_*.txt)
```

确认 ZIP 根目录恰有 24 个 `val_*.txt` 后提交。当前已生成：
`log/challenge2/m13e4_m10low30000_p18low30000_f055.zip`。

### 重训当前 M13

```bash
M5_CKPT="$PROJECT_DIR/checkpoints/m4_dacc_m5_best_loss_seed42.pt"
M13_ROOT="$PROJECT_DIR/log/m13_dense_views4_ft4_seed42"

python train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=42 TRAIN.epochs=4 TRAIN.batch_size=1 TRAIN.lr=0.00002 \
  TRAIN.scheduler=cosine TRAIN.scheduler_min_lr=0.000001 \
  TRAIN.checkpoint_interval=1 TRAIN.model_save_root="$M13_ROOT" \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$M5_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0 \
  TEMPORAL_MEMORY.temporal_memory_memory_lr_multiplier=1.0 \
  TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff=200000 \
  TEMPORAL_MEMORY.temporal_memory_dense_view_multiplier=4 \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_weight=0.05 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_margin_logit=1.0 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_min_points=3 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_warmup_epochs=3
```

训练结束后应逐个评估 `epoch_001` 至 `epoch_004`。不要以 `best_loss_seed42.pt` 或
`last_seed42.pt` 替代当前最优 `epoch_004_seed42.pt`。

## 预训练权重（免训练复现报告分数）

| 文件 | 说明 | 用途 |
| --- | --- | --- |
| `checkpoints/m4_dacc_m5_best_loss_seed42.pt` | M4+DACC+M5，50 epoch，seed 42 | 训练 M10/M11/M13 的初始化权重 |
| `checkpoints/p23_baseline_5ep_seed42.pt` | P23 baseline，5 epoch，seed 42 | 重训 M4 的初始化权重 |
| `checkpoints/m10_dense_views2_epoch_002_seed42.pt` | M10，高密度连续采样第 2 轮，seed 42 | 当前方案在 `event_count <= 30000` 时的路由模型 |
| `checkpoints/m13_dense_views4_epoch_004_seed42.pt` | M13，高密度连续采样第 4 轮，seed 42 | 当前方案在 `event_count > 30000` 时的主模型 |

四个必要权重均已随仓库发布，因此无需下载本地 `log/` 目录即可复现报告中的验证结果。

## 方法

- **P23 baseline**：轻量 2D U-Net（`width=16`），按 50 时间单位分箱，对中心箱前后 5 个
  时间箱构建 10 通道事件计数帧，并以平衡 BCE 进行事件级二分类。
- **M4 双向时序记忆**：对整段视频的 bottleneck 特征做正反向 ConvGRU 传播，使用零初始化
  残差投影注入 decoder，使继承的 P23 表达在训练初期保持稳定。
- **DACC**：以事件帧全局密度生成 decoder 通道权重，末层偏置初始化为 4.0，使初始权重约为
  0.98，接近恒等映射。
- **M5 轨迹外推损失**：训练期基于 `target_id` 的线性轨迹，在序列中未标注的合理位置施加
  hinge 正约束；仅在连续 16 个时间箱的训练序列上使用。
- **M13 高密度采样**：把最密的训练视频采样频率提高，改善其时序记忆和背景判别；测试阶段
  不要求知道视频身份。
- **密度路由**：低密度视频保留 M10 的召回优势，高密度及中密度视频采用 M13 的 IoU/Fa
  优势。输入事件数在完整事件流读取后即可获得。

更完整的 M4、DACC、M5 数学细节、初始化机制和历史消融见
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)。

## 环境配置

已验证环境：WSL/Ubuntu、Python 3.9、PyTorch 1.9.1 + CUDA 11.1、torchvision 0.10.1、
`spconv-cu111`、NumPy 1.23.5。RTX 3050 Laptop 4GB 可以运行完整时序推理，两个路由模型
按视频顺序推理，不会同时做大批量前向。

除 PyTorch 的 CUDA 运行时外，编译 `HAIS_OP` 还需要安装与 PyTorch 兼容的 **CUDA Toolkit**
（含 `nvcc`，本实验为 CUDA 11.1）和 C++ 编译器。仅安装 `torch==1.9.1+cu111` 不会提供
`nvcc`。

```bash
conda create -n EV39 python=3.9 pip -y
conda activate EV39
python -m pip install --upgrade pip
python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install numpy==1.23.5 pyyaml==6.0.2 tqdm==4.66.5 pandas==2.0.3 \
  opencv-python==4.8.1.78 mlflow==2.17.2 spconv-cu111 \
  typing-extensions==4.12.2 pillow==10.4.0

export PROJECT_DIR=/absolute/path/to/EVC
export CUDA_HOME=/usr/local/cuda-11.1  # 改为本机 CUDA Toolkit 的实际路径
export PATH="$CUDA_HOME/bin:$PATH"
cd "$PROJECT_DIR/lib/hais_ops"
python setup.py build_ext --inplace
export PYTHONPATH="$PROJECT_DIR/lib/hais_ops:$PROJECT_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
cd "$PROJECT_DIR"
```

`lib/hais_ops` 是项目的 CUDA 扩展，首次使用前必须在目标机器上编译。CUDA、PyTorch
和编译器版本需彼此兼容；未设置 `LD_LIBRARY_PATH` 时，WSL 环境可能无法加载 PyTorch CUDA
动态库。

## 数据准备

从官方渠道下载 EV-UAV Challenge 2 数据后，目录结构应为：

```text
dataset/训练集、验证集/
  train/
    train_000.npz
    ...
  val/
    val_000.npz
    ...
```

上文命令通过 `DATA.root` 指向 `dataset/训练集、验证集`。

## 复现流程

1. 使用 `p23_baseline_5ep_seed42.pt` 按项目原始 P23 流程训练或评估基线。
2. 使用 `m4_dacc_m5_best_loss_seed42.pt` 作为 `temporal_memory_init_model_path`，按上文 M13
   命令训练 4 个 epoch，并保留每个 epoch checkpoint。
3. 选取 `epoch_004_seed42.pt` 为主模型，选取 M10 第 2 轮为低密度模型，运行固定验证命令。
4. 使用相同参数运行提交命令，压缩根目录的 24 个 `val_*.txt`。

## 关键配置（`configs/evisseg_evuav.yaml`）

```text
TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0
TEMPORAL_MEMORY.temporal_memory_memory_lr_multiplier=1.0
TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=true
TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff=200000
TEMPORAL_MEMORY.temporal_memory_dense_view_multiplier=4
TEMPORAL_MEMORY.temporal_memory_secondary_model_path=<M10 epoch_002>
TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000
```

`temporal_memory_secondary_model_path` 为空或
`temporal_memory_secondary_max_event_count=0` 时，密度路由完全关闭，单模型结果可按历史命令复现。

## 与官方 EVC 的代码改动

- `model/temporal_memory_net.py`：M4 双向 ConvGRU 时序记忆网络。
- `model/temporal_frame_net.py`：DACC 密度自适应通道校准。
- `utils/temporal_frame_loss.py`：M5 轨迹外推训练损失。
- `dataset/temporal_memory.py`：支持按视频事件数增加连续训练视图。
- `train_temporal_memory.py`：保存每轮 checkpoint，便于按官方分数选择权重。
- `utils/temporal_memory_inference.py`：支持第二个全流模型的分数融合或基于事件数的直接路由。
- `test2.py`、`submit_challenge2.py`：评估和提交使用相同的全流时序记忆与密度路由逻辑。

## 仓库结构

```text
configs/                    YAML 配置
dataset/                    数据读取、时序帧和训练采样
model/                      P23、M4、DACC 模型实现
utils/                      损失、推理、评估、后处理工具
lib/hais_ops/               CUDA 自定义算子源码（首次运行前编译）
train_temporal_memory.py    M4/DACC/M5/M13 训练入口
test2.py                    Challenge 2 验证入口
submit_challenge2.py        Challenge 2 TXT 提交生成入口
checkpoints/                M4/M5、P23、M10、M13 复现所需权重
log/                        本地训练、验证与提交产物（不纳入版本控制）
```

## 引用

```bibtex
@article{chen2025event,
  title={Event-based Tiny Object Detection: A Benchmark Dataset and Baseline},
  author={Chen, Nuo and others},
  journal={arXiv preprint arXiv:2506.23575},
  year={2025}
}
```
