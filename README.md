# EVSOD

## EV-UAV Challenge 2 当前最优方案复现

本分支保存 EV-UAV Challenge 2 当前已验证的事件级微小目标检测方案：按输入事件数路由的
全事件流双向时序记忆网络，并在高密度分支加入时序自注意力残差。仓库包含复现当前分数所需的
代码、固定配置、验证脚本、提交生成脚本和 checkpoint；无需重新训练即可直接验证。

项目基于 ICCV 2025 EV-UAV 官方基线实现整理。EV-SpSegNet、EV-UAV 数据集和原始预训练
资源的版权归原论文作者所有。

以下分数来自 `val/` 的 24 个视频，是本地验证结果，不代表未知官方测试集分数。不同 CUDA、
PyTorch、spconv 或 HAIS_OP 编译版本可能造成轻微数值差异。

## M20 已验证结果

| 指标 | 数值 |
| --- | ---: |
| IoU | 0.9422550201 |
| Acc | 0.9767196774 |
| Pd | 0.9762704746 |
| Fa | 4.6929172975e-06 |
| Score_Fa | 0.9541549752 |
| Score | **0.9628776542** |

评分由仓库内的 Challenge 2 评估器计算：

```text
Score_Fa = exp(-10000 * Fa)
Score = 0.4 * Pd + 0.3 * Score_Fa + 0.2 * IoU + 0.1 * Acc
```

M20 的 12 个保存点均已通过完整 `test2.py` 验证。最佳 Challenge 2 checkpoint 是
**epoch 003**；训练 loss 最低的是 epoch 011，不能以它替换 epoch 003。

## M20 方案组成

| 环节 | 固定设置 | 作用 |
| --- | --- | --- |
| 低密度路由 | `event_count <= 30000` 使用 M10 epoch 002 | 保留低密度视频上更稳定的专家 |
| 高密度路由 | `event_count > 30000` 使用 M20 attention epoch 003 | 当前主模型 |
| M20 基础 | M15 双向 ConvGRU 时序记忆网络 | 继承已验证的全事件流时序表征 |
| M20 新增模块 | bottleneck 处零初始化的多头时序自注意力残差 | 让任意两个时间帧直接交换证据 |
| 训练采样 | `event_count > 200000` 的视频每轮使用 8 个确定性视图 | 提高高密度输入的时序覆盖 |
| P6 阈值 | 低密度 `0.718`，其他 `0.719` | 生成最终事件标签的阈值 |
| P0/P0c | 半径 2、相邻 1 个时间箱、最少 3 事件和 5 时间箱、保留分数 0.95 | 过滤弱时空连通簇，同时保留高置信小簇 |
| P18 | 最多 35000 事件、候选下限 0.53、半径 5、连接距离 8、最少 4 时间箱 | 仅恢复稳定的弱轨迹 |

M20 的 attention 输出投影采用零初始化。因此从 M15 加载时预测不变，新增残差会在训练中逐步
学习，而不会一开始破坏已经收敛的时序记忆模型。

## 固定推理顺序

1. 读取完整原始事件流，按宽度 `50` 的时间箱构建上下文为 5 的时序输入帧。
2. 事件数不超过 30000 的视频路由到 M10，其余视频使用 M20。
3. 对每个原始事件输出连续目标概率。
4. 按 P6 的密度自适应阈值生成初始二值事件。
5. 应用 P0 时空连通簇过滤，再应用 P0c 高置信恢复。
6. 在满足事件数范围的视频上应用 P18 弱轨迹恢复。
7. 生成提交时保留原始 `x y t p`，仅写入最终二值 `label`。

所有路由条件都只依赖可观察的输入事件数。推理过程中不读取验证标签、目标 ID 或视频名称规则。

## 已包含权重

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `checkpoints/m10_dense_views2_epoch_002_seed42.pt` | 固定低密度路由模型 | `5C89C89A165469C0A4E8286D4644D60D2F82CF5775EDBB724F626E24E67D8935` |
| `checkpoints/m20_attn_dense_views8_epoch_003_seed48.pt` | 固定高密度 M20 模型 | `4B8B2B19EA9D913EE4E52CB21AE52BF945B2B0F3CEFD5CB5AB6F64D51BF49849` |

直接评估当前最优方案只需要上表两份权重。

## 仓库结构

```text
EVSOD-main/
|-- checkpoints/                 # 版本管理的 M10、M20 权重
|-- configs/evisseg_evuav.yaml   # 固定配置
|-- dataset/                     # 数据集目录，不上传 Git
|-- model/temporal_memory_net.py # ConvGRU 时序记忆和时序自注意力
|-- utils/                       # 全事件流推理与 Challenge 2 评估器
|-- train_temporal_memory.py     # M20 训练入口
|-- test2.py                     # 本地 Challenge 2 验证
|-- submit_challenge2.py         # 提交 TXT 生成
|-- note.md                      # 本地实验日志，不上传 Git
`-- README.md                    # 当前最优方案复现文档
```

`log/`、数据集、生成的提交文件以及本地 HAIS_OP 编译产物均不上传 Git。首次使用时需要在目标
环境中编译 HAIS_OP。

## 环境配置

已验证环境：WSL/Ubuntu、Python 3.9、PyTorch 1.9.1 + CUDA 11.1、torchvision 0.10.1、
spconv-cu111、NumPy 1.23.5，以及 CUDA 11.x Toolkit。

```bash
git clone --branch evsod-main https://github.com/Picasso9jiu/EVC.git EVSOD-main
cd EVSOD-main

conda create -n EV39 python=3.9 pip -y
conda activate EV39

python -m pip install --upgrade pip
python -m pip install \
  torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install \
  numpy==1.23.5 pyyaml==6.0.2 tqdm==4.66.5 pandas==2.0.3 \
  opencv-python==4.8.1.78 mlflow==2.17.2 spconv-cu111 \
  typing-extensions==4.12.2 pillow==10.4.0
```

安装 PyTorch 后首次编译 HAIS_OP。系统需要兼容的 CUDA Toolkit、C++ 编译器和
`libsparsehash-dev`：

```bash
sudo apt update
sudo apt install -y build-essential libsparsehash-dev ninja-build

export PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR/lib/hais_ops"
python setup.py build_ext develop

export PYTHONPATH="$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
cd "$PROJECT_DIR"
python -c "import torch; import spconv.pytorch; import HAIS_OP; print(torch.cuda.is_available(), 'HAIS_OP: ok')"
```

每次打开新终端后，重新激活环境并设置路径：

```bash
conda activate EV39
export PROJECT_DIR=/absolute/path/to/EVSOD-main
export DATA_ROOT="$PROJECT_DIR/dataset/训练集、验证集"
export PYTHONPATH="$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
cd "$PROJECT_DIR"
```

## 数据准备

从官方渠道下载 EV-UAV Challenge 2 数据包，放置为：

```text
dataset/训练集、验证集/
|-- train/       # 99 个 .npz 视频
|-- val/         # 24 个 .npz 视频
`-- val_Challenge2.py
```

数据集不随 Git 发布。官方数据可从 EV-UAV benchmark 发布页提供的百度网盘或 Google Drive
链接下载。

## **免训练评估**

下列命令直接使用仓库中的 M10/M20 权重，按当前固定策略在 24 个验证视频上评估，不需要训练。
在已验证 GPU 环境中通常需要数分钟。

```bash
M10_CKPT="$PROJECT_DIR/checkpoints/m10_dense_views2_epoch_002_seed42.pt"
M20_CKPT="$PROJECT_DIR/checkpoints/m20_attn_dense_views8_epoch_003_seed48.pt"

python test2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.eval=true TEST.roc=true TEST.prediction_threshold=0.719 \
  TEMPORAL_FRAME.temporal_frame_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_model_path="$M20_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_model_path="$M10_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000 \
  TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0 \
  TEMPORAL_MEMORY.temporal_memory_inference_batch_size=8 \
  POSTPROCESS.p0_enabled=true POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 POSTPROCESS.p0_min_duration_bins=5 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.95 POSTPROCESS.p0b_enabled=false \
  POSTPROCESS.p18_score_track_recovery_enabled=true \
  POSTPROCESS.p18_event_count_cutoff=1 POSTPROCESS.p18_max_event_count=35000 \
  POSTPROCESS.p18_candidate_floor=0.53 POSTPROCESS.p18_spatial_radius=5 \
  POSTPROCESS.p18_temporal_bin_size=50 POSTPROCESS.p18_max_link_distance=8.0 \
  POSTPROCESS.p18_max_gap_bins=1 POSTPROCESS.p18_min_track_bins=4 \
  POSTPROCESS.p18_restore_mode=best \
  POSTPROCESS.p6_density_threshold_enabled=true \
  POSTPROCESS.p6_event_count_cutoff=30000 \
  POSTPROCESS.p6_low_density_threshold=0.718 \
  POSTPROCESS.p6_high_density_threshold=0.719
```

预期输出接近：

```text
IoU:      0.9422550201
Acc:      0.9767196774
Pd:       0.9762704746
Fa:       4.6929172975e-06
Score_Fa: 0.9541549752
Score:    0.9628776542
```

## 生成 Challenge 2 提交文件

提交必须使用与 **免训练评估** 完全相同的 M10/M20 权重及固定参数，只将验证选项替换为输出目录：

```bash
OUTPUT_DIR="$PROJECT_DIR/log/challenge2/m20_e3_m10low30000"
M10_CKPT="$PROJECT_DIR/checkpoints/m10_dense_views2_epoch_002_seed42.pt"
M20_CKPT="$PROJECT_DIR/checkpoints/m20_attn_dense_views8_epoch_003_seed48.pt"

python submit_challenge2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.challenge_output_dir="$OUTPUT_DIR" TEST.prediction_threshold=0.719 \
  TEMPORAL_FRAME.temporal_frame_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_model_path="$M20_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_model_path="$M10_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000 \
  TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0 \
  TEMPORAL_MEMORY.temporal_memory_inference_batch_size=8 \
  POSTPROCESS.p0_enabled=true POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 POSTPROCESS.p0_min_duration_bins=5 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.95 POSTPROCESS.p0b_enabled=false \
  POSTPROCESS.p18_score_track_recovery_enabled=true \
  POSTPROCESS.p18_event_count_cutoff=1 POSTPROCESS.p18_max_event_count=35000 \
  POSTPROCESS.p18_candidate_floor=0.53 POSTPROCESS.p18_spatial_radius=5 \
  POSTPROCESS.p18_temporal_bin_size=50 POSTPROCESS.p18_max_link_distance=8.0 \
  POSTPROCESS.p18_max_gap_bins=1 POSTPROCESS.p18_min_track_bins=4 \
  POSTPROCESS.p18_restore_mode=best \
  POSTPROCESS.p6_density_threshold_enabled=true \
  POSTPROCESS.p6_event_count_cutoff=30000 \
  POSTPROCESS.p6_low_density_threshold=0.718 \
  POSTPROCESS.p6_high_density_threshold=0.719

cd "$OUTPUT_DIR"
zip -j ../m20_e3_m10low30000.zip val_*.txt
```

## 实验日志

`note.md` 是本地实验日志，持续记录每次训练配置、保存点的完整验证分数、融合扫描、采用或淘汰
结论，以及下一步的研究假设。它不上传 Git，不作为公开复现材料；当前可复现方案始终以本 README
为准。
