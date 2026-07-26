# EVC

## EV-UAV Challenge 2 最优方案复现

本仓库包含当前 EV-SpSegNet 最优方案的源码、配置、训练和提交流程，适用于
Linux/WSL + NVIDIA GPU 环境。目标是在 EV-UAV Challenge 2 的事件级微小目标检测
任务上复现当前冻结方案。

> 本仓库基于 ICCV 2025 论文 *Event-based Tiny Object Detection: A Benchmark
> Dataset and Baseline* 的官方实现整理。EV-SpSegNet、EV-UAV 数据集和原始预训练
> 资源的贡献归原论文作者所有。

## 当前冻结结果

当前方案在 Challenge 2 的 `val/` 验证集 24 个视频上得到：

| 指标 | 数值 |
| --- | ---: |
| IoU | 0.7831628323 |
| Acc | 0.8029493690 |
| Pd | 0.8357832843 |
| Fa | 3.1967536389e-06 |
| Score_Fa | 0.9685380238 |
| Score | **0.8618022242** |

该结果用于冻结当前提交候选，不代表未知测试集上的保证分数。

Git 仓库只包含代码、配置和文档，不包含数据集、checkpoint、训练日志和提交 TXT。
按下文训练命令可复现完整方法；要精确复测上表数值，需要使用记录的两份权重。
不同 CUDA、PyTorch、spconv 或 HAIS_OP 构建版本可能产生数值差异。

## 最优方案组成

| 环节 | 方法与固定参数 | 作用 |
| --- | --- | --- |
| 基础网络 | EV-SpSegNet，4 GB 配置，`width=12` | 输出每个事件属于目标的概率。 |
| 主模型训练 | P1b 目标保持采样，`max_events_num=100000`，100 epoch，Adam，`lr=0.001`，余弦退火最小学习率 `1e-5` | 在事件预算内优先保留目标事件，其余位置由背景事件补足。 |
| 次模型训练 | 主模型配方 + P15 训练期水平翻转，概率 `0.5` | 产生与主模型互补的次模型。 |
| E1 概率集成 | 主/次模型权重 `0.895/0.105` | 在二值化前融合两份事件分数。 |
| P8 稠密视频分块推理 | 仅事件数 `>100000`；`chunk_size=100000`；随机种子 `[37,73,101]` | 每个随机划分覆盖全部事件一次，再平均多个划分的分数。 |
| P14 水平翻转 TTA | 原始/镜像权重 `0.5/0.5` | 对原始事件流和水平镜像事件流的分数进行无标签平均。 |
| P6 密度自适应阈值 | 截止值 `100000`；低密度阈值 `0.45`；高密度阈值 `0.92` | 根据可观测的事件总数选择二值化阈值。 |
| P0/P0c 后处理 | `spatial_radius=2`，`temporal_bin_size=50`，`temporal_radius_bins=1`，`min_cluster_events=3`，`min_duration_bins=1`，高置信恢复阈值 `0.975` | 删除孤立小正簇，同时恢复置信度足够高的被删簇。 |

### 方法原理

**EV-SpSegNet** 将事件流表示为稀疏时空体素，网络输出与原始事件一一对应的目标
概率。所有后续模块都只处理这些事件级分数或可从事件流直接观测到的信息，不改动
官方评分代码。

**P1b 目标保持采样** 只在训练阶段使用标签。一个训练视频超过 100000 个事件时，
先保留预算内可容纳的目标事件，再从背景事件中随机补齐至固定预算。这样可以避免在
均匀随机截断时丢失本就稀少的目标轨迹，同时仍保留足够的背景用于学习判别边界。
验证和提交阶段不使用标签采样。

**P15 训练期水平翻转** 以 0.5 概率将事件的 x 坐标和对应的归一化 x 特征同时
镜像，事件顺序与标签含义保持不变。它不改变网络结构，而是让次模型看到与主模型
不同的几何分布，从而让两份模型在部分召回和误报位置上形成互补。

**E1 概率集成** 在二值化前逐事件融合两份模型的输出：

```text
s = 0.895 * s_primary + 0.105 * s_secondary
```

融合的是连续概率而不是二值标签，因此后续阈值和后处理仍能利用完整置信度信息。

**P8 随机分块推理** 面向超过训练预算的稠密视频。对每个固定随机种子，先生成一个
覆盖全部事件且互不重叠的随机排列分块；每个分块独立完成网络前向和 E1 融合，再把
分数写回原始事件索引。三个种子的完整覆盖结果再做平均，因此每个事件都能获得分数，
并减少单次截断边界带来的波动。

**P14 水平翻转 TTA** 在原始事件流和镜像事件流上分别运行同一套 P8/E1 推理。镜像
输入保持原事件顺序，得到的分数可直接对应回原始事件，再按下式平均：

```text
s = 0.5 * s_original + 0.5 * s_flipped
```

该过程只变换输入坐标和特征，不读取验证或测试标签。

**P6 密度自适应阈值** 以视频原始事件数为唯一判据：事件数不超过 100000 时使用
0.45，以外时使用 0.92。较稠密的视频通常包含更多背景活动，较高阈值有助于抑制
过量正预测；较低密度视频则保留更多弱目标响应。该规则在验证和提交中固定一致。

**P0/P0c 时空后处理** 先以阈值后的正事件建立时空连通图：时间被分到宽度为 50
的时间箱，空间半径为 2、时间相邻半径为 1 的事件单元可连通。P0 保留至少含 3 个
事件的连通簇，并将其他簇的分数置零。P0c 是保护机制：若一个本应删除的簇中最高
事件分数不低于 0.975，则恢复整簇，避免过度过滤极高置信的小目标。

### 推理顺序

1. 分别准备原始事件流和水平镜像事件流；稠密视频先由 P8 划为多个随机分块，其他视频直接完整前向。
2. 每个完整输入或分块输入中，主模型和次模型分别推理，并按 E1 权重融合。
3. P8 将分块分数还原到原始事件顺序，并平均 `[37,73,101]` 三个种子的结果。
4. P14 对原始流和镜像流的分数按 `0.5/0.5` 平均。
5. P6 根据视频事件总数选择 `0.45` 或 `0.92` 作为该视频的决策阈值。
6. P0 对阈值后的时空连通簇过滤，P0c 恢复最大分数不低于 `0.975` 的被删簇。
7. `submit_challenge2.py` 保留原始 `x y t p`，只写入最终二值 `label`。

## 仓库内容

```text
EVC/
|-- configs/                         # 训练和 Challenge 2 配置
|-- dataset/                         # 数据读取和训练采样实现
|-- lib/hais_ops/                    # HAIS_OP CUDA 扩展源码
|-- model/                           # EV-SpSegNet 网络实现
|-- utils/                           # 集成、P0/P0c、P6、P8、P14 和评估工具
|-- train.py                         # 训练入口
|-- test2.py                         # Challenge 2 本地验证与评分
|-- submit_challenge2.py             # Challenge 2 提交 TXT 生成
`-- README.md
```

`.gitignore` 会排除数据集、权重、日志、压缩包和本地 CUDA 编译产物。请不要强制将
这些文件加入 Git。

## 环境配置

已验证环境：WSL/Ubuntu、Python 3.9、PyTorch 1.9.1 + CUDA 11.1、
`torchvision` 0.10.1、`spconv-cu111`、NumPy 1.23.5，以及用于编译 HAIS_OP 的
CUDA 11.x Toolkit。

```bash
git clone https://github.com/Picasso9jiu/EVC.git
cd EVC

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

首次克隆没有 HAIS_OP 二进制文件，需要在安装 PyTorch 后编译。系统需有兼容的
CUDA Toolkit、C++ 编译器和 `libsparsehash-dev`。

```bash
sudo apt update
sudo apt install -y build-essential libsparsehash-dev ninja-build

export PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR/lib/hais_ops"
python setup.py build_ext develop

export PYTHONPATH=$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "$PROJECT_DIR"
python -c "import torch; import spconv.pytorch; import HAIS_OP; print(torch.cuda.is_available(), 'HAIS_OP: ok')"
```

## 数据集准备

从官方渠道下载 EV-UAV 数据包：[百度网盘](https://pan.baidu.com/s/15pAlu3KP1uXych-c3SC5qA?pwd=sbr2)
（提取码 `sbr2`）或 [Google Drive](https://drive.google.com/drive/folders/1VIkBFx5Po0KPIFBYOL_appLVie5wgdyi?usp=drive_link)。
当前方案使用 Challenge 2 数据目录：

```text
dataset/训练集、验证集/
|-- train/
|-- val/
`-- val_Challenge2.py
```

HAIS_OP 编译完成后，每个新终端都在仓库根目录执行：

```bash
cd /path/to/EVC
export PROJECT_DIR="$(pwd)"
export DATA_ROOT="$PROJECT_DIR/dataset/训练集、验证集"
export PYTHONPATH=$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

后续命令显式覆盖 `DATA.root`，因此仓库可克隆到任意 WSL 路径。

## 训练两份模型

两次训练都使用冻结的 100000 事件配方。训练命令中的 P0 只用于验证阶段选择
`best_score_seed37.pt`，不参与反向传播；最终 P0/P0c/P6/P8/P14 参数只在下文推理
和提交时启用。

### 主模型：P1b

```bash
PRIMARY_ROOT="$PROJECT_DIR/log/e1v2_p1b_cosine100_4gb_seed37"

python train.py --config configs/evisseg_evuav_4gb.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=37 \
  TRAIN.max_events_num=100000 \
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

训练完成后记录控制台输出的 `best Score checkpoint:` 路径，作为主模型权重。

### 次模型：P1b + P15

```bash
SECONDARY_ROOT="$PROJECT_DIR/log/p15_flip_p1b_cosine100_4gb_seed37"

python train.py --config configs/evisseg_evuav_4gb.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=37 \
  TRAIN.max_events_num=100000 \
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

训练完成后记录控制台输出的 `best Score checkpoint:` 路径，作为次模型权重。

### 当前冻结分数使用的权重

以下是产生 `0.8618022242` 的原始本地权重路径和哈希。它们不在 Git 仓库中；如持有
这两份文件，可通过 SHA-256 确认一致。

```text
主模型：log/e1v2_p1b_cosine100_4gb_seed37/runs/20260724-182346_seed37_pid544/best_score_seed37.pt
SHA-256：8ae3687bcc1e508df8cd1dc4bef1fdf4f08354c4197bd4f5378c6a48b35afdab

次模型：log/p15_flip_p1b_cosine100_4gb_seed37/runs/20260726-114604_seed37_pid5970/best_score_seed37.pt
SHA-256：37649c4017a73cef3ac0f9f01e8c0e2db0cf6d0a23e4c22e116b87783b89f6d9
```

## 验证当前冻结方案

将 `PRIMARY` 和 `SECONDARY` 分别设置为上面两份原始权重，或设置为自行训练得到的
两个 `best_score_seed37.pt`。

```bash
PRIMARY=/absolute/path/to/primary/best_score_seed37.pt
SECONDARY=/absolute/path/to/secondary/best_score_seed37.pt

# 使用原始权重时可核对哈希。
sha256sum "$PRIMARY" "$SECONDARY"

python test2.py --config configs/evisseg_evuav_challenge2.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.model_path="$PRIMARY" \
  TEST.prediction_threshold=0.900 \
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

使用上面记录的两份权重时，预期输出为：

```text
IoU:      0.7831628323
Acc:      0.8029493690
Pd:       0.8357832843
Fa:       3.1967536389e-06
Score_Fa: 0.9685380238
Score:    0.8618022242
```

## 生成比赛提交 TXT

提交时必须使用与验证完全相同的两份权重和全部推理参数。

```bash
OUTPUT_DIR="$PROJECT_DIR/log/challenge2/val-pred-txt-score8618"

python submit_challenge2.py --config configs/evisseg_evuav_challenge2.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.model_path="$PRIMARY" \
  TEST.challenge_output_dir="$OUTPUT_DIR" \
  TEST.prediction_threshold=0.900 \
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
zip -j ../evc_score8618022.zip val_*.txt
```

`test2.py` 只计算本地验证指标，不写提交文件；`submit_challenge2.py` 会为每个视频
生成一个 `val_*.txt`，其推理实现与 `test2.py` 一致。

## 引用

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
