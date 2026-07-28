# EVC

## EV-UAV Challenge 2 当前最优方案复现

本仓库包含 EV-UAV Challenge 2 事件级微小目标检测方案的代码、配置、训练入口、
本地验证脚本和提交 TXT 生成脚本。本文档记录当前已整理的 P23 全事件流时序帧方案，
包括完整的训练、验证和提交流程，不包含未采用的实验方法。

本项目适用于 Linux/WSL + NVIDIA GPU 环境。仓库基于 ICCV 2025 论文
*Event-based Tiny Object Detection: A Benchmark Dataset and Baseline* 的官方实现
整理；EV-SpSegNet、EV-UAV 数据集和原始预训练资源的版权归原论文作者所有。

## P23 已验证方案

P23 方案在 Challenge 2 的 `val/` 验证集 24 个视频上得到：

| 指标 | 数值 |
| --- | ---: |
| IoU | 0.9066961408 |
| Acc | 0.9520959854 |
| Pd | 0.9493910122 |
| Fa | 6.2246333599e-06 |
| Score_Fa | 0.9396513906 |
| Score | **0.9382006487** |

评分使用项目中的官方 Challenge 2 计算方式：

```text
Score_Fa = exp(-10000 * Fa)
Score = 0.4 * Pd + 0.3 * Score_Fa + 0.2 * IoU + 0.1 * Acc
```

该分数是本地 `val/` 验证结果，不代表未知官方测试集的保证分数。不同 CUDA、
PyTorch、spconv 或 HAIS_OP 构建版本可能产生轻微数值差异。

Git 仓库不包含数据集、checkpoint、训练日志和提交 TXT。按下文流程可以重新训练
P23 模型并复现该历史方案；要逐项核对上表数值，应使用相同的训练配置、随机种子和
`best_loss_seed37.pt` 权重。

## P23 方案组成

P23 分数由以下方法组成：

| 环节 | 固定设置 | 是否影响最终分数 |
| --- | --- | --- |
| P23 全事件流时序帧专家 | 时间箱 `50`，上下文 `5` 个时间箱，`width=16` | 是，核心模型 |
| P23 训练采样 | 每个训练视频每轮抽取 `8` 个时间视图，正时间箱概率 `0.75` | 是，训练方式 |
| P23 训练损失 | 每个时间视图独立平衡 BCE，正样本损失质量 `0.20`，正样本权重上限 `16` | 是，训练方式 |
| 推理阈值 | 全局固定阈值 `0.600` | 是 |
| P0 时空连通簇过滤 | `spatial_radius=2`，`temporal_bin_size=50`，`temporal_radius_bins=1`，`min_cluster_events=3`，`min_duration_bins=1` | 是 |
| P0c 高置信恢复 | `retain_min_score=0.975` | 是 |

最终预测不再依赖 `100000` 事件预算内的稀疏模型分数，而是由完整事件流构建时序帧
并进行全画面预测。设置 `temporal_frame_sparse_weight=0.0` 后，`test2.py` 与
`submit_challenge2.py` 会自动跳过稀疏模型加载和 P8 分块推理，因此完整复现只需要
P23 的时序帧 checkpoint。

## P23 方法原理

### 1. 完整事件流时序表征

稀疏事件模型通常需要在固定事件预算内截断输入。对于稠密视频，这可能丢失目标事件
或破坏目标在相邻时间内的上下文。P23 不对整段视频做 `100000` 事件截断，而是
直接读取完整事件流。

把时间按照官方 Pd 使用的 `50` 时间单位划分为时间箱。对每一个中心时间箱，取前后
共 `5` 个时间箱，并在每个时间箱分别累积负极性和正极性事件数，得到 `10` 个输入
通道：

```text
5 个时间箱 * 2 个极性通道 = 10 个通道
```

每个像素的事件数先执行 `log1p`，再以 `4.0` 截断并归一化。输入只来自：

```text
ev_loc                 # x, y, t
evs_norm[:, 3]         # 极性
```

推理阶段不读取标签、目标 ID 或视频名称规则。

### 2. 全画面时序帧网络

P23 使用一个轻量级二维 U-Net 风格网络，输入大小为 `346x260`，基础通道宽度为
`16`。网络包含残差卷积块、下采样、空洞上下文块和上采样结构，输出每个像素的
目标 logit。

网络并不把整张画面的所有像素都当作监督目标，而是只在中心时间箱中原始事件出现的
坐标上提取 logit，并对这些事件计算损失。这样既保留了完整空间上下文，又使训练
目标与最终的事件级提交格式严格对齐。

对于事件 `e`，网络输出的连续概率记为：

```text
s_frame(e) = sigmoid(logit(x_e, y_e | temporal_context(t_e)))
```

每个非空时间箱都在推理时被处理一次，因此完整事件流中的每个原始事件最终都有一个
时序帧分数。

### 3. 训练采样与损失

训练脚本 `train_temporal_frame.py` 只读取 `train/`，不在训练过程中读取验证集，也不
根据验证视频选择权重。每个视频每轮生成 `8` 个确定性随机时间视图，其中 `75%` 的
视图优先从包含目标事件的时间箱中抽取，剩余视图从有事件的时间箱中抽取。

损失在每个时间视图内独立计算。正事件的权重根据该视图的正负事件数量计算，使正样本
最多占该视图总损失质量的 `20%`，同时限制正样本权重不超过 `16`。这种有上限的平衡
方式提高稀少目标事件的学习强度，但不会让极少数正事件无限放大整个 batch 的梯度。

训练设置为 AdamW、学习率 `0.0001`、余弦退火、最小学习率 `0.000001`、固定 50 轮，
最终使用训练损失最低的 `best_loss_seed37.pt`，不使用验证集挑选 epoch。

### 4. P0/P0c 时空后处理

先将时序帧分数与全局阈值 `0.600` 比较，得到初始正事件。然后按以下规则构建时空
连通簇：

- 空间邻域半径为 `2`；
- 时间按宽度 `50` 的时间箱组织；
- 相邻时间箱距离不超过 `1` 时允许连接；
- 连通簇至少包含 `3` 个事件；
- 连通簇至少覆盖 `1` 个时间箱。

不满足条件的正簇被过滤。P0c 作为保护机制检查被 P0 删除的簇，如果簇内最高连续
分数不低于 `0.975`，则恢复该簇，避免过滤极高置信度的小目标。

在当前验证结果中，P0/P0c 将初始正事件数从 `66837` 调整为 `65648`，在保持高
召回的同时降低了误报，最终得到 `Score=0.9382006487`。

## P23 固定推理顺序

1. 读取完整原始事件流，并按 `50` 时间单位建立非空时间箱。
2. 对每个中心时间箱构建前后共 5 个时间箱的 10 通道极性计数帧。
3. 使用 P23 网络输出像素图，并在原始事件坐标提取事件级连续分数。
4. 使用全局阈值 `0.600` 生成初始二值事件。
5. 应用 P0 时空连通簇过滤。
6. 应用 P0c 高置信恢复。
7. `submit_challenge2.py` 保留原始事件的 `x y t p`，只写入最终二值 `label`。

## 仓库内容

```text
EVC/
|-- configs/
|   `-- evisseg_evuav_challenge2.yaml
|-- dataset/
|   |-- ev_uav.py
|   `-- temporal_frame.py
|-- model/
|   |-- evspsegnet.py
|   `-- temporal_frame_net.py
|-- utils/
|   |-- temporal_frame_inference.py
|   |-- temporal_frame_loss.py
|   |-- postprocess.py
|   `-- challenge_eval.py
|-- train.py
|-- train_temporal_frame.py
|-- test2.py
|-- test_temporal_frame.py
|-- submit_challenge2.py
`-- README.md
```

`.gitignore` 会排除数据集、权重、日志、压缩包和本地 CUDA 编译产物。不要强制将这些
大文件加入 Git。

## 环境配置

已验证环境：WSL/Ubuntu、Python 3.9、PyTorch 1.9.1 + CUDA 11.1、torchvision 0.10.1、
`spconv-cu111`、NumPy 1.23.5，以及用于编译 HAIS_OP 的 CUDA 11.x Toolkit。

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

首次克隆没有 HAIS_OP 二进制文件，需要在安装 PyTorch 后编译。系统需有兼容的 CUDA
Toolkit、C++ 编译器和 `libsparsehash-dev`：

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

每次打开新的 WSL 终端后，在仓库根目录重新执行环境变量：

```bash
export PROJECT_DIR=/absolute/path/to/EVC
export DATA_ROOT="$PROJECT_DIR/dataset/训练集、验证集"
export PYTHONPATH=$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "$PROJECT_DIR"
```

## 数据集准备

从官方渠道下载 EV-UAV 数据包：[百度网盘](https://pan.baidu.com/s/15pAlu3KP1uXych-c3SC5qA?pwd=sbr2)
（提取码 `sbr2`）或 [Google Drive](https://drive.google.com/drive/folders/1VIkBFx5Po0KPIFBYOL_appLVie5wgdyi?usp=drive_link)。
解压后将 Challenge 2 数据放在：

```text
dataset/训练集、验证集/
|-- train/
|-- val/
`-- val_Challenge2.py
```

训练目录应包含 99 个 `.npz` 视频文件，验证目录应包含 24 个 `.npz` 视频文件。

## 复现流程

### 1. 训练 P23 时序帧模型

P23 是独立的全事件流模型，不需要用稀疏模型权重初始化。训练过程中只使用 `train/`，
不会逐轮验证官方 `val/`，也不会根据视频名称制定规则。

```bash
conda activate EV39

export PROJECT_DIR=/mnt/d/AI/ESOD/EV-UAV-main
export DATA_ROOT="$PROJECT_DIR/dataset/训练集、验证集"
export PYTHONPATH=$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH

cd "$PROJECT_DIR"

TEMPORAL_ROOT="$PROJECT_DIR/log/p23_fullstream_temporal_frame_w16_e50_seed37"

python train_temporal_frame.py --config configs/evisseg_evuav_4gb.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=37 \
  TRAIN.epochs=50 \
  TRAIN.lr=0.0001 \
  TRAIN.scheduler=cosine \
  TRAIN.scheduler_min_lr=0.000001 \
  TRAIN.model_save_root="$TEMPORAL_ROOT" \
  TEMPORAL_FRAME.temporal_frame_enabled=true \
  TEMPORAL_FRAME.temporal_frame_bin_size=50 \
  TEMPORAL_FRAME.temporal_frame_context_bins=5 \
  TEMPORAL_FRAME.temporal_frame_width=16 \
  TEMPORAL_FRAME.temporal_frame_batch_size=4 \
  TEMPORAL_FRAME.temporal_frame_train_views_per_video=8 \
  TEMPORAL_FRAME.temporal_frame_positive_frame_probability=0.75 \
  TEMPORAL_FRAME.temporal_frame_target_positive_loss_mass=0.20 \
  TEMPORAL_FRAME.temporal_frame_max_positive_weight=16 \
  TEMPORAL_FRAME.temporal_frame_log_count_clip=4 \
  TEMPORAL_FRAME.temporal_frame_cache_all_videos=true \
  TEMPORAL_FRAME.temporal_frame_train_workers=0
```

训练结束后使用控制台输出的：

```text
best loss checkpoint: .../best_loss_seed37.pt
```

不要使用 `last_seed37.pt` 替换它，除非重新进行完整验证。当前已验证权重为：

```text
/mnt/d/AI/ESOD/EV-UAV-main/log/p23_fullstream_temporal_frame_w16_e50_seed37/runs/20260727-134621_seed37_pid16232/best_loss_seed37.pt
SHA-256: 670465B729DFF5D7BAC48568651C92883C21862C3525CAA38181E6C9095CA54B
```

### 2. 本地验证 P23 方案

P23 为纯时序帧推理，不需要 EV-SpSegNet、E1、P8 或任何旧稀疏模型
checkpoint。`TEST.model_path` 即使保留默认值也不会被加载。

```bash
TEMPORAL="$PROJECT_DIR/log/p23_fullstream_temporal_frame_w16_e50_seed37/runs/20260727-134621_seed37_pid16232/best_loss_seed37.pt"

python test2.py --config configs/evisseg_evuav_challenge2.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.prediction_threshold=0.600 \
  ENSEMBLE.ensemble_enabled=false \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 \
  POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.975 \
  POSTPROCESS.p6_density_threshold_enabled=false \
  INFERENCE_CHUNK.p8_enabled=false \
  INFERENCE_TTA.p14_horizontal_flip_enabled=false \
  TEMPORAL_FRAME.temporal_frame_enabled=true \
  TEMPORAL_FRAME.temporal_frame_model_path="$TEMPORAL" \
  TEMPORAL_FRAME.temporal_frame_width=16 \
  TEMPORAL_FRAME.temporal_frame_inference_batch_size=8 \
  TEMPORAL_FRAME.temporal_frame_sparse_weight=0.0
```

预期输出接近：

```text
IoU:      0.9066961408
Acc:      0.9520959854
Pd:       0.9493910122
Fa:       6.2246333599e-06
Score_Fa: 0.9396513906
Score:    0.9382006487
```

### 3. 生成提交 TXT

验证和提交必须使用完全相同的模型、阈值和 P0/P0c 参数。运行：

```bash
OUTPUT_DIR="$PROJECT_DIR/log/challenge2/temporal_frame_p23_score9382"

python submit_challenge2.py --config configs/evisseg_evuav_challenge2.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.challenge_output_dir="$OUTPUT_DIR" \
  TEST.prediction_threshold=0.600 \
  ENSEMBLE.ensemble_enabled=false \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 \
  POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.975 \
  POSTPROCESS.p6_density_threshold_enabled=false \
  INFERENCE_CHUNK.p8_enabled=false \
  INFERENCE_TTA.p14_horizontal_flip_enabled=false \
  TEMPORAL_FRAME.temporal_frame_enabled=true \
  TEMPORAL_FRAME.temporal_frame_model_path="$TEMPORAL" \
  TEMPORAL_FRAME.temporal_frame_width=16 \
  TEMPORAL_FRAME.temporal_frame_inference_batch_size=8 \
  TEMPORAL_FRAME.temporal_frame_sparse_weight=0.0

cd "$OUTPUT_DIR"
zip -j ../temporal_frame_p23_score9382.zip val_*.txt
```

`submit_challenge2.py` 会生成 24 个 `val_*.txt` 文件，每行格式为：

```text
x y t p label
```

其中 `x y t p` 来自原始事件，`label` 是经过时序帧预测、阈值、P0 和 P0c 后得到的
最终二值标签。`test2.py` 只计算本地验证指标，不生成提交文件；不要使用旧的
`test.py` 替代上述 Challenge 2 流程。

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
