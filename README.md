# EVSOD

EVSOD 是一个用于复现并改进 **EV-SpSegNet** 的研究仓库，面向 EV-UAV 事件相机微小目标检测基准。仓库提供可运行的基线代码、显存受限配置和后续改进实验的基础规范。

> 本仓库基于 ICCV 2025 论文 *Event-based Tiny Object Detection: A Benchmark Dataset and Baseline* 的官方实现整理。EV-SpSegNet、EV-UAV 数据集及其预训练权重的原始贡献均属于原论文作者；本仓库不将基线方法或数据集作为新的方法或数据集主张。

## 方法摘要

EV-SpSegNet 将事件相机微小目标检测建模为稀疏点云分割问题。运动目标在时空事件点云中通常形成连续轨迹，而背景噪声更常表现为孤立、弱相关的事件。

基线由以下部分构成：

- **GDSCA**：分组空洞稀疏卷积，用于提取多尺度时空特征。
- **Sp-SE**：稀疏特征融合模块。
- **Patch Attention**：用于体素下采样和全局上下文建模。
- **STC Loss**：时空相关损失，保留具有连续性的目标事件并抑制孤立背景事件。

![EV-SpSegNet 网络结构](imgs/framework.png)

EV-UAV 基准包含 147 段带事件级标注的序列。原论文报告的无人机目标平均尺寸约为 6.8 x 5.4 像素，属于极小目标检测场景。

## 代码结构

```text
EVSOD/
|-- configs/                 # 原始基线、从头训练、烟雾测试、4 GB 显存配置
|-- dataset/
|   |-- EV-UAV-dataset/      # 本地数据集目录，不提交到 Git
|   `-- ev_uav.py            # EV-UAV 数据集读取逻辑
|-- lib/hais_ops/            # 自定义 CUDA 扩展 HAIS_OP
|-- model/                   # EV-SpSegNet 网络实现
|-- utils/                   # STC Loss 与评估工具
|-- train.py                 # 训练入口
|-- test.py                  # 测试入口
`-- log/                     # 本地权重和结果，不提交到 Git
```

## 环境配置

本地复现使用 WSL Ubuntu 和支持 CUDA 的 NVIDIA GPU。已经跑通的核心环境如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.9 |
| PyTorch | 1.9.1 + CUDA 11.1 (`cu111`) |
| torchvision | 0.10.1 + CUDA 11.1 (`cu111`) |
| 编译 HAIS_OP 使用的 CUDA Toolkit | CUDA 11.x（仅从源码编译时需要） |
| NumPy | `< 2` |

下面的命令以项目目录 `/mnt/d/AI/ESOD/EVSOD` 为例。如果你保留当前目录名，请将它替换为 `/mnt/d/AI/ESOD/EV-UAV-main`。开始前先确认 WSL 能识别显卡：

```bash
nvidia-smi
```

### 1. 下载 CUDA 版 PyTorch wheel

为避免 conda 自动解析到 CPU 版 PyTorch，使用与 Python 3.9 对应的 CUDA 11.1 wheel。下载中断后重新执行同一条命令即可从已下载位置继续：

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

若 `curl` 长时间保持 `0 B/s`，通常是镜像连接暂时无响应；可中断后晚些时候执行相同命令续传。不要安装不完整的 `.whl` 文件。

### 2. 创建隔离 conda 环境

```bash
conda create -n EV39 python=3.9 pip -y
conda activate EV39
```

如果此前创建过同名环境并且准备完全重建，先执行以下命令，再运行上面的创建命令：

```bash
conda env remove -n EV39 -y
```

### 3. 安装 Python 依赖

以下命令使用阿里云 PyPI 镜像安装除 PyTorch 之外的依赖。这里不使用 conda 安装 PyTorch，以避免被解析为 CPU 构建：

```bash
python -m pip install --upgrade pip
python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ \
  numpy==1.23.5 pyyaml==6.0.2 tqdm==4.66.5 pandas==2.0.3 \
  opencv-python==4.8.1.78 mlflow==2.17.2 spconv-cu111 \
  typing-extensions==4.12.2 pillow==10.4.0
```

### 4. 从本地 wheel 安装 CUDA 版 PyTorch

```bash
python -m pip install --no-deps \
  "$HOME/.cache/evuav-wheels/torch-1.9.1+cu111-cp39-cp39-linux_x86_64.whl" \
  "$HOME/.cache/evuav-wheels/torchvision-0.10.1+cu111-cp39-cp39-linux_x86_64.whl"
```

### 5. 验证 PyTorch 能调用 GPU

```bash
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

输出中必须包含 `cuda: 11.1` 和 `available: True`。如果 `available` 为 `False`，先解决 WSL GPU 驱动或 PyTorch 安装问题，不要继续运行训练和测试。

### 6. 配置 HAIS_OP

`HAIS_OP` 是项目使用的自定义 CUDA 扩展。`lib/hais_ops/build/` 是本地编译产物，已被 `.gitignore` 排除，因此 **GitHub 新克隆的仓库不包含预编译二进制文件**。

如果你在已跑通的本地工作区中保留了 Python 3.9 的预编译产物，可直接加载：

```bash
export PROJECT_DIR=/mnt/d/AI/ESOD/EV-UAV-main
export PYTHONPATH=$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
python -c "import torch; import spconv.pytorch; import HAIS_OP; print('HAIS_OP ok')"
```

如果是全新克隆，或上面的目录不存在，则必须从源码编译。系统需要安装 `libsparsehash-dev`、兼容的 C++ 编译器和完整 CUDA Toolkit；`nvcc --version` 必须成功。PyTorch wheel 内的 CUDA 运行时不能替代 `nvcc` 和 CUDA 头文件。安装好系统依赖后执行：

```bash
sudo apt update
sudo apt install -y build-essential libsparsehash-dev ninja-build

cd /mnt/d/AI/ESOD/EVSOD/lib/hais_ops
export CUDA_HOME=/path/to/your/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/usr/include:$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
python setup.py build_ext develop
python -c "import HAIS_OP; print('HAIS_OP ok')"
```

CUDA 11.x 对宿主编译器版本有约束。若编译报出 `g++` 版本过高，请安装与本机 CUDA Toolkit 兼容的 GCC/G++，并通过 `CC`、`CXX` 和 `CUDAHOSTCXX` 显式指定，而不要修改项目源码来绕过编译器检查。

### 7. 每次打开新 WSL 终端

在已有预编译 `HAIS_OP` 的工作区中，每次新开终端至少执行：

```bash
conda activate EV39
export PROJECT_DIR=/mnt/d/AI/ESOD/EV-UAV-main
export PYTHONPATH=$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd $PROJECT_DIR
```

若通过源码编译并使用 `develop` 安装扩展，通常不需要设置 `PYTHONPATH`；但仍需要激活正确的 conda 环境。

## 数据集与预训练权重

数据集、预训练权重、训练日志和实验结果均不会提交到本仓库。请通过原项目的官方链接下载：

- EV-UAV 数据集：[百度网盘](https://pan.baidu.com/s/15pAlu3KP1uXych-c3SC5qA?pwd=sbr2)（提取码：`sbr2`）或 [Google Drive](https://drive.google.com/drive/folders/1VIkBFx5Po0KPIFBYOL_appLVie5wgdyi?usp=drive_link)
- EV-SpSegNet 预训练权重：[百度网盘](https://pan.baidu.com/s/1e6a_Ool5WZ3cBMPvoJvWbg?pwd=ztp4)（提取码：`ztp4`）或 [Google Drive](https://drive.google.com/file/d/1nNZsckiN0qp2oo1uX40tU6oz3mUcrSHq/view?usp=drive_link)

建议放置为以下本地目录结构：

```text
dataset/EV-UAV-dataset/
|-- train/
|-- val/
`-- test/

log/model/best_iou_seed37.pt
```

运行前，请在所选 YAML 配置中修改以下路径为自己的 WSL 路径：`DATA.root`、`TRAIN.model_save_root` 和 `TEST.model_path`。

## 快速运行

### 1. 测试官方预训练基线

在 `configs/evisseg_evuav.yaml` 中将 `TEST.model_path` 指向 `best_iou_seed37.pt`，然后执行：

```bash
cd /mnt/d/AI/ESOD/EVSOD
conda activate EV39
mkdir -p log/model
python test.py --config configs/evisseg_evuav.yaml
```

本地已完成 24 个测试视频的推理，得到以下参考结果：

```text
iou: 0.5843424201011658
seg_acc: 0.6784908771514893
pd: 0.7846212700841622
fa: 8.493834145404406e-06
```

上述数值仅为本地运行参考，不替代完整、独立的复现实验流程。

### 2. 训练烟雾测试

烟雾测试配置每段训练序列最多采样 100,000 个事件，只训练 1 个 epoch。它用于验证数据读取、CUDA 扩展、前向和反向传播、权重保存以及评估流程是否都能正常运行。

```bash
python train.py --config configs/evisseg_evuav_smoke.yaml
python test.py --config configs/evisseg_evuav_smoke.yaml
```

单个 epoch 的烟雾测试不应被视为有意义的检测性能结果。

### 3. 从头训练基线

使用原始事件上限并训练 50 个 epoch：

```bash
python train.py --config configs/evisseg_evuav_scratch.yaml
python test.py --config configs/evisseg_evuav_scratch.yaml
```

原始配置使用 `max_events_num: 700000`，显存占用较高。`configs/evisseg_evuav_4gb.yaml` 使用 `max_events_num: 100000` 并写入独立的输出目录，面向显存受限情况下的训练：

```bash
python train.py --config configs/evisseg_evuav_4gb.yaml
python test.py --config configs/evisseg_evuav_4gb.yaml
```

降低 `max_events_num` 会改变训练数据分布和最终指标，可用于功能验证或显存受限实验；与完整事件基线比较时必须单独报告该设置。

## 改进方向

下表中的方向是根据基线局限性提出的研究假设，而非已经证实的结论。每项改动都应独立完成消融实验，并与不改动的基线在相同事件上限下对比。

| 优先级 | 方向 | 基线局限 | 可尝试的改进 | 核心验证指标 |
| --- | --- | --- | --- | --- |
| P0 | 后处理 | 基线没有专门的后处理，仍有少量孤立虚警，部分短轨迹会断裂。 | 对预测为前景的事件做时空聚类，删除过小或时间跨度过短的噪声簇；再利用速度估计和运动连续性连接短断轨迹。 | 首先观察 `FA` 是否下降，同时保证或提升 `PD`；测量额外推理耗时。 |
| P1 | 特征提取模块 | GDSCA 的空洞率固定，难以随目标速度、尺度和局部事件密度变化调整感受野，极小目标可能漏检。 | 将固定 GDSCA 替换为可变形分组空洞稀疏卷积，或由局部事件密度预测每组的空洞率/采样偏移。 | `IoU`、`PD`，并按目标尺度和速度分桶分析漏检率。 |
| P1 | 损失函数 | STC 的邻域大小固定，难以同时适配快慢目标；目标和背景事件数量高度不均衡。 | 设计自适应时空邻域的 STC；与类别平衡权重、Focal 类权重或难例挖掘结合。 | `IoU`、`seg_acc`、`PD`、`FA`，以及正负事件召回率。 |
| P2 | 多表示融合 | 当前网络只有点云单一表示，强噪声或稀疏事件场景中的特征表达可能不足。 | 增加轻量事件帧分支，与点云分支在中间层进行门控融合或跨注意力融合。 | 全集指标及按光照、噪声场景分组的鲁棒性；显存和延迟。 |

### 可进一步拓展的方向

| 方向 | 动机 | 建议的第一步 |
| --- | --- | --- |
| 自适应事件窗口与事件预算 | 固定 `whole_t` 和固定事件上限会让快慢目标获得不均衡的时间上下文，并影响显存。 | 用局部事件率或目标运动估计动态选择时间窗口；先只在数据加载阶段实现，保持网络不变。 |
| 时序关联与轨迹级评估 | 事件级分割指标不能完全反映目标轨迹是否连续。 | 在测试阶段为前景簇增加跨窗口关联，并补充轨迹连续率、断轨次数和每段序列的 `PD/FA`。 |
| 面向事件噪声的数据增强 | 强光、低照度和背景抖动会改变事件密度，单一训练分布可能不稳健。 | 依次测试随机事件丢弃、极性扰动、时间抖动和背景噪声注入；每次只启用一种增强。 |
| 轻量化与部署优化 | 点云稀疏卷积和后处理可能限制实时性。 | 统计各阶段耗时，随后尝试减少通道宽度、知识蒸馏、混合精度和后处理的在线聚类实现。 |

后续实验应固定训练/验证/测试划分、随机种子、评估配置和事件上限，并同时报告 `IoU`、`seg_acc`、`PD`、`FA`、显存占用、训练时间和推理时间。对于新模块，还应报告其参数量和额外延迟。

## 实验规范

- 保持基线配置不变作为公平对照；每个消融实验新建独立 YAML 配置。
- 权重不提交到 Git；每次实验记录对应配置、随机种子和 Git commit。
- 先验证官方预训练权重的推理结果，再完成从头训练复现，最后比较改进方法。
- 不要将烟雾测试结果当作最终模型性能。

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

原始实现基于 [HAIS](https://github.com/hustvl/HAIS) 和 [spconv](https://github.com/traveller59/spconv)。使用本仓库时，请同时遵守原项目、数据集、预训练权重、HAIS 与 spconv 的许可证和使用条款。
