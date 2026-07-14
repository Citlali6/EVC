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
|-- configs/                 # 原始基线、从头训练、冒烟测试、4 GB 显存配置
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
| 编译 HAIS_OP 使用的 CUDA Toolkit | CUDA 11.x |
| NumPy | `< 2` |

此外还需要安装与 PyTorch/CUDA 匹配的 `spconv`、`PyYAML`、`mlflow`、`tqdm`、`libsparsehash-dev`，并编译自定义扩展 `HAIS_OP`。

激活 conda 环境后，在 WSL 中编译 `HAIS_OP`：

```bash
cd /mnt/d/AI/ESOD/EVSOD/lib/hais_ops
export CUDA_HOME=/path/to/your/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/usr/include:$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
python setup.py build_ext develop
python -c "import HAIS_OP; print('HAIS_OP ok')"
```

编译前 `nvcc --version` 必须能正常输出版本。PyTorch wheel 自带的 CUDA 运行时不能替代编译 `HAIS_OP` 所需的 CUDA 编译器和头文件。

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

### 2. 训练冒烟测试

冒烟测试配置每段训练序列最多采样 100,000 个事件，只训练 1 个 epoch。它用于验证数据读取、CUDA 扩展、前向和反向传播、权重保存以及评估流程是否都能正常运行。

```bash
python train.py --config configs/evisseg_evuav_smoke.yaml
python test.py --config configs/evisseg_evuav_smoke.yaml
```

单个 epoch 的冒烟测试不应被视为有意义的检测性能结果。

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

以下方向来自对基线局限性的分析假设，具体收益必须通过严格对照实验验证。

1. **特征提取模块**：当前 GDSCA 的空洞率固定，对不同速度、不同尺度的目标适配不足，极小目标可能漏检。可尝试可变形分组空洞稀疏卷积，根据局部事件密度动态调整感受野。
2. **损失函数**：当前 STC 的邻域大小固定，难以同时适配快慢不同的目标，且正负样本存在严重不均衡。可设计自适应邻域的 STC，并引入类别平衡权重。
3. **后处理**：这是成本较低、预期收益较高的优先方向。基线没有专门后处理，输出中仍有少量孤立虚警，部分短轨迹也会断裂。可采用时空事件聚类删除过小的孤立噪声簇，再用运动连续性补全断裂轨迹，以降低虚警并提升检测率。
4. **多表示融合**：当前模型只有点云单一表示，在强噪声场景下特征可能不足。可增加轻量事件帧分支，并与点云分支进行特征融合，利用两种表示的互补性提升鲁棒性。

后续实验应固定训练/验证/测试划分、随机种子、评估配置和事件上限，并同时报告 `IoU`、`seg_acc`、`PD`、`FA`、显存占用、训练时间和推理时间。

## 实验规范

- 保持基线配置不变作为公平对照；每个消融实验新建独立 YAML 配置。
- 权重不提交到 Git；每次实验记录对应配置、随机种子和 Git commit。
- 先验证官方预训练权重的推理结果，再完成从头训练复现，最后比较改进方法。
- 不要将冒烟测试结果当作最终模型性能。

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
