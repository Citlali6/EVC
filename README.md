# EVSOD

EVSOD is a research repository for reproducing and extending **EV-SpSegNet** on the EV-UAV event-based tiny-object detection benchmark. It provides a runnable baseline, constrained-memory configurations, and a starting point for controlled improvement experiments.

> This repository is based on the official implementation of *Event-based Tiny Object Detection: A Benchmark Dataset and Baseline* (ICCV 2025). EV-SpSegNet, the EV-UAV dataset, and the pretrained checkpoint remain the work of the original authors. This repository does not claim the baseline method or dataset as new work.

## Method Summary

EV-SpSegNet formulates event-based tiny-object detection as sparse point-cloud segmentation. Moving targets form continuous trajectories in the spatiotemporal event cloud; background noise is more often isolated and weakly correlated.

The baseline consists of:

- **GDSCA**: grouped dilated sparse convolutions for multi-scale spatiotemporal feature extraction.
- **Sp-SE**: sparse feature fusion.
- **Patch Attention**: voxel downsampling and global-context modeling.
- **STC loss**: a spatiotemporal-correlation loss that retains coherent target events and suppresses isolated background events.

![EV-SpSegNet architecture](imgs/framework.png)

The EV-UAV benchmark contains 147 event sequences with event-level annotations. The original paper reports extremely small UAV targets, averaging approximately 6.8 x 5.4 pixels.

## Repository Layout

```text
EVSOD/
|-- configs/                 # Baseline, scratch, smoke-test, and 4 GB configurations
|-- dataset/
|   |-- EV-UAV-dataset/      # Local dataset only; ignored by Git
|   `-- ev_uav.py            # EV-UAV dataset loader
|-- lib/hais_ops/            # Custom CUDA extension (HAIS_OP)
|-- model/                   # EV-SpSegNet implementation
|-- utils/                   # STC loss and evaluation utilities
|-- train.py                 # Training entry point
|-- test.py                  # Evaluation entry point
`-- log/                     # Local checkpoints and outputs; ignored by Git
```

## Environment

The local reproduction was run in WSL Ubuntu with a CUDA-capable NVIDIA GPU. The verified Python stack is:

| Component | Version |
| --- | --- |
| Python | 3.9 |
| PyTorch | 1.9.1 + CUDA 11.1 (`cu111`) |
| torchvision | 0.10.1 + CUDA 11.1 (`cu111`) |
| CUDA toolkit used to build HAIS_OP | CUDA 11.x |
| NumPy | `< 2` |

The project also needs `spconv`, `PyYAML`, `mlflow`, `tqdm`, `libsparsehash-dev`, and the custom `HAIS_OP` extension. Install a `spconv` build compatible with the selected PyTorch and CUDA version.

After activating the conda environment, compile the extension in WSL:

```bash
cd /mnt/d/AI/ESOD/EVSOD/lib/hais_ops
export CUDA_HOME=/path/to/your/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/usr/include:$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
python setup.py build_ext develop
python -c "import HAIS_OP; print('HAIS_OP ok')"
```

`nvcc --version` must succeed before compiling. A CUDA runtime embedded in a PyTorch wheel is not a replacement for the CUDA compiler and headers required by `HAIS_OP`.

## Dataset and Checkpoints

Dataset archives and checkpoints are intentionally not tracked in this repository. Download them from the official project and place them locally:

- EV-UAV dataset: [Baidu Netdisk](https://pan.baidu.com/s/15pAlu3KP1uXych-c3SC5qA?pwd=sbr2) (code: `sbr2`) or [Google Drive](https://drive.google.com/drive/folders/1VIkBFx5Po0KPIFBYOL_appLVie5wgdyi?usp=drive_link)
- EV-SpSegNet pretrained weights: [Baidu Netdisk](https://pan.baidu.com/s/1e6a_Ool5WZ3cBMPvoJvWbg?pwd=ztp4) (code: `ztp4`) or [Google Drive](https://drive.google.com/file/d/1nNZsckiN0qp2oo1uX40tU6oz3mUcrSHq/view?usp=drive_link)

Expected local layout:

```text
dataset/EV-UAV-dataset/
|-- train/
|-- val/
`-- test/

log/model/best_iou_seed37.pt
```

Before running, update `DATA.root`, `TRAIN.model_save_root`, and `TEST.model_path` in the selected YAML file to your own WSL paths.

## Quick Start

### 1. Evaluate the official pretrained baseline

Set `TEST.model_path` in `configs/evisseg_evuav.yaml` to `best_iou_seed37.pt`, then run:

```bash
cd /mnt/d/AI/ESOD/EVSOD
conda activate EV39
python test.py --config configs/evisseg_evuav.yaml
```

The local run completed all 24 test videos and produced:

```text
iou: 0.5843424201011658
seg_acc: 0.6784908771514893
pd: 0.7846212700841622
fa: 8.493834145404406e-06
```

These values are a local reference, not a replacement for a complete independent reproduction protocol.

### 2. Smoke-test training

The smoke configuration trains for one epoch with 100,000 sampled events per training sequence. It verifies the data pipeline, CUDA extension, forward and backward pass, checkpoint writing, and evaluation pipeline.

```bash
python train.py --config configs/evisseg_evuav_smoke.yaml
python test.py --config configs/evisseg_evuav_smoke.yaml
```

A one-epoch smoke test is not expected to yield meaningful detection quality.

### 3. Train the baseline from scratch

For the original event cap and 50 epochs:

```bash
python train.py --config configs/evisseg_evuav_scratch.yaml
python test.py --config configs/evisseg_evuav_scratch.yaml
```

The original configuration uses `max_events_num: 700000` and is memory intensive. `configs/evisseg_evuav_4gb.yaml` uses `max_events_num: 100000` with separate output paths for limited-memory training:

```bash
python train.py --config configs/evisseg_evuav_4gb.yaml
python test.py --config configs/evisseg_evuav_4gb.yaml
```

Reducing `max_events_num` changes the training data distribution and final metrics. It is useful for functional validation and constrained-hardware experiments, but it must be reported separately from the full-event baseline.

## Planned Improvements

The following are hypotheses based on the baseline's limitations. Their effect must be established through controlled experiments.

1. **Feature extraction**: GDSCA uses fixed dilation rates, which may not adapt to targets with different speeds and scales. Replace it with deformable grouped dilated sparse convolution, using local event density to dynamically adjust the receptive field. This may reduce missed detections of extremely small targets.
2. **Loss function**: STC uses a fixed neighborhood, which may not suit both fast and slow targets, while target/background events are highly imbalanced. Investigate adaptive-neighborhood STC with class-balanced weighting.
3. **Post-processing**: The baseline has no dedicated post-processing. Apply spatiotemporal event clustering to remove small isolated noise clusters, then use motion continuity to reconnect short broken trajectories. This is the lowest-cost direction and should be evaluated first for false-alarm reduction and detection-rate gains.
4. **Multi-representation fusion**: The baseline uses only a point-cloud representation. Add a lightweight event-frame branch and fuse it with the point-cloud branch to provide complementary features under strong-noise conditions and improve robustness.

For every improvement, retain the same train/validation/test split, seed, evaluation configuration, and event budget as the reference experiment. Report accuracy metrics (`IoU`, `seg_acc`, `PD`, `FA`) as well as GPU memory, training time, and inference time.

## Experiment Practice

- Keep the baseline configuration unchanged for fair comparisons; create a new YAML file for each ablation.
- Save checkpoints outside Git and record the matching configuration, seed, and code commit.
- Evaluate the official pretrained checkpoint first, then reproduce scratch training before comparing a modification.
- Do not interpret smoke-test results as final model performance.

## Citation and Acknowledgements

Please cite the original work when using the EV-UAV dataset or EV-SpSegNet:

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

The original implementation builds on [HAIS](https://github.com/hustvl/HAIS) and [spconv](https://github.com/traveller59/spconv). Follow the original project license and the licenses and terms of the dataset, pretrained weights, HAIS, and spconv when using this repository.
