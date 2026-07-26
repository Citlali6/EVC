# EVC

## EV-UAV Challenge 2 Reproducible Pipeline

EVC contains the code, configuration, tests, and documentation for the current
best EV-SpSegNet pipeline on the EV-UAV Challenge 2 validation split. It is
intended for WSL/Linux with an NVIDIA GPU.

> This repository builds on the official implementation and dataset from
> *Event-based Tiny Object Detection: A Benchmark Dataset and Baseline* (ICCV
> 2025). EV-SpSegNet, EV-UAV, and the original pretrained assets remain the
> work of the paper authors.

## Frozen Validation Result

The frozen candidate below reached **Score `0.8618022242`** on the Challenge 2
`val/` split with 24 videos. This is a validation result used to select the
submission candidate, not a guarantee of performance on an unseen test split.

| Metric | Value |
| --- | ---: |
| IoU | 0.7831628323 |
| Acc | 0.8029493690 |
| Pd | 0.8357832843 |
| Fa | 3.1967536389e-06 |
| Score_Fa | 0.9685380238 |
| Score | **0.8618022242** |

The Git repository intentionally excludes datasets, checkpoints, logs, and
generated prediction files. It contains everything needed to build the
environment, train the two models, run validation, and generate submission
TXT files. Exact reproduction of the table above requires the two recorded
checkpoints listed below; re-training reproduces the method but can produce a
different checkpoint because CUDA sparse operators can vary across builds.

## Best Method

| Stage | Method and fixed parameters | Purpose |
| --- | --- | --- |
| Base network | EV-SpSegNet, 4 GB configuration (`width=12`) | Produces an event-level target probability. |
| Primary training | P1b target-preserving sampling, `max_events_num=100000`, 100 epochs, Adam `lr=0.001`, cosine minimum LR `1e-5` | Keeps foreground events within the 100000-event budget and fills remaining capacity with background events. |
| Secondary training | Primary recipe plus P15 label-preserving horizontal-flip augmentation with probability `0.5` | Produces a complementary model for probability fusion. |
| E1 ensemble | Primary/secondary weights `0.895/0.105` | Reduces model-specific prediction error before thresholding. |
| P8 dense-video inference | For videos with more than `100000` events: chunk size `100000`, seeds `[37,73,101]` | Covers all events in each deterministic random partition and averages partition scores. |
| P14 flip TTA | Original/flipped score weights `0.5/0.5` | Averages label-free original and horizontal-mirror predictions. |
| P6 density threshold | Cutoff `100000`; thresholds `0.45` / `0.92` | Uses a lower threshold for low-density videos and a stricter one for dense videos. |
| P0/P0c postprocess | `spatial_radius=2`, `temporal_bin_size=50`, `temporal_radius_bins=1`, `min_cluster_events=3`, `min_duration_bins=1`, recovery score `0.975` | Removes isolated positive clusters while retaining very high-confidence clusters that would otherwise be removed. |

### Inference Order

1. For each original or horizontally mirrored event stream, run the primary
   and secondary models and form the E1 weighted score.
2. For a video above the event cutoff, P8 runs the score inference over each
   random partition and restores one score per source event; otherwise it runs
   one full-video forward pass.
3. P14 averages scores from the original and mirrored streams.
4. P6 chooses the per-video decision threshold from the observable event
   count.
5. P0 filters thresholded spatiotemporal clusters and P0c restores a removed
   cluster when its maximum score is at least `0.975`.
6. `submit_challenge2.py` writes the original `x y t p` columns plus the final
   binary `label` column.

## Repository Contents

```text
EVC/
|-- configs/                         # Training and Challenge 2 configs
|-- dataset/                         # Dataset readers and sampling code
|-- lib/hais_ops/                    # HAIS_OP CUDA extension source
|-- model/                           # EV-SpSegNet implementation
|-- utils/                           # Ensemble, P0/P0c, P6, P8, P14, metrics
|-- tests/                           # Unit tests for reproducible helpers
|-- train.py                         # Training entry point
|-- test2.py                         # Challenge 2 validation and scoring
|-- submit_challenge2.py             # Challenge 2 TXT export
`-- README.md
```

`.gitignore` excludes all datasets, checkpoints, logs, archives, and local
CUDA build products. Do not force-add those artifacts to Git.

## Environment

The validated stack is WSL/Ubuntu, Python 3.9, PyTorch 1.9.1 with CUDA 11.1,
`torchvision` 0.10.1, `spconv-cu111`, NumPy 1.23.5, and a CUDA 11.x toolkit
for building HAIS_OP.

```bash
git clone https://github.com/Picasso9jiu/EVC.git
cd EVC

conda create -n EV39 python=3.9 pip -y
conda activate EV39

python -m pip install --upgrade pip
python -m pip install \
  numpy==1.23.5 pyyaml==6.0.2 tqdm==4.66.5 pandas==2.0.3 \
  opencv-python==4.8.1.78 mlflow==2.17.2 spconv-cu111 \
  typing-extensions==4.12.2 pillow==10.4.0
python -m pip install \
  torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

Build HAIS_OP after PyTorch is installed. A CUDA toolkit, compatible C++
compiler, and `libsparsehash-dev` are required for a fresh build.

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

## Data Layout

Download the official EV-UAV data package from [Baidu Netdisk](https://pan.baidu.com/s/15pAlu3KP1uXych-c3SC5qA?pwd=sbr2)
(code `sbr2`) or [Google Drive](https://drive.google.com/drive/folders/1VIkBFx5Po0KPIFBYOL_appLVie5wgdyi?usp=drive_link).
The frozen recipe uses the Challenge 2 package at:

```text
dataset/训练集、验证集/
|-- train/
|-- val/
`-- val_Challenge2.py
```

Run the following in every new terminal after HAIS_OP has been built. The
commands below pass `DATA.root` explicitly, so a clone can live at any WSL
path.

```bash
export PROJECT_DIR="$(pwd)"
export DATA_ROOT="$PROJECT_DIR/dataset/训练集、验证集"
export PYTHONPATH=$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

## Train the Two Models

Both commands train from scratch with the frozen 100000-event recipe. P0 is
enabled only so the validation loop selects `best_score_seed37.pt` using the
same historical model-selection setting; it does not alter gradients. The
final P0/P0c/P6/P8/P14 inference parameters are applied only below.

### Primary: P1b

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

Use the printed `best Score checkpoint:` as the primary checkpoint.

### Secondary: P1b + P15

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

Use the printed `best Score checkpoint:` as the secondary checkpoint.

### Recorded Checkpoints for the Frozen Score

These are the original local artifact paths used for the recorded score. They
are not part of Git. If they are available, their SHA-256 values are:

```text
Primary:   log/e1v2_p1b_cosine100_4gb_seed37/runs/20260724-182346_seed37_pid544/best_score_seed37.pt
SHA-256:   8ae3687bcc1e508df8cd1dc4bef1fdf4f08354c4197bd4f5378c6a48b35afdab

Secondary: log/p15_flip_p1b_cosine100_4gb_seed37/runs/20260726-114604_seed37_pid5970/best_score_seed37.pt
SHA-256:   37649c4017a73cef3ac0f9f01e8c0e2db0cf6d0a23e4c22e116b87783b89f6d9
```

## Validate the Frozen Pipeline

Set `PRIMARY` and `SECONDARY` to the two recorded artifacts above or to the
two `best_score_seed37.pt` files produced by the training commands.

```bash
PRIMARY=/absolute/path/to/primary/best_score_seed37.pt
SECONDARY=/absolute/path/to/secondary/best_score_seed37.pt

# Check against the two recorded SHA-256 values when using the original artifacts.
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

With the recorded artifacts, the expected output is:

```text
IoU:      0.7831628323
Acc:      0.8029493690
Pd:       0.8357832843
Fa:       3.1967536389e-06
Score_Fa: 0.9685380238
Score:    0.8618022242
```

## Generate Submission TXT Files

Use exactly the same two checkpoints and inference options as validation.

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

`test2.py` evaluates local validation metrics and does not write submission
files. `submit_challenge2.py` writes one `val_*.txt` file per video using the
same inference implementation.

## Tests

Run the repository tests in the configured `EV39` environment:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Citation

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
