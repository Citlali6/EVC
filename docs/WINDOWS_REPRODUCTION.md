# Windows 复现与首轮训练记录

本文记录在 RTX 4060 Laptop、Windows、Conda `yolo` 环境上复现 EV-UAV Challenge 2
发布模型的已验证流程。基线阶段不修改模型结构，也不依赖 `HAIS_OP` 或 `spconv`。

## 本机资产

- 项目：`F:\小目标检测\EVC-work`
- 数据：`F:\小目标检测\datasets\EV-UAV-Challenge2`
- Python：`C:\Users\CSK\.conda\envs\yolo\python.exe`
- M10：`checkpoints\m10_dense_views2_epoch_002_seed42.pt`
- M20：`checkpoints\m20_attn_dense_views8_epoch_003_seed48.pt`

官方数据已逐文件通过 Google Drive 的 size 和 MD5 校验：`train=99`、`val=24`、
`test=24`。完整清单位于数据根目录的 `official_google_drive_manifest.json`。

官方 Google Drive 根目录不包含 README 中提到的 `val_Challenge2.py`，上游官方仓库和
当前方案仓库也没有该文件。当前复现使用仓库内的 Challenge 2 指标实现；这不影响纯时序
M20 验证、训练或 TXT 生成。

## 数据与提交资产校验

```powershell
conda activate yolo
Set-Location "F:\小目标检测\EVC-work"

python validate_challenge2_assets.py `
  "F:\小目标检测\datasets\EV-UAV-Challenge2"
```

校验器逐文件检查 `ev`、`evs_norm`、`ev_loc`、事件行数、346x260 坐标范围、极性和
标签。提交目录或 ZIP 可用以下命令追加检查：

```powershell
python validate_challenge2_assets.py `
  "F:\小目标检测\datasets\EV-UAV-Challenge2" `
  --submission "F:\小目标检测\results\submission_m20_golden\m20_golden_val24.zip"
```

## M20/M10 共同参数

先在 PowerShell 中定义共同参数：

```powershell
$Project = "F:\小目标检测\EVC-work"
$Data = "F:/小目标检测/datasets/EV-UAV-Challenge2"
$M10 = "F:/小目标检测/EVC-work/checkpoints/m10_dense_views2_epoch_002_seed42.pt"
$M20 = "F:/小目标检测/EVC-work/checkpoints/m20_attn_dense_views8_epoch_003_seed48.pt"

$M20Common = @(
  "DATA.root=$Data"
  "TEST.prediction_threshold=0.719"
  "TEMPORAL_FRAME.temporal_frame_enabled=false"
  "TEMPORAL_MEMORY.temporal_memory_enabled=true"
  "TEMPORAL_MEMORY.temporal_memory_model_path=$M20"
  "TEMPORAL_MEMORY.temporal_memory_secondary_model_path=$M10"
  "TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000"
  "TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true"
  "TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0"
  "TEMPORAL_MEMORY.temporal_memory_inference_batch_size=8"
  "POSTPROCESS.p0_enabled=true"
  "POSTPROCESS.p0_spatial_radius=2"
  "POSTPROCESS.p0_temporal_bin_size=50"
  "POSTPROCESS.p0_temporal_radius_bins=1"
  "POSTPROCESS.p0_min_cluster_events=3"
  "POSTPROCESS.p0_min_duration_bins=5"
  "POSTPROCESS.p0c_high_confidence_recovery_enabled=true"
  "POSTPROCESS.p0c_retain_min_score=0.95"
  "POSTPROCESS.p0b_enabled=false"
  "POSTPROCESS.p18_score_track_recovery_enabled=true"
  "POSTPROCESS.p18_event_count_cutoff=1"
  "POSTPROCESS.p18_max_event_count=35000"
  "POSTPROCESS.p18_candidate_floor=0.53"
  "POSTPROCESS.p18_spatial_radius=5"
  "POSTPROCESS.p18_temporal_bin_size=50"
  "POSTPROCESS.p18_max_link_distance=8.0"
  "POSTPROCESS.p18_max_gap_bins=1"
  "POSTPROCESS.p18_min_track_bins=4"
  "POSTPROCESS.p18_restore_mode=best"
  "POSTPROCESS.p6_density_threshold_enabled=true"
  "POSTPROCESS.p6_event_count_cutoff=30000"
  "POSTPROCESS.p6_low_density_threshold=0.718"
  "POSTPROCESS.p6_high_density_threshold=0.719"
)
```

## 免训练验证

```powershell
Set-Location $Project
$ValidationArgs = $M20Common + @("TEST.eval=true", "TEST.roc=true")
& python test2.py --config configs/evisseg_evuav.yaml --set @ValidationArgs
```

本机实测结果：

| 指标 | 发布值 | 本机值 |
| --- | ---: | ---: |
| IoU | 0.9422550201 | 0.9422550201 |
| Acc | 0.9767196774 | 0.9767196774 |
| Pd | 0.9762704746 | 0.9762704746 |
| Fa | 4.6929172975e-06 | 4.6929172975e-06 |
| Score_Fa | 0.9541549752 | 0.9541549752 |
| Score | 0.9628776542 | 0.9628776542 |

结果逐位一致，因此当前机器不需要切换 WSL。完整输出保存在
`F:\小目标检测\results\baseline_m20\validation.log`。

## 生成 TXT 和平铺 ZIP

```powershell
$Output = "F:/小目标检测/results/submission_m20_golden/txt"
$SubmissionArgs = $M20Common + @("TEST.challenge_output_dir=$Output")
& python submit_challenge2.py --config configs/evisseg_evuav.yaml --set @SubmissionArgs

$ZipPath = Join-Path `
  "F:\小目标检测\results\submission_m20_golden" `
  ("m20_golden_val24_{0}.zip" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Push-Location "F:\小目标检测\results\submission_m20_golden\txt"
Compress-Archive -Path "*.txt" `
  -DestinationPath $ZipPath `
  -CompressionLevel Optimal
Pop-Location

python validate_challenge2_assets.py `
  "F:\小目标检测\datasets\EV-UAV-Challenge2" `
  --submission $ZipPath
```

已生成的 ZIP 含 24 个根目录 TXT、1,424,330 行，大小 9,486,186 字节，SHA-256 为：

```text
987DE067AE969269374B5CFBF7D2AD75BF0B81EE7701E26CB2FBB477A397284C
```

## 三视频 1 epoch smoke training

按事件数选择了三个训练视频：

| 密度 | 文件 | 事件数 |
| --- | --- | ---: |
| 低 | `train_080.npz` | 7,230 |
| 中位 | `train_067.npz` | 32,582 |
| 高 | `train_096.npz` | 625,178 |

三文件副本位于 `F:\小目标检测\datasets\EV-UAV-Challenge2-smoke\train`。训练必须保留以下
内存安全参数：

```text
TEMPORAL_MEMORY.temporal_memory_cache_all_videos=false
TEMPORAL_MEMORY.temporal_memory_cache_video_count=2
TEMPORAL_MEMORY.temporal_memory_train_workers=0
```

本次 smoke training 的完整 PowerShell 命令如下：

```powershell
$SmokeData = "F:/小目标检测/datasets/EV-UAV-Challenge2-smoke"
$SmokeOutput = "F:/小目标检测/results/smoke_training_m4_3videos/model"
$M4 = "F:/小目标检测/EVC-work/checkpoints/m4_dacc_m5_best_loss_seed42.pt"

$SmokeArgs = @(
  "DATA.root=$SmokeData"
  "TRAIN.seed=42"
  "TRAIN.epochs=1"
  "TRAIN.batch_size=1"
  "TRAIN.lr=0.00002"
  "TRAIN.scheduler=cosine"
  "TRAIN.scheduler_min_lr=0.000001"
  "TRAIN.checkpoint_interval=1"
  "TRAIN.model_save_root=$SmokeOutput"
  "TEMPORAL_MEMORY.temporal_memory_enabled=true"
  "TEMPORAL_MEMORY.temporal_memory_init_model_path=$M4"
  "TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0"
  "TEMPORAL_MEMORY.temporal_memory_memory_lr_multiplier=1.0"
  "TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled=false"
  "TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=false"
  "TEMPORAL_MEMORY.temporal_memory_train_views_per_video=1"
  "TEMPORAL_MEMORY.temporal_memory_cache_all_videos=false"
  "TEMPORAL_MEMORY.temporal_memory_cache_video_count=2"
  "TEMPORAL_MEMORY.temporal_memory_train_workers=0"
  "TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true"
  "TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=true"
  "TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_weight=0.05"
  "TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_margin_logit=1.0"
  "TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_min_points=3"
  "TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_warmup_epochs=3"
)

Set-Location $Project
& python train_temporal_memory.py `
  --config configs/evisseg_evuav.yaml `
  --set @SmokeArgs
```

本机 smoke run 完成 3/3 次前向、反向和优化器更新，epoch loss 为 `0.029870598`。
`best_loss_seed42.pt`、`last_seed42.pt`、`epoch_001_seed42.pt` 均已保存，严格重载成功；
相对 M4 初始化权重，83 个公共张量中 79 个发生更新，最大绝对更新量为
`6.00889325e-05`。日志和 checkpoint 位于：

```text
F:\小目标检测\results\smoke_training_m4_3videos
```

## 后续优化纪律

1. 保留 `Score=0.9628776542` 作为 golden baseline；每个实验只改一个因素并保存完整参数。
2. 先做阈值、密度路由和后处理消融，再投入完整训练链，能更快识别有效方向。
3. DACC 当前发布权重的门控实际恒为 0.5。修复它会改变模型定义，必须建独立实验分支、
   重新训练并与 golden baseline 比较，不能覆盖发布 M20。
4. 当前 `train_views_per_video=2`；`dense_view_multiplier=8` 实际给高密度视频生成 16 个视图。
5. 完整 M13 -> M15 -> M20 训练前先释放至少 6 GB 物理内存，并继续关闭全数据缓存。
