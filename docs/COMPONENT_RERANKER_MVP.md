# Dense component reranker MVP

## Status and scope

This branch contains an **opt-in, not-yet-promoted** reranker.  It is disabled
in every shipped YAML.  With the default settings no artifact is opened and
the prediction tensor is returned unchanged.

The runtime stage is fixed as:

`M10/M20 scores -> P0/P0c -> component reranker -> P18 -> threshold`

The reranker is eligible only when the complete-video event count is strictly
greater than `POSTPROCESS.component_reranker_event_count_cutoff` (default
`100000`).  It can suppress a small retained component; it never raises a
score.  Runtime features contain only score, `x/y/t`, component/short-track
statistics, and complete-video event count.  File name, target ID, and label
are not features.

The current artifact contract intentionally supports only the released pure
temporal-memory M20 primary route.  It rejects sparse/temporal-frame blends,
high-density temporal blend models, and dense experts.  At load time it checks:

- expected artifact SHA-256;
- M20 checkpoint SHA-256 recorded by the train cache;
- prediction threshold;
- deployment event-count cutoff;
- the effective P0/P0c input mapping and its canonical SHA-256.

## GPU gate: build a train-only cache

Do not run this phase until its output directory, event-count training rule,
and expected selected-video count have been preregistered.  The selection rule
is deliberately configurable.  For example, `>30000` includes more independent
training videos than `>100000`, while deployment may remain `>100000`.

```powershell
& 'C:\Users\CSK\.conda\envs\yolo\python.exe' train_component_reranker.py cache `
  --config configs\evisseg_evuav.yaml `
  --override 'DATA.root=F:/小目标检测/datasets/EV-UAV-Challenge2' `
  --checkpoint checkpoints\m20_attn_dense_views8_epoch_003_seed48.pt `
  --data-root F:\小目标检测\datasets\EV-UAV-Challenge2 `
  --output-cache-dir <NEW_EXPERIMENT_DIR>\train_cache `
  --min-event-count-exclusive 30000 `
  --expected-total-videos 99 `
  --expected-selected-videos 54
```

On the current official train split, the read-only metadata audit gives
`>30000: 54 videos / 8,555,762 events` and
`>100000: 15 videos / 6,819,439 events`.

This performs real M20 GPU inference on each complete selected train video; it
does not use `EvUAV(mode='train')`, so oversized sources are not silently
downsampled.  The command writes compact per-video NPZ files and writes the
manifest last.  Before creating the cache directory or loading the GPU model,
it hashes all 99 canonical raw train files and requires the independently
verified official semantic SHA-256
`e94aaeae451113943a464feec7b1500968601a835ce8eeee914129ed2456625f`;
it repeats that check before publishing the manifest.  It has no
validation-data input path.

## CPU fit gate

The fit command has no automatic validation sweep or OOF selection.  Positive
class weight and keep probability are required explicit inputs so they can be
recorded before the result is observed.

```powershell
& 'C:\Users\CSK\.conda\envs\yolo\python.exe' train_component_reranker.py fit `
  --config configs\evisseg_evuav.yaml `
  --override 'POSTPROCESS.p0_enabled=true' `
  --override 'POSTPROCESS.p0_spatial_radius=2' `
  --override 'POSTPROCESS.p0_temporal_bin_size=50' `
  --override 'POSTPROCESS.p0_temporal_radius_bins=1' `
  --override 'POSTPROCESS.p0_min_cluster_events=3' `
  --override 'POSTPROCESS.p0_min_duration_bins=5' `
  --override 'POSTPROCESS.p0c_high_confidence_recovery_enabled=true' `
  --override 'POSTPROCESS.p0c_retain_min_score=0.95' `
  --override 'POSTPROCESS.p0c_density_retain_enabled=false' `
  --override 'POSTPROCESS.p0c_density_event_count_cutoff=100000' `
  --override 'POSTPROCESS.p0c_density_retain_min_score=0.97' `
  --cache-dir <NEW_EXPERIMENT_DIR>\train_cache `
  --output-model <NEW_EXPERIMENT_DIR>\component_reranker.json `
  --prediction-threshold 0.719 `
  --deployment-event-count-cutoff 100000 `
  --positive-weight <PREREGISTERED_WEIGHT> `
  --keep-probability <PREREGISTERED_PROBABILITY>
```

The fit example deliberately freezes the canonical C00 P0c contract
(`p0c_retain_min_score=0.95`, density retain disabled).  C09's dense `.97`
retain rule is a separate later comparison and is not the training starting
point.

The artifact reports in-sample train diagnostics only.  They are not evidence
of a validation or leaderboard improvement.

The separately preregistered train-only cross-source consistency experiment
is implemented by `crossfit_component_reranker.py`.  Its immutable folds,
eight-candidate grid, balanced weighting, official-count gates, and
pass-only artifact rule are specified in
`docs/COMPONENT_RERANKER_CROSSFIT_PROTOCOL.md`; runnable commands are in
`docs/COMPONENT_RERANKER_CROSSFIT.md`.  It must not be described as an
unbiased independent OOF estimate because each held train block also selects
its candidate.

## Promotion gate

Before enabling the artifact in a full validation or submission command:

1. record the artifact SHA-256 printed by the fit command;
2. enable the reranker only through config overrides and provide that exact
   SHA-256;
3. replay the complete 24-video validation once with all M10/M20, threshold,
   P0/P0c, and P18 settings frozen;
4. compare exact official metrics and per-video sufficient counts against the
   frozen C00 and exploratory C09 references;
5. do not promote if the preregistered score/recall gate fails.

CPU contract tests are in `tests/test_component_reranker.py`.  They lock the
default-off identity, strict `>100000` boundary, 8-connected component radius,
label-free feature derivation, strict JSON/checkpoint/P0 binding, runtime stage
order, and a real deterministic optimizer update.
