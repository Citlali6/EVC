# Component reranker train-only cross-fit protocol

## Status

This document defines two explicitly named experiment profiles.  The original
`conservative_v1` profile was frozen before its cross-fit result was produced.
The later `posthoc_pw4_kp040_v2` profile is deliberately and permanently
marked retrospective: it was frozen only after the v1 train-only no-op report
and a train-only threshold diagnostic.  It is not presented as independent or
unbiased OOF evidence.  Both profiles use only the official `train` split and
an immutable M20 train-cache.  Validation labels, leaderboard results,
file-name identity, target ID, and labels are never runtime features.  A model
artifact is emitted only when every gate below passes.

The machine-readable protocol produced by
`crossfit_component_reranker.py preregister` records this document's SHA-256,
the cache manifest SHA-256, the relevant source-code SHA-256 values, and a
canonical protocol-definition SHA-256.  The `run` phase requires the exact
protocol-file SHA-256 on its command line and rejects any mismatch.

## Frozen population and folds

- Cache selection: complete official train videos with strict
  `event_count > 30000`.
- The cache builder hashes all 99 raw train files before creating its output
  directory or starting GPU work, and again before publishing the manifest.
  The frozen semantic scheme is SHA-256 over each canonical
  `name_utf8 + NUL + raw_file_sha256_bytes` entry in order.  The independently
  verified official digest is
  `e94aaeae451113943a464feec7b1500968601a835ce8eeee914129ed2456625f`.
- Required selected population: exactly 54 canonical videos and exactly
  `8,555,762` events: indices `000-014`, `028-032`, `040-047`, `059-065`,
  `067-074`, and `088-098`.
- Deployment eligibility remains strict `event_count > 100000`.
- `H1`: `train_044.npz` through `train_047.npz` (four videos).
- `H2`: `train_088.npz` through `train_098.npz` (eleven videos).
- `middle`: the other 39 selected videos.
- Fold `holdout_h1` fits on `H2 + middle` and evaluates only `H1`.
- Fold `holdout_h2` fits on `H1 + middle` and evaluates only `H2`.
- Middle-video OOF deployment is identity with respect to the reranker.
  Middle videos still run through the complete frozen C00 postprocessor,
  including P18 for videos in its 30k--35k routing range.

## Frozen model candidates: `conservative_v1`

Exactly eight candidates are preregistered:

- positive component weight: `4` or `8`;
- keep probability: `0.02`, `0.05`, `0.10`, or `0.20`;
- L2 penalty: `0.1`;
- deterministic Newton iteration ceiling: `50`;
- topology: 8-connected per-bin components (`spatial_radius=1`),
  `temporal_bin_size=50`, greedy link distance `6`, maximum gap `1`, and
  candidate component size at most three events.

Each fold fits its scaler and logistic model only on that fold's fit videos.
The weighted scaler uses the preregistered base weights.  High-domain and
middle-domain components receive total base mass `0.5` each.  Within a domain,
each video receives equal mass, and within a video each candidate component
receives equal mass.  Positive component weights are multiplied only after
the base weights are assigned.  Block and source identity are used only for
fold membership and sample weighting and are never model features.

For `conservative_v1`, the winner for each fold is the candidate with the
highest held-block Score; candidate ID is the deterministic tie-breaker.  No
validation result can alter the candidate set, topology, weighting, winner
rule, or gates.

## Retrospective singleton: `posthoc_pw4_kp040_v2`

The first v1 run produced eight exact no-ops and no artifact.  Its immutable
report is
`experiments/20260810_component_reranker_crosssource_v1/crossfit_report.json`
with SHA-256
`e06182a03667e169b16fee7b02e7e44dd636457bc2b4d6f62265dcb6300577d0`.
The post-hoc train-only diagnostic that generated the follow-up hypothesis is
`experiments/20260810_component_reranker_crosssource_v1/posthoc_threshold_diagnostic.json`
with SHA-256
`5ba11e24c8f820773b9baf2c1b778131cdf30876dffbc64b4f2d4c06b17ef8e3`.
The tool requires both exact files and validates the diagnostic schema
`ev-uav-component-reranker-posthoc-train-diagnostic-v1`, evidence class
`retrospective_train_only_not_independent_oof`, source protocol/report/cache
SHA-256 lineage, and the v1 all-candidate no-op outcome.

V2 contains exactly one candidate and therefore performs no new
hyperparameter selection:

- candidate ID `pw04_kp400`;
- positive component weight `4.0`;
- keep probability `0.40`;
- L2 penalty `0.1`;
- the same feature semantics, topology, folds, fitting weights, C00 path, and
  promotion gates as v1.

The hypothesis origin is frozen as
`retrospective_train_only_after_v1_noop`.  H1 and H2 are reused only as
cross-source consistency gates.  Because those same held train blocks were
inspected to create the `0.40` hypothesis, the v2 result must never be called
an independent confirmation or unbiased OOF estimate.  In particular, v2
must not scan `0.39`, `0.41`, or any other threshold.

Before any v2 artifact or validation replay, the validation decision rule was
separately frozen at
`experiments/20260810_component_reranker_posthoc_singleton_v2/validation_acceptance_policy.json`
with SHA-256
`8f26a10b8585e31a2bef26177a2bc6390292549f66ce4e0b39f6c51eecf9d987`.
The protocol records this path and digest.  If the train-only gates pass, at
most one complete 24-video validation replay is allowed.  No validation-based
threshold or hyperparameter search is allowed.  If that single replay fails
the already-frozen acceptance policy, this component-suppression branch is
archived without tuning it on validation.

## Frozen C00 and scoring path

Input scores are the cache-bound primary M20 scores.  Every video runs through
the frozen C00 stages in deployment order:

`M20 score -> P0/P0c -> optional held-fold reranker -> P18 -> threshold`

The cache checkpoint is required to equal the released M20 SHA-256
`4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849`.
A cache produced by any other checkpoint is rejected during preregistration.

The canonical threshold is `0.719`.  C00 uses P0 radius `2`, temporal bin
`50`, temporal radius `1`, minimum cluster size `3`, minimum duration `5`,
P0c high-confidence recovery at `0.95`, and no density-retain override.  P18
is enabled only by its frozen route (`event_count > 1` and `<=35000`) with
candidate floor `0.53`, radius `5`, temporal bin `50`, link distance `8`, gap
`1`, minimum four track bins, and `best` restore mode.  The preregistration
command rejects a configuration that differs from this contract.  Cross-fit
cache input is explicitly primary-M20-only: secondary model path must be
empty, secondary cutoff must be zero, and primary weight must be one.  A later
runtime may still route M10 only at `<=30000`, below both the cache population
and reranker deployment boundary.

Metrics use the unchanged Challenge 2 definitions and sufficient counts:
event TP/FP/FN, target-frame detections, equivalent frame count, and
8-connected false-alarm components.  H1 and H2 held predictions plus all 39
middle identity predictions are pooled once before computing the nonlinear
Score.

One existing runtime/evaluator boundary difference is frozen explicitly.
Reranker feature components use integer `floor_divide(t, 50)`, so an event at
exactly `t % 50 == 0` belongs to a feature bin.  The unchanged official Pd/Fa
evaluator uses open frame intervals and therefore excludes that event from
Pd/Fa frame-component counts, while IoU/Acc still count it as an event.  The
cross-fit does not silently redefine either behavior: candidate topology stays
runtime-identical, and promotion uses the official sufficient counts so the
actual nonlinear tradeoff is measured.

## Promotion gates

All of the following must pass:

1. H1 held-block Score delta is strictly positive.
2. H2 held-block Score delta is strictly positive.
3. Pooled Pd does not decrease.
4. Pooled IoU does not decrease.
5. False components do not increase in either held block.
6. Combined H1+H2 false components decrease by at least `1%`.
7. Pooled Score delta is at least `0.0002`.
8. The independently selected H1 and H2 winner candidate IDs are identical.

If any gate fails, the run writes an audit report but no deployable model.  If
all gates pass, the common v1 winner or the sole v2 candidate is refit once on
all 54 videos with the same domain/video/component weighting.  The strict
JSON artifact is compatible with the existing default-off runtime and records
the train-only consistency report, protocol, cache, document, and code
SHA-256 provenance.  A v2 artifact additionally records its retrospective,
non-independent hypothesis report, diagnostic, and validation-policy lineage.
NumPy, PyTorch, and OpenCV versions are frozen into the protocol and artifact.
For a passing run, the artifact and report are fully serialized and staged
before either final path is published; if the second publish fails, the first
is removed so no unaudited orphan model remains.

## Command boundary

`preregister` is run and its printed protocol-file SHA-256 is recorded before
`run` is allowed.  `run` requires that SHA-256 explicitly.  Neither command
accepts a validation-data path, and neither command performs GPU inference.

`--candidate-profile` defaults to `conservative_v1`, which retains the exact
eight-candidate v1 behavior.  V2 must be requested explicitly and must supply
both frozen hypothesis sources and their exact digests:

```powershell
python crossfit_component_reranker.py preregister `
  --candidate-profile posthoc_pw4_kp040_v2 `
  --hypothesis-source-report "F:\小目标检测\experiments\20260810_component_reranker_crosssource_v1\crossfit_report.json" `
  --expected-hypothesis-source-report-sha256 e06182a03667e169b16fee7b02e7e44dd636457bc2b4d6f62265dcb6300577d0 `
  --hypothesis-source-diagnostic "F:\小目标检测\experiments\20260810_component_reranker_crosssource_v1\posthoc_threshold_diagnostic.json" `
  --expected-hypothesis-source-diagnostic-sha256 5ba11e24c8f820773b9baf2c1b778131cdf30876dffbc64b4f2d4c06b17ef8e3 `
  --cache-dir <immutable-train-cache> `
  --config configs/evisseg_evuav.yaml `
  --expected-selected-videos 54 `
  --output-protocol <new-v2-protocol.json> `
  <the exact frozen C00 --override arguments>
```

The preregistered protocol embeds these source paths and hashes, the fixed
validation-policy path and hash, and all shared cache/config/code/document
lineage.  `run` rebuilds and revalidates that entire definition before and
after fitting.  It does not accept a replacement diagnostic, report, profile,
or candidate from the command line.
