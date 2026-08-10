# Track-edge weak-event recovery: frozen train-only MVP protocol

## Scope and non-claims

This experiment is a conservative recovery counterpart to the suppression-only
component reranker.  It consumes the immutable M20 official-train cache and
does not accept, read, or score a validation path.  It is an incremental-module
cross-source audit conditional on the already trained M20 checkpoint, not an
unbiased end-to-end OOF estimate.

This MVP has no runtime integration and emits no deployable artifact.  Passing
the gates only authorizes a later code review about default-off runtime wiring;
it does not authorize validation access or a submission.

## Immutable training population

- Cache split: official `train` only.
- Cache selection: complete-video `event_count > 30000`.
- Exact population: 54 videos and 8,555,762 events.
- Base checkpoint SHA-256:
  `4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849`.
- High source block H1: `train_044.npz` through `train_047.npz`.
- High source block H2: `train_088.npz` through `train_098.npz`.
- The remaining 39 cached videos form the middle-density auxiliary fit domain.

The protocol builder verifies the complete official-source manifest, every
selected record path and SHA, the cache manifest SHA, config, code, and M20
identity before it writes a preregistration file.  It also freezes the NumPy,
PyTorch, and OpenCV versions because official false-alarm connected-component
counts use OpenCV, and it rejects any mismatch between the candidate temporal
bin width and the official Pd detection interval.

## Label-free candidate semantics

Candidate extraction accepts only raw M20 scores, complete C00 scores, `x/y/t`
locations, and complete-video event count.  Its function signature does not
accept labels or target IDs.

The frozen observable rules are:

- deployment eligibility is `event_count > 100000` (middle videos are used only
  as a separately weighted fit domain);
- weak scores satisfy `0.53 <= raw_score < 0.719`;
- seed scores are final C00 scores `>= 0.719`;
- temporal bins have width 50;
- spatial components use radius 5;
- seed components link only across the adjacent bin and within 8 pixels;
- a seed track contains at least four linked bins;
- weak components are considered only in the bin immediately before the first
  seed bin or immediately after the last seed bin;
- a weak component that could attach to two endpoints is assigned once by
  motion residual, endpoint distance, then deterministic IDs;
- each weak component proposes its highest-raw-score event;
- later selection can recover at most one event for each track endpoint.

The 15 inference-observable features contain score/component statistics,
seed-track length and endpoint confidence, last velocity, constant-velocity
residual, turning residual, direction cosine, speed ratio, local event density,
event count, and endpoint side.  Absolute coordinates, absolute timestamps,
file/source identity, labels, and target IDs are not features.

## Train-only targets and metric utility

After features are frozen, a separate training function attaches event labels,
target IDs, and the exact local 8-connected false-component delta, including
the unchanged evaluator's uint8 per-pixel accumulation semantics.  It rejects
any purported recovery candidate that is already baseline-positive.  For Pd it
counts the complete official open-interval target group and its existing C00
correct events, then marks a recovery only for the exact transition
`correct/total < 0.0001` to `(correct+1)/total >= 0.0001`.  It therefore handles
both a one-event recovery that is still insufficient and an existing partial
hit that crosses the official fraction only after this candidate.

For each fit fold, candidate weight is the absolute one-action change in the
official Challenge score computed from that fold's aggregate fit-side
sufficient counts.  A positive event converts one FN to TP and may recover one
Pd group.  A false event adds one FP and its non-negative false-component cost.
An accidental merge of existing false components is clipped to zero so a false
event can never receive a positive reward.

Hierarchical base mass is fixed:

- fit-high domain: 0.5;
- middle domain: 0.5;
- candidate-bearing videos are equal within a domain;
- inferred track endpoints are equal within a video;
- candidates are equal within an endpoint.

There is no positive oversampling, held-label early stopping, candidate grid,
or decision-threshold sweep.

## Real optimization

Each fold trains exactly the following CPU model:

```text
15 standardized features -> Linear(15,8) -> tanh -> Linear(8,1)
```

- trainable parameters: 137;
- loss: official-marginal-utility-weighted binary cross entropy;
- optimizer: AdamW;
- seed: 53;
- learning rate: 0.002;
- weight decay: 0.001;
- betas: `(0.9, 0.999)`;
- full batch, exactly 200 optimizer steps;
- action decision: logit `>= 0`;
- no model selection and no early stopping.

The report records initial/final loss, a 25-step loss trace, parameter delta,
initial/final state SHA, optimizer step, Adam moment tensor count/norm, class
counts, and complete fold model weights.  Missing/non-finite moments, unchanged
parameters, or non-decreasing loss fails the run.

## Cross-source folds

- `holdout_h1`: fit H2 plus all middle auxiliary videos; score only H1.
- `holdout_h2`: fit H1 plus all middle auxiliary videos; score only H2.

Fit and held source paths are checked for disjointness.  Standardization,
utility counts, model parameters, and optimizer state are fit independently in
each fold.  Middle videos receive identity recovery in pooled OOF scoring.

## Promotion gates

All checks must pass:

1. H1 and H2 each have strictly positive Score delta.
2. H1 and H2 each recover at least one new Pd group.
3. H1 and H2 each have non-decreasing IoU.
4. Added false components per new Pd group are at most 4 in each fold and at
   most 3 after pooling both high blocks.
5. Each held block has positive candidates in at least two distinct videos.
6. Both folds contain real, finite 200-step AdamW evidence.
7. H1+H2 OOF plus middle identity has strictly higher Pd, non-decreasing IoU,
   and Score delta at least `0.0002`.

If any gate fails, preserve the negative report and do not change geometry,
score floors, features, loss, or decision threshold based on the result.

## Execution order

Use the same C00 overrides listed in
`docs/COMPONENT_RERANKER_CROSSFIT.md`, then preregister before reading any real
candidate or held-block result:

```powershell
& 'C:\Users\CSK\.conda\envs\yolo\python.exe' train_track_edge_recovery.py preregister `
  --cache-dir '<AUDIT_DIR>\train_cache_gt30000' `
  --config 'configs\evisseg_evuav.yaml' `
  --output-protocol '<AUDIT_DIR>\track_edge_protocol.json' `
  <C00 --override arguments>
```

Record the printed protocol-file SHA externally, then run exactly that file:

```powershell
& 'C:\Users\CSK\.conda\envs\yolo\python.exe' train_track_edge_recovery.py run `
  --protocol '<AUDIT_DIR>\track_edge_protocol.json' `
  --expected-protocol-sha256 '<RECORDED_SHA256>' `
  --cache-dir '<AUDIT_DIR>\train_cache_gt30000' `
  --output-report '<AUDIT_DIR>\track_edge_crossfit_report.json'
```
