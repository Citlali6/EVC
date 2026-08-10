# Temporal input route: one-shot 24-val protocol

## Frozen candidate

This protocol is opt-in and does not change `submit_challenge2.py` or any
default submission configuration.  The candidate is fixed as follows:

| Complete-video input condition | Checkpoint / temporal mode | Threshold |
| --- | --- | ---: |
| `event_count <= 30000` | released M10, full T160 | `0.718` |
| `30000 < event_count <= 200000` | released M20, full T160 | `0.719` |
| `event_count > 200000` and polarity minority fraction `< 0.20` | released M20, full T160 | `0.719` |
| `event_count > 200000` and polarity minority fraction `>= 0.20` | released M20, T32 / stride 16 | `0.719` |

The polarity minority fraction is
`min(mean(polarity > 0.5), 1 - mean(polarity > 0.5))`, computed over every
input event exactly once.  It uses neither labels nor source names.  The
postprocessor is exact released C00.  The persistent-pixel second stage is
disabled.

The machine-readable science protocol is
`protocols/temporal_input_route_val24_science.json`.  It embeds the exact 24
official filenames, sizes and SHA-256 values copied from the earlier frozen T32
execution protocol; creating it required no new validation-file read.

## Train prerequisite

Formal validation is fail-closed unless the completed 99-train v3 evidence is
present at its canonical path and has all three exact hashes:

| Evidence | SHA-256 |
| --- | --- |
| score-cache manifest v3 | `78ca63efd1fd8fda62dcccb1203f0e69000007454a391b7d46455f9952cf2dc7` |
| fixed-route evaluation v3 | `4e3610a057c9c330b18f9d0712d57fe80a1463bc4ae567f64172986409d6e956` |
| frozen train protocol v3 | `ddd027961bc36f2756a62cd62914c5be3400a2ddd965d53ab2ff066b331f36d1` |

The runner also parses the hashed evidence and requires all train gates:

- all 99 sources are present, with 45 M10/full, 43 M20/full and 11 M20/T32;
- all 88 non-H2 score vectors are bitwise identical to baseline;
- all 11 H2 score vectors differ from baseline;
- pooled Score delta is `+0.005269895410325631`;
- pooled Pd and IoU do not decrease, and Fa does not increase.

The complete-train result is selection evidence, not an independent score
claim.  The single frozen 24-val attempt is the held-data confirmation.

One P2 provenance limitation is frozen explicitly: the historical train-v3
manifest recorded the direct runner/router/window/inference/postprocess/eval
hashes, but not every transitive dataset/model/multiscale/reranker file.  Exact
historical-to-val equivalence for those omitted files cannot be reconstructed
after the fact.  This protocol does not rerun or rewrite train-v3; it mitigates
the gap by binding the formal replay to a clean git commit and an expanded set
of current runtime code hashes.  This limitation must remain attached to any
claim made from the one-shot result.

## Immutable validation inputs

The execution protocol binds:

- released M10 checkpoint SHA-256
  `5c89c89a165469c0a4e8286d4644d60d2f82cf5775edbb724f626e24e67d8935`;
- released M20 checkpoint SHA-256
  `4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849`;
- frozen M10 raw cache SHA-256
  `96a9dfa8833e6f609d29f4db9d8f7196c84c7e92c7026cce734b97ddf133622f`;
- frozen M20 raw cache SHA-256
  `6c9b4a8e33217aac7a05c78590a7feb6db6e6fc332b6411d7603264687710304`;
- golden C00 report SHA-256
  `da6004ddd22731b8e848c9ed0c561961abbc04b4e3f66cd07b1e085d26f9f383`;
- the route-policy digest, exact C00 mapping, thresholds, persistence-off
  contract, current git commit and every runtime code-file SHA-256.

`freeze`, `preflight`, and `runtime-preflight` intentionally do not open the
official manifest, either raw validation cache, a validation NPZ, or a label.
The GPU runtime preflight checks the exact train-v3 Python/Torch/NumPy/Windows
environment, CUDA device, checkpoint loader, route API, and a synthetic M20
T32/stride-16 forward pass.  `run` repeats that smoke check before creating and
flushing the exclusive attempt claim; only then can the first validation-bound
read occur.

## Cache reuse and one-shot execution

The baseline is reconstructed from the frozen score caches: M10 for
`event_count <= 30000`, otherwise M20.  Candidate records in low, middle and H1
reuse the same underlying baseline storage and must pass a bitwise identity
check.  Only H2 records may invoke the model, and only with M20 T32 / stride 16.
The runner writes a partial H2-only cache; it never regenerates low, middle, H1,
M10-full or M20-full scores.

The claim is created with exclusive-create semantics and is never removed,
including on a crash.  Its presence consumes attempt 1 of 1.  A failure report
is written when possible, and the candidate is archived without validation
retuning or a second attempt.

## Promotion gates

All gates are mandatory:

1. the cache replay reproduces the released golden counts and metrics exactly;
2. candidate `Score > golden Score + 0.0001` (strict inequality);
3. candidate Pd is not below golden Pd;
4. candidate IoU is not below golden IoU;
5. candidate Fa is not above golden Fa;
6. every non-H2 score vector is bitwise preserved;
7. the number of inference calls equals the number of H2 routes;
8. persistence remains disabled.

No threshold grid, profile selection, ZIP creation or platform upload is
authorized.

## Commands (prepared, not executed)

First commit the reviewed candidate and runner so the worktree is clean.  Then
freeze the execution protocol; this stage is CPU-only and validation-blind:

```powershell
Set-Location 'F:\小目标检测\EVC-work'
python .\evaluate_temporal_input_route_validation.py freeze
```

Copy the printed execution-protocol SHA-256 verbatim.  The following preflight
is also CPU-only and validation-blind:

```powershell
$ProtocolSha = '<exact SHA-256 printed by freeze>'
python .\evaluate_temporal_input_route_validation.py preflight `
  --expected-execution-protocol-sha256 $ProtocolSha
```

Stop here until the one-shot validation run is explicitly authorized.  After
authorization, create the immutable runtime receipt with the known working
CUDA/spconv Python.  A failure here consumes no validation attempt:

```powershell
$GpuPython = 'C:\Users\CSK\.conda\envs\yolo\python.exe'
& $GpuPython .\evaluate_temporal_input_route_validation.py runtime-preflight `
  --expected-execution-protocol-sha256 $ProtocolSha
```

Only after that receipt passes, use the same Python for the formal command:

```powershell
& $GpuPython .\evaluate_temporal_input_route_validation.py run `
  --expected-execution-protocol-sha256 $ProtocolSha
```

Canonical outputs are under
`F:\小目标检测\experiments\20260810_temporal_input_route_frozen_val24`:

- `preregistered_execution_protocol.json`;
- `runtime_preflight_receipt.json`;
- `validation_attempt_claim.json`;
- `raw_m20_t32_stride16_h2_only.pt`;
- `frozen_validation_report.json`.
