"""Feature separability of pure-FP vs GT-containing components on Val24.

For the merged per-video selection, extracts post-C00 atomic components and
compares observable features (event count, temporal duration, spatial size,
score stats, polarity, per-video density) between pure-FP components
(no GT events) and components containing GT events.  Goal: find a label-free
rule that approximates the pure-FP deletion gain.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
EVC_ROOT = Path(__file__).resolve().parent
M20_CACHE = WORKSPACE / "experiments" / "20260810_baseline_fine_sweep" / "m20_val24_raw.pt"
M10_CACHE = WORKSPACE / "experiments" / "20260810_dacc_v2_projection_only_seed49" / "replay" / "m10_val24_raw.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(EVC_ROOT))
    import torch
    import run_temporal_memory_input_route_train as routed
    import replay_temporal_memory_validation as replay
    from utils.atomic_component_deletion import extract_atomic_components

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))["final"]["selection"]
    cfg = SimpleNamespace(**routed.c00_config().__dict__, roc=True, correct_thresh=0.0001)
    m20 = replay.load_cache(M20_CACHE)
    m10 = replay.load_cache(M10_CACHE)
    recs10 = {r["file_name"]: r for r in m10["records"]}
    recs20 = {r["file_name"]: r for r in m20["records"]}

    rows = []
    for fn in sorted(selection):
        source, thr = selection[fn]
        rec10 = recs10[fn]
        rec20 = recs20[fn]
        scores = rec10 if source == "m10" else rec20
        rec = replay.RoutedRecord(
            file_name=fn, event_count=int(rec10["event_count"]),
            scores=scores["scores"].clone(), seg_label=rec10["seg_label"].clone(),
            locs=rec10["locs"].clone(), idx_label=np.ascontiguousarray(rec10["idx_label"]),
            source_sha256=str(rec10["source_sha256"]), score_source=source,
        )
        postprocessor = replay.ChallengePostprocessor.from_cfg(cfg, float(thr), event_count=rec.event_count)
        final, _ = postprocessor.apply(rec.scores.clone(), rec.locs)
        final = final.numpy().astype(np.float32, copy=True)
        comps = extract_atomic_components(
            final, rec.locs, float(thr),
            spatial_radius=2, temporal_bin_size=50, temporal_radius_bins=1,
        ).event_indices
        labels_np = rec.seg_label.numpy().reshape(-1)
        scores_np = scores["scores"].numpy().reshape(-1)
        locs_np = rec.locs.numpy()
        ts = locs_np[:, 3]
        # polarity from event coordinate p: ev_loc column 2 is p in source cache? locs[:, 0] is batch, [1] x, [2] y, [3] t. polarity comes from evs_norm col 3.
        for indices in comps:
            idx = np.asarray(indices, dtype=np.int64)
            n_gt = int(labels_np[idx].sum())
            n = idx.size
            bins = np.unique(ts[idx] // 50)
            x = locs_np[idx, 1]
            y = locs_np[idx, 2]
            bbox = float(np.hypot(x.max() - x.min(), y.max() - y.min()))
            rows.append({
                "video": fn,
                "n_events": n,
                "duration_bins": int(len(bins)),
                "bbox": bbox,
                "score_min": float(scores_np[idx].min()),
                "score_mean": float(scores_np[idx].mean()),
                "score_max": float(scores_np[idx].max()),
                "n_gt": n_gt,
                "pure_fp": int(n_gt == 0),
            })
        print("{}: {} comps".format(fn, len(comps)), flush=True)

    arr = np.array([(r["n_events"], r["duration_bins"], r["bbox"], r["score_min"], r["score_mean"], r["score_max"], r["pure_fp"]) for r in rows])
    pure = arr[arr[:, 6] == 1]
    gt = arr[arr[:, 6] == 0]
    print("total comps: {} pure_fp: {} gt-containing: {}".format(len(arr), len(pure), len(gt)))
    names = ["n_events", "duration_bins", "bbox", "score_min", "score_mean", "score_max"]
    for i, name in enumerate(names):
        print("{}: pure_fp mean={:.3f} median={:.3f} | gt mean={:.3f} median={:.3f}".format(
            name, pure[:, i].mean(), np.median(pure[:, i]), gt[:, i].mean(), np.median(gt[:, i])))

    # simple label-free candidate rules, evaluated by hit/miss against label truth
    rules = {
        "n<=3": arr[:, 0] <= 3,
        "n<=2": arr[:, 0] <= 2,
        "n==1": arr[:, 0] == 1,
        "dur<=2": arr[:, 1] <= 2,
        "dur==1": arr[:, 1] == 1,
        "bbox<=10": arr[:, 2] <= 10.0,
        "bbox<=6": arr[:, 2] <= 6.0,
        "bbox<=3": arr[:, 2] <= 3.0,
        "score_max<0.8": arr[:, 5] < 0.8,
        "score_max<0.75": arr[:, 5] < 0.75,
        "score_mean<0.75": arr[:, 4] < 0.75,
        "n<=2&dur<=1&bbox<=6": (arr[:, 0] <= 2) & (arr[:, 1] <= 1) & (arr[:, 2] <= 6.0),
        "n<=3&dur<=2&bbox<=8": (arr[:, 0] <= 3) & (arr[:, 1] <= 2) & (arr[:, 2] <= 8.0),
        "n<=4&dur<=2": (arr[:, 0] <= 4) & (arr[:, 1] <= 2),
    }
    print("\nlabel-free rule sweep (on Val24 components):")
    for name, mask in rules.items():
        deleted = mask
        kept = ~mask
        tp_loss = int(((deleted) & (arr[:, 6] == 0)).sum())   # gt components wrongly deleted
        fp_removed = int(((deleted) & (arr[:, 6] == 1)).sum())  # pure fp removed
        fp_kept = int(((kept) & (arr[:, 6] == 1)).sum())
        print("  {}: deleted={} fp_removed={} tp_loss={} fp_kept={} precision={:.3f}".format(
            name, int(deleted.sum()), fp_removed, tp_loss, fp_kept,
            fp_removed / max(1, int(deleted.sum()))))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "ev-uav-component-features-val24-v1", "components": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
