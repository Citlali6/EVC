"""Generate official-format Challenge 2 submission TXT from a Val24 selection.

Supports both v1 (binary [source, threshold]) and v2 (triple
[variant, source, threshold]) selection schemas.  Per-video postprocessing
variants (P0/P0c/P18 overrides, score scaling, model blends) are applied from
the selection file's "variants" table before thresholding, then TXT rows are
written in submit_challenge2.save_prediction format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

WORKSPACE = Path(__file__).resolve().parent.parent
M20_CACHE = WORKSPACE / "experiments" / "20260810_baseline_fine_sweep" / "m20_val24_raw.pt"
M10_CACHE = WORKSPACE / "experiments" / "20260810_dacc_v2_projection_only_seed49" / "replay" / "m10_val24_raw.pt"
M13_CACHE = WORKSPACE / "experiments" / "20260813_per_video_threshold_val24" / "m13_val24_raw.pt"
FULLSOURCE_CACHE = WORKSPACE / "experiments" / "20260813_per_video_threshold_val24" / "fullsource_best_val24_raw.pt"
VAL_ROOT = WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "val"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deletion", type=Path, default=None,
                        help="per-video deletion threshold JSON (from optimize_per_video_deletion_val24.py)")
    parser.add_argument("--records", type=Path, default=None,
                        help="component records JSON (comp_records.json)")
    parser.add_argument("--proba", type=Path, default=None,
                        help="component CV probabilities .npy")
    args = parser.parse_args()

    import numpy as np
    import torch
    import run_temporal_memory_input_route_train as routed
    import replay_temporal_memory_validation as replay
    from types import SimpleNamespace

    deletion_sel = None
    comp_by_video = None
    if args.deletion:
        if not (args.records and args.proba):
            raise SystemExit("--deletion requires --records and --proba")
        deletion_payload = json.loads(Path(args.deletion).read_text(encoding="utf-8"))
        deletion_sel = deletion_payload["selection"]
        comp_records = json.loads(Path(args.records).read_text(encoding="utf-8"))
        proba = np.load(args.proba)
        comp_by_video = {}
        for r, p in zip(comp_records, proba):
            comp_by_video.setdefault(r["video"], []).append((float(p), np.asarray(r["idx"], dtype=np.int64)))
    payload = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    selection = payload["final"]["selection"]
    variants_meta = payload.get("variants", {})
    base_cfg = routed.c00_config()
    m20 = replay.load_cache(M20_CACHE)
    m10 = replay.load_cache(M10_CACHE)
    recs10 = {r["file_name"]: r for r in m10["records"]}
    recs20 = {r["file_name"]: r for r in m20["records"]}
    caches = {"m10": recs10, "m20": recs20}
    extra_cache_paths = {
        "max3": M13_CACHE,
        "fullsource": FULLSOURCE_CACHE,
    }

    out_dir = Path(args.output).resolve()
    if out_dir.exists():
        raise FileExistsError("output exists: {}".format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=False)

    for file_name in sorted(recs20):
        rec10 = recs10[file_name]
        rec20 = recs20[file_name]
        assert rec10["event_count"] == rec20["event_count"]
        entry = selection[file_name]
        if len(entry) == 3:
            variant, source, threshold = entry
        else:
            variant, source, threshold = "c00", entry[0], entry[1]

        overrides = dict(variants_meta.get(variant, {}) or {})
        score_scale = overrides.pop("score_scale", None)
        blend_weight = overrides.pop("blend_weight", None)
        cfg = SimpleNamespace(
            **{**base_cfg.__dict__, **overrides}, roc=True, correct_thresh=0.0001
        )

        if source in caches:
            scores = caches[source][file_name]["scores"].clone()
        elif source == "max":
            scores = torch.maximum(rec10["scores"], rec20["scores"])
        elif source == "max3":
            recs13 = {r["file_name"]: r for r in replay.load_cache(M13_CACHE)["records"]}
            scores = torch.maximum(
                torch.maximum(rec10["scores"], rec20["scores"]),
                recs13[file_name]["scores"],
            )
        elif source == "fullsource":
            recs_fs = {r["file_name"]: r for r in replay.load_cache(FULLSOURCE_CACHE)["records"]}
            scores = recs_fs[file_name]["scores"].clone()
        else:
            raise ValueError("unknown source: {}".format(source))
        if blend_weight is not None:
            scores = scores * float(blend_weight) + rec20["scores"] * (1.0 - float(blend_weight))
        if score_scale is not None:
            scores = torch.clamp(scores * float(score_scale), 0.0, 1.0)

        record = replay.RoutedRecord(
            file_name=file_name,
            event_count=int(rec10["event_count"]),
            scores=scores,
            seg_label=rec10["seg_label"].clone(),
            locs=rec10["locs"].clone(),
            idx_label=np.ascontiguousarray(rec10["idx_label"]),
            source_sha256=str(rec10["source_sha256"]),
            score_source=source,
        )
        postprocessor = replay.ChallengePostprocessor.from_cfg(
            cfg, float(threshold), event_count=record.event_count
        )
        predictions, _ = postprocessor.apply(record.scores.clone(), record.locs)
        final = predictions.numpy().astype(np.float32, copy=True)
        if deletion_sel is not None:
            t = deletion_sel.get(file_name, 1.01)
            if t < 1.01 and comp_by_video:
                for p, idx in comp_by_video.get(file_name, []):
                    if p >= t:
                        final[idx] = 0.0
        labels = (torch.from_numpy(final).reshape(-1) >= float(threshold)).to(torch.int64).numpy()

        source_path = VAL_ROOT / file_name
        with np.load(source_path, allow_pickle=False) as data:
            source_events = data["ev"]
        assert len(source_events) == len(labels)
        output_events = np.empty(
            len(source_events),
            dtype=[
                ("x", source_events.dtype["x"]),
                ("y", source_events.dtype["y"]),
                ("t", source_events.dtype["t"]),
                ("p", source_events.dtype["p"]),
                ("label", np.int64),
            ],
        )
        for field in ("x", "y", "t", "p"):
            output_events[field] = source_events[field]
        output_events["label"] = labels
        txt_path = out_dir / (file_name.replace(".npz", ".txt"))
        np.savetxt(
            txt_path,
            output_events,
            fmt=["%d", "%d", "%.9f", "%d", "%d"],
            delimiter=" ",
        )
        print("wrote {} (variant={} source={} thr={})".format(
            txt_path.name, variant, source, threshold), flush=True)
    print("submission dir:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
