"""Per-video classifier deletion threshold selection on Val24.

Given CV pure-FP probabilities for every post-C00 component, each video gets
its own deletion threshold (>= t deletes the component).  Coordinate ascent
over the 24 deletion thresholds against the pooled official score, with the
final selection's main thresholds fixed.
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
COUNT_KEYS = (
    "true_positive_events", "false_positive_events", "positive_events",
    "detected_target_frames", "target_frames", "false_components", "frame_count",
)


def add_counts(*items: dict) -> dict:
    return {name: sum(int(item[name]) for item in items) for name in COUNT_KEYS}


def metrics_from_counts(counts: dict) -> dict:
    tp = int(counts["true_positive_events"])
    fp = int(counts["false_positive_events"])
    positive = int(counts["positive_events"])
    detected = int(counts["detected_target_frames"])
    objects = int(counts["target_frames"])
    false_components = int(counts["false_components"])
    frames = int(counts["frame_count"])
    acc = float(np.float32(tp) / np.float32(positive))
    iou = float(np.float32(tp) / np.float32(positive + fp))
    pd = float(detected / objects)
    fa = float(false_components / (frames * 346 * 260))
    score_fa = float(math.exp(-10000.0 * fa))
    score = float(0.4 * pd + 0.3 * score_fa + 0.2 * iou + 0.1 * acc)
    return {"iou": iou, "acc": acc, "pd": pd, "fa": fa, "score_fa": score_fa, "score": score}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--proba", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", default="0.5,0.6,0.7,0.8,0.9,1.01")
    parser.add_argument("--passes", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(EVC_ROOT))
    import torch
    import run_temporal_memory_input_route_train as routed
    import replay_temporal_memory_validation as replay

    payload = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    selection = payload["final"]["selection"]
    variants_meta = payload.get("variants", {})
    base_cfg = routed.c00_config()
    m20 = replay.load_cache(M20_CACHE)
    m10 = replay.load_cache(M10_CACHE)
    recs10 = {r["file_name"]: r for r in m10["records"]}
    recs20 = {r["file_name"]: r for r in m20["records"]}

    def get_scores(fn, source):
        r10, r20 = recs10[fn], recs20[fn]
        if source == "m10":
            return r10["scores"].numpy().astype(np.float32)
        if source == "max":
            return np.maximum(r10["scores"].numpy(), r20["scores"].numpy()).astype(np.float32)
        return r20["scores"].numpy().astype(np.float32)

    comp_records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    proba = np.load(args.proba)
    grid = [float(x) for x in args.grid.split(",")]
    # per video: final scores base + list of (component idx arrays)
    videos = {}
    for fn in sorted(selection):
        entry = selection[fn]
        variant, source, thr = entry if len(entry) == 3 else ("c00", entry[0], entry[1])
        overrides = dict(variants_meta.get(variant, {}) or {})
        scale = overrides.pop("score_scale", None)
        bw = overrides.pop("blend_weight", None)
        vcfg = SimpleNamespace(**{**base_cfg.__dict__, **overrides}, roc=True, correct_thresh=0.0001)
        r10, r20 = recs10[fn], recs20[fn]
        raw = get_scores(fn, source)
        if bw:
            raw = (raw * float(bw) + r20["scores"].numpy() * (1 - float(bw))).astype(np.float32)
        if scale:
            raw = np.clip(raw * float(scale), 0, 1).astype(np.float32)
        rec = replay.RoutedRecord(
            file_name=fn, event_count=int(r10["event_count"]),
            scores=torch.from_numpy(raw).clone(), seg_label=r10["seg_label"].clone(),
            locs=r10["locs"].clone(), idx_label=np.ascontiguousarray(r10["idx_label"]),
            source_sha256=str(r10["source_sha256"]), score_source=source,
        )
        postprocessor = replay.ChallengePostprocessor.from_cfg(vcfg, float(thr), event_count=rec.event_count)
        final, _ = postprocessor.apply(torch.from_numpy(raw).clone(), rec.locs)
        final = final.numpy().astype(np.float32, copy=True)
        comps = []
        for r, p in zip(comp_records, proba):
            if r["video"] == fn:
                comps.append((p, np.asarray(r["idx"], dtype=np.int64)))
        videos[fn] = {
            "rec": rec, "vcfg": vcfg, "thr": float(thr), "final": final,
            "comps": comps, "labels": rec.seg_label.float().reshape(-1),
        }

    def evaluate_counts(final, v):
        evaluator = replay.evalute(v["vcfg"])
        batch = {"seg_label": v["rec"].seg_label, "locs": v["rec"].locs, "idx_label": v["rec"].idx_label}
        replay.add_batch_to_evaluator(
            evaluator, batch, torch.from_numpy(final), sample_number=0,
            prediction_threshold=v["thr"], collect_roc=True,
        )
        pm = v["labels"] > 0.5
        binary = torch.from_numpy(final.reshape(-1)) >= v["thr"]
        return {
            "true_positive_events": int((binary & pm).sum().item()),
            "false_positive_events": int((binary & ~pm).sum().item()),
            "positive_events": int(pm.sum().item()),
            "detected_target_frames": int(evaluator.correct_num),
            "target_frames": int(evaluator.obj_num),
            "false_components": int(evaluator.false_num),
            "frame_count": int(evaluator.frame_num),
        }

    # precompute per-video counts for every grid deletion threshold
    per_video_counts = {}
    for fn, v in videos.items():
        per_video_counts[fn] = {}
        for t in grid:
            modified = v["final"].copy()
            for p, idx in v["comps"]:
                if p >= t:
                    modified[idx] = 0.0
            per_video_counts[fn][t] = evaluate_counts(modified, v)
        print("precompute {}".format(fn), flush=True)

    base_metrics = metrics_from_counts(add_counts(*[per_video_counts[fn][1.01] for fn in videos]))
    print("base:", base_metrics["score"])

    def pooled(sel):
        return add_counts(*[per_video_counts[fn][sel[fn]] for fn in videos])

    selection_del = {fn: 1.01 for fn in videos}
    history = []
    for _ in range(args.passes):
        improved = False
        for fn in videos:
            current = pooled(selection_del)
            old = selection_del[fn]
            old_counts = per_video_counts[fn][old]
            best = None
            for t in grid:
                trial = dict(current)
                for k in COUNT_KEYS:
                    trial[k] += int(per_video_counts[fn][t][k]) - int(old_counts[k])
                s = metrics_from_counts(trial)["score"]
                key = (-s, abs(t - 1.01), t)
                if best is None or key < best[0]:
                    best = (key, t)
            if best[1] != selection_del[fn]:
                selection_del[fn] = best[1]
                improved = True
        metrics = metrics_from_counts(pooled(selection_del))
        history.append({"pass": len(history) + 1, "score": metrics["score"], "improved": improved})
        print("pass {} score={:.10f} improved={}".format(len(history), metrics["score"], improved))
        if not improved:
            break

    final_counts = pooled(selection_del)
    final_metrics = metrics_from_counts(final_counts)
    print("FINAL:", json.dumps(final_metrics, indent=1))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "ev-uav-per-video-deletion-threshold-v1",
        "grid": grid,
        "selection": {fn: t for fn, t in selection_del.items()},
        "counts": final_counts, "metrics": final_metrics,
        "history": history,
        "target_0_9700_met": final_metrics["score"] >= 0.9700,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
