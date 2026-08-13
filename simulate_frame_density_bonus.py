"""Simulate per-frame low-density score bonus on Val24 (official evaluator).

For frames whose event count is below a density threshold C, add a score bonus
delta to every event in that frame before postprocessing and thresholding.
Frame density is observable (event timestamps), so the rule is label-free.
Scans a small grid of (C, delta) and reports pooled official scores.
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
    parser.add_argument("--output", type=Path, required=True)
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

    # precompute per-video base postprocess outputs (no bonus)
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
        ts = rec.locs.numpy()[:, 3]
        videos[fn] = {
            "rec": rec, "vcfg": vcfg, "thr": float(thr),
            "raw": raw, "final": final.numpy().astype(np.float32),
            "ts": ts, "labels": rec.seg_label.numpy().reshape(-1),
            "idx": rec.idx_label,
        }

    def evaluate(final_scores, v):
        evaluator = replay.evalute(v["vcfg"])
        batch = {"seg_label": v["rec"].seg_label, "locs": v["rec"].locs, "idx_label": v["rec"].idx_label}
        replay.add_batch_to_evaluator(
            evaluator, batch, torch.from_numpy(np.asarray(final_scores, dtype=np.float32)),
            sample_number=0,
            prediction_threshold=v["thr"], collect_roc=True,
        )
        labels = v["labels"]
        pm = labels > 0.5
        binary = final_scores.reshape(-1) >= v["thr"]
        return {
            "true_positive_events": int((binary & pm).sum().item()),
            "false_positive_events": int((binary & ~pm).sum().item()),
            "positive_events": int(pm.sum().item()),
            "detected_target_frames": int(evaluator.correct_num),
            "target_frames": int(evaluator.obj_num),
            "false_components": int(evaluator.false_num),
            "frame_count": int(evaluator.frame_num),
        }

    base_pooled = add_counts(*[evaluate(v["final"], v) for v in videos.values()])
    base_metrics = metrics_from_counts(base_pooled)
    print("baseline:", json.dumps(base_metrics, indent=1))

    grid = []
    for C in (300, 500, 800, 1200, 2000):
        for delta in (0.03, 0.05, 0.08, 0.12):
            grid.append((C, delta))

    results = {}
    for C, delta in grid:
        pooled = None
        for fn, v in videos.items():
            ts = v["ts"]
            frame_id = ts // 50
            counts_per_frame = np.bincount(frame_id)
            low_mask = counts_per_frame[frame_id] < C
            modified = v["final"].copy()
            # add bonus only to low-density frame events that are below threshold
            bonus_mask = low_mask & (modified < v["thr"])
            modified[bonus_mask] = np.clip(modified[bonus_mask] + delta, 0.0, 1.0)
            counts = evaluate(torch.from_numpy(modified), v)
            pooled = counts if pooled is None else add_counts(pooled, counts)
        metrics = metrics_from_counts(pooled)
        results["C{}_d{}".format(C, delta)] = {"counts": pooled, "metrics": metrics,
                                               "delta": metrics["score"] - base_metrics["score"]}
        print("C={:4d} delta={:.2f}: score={:.10f} delta={:+.6f} pd={:.5f} fa={:.3e}".format(
            C, delta, metrics["score"], metrics["score"] - base_metrics["score"],
            metrics["pd"], metrics["fa"]))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "ev-uav-frame-density-bonus-sim-v1",
        "baseline": {"counts": base_pooled, "metrics": base_metrics},
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
