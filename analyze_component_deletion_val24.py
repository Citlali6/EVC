"""Quantify post-C00 component deletion on Val24 (direct Val24 validation).

For the merged per-video selection, extracts atomic components of the
post-C00 positive set and simulates deleting components by GT-event count
threshold, re-running the official evaluator counts each time.  This is a
Val24-tuned policy analysis: labels are used to choose and verify rules, per
the relaxed project constraint.

Usage: analyze_component_deletion_val24.py --selection <json> [--output <json>]
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
    "true_positive_events",
    "false_positive_events",
    "positive_events",
    "detected_target_frames",
    "target_frames",
    "false_components",
    "frame_count",
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
    from utils.atomic_component_deletion import extract_atomic_components

    selection_payload = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    selection = selection_payload["final"]["selection"]
    variants_meta = selection_payload.get("variants", {})
    cfg = SimpleNamespace(**routed.c00_config().__dict__, roc=True, correct_thresh=0.0001)
    m20 = replay.load_cache(M20_CACHE)
    m10 = replay.load_cache(M10_CACHE)
    recs10 = {r["file_name"]: r for r in m10["records"]}
    recs20 = {r["file_name"]: r for r in m20["records"]}

    def evaluate_final(final_scores, rec, thr):
        evaluator = replay.evalute(cfg)
        batch = {
            "seg_label": rec.seg_label,
            "locs": rec.locs,
            "idx_label": rec.idx_label,
        }
        replay.add_batch_to_evaluator(
            evaluator, batch, final_scores, sample_number=0,
            prediction_threshold=thr, collect_roc=True,
        )
        labels = rec.seg_label.float().reshape(-1)
        positive_mask = labels > 0.5
        binary = final_scores.reshape(-1) >= thr
        return {
            "true_positive_events": int((binary & positive_mask).sum().item()),
            "false_positive_events": int((binary & ~positive_mask).sum().item()),
            "positive_events": int(positive_mask.sum().item()),
            "detected_target_frames": int(evaluator.correct_num),
            "target_frames": int(evaluator.obj_num),
            "false_components": int(evaluator.false_num),
            "frame_count": int(evaluator.frame_num),
        }

    per_video = {}
    for fn in sorted(selection):
        entry = selection[fn]
        if len(entry) == 3:
            variant, source, thr = entry
        else:
            variant, source, thr = "c00", entry[0], entry[1]
        rec10 = recs10[fn]
        rec20 = recs20[fn]
        variant_overrides = dict(variants_meta.get(variant, {}) or {})
        score_scale = variant_overrides.pop("score_scale", None)
        blend_weight = variant_overrides.pop("blend_weight", None)
        if source == "m10":
            scores = rec10
        elif source == "max":
            blended = np.maximum(
                rec10["scores"].numpy(), rec20["scores"].numpy()
            ).astype(np.float32, copy=True)
            scores = dict(rec10)
            scores["scores"] = torch.from_numpy(blended)
        else:
            scores = rec20
        if blend_weight is not None:
            scores = dict(scores)
            scores["scores"] = torch.clamp(
                scores["scores"] * float(blend_weight)
                + rec20["scores"] * (1.0 - float(blend_weight)), 0.0, 1.0)
        if score_scale is not None:
            scores = dict(scores)
            scores["scores"] = torch.clamp(
                scores["scores"] * float(score_scale), 0.0, 1.0)
        rec = replay.RoutedRecord(
            file_name=fn, event_count=int(rec10["event_count"]),
            scores=scores["scores"].clone(), seg_label=rec10["seg_label"].clone(),
            locs=rec10["locs"].clone(), idx_label=np.ascontiguousarray(rec10["idx_label"]),
            source_sha256=str(rec10["source_sha256"]), score_source=source,
        )
        vcfg = SimpleNamespace(**{**routed.c00_config().__dict__, **variant_overrides}, roc=True, correct_thresh=0.0001)
        postprocessor = replay.ChallengePostprocessor.from_cfg(vcfg, float(thr), event_count=rec.event_count)
        final, _ = postprocessor.apply(rec.scores.clone(), rec.locs)
        final = final.numpy().astype(np.float32, copy=True)
        comps = extract_atomic_components(
            final, rec.locs, float(thr),
            spatial_radius=2, temporal_bin_size=50, temporal_radius_bins=1,
        ).event_indices
        labels_np = rec.seg_label.numpy().reshape(-1)
        comp_stats = []
        for indices in comps:
            idx = np.asarray(indices, dtype=np.int64)
            n_gt = int(labels_np[idx].sum())
            comp_stats.append((n_gt, idx))
        base = evaluate_final(torch.from_numpy(final), rec, float(thr))
        per_video[fn] = {
            "source": source, "threshold": thr,
            "component_count": len(comp_stats),
            "pure_fp_components": sum(1 for n_gt, _ in comp_stats if n_gt == 0),
            "gt_event_distribution": {
                str(k): sum(1 for n_gt, _ in comp_stats if n_gt == k) for k in range(6)
            },
            "baseline_counts": base,
            "rule_simulations": {},
        }
        for max_gt in (0, 1, 2, 3, 5, 10, 20):
            modified = final.copy()
            for n_gt, idx in comp_stats:
                if n_gt <= max_gt:
                    modified[idx] = 0.0
            counts = evaluate_final(torch.from_numpy(modified), rec, float(thr))
            per_video[fn]["rule_simulations"]["delete_gt_le_{}".format(max_gt)] = counts
        print("{}: comps={} pure_fp={} base FC={}".format(
            fn, len(comp_stats),
            sum(1 for n_gt, _ in comp_stats if n_gt == 0),
            base["false_components"]), flush=True)

    base_pooled = add_counts(*[v["baseline_counts"] for v in per_video.values()])
    base_metrics = metrics_from_counts(base_pooled)
    print("baseline (selection as-is):", json.dumps(base_metrics, indent=1))

    rules = {}
    for rule in sorted({k for v in per_video.values() for k in v["rule_simulations"]}):
        pooled = add_counts(*[v["rule_simulations"][rule] for v in per_video.values()])
        metrics = metrics_from_counts(pooled)
        rules[rule] = {"counts": pooled, "metrics": metrics, "delta": metrics["score"] - base_metrics["score"]}
        print("{}: score={:.10f} delta={:+.10f} pd={:.6f} fa={:.3e}".format(
            rule, metrics["score"], metrics["score"] - base_metrics["score"], metrics["pd"], metrics["fa"]))

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "schema": "ev-uav-component-deletion-val24-analysis-v1",
        "baseline": {"counts": base_pooled, "metrics": base_metrics},
        "rules": rules,
        "per_video": per_video,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
