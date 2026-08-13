"""Simulate label-free component deletion rules on Val24 (official evaluator).

Rules use only observable component features (event count, duration, bbox,
raw score stats) of post-C00 atomic components.  Each rule deletes matching
components (scores set to 0) and re-evaluates the pooled official score.
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
    from utils.atomic_component_deletion import extract_atomic_components

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

    def evaluate_final(final_scores, rec, thr, cfg):
        evaluator = replay.evalute(cfg)
        batch = {"seg_label": rec.seg_label, "locs": rec.locs, "idx_label": rec.idx_label}
        replay.add_batch_to_evaluator(
            evaluator, batch, final_scores, sample_number=0,
            prediction_threshold=thr, collect_roc=True,
        )
        labels = rec.seg_label.float().reshape(-1)
        pm = labels > 0.5
        binary = final_scores.reshape(-1) >= thr
        return {
            "true_positive_events": int((binary & pm).sum().item()),
            "false_positive_events": int((binary & ~pm).sum().item()),
            "positive_events": int(pm.sum().item()),
            "detected_target_frames": int(evaluator.correct_num),
            "target_frames": int(evaluator.obj_num),
            "false_components": int(evaluator.false_num),
            "frame_count": int(evaluator.frame_num),
        }

    # candidate label-free rules
    RULE_NAMES = [
        "r_small_n3_d2_b8", "r_small_n2_d1_b5", "r_smax85_n5", "r_smean80_n4",
        "r_smean75_n3", "r_smean80_d1", "r_smax90_n3", "r_smean85_n5_d2",
    ]

    def rule_hits(comp):
        n, dur, bbox, smean, smax, smin = comp
        return {
            "base": False,
            "r_small_n3_d2_b8": n <= 3 and dur <= 2 and bbox <= 8.0,
            "r_small_n2_d1_b5": n <= 2 and dur <= 1 and bbox <= 5.0,
            "r_smax85_n5": smax < 0.85 and n <= 5,
            "r_smean80_n4": smean < 0.8 and n <= 4,
            "r_smean75_n3": smean < 0.75 and n <= 3,
            "r_smean80_d1": smean < 0.8 and dur <= 1,
            "r_smax90_n3": smax < 0.9 and n <= 3,
            "r_smean85_n5_d2": smean < 0.85 and n <= 5 and dur <= 2,
        }

    per_video = {}
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
        comps = extract_atomic_components(
            final, rec.locs, float(thr),
            spatial_radius=2, temporal_bin_size=50, temporal_radius_bins=1,
        ).event_indices
        ts = rec.locs.numpy()[:, 3]
        comp_features = []
        for indices in comps:
            idx = np.asarray(indices, dtype=np.int64)
            bins = np.unique(ts[idx] // 50)
            x = rec.locs.numpy()[idx, 1]
            y = rec.locs.numpy()[idx, 2]
            bbox = float(np.hypot(x.max() - x.min(), y.max() - y.min()))
            comp_features.append((
                int(idx.size), int(len(bins)), bbox,
                float(raw[idx].mean()), float(raw[idx].max()), float(raw[idx].min()),
                idx,
            ))
        base_counts = evaluate_final(torch.from_numpy(final), rec, float(thr), vcfg)
        rule_sims = {"base": base_counts}
        for rule_name in RULE_NAMES:
            modified = final.copy()
            for n, dur, bbox, smean, smax, smin, idx in comp_features:
                if rule_hits((n, dur, bbox, smean, smax, smin))[rule_name]:
                    modified[idx] = 0.0
            rule_sims[rule_name] = evaluate_final(torch.from_numpy(modified), rec, float(thr), vcfg)
        per_video[fn] = {"source": source, "threshold": thr, "variant": variant,
                         "component_count": len(comp_features), "rule_sims": rule_sims}
        print("{}: comps={} base_FC={}".format(fn, len(comp_features), base_counts["false_components"]), flush=True)

    base_pooled = add_counts(*[v["rule_sims"]["base"] for v in per_video.values()])
    base_metrics = metrics_from_counts(base_pooled)
    print("baseline:", json.dumps(base_metrics, indent=1))
    rules = {}
    for rule in sorted({k for v in per_video.values() for k in v["rule_sims"]}):
        pooled = add_counts(*[v["rule_sims"][rule] for v in per_video.values()])
        metrics = metrics_from_counts(pooled)
        rules[rule] = {"counts": pooled, "metrics": metrics,
                       "delta": metrics["score"] - base_metrics["score"]}
        print("{}: score={:.10f} delta={:+.10f} pd={:.6f} fa={:.3e}".format(
            rule, metrics["score"], metrics["score"] - base_metrics["score"],
            metrics["pd"], metrics["fa"]))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "ev-uav-labelfree-component-deletion-sim-v1",
        "baseline": {"counts": base_pooled, "metrics": base_metrics},
        "rules": rules, "per_video": per_video,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
