"""Official-score simulation of CV-trained component deletion policies.

Deletes components whose CV pure-FP probability exceeds a threshold and
re-evaluates the pooled official Challenge 2 score under the final selection.
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
    parser.add_argument("--recall", type=float, default=0.4, help="target pure-FP recall")
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
    assert len(proba) == len(comp_records)
    y = np.array([r["n_gt"] == 0 for r in comp_records], dtype=np.int64)
    q = np.quantile(proba[y == 1], 1.0 - args.recall)
    hit_mask = proba >= q
    print("policy recall={}: threshold={:.4f} deleted={} fp_rm={} tp_loss={}".format(
        args.recall, q, int(hit_mask.sum()),
        int(hit_mask[y == 1].sum()), int(hit_mask[y == 0].sum())))

    # group component deletions by video
    deletes = {}
    for r, hit in zip(comp_records, hit_mask):
        if hit:
            deletes.setdefault(r["video"], []).append(np.asarray(r["idx"], dtype=np.int64))

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
        for idx in deletes.get(fn, []):
            final[idx] = 0.0
        evaluator = replay.evalute(vcfg)
        batch = {"seg_label": rec.seg_label, "locs": rec.locs, "idx_label": rec.idx_label}
        replay.add_batch_to_evaluator(
            evaluator, batch, torch.from_numpy(final), sample_number=0,
            prediction_threshold=float(thr), collect_roc=True,
        )
        labels = rec.seg_label.float().reshape(-1)
        pm = labels > 0.5
        binary = torch.from_numpy(final.reshape(-1)) >= float(thr)
        per_video[fn] = {
            "true_positive_events": int((binary & pm).sum().item()),
            "false_positive_events": int((binary & ~pm).sum().item()),
            "positive_events": int(pm.sum().item()),
            "detected_target_frames": int(evaluator.correct_num),
            "target_frames": int(evaluator.obj_num),
            "false_components": int(evaluator.false_num),
            "frame_count": int(evaluator.frame_num),
        }
        print("eval {}".format(fn), flush=True)

    pooled = add_counts(*per_video.values())
    metrics = metrics_from_counts(pooled)
    print("FINAL:", json.dumps(metrics, indent=1))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "ev-uav-classifier-deletion-official-sim-v1",
        "recall": args.recall, "threshold": q,
        "deleted": int(hit_mask.sum()),
        "counts": pooled, "metrics": metrics,
        "target_0_9700_met": metrics["score"] >= 0.9700,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
