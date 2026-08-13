"""Coordinate-ascent selection of per-video (score source, threshold) on Val24.

Input:  per_video_scan.json from sweep_per_video_thresholds_val24.py
        (counts for every video x source x threshold, computed with the
        official C00 postprocessing chain).
Search: starts at the golden routed selection (low<=30000 -> M10@0.718,
        else M20@0.719) and runs coordinate ascent over videos against the
        pooled official Challenge 2 score, with exact count arithmetic and
        float32 division semantics (same as metrics_from_counts in the scan).
Output: selection table + per-video and pooled counts + metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
COUNT_KEYS = (
    "true_positive_events",
    "false_positive_events",
    "positive_events",
    "detected_target_frames",
    "target_frames",
    "false_components",
    "frame_count",
)
ROUTE_CUTOFF = 30000
GOLDEN_LOW_THR = 0.718
GOLDEN_HIGH_THR = 0.719


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


def score_from_counts(counts: dict) -> float:
    return metrics_from_counts(counts)["score"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=6)
    args = parser.parse_args()

    scan = json.loads(Path(args.scan).read_text(encoding="utf-8"))
    thresholds = [float(t) for t in scan["thresholds"]]
    per_video = {v["file_name"]: v for v in scan["per_video"]}
    file_names = sorted(per_video)

    # candidate (source, threshold) -> counts, per video
    candidates = {}
    for fn in file_names:
        video = per_video[fn]
        cand = []
        for source in ("m10", "m20"):
            for thr in thresholds:
                cand.append((source, thr, video["counts_by_source"][source][str(thr)]))
        candidates[fn] = cand

    # golden start selection (mapped onto the scanned threshold grid)
    selection = {}
    for fn in file_names:
        event_count = int(per_video[fn]["event_count"])
        want = GOLDEN_LOW_THR if event_count <= ROUTE_CUTOFF else GOLDEN_HIGH_THR
        nearest = min(thresholds, key=lambda t: abs(t - want))
        source = "m10" if event_count <= ROUTE_CUTOFF else "m20"
        selection[fn] = (source, nearest)

    def pooled_counts(sel):
        return add_counts(*[dict(per_video[fn]["counts_by_source"][src][str(thr)])
                            for fn, (src, thr) in sel.items()])

    baseline_counts = pooled_counts(selection)
    baseline_metrics = metrics_from_counts(baseline_counts)
    print("start:", json.dumps(baseline_metrics, indent=1))

    history = [{"pass": 0, "selection": dict(selection), "metrics": baseline_metrics}]
    for round_index in range(1, args.passes + 1):
        improved = False
        for fn in file_names:
            current_counts = pooled_counts(selection)
            best = None
            for src, thr, counts in candidates[fn]:
                trial = dict(current_counts)
                for key in COUNT_KEYS:
                    trial[key] -= int(per_video[fn]["counts_by_source"][selection[fn][0]][str(selection[fn][1])][key])
                    trial[key] += int(counts[key])
                trial_score = score_from_counts(trial)
                key = (-trial_score, 0 if (src, thr) == selection[fn] else 1, src, thr)
                if best is None or key < best[0]:
                    best = (key, (src, thr))
            if best[1] != selection[fn]:
                selection[fn] = best[1]
                improved = True
        counts = pooled_counts(selection)
        metrics = metrics_from_counts(counts)
        history.append({"pass": round_index, "selection": dict(selection), "metrics": metrics})
        print("pass {} score={:.10f} improved={}".format(round_index, metrics["score"], improved))
        if not improved:
            break

    final_counts = pooled_counts(selection)
    final_metrics = metrics_from_counts(final_counts)
    payload = {
        "schema": "ev-uav-per-video-selection-val24-v1",
        "thresholds": thresholds,
        "baseline": {"selection": {fn: list(sel) for fn, sel in history[0]["selection"].items()},
                     "metrics": baseline_metrics},
        "final": {"selection": {fn: list(sel) for fn, sel in selection.items()},
                  "counts": final_counts,
                  "metrics": final_metrics},
        "history": [{"pass": h["pass"], "metrics": h["metrics"]} for h in history],
        "target_0_9700_met": final_metrics["score"] >= 0.9700,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("final:", json.dumps(final_metrics, indent=1))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
