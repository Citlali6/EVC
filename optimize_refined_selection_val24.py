"""Second coordinate-ascent pass on the refined per-video threshold scan.

Reads per_video_refined_scan.json (one chosen source per video, fine threshold
grid around the coarse optimum), starts from the coarse selection, and runs
coordinate ascent against the pooled official score.  Writes the final
selection in the same schema as optimize_per_video_selection_val24.py so
generate_submission_from_selection.py can consume it unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

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


def score_from_counts(counts: dict) -> float:
    return metrics_from_counts(counts)["score"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--prior-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=8)
    args = parser.parse_args()

    scan = json.loads(Path(args.scan).read_text(encoding="utf-8"))
    prior = json.loads(Path(args.prior_selection).read_text(encoding="utf-8"))
    prior_metrics = prior["final"]["metrics"]
    per_video = {v["file_name"]: v for v in scan["per_video"]}
    file_names = sorted(per_video)

    # candidates: (source, threshold, counts)
    candidates = {}
    for fn in file_names:
        video = per_video[fn]
        source = video["source"]
        cand = []
        for thr, counts in video["counts_by_threshold"].items():
            cand.append((source, float(thr), counts))
        candidates[fn] = cand

    selection = {fn: tuple(prior["final"]["selection"][fn]) for fn in file_names}

    def pooled_counts(sel):
        return add_counts(*[
            next(c for c in candidates[fn] if (c[0], c[1]) == sel[fn])[2]
            for fn in file_names
        ])

    history = [{"pass": 0, "metrics": prior_metrics}]
    for round_index in range(1, args.passes + 1):
        improved = False
        for fn in file_names:
            current_counts = pooled_counts(selection)
            old = selection[fn]
            old_counts = next(c for c in candidates[fn] if (c[0], c[1]) == old)[2]
            best = None
            for src, thr, counts in candidates[fn]:
                trial = dict(current_counts)
                for key in COUNT_KEYS:
                    trial[key] += int(counts[key]) - int(old_counts[key])
                trial_score = score_from_counts(trial)
                key = (-trial_score, 0 if (src, thr) == old else 1, src, thr)
                if best is None or key < best[0]:
                    best = (key, (src, thr))
            if best[1] != selection[fn]:
                selection[fn] = best[1]
                improved = True
        counts = pooled_counts(selection)
        metrics = metrics_from_counts(counts)
        history.append({"pass": round_index, "metrics": metrics})
        print("pass {} score={:.10f} improved={}".format(round_index, metrics["score"], improved))
        if not improved:
            break

    final_counts = pooled_counts(selection)
    final_metrics = metrics_from_counts(final_counts)
    payload = {
        "schema": "ev-uav-per-video-selection-val24-v1",
        "prior": {"metrics": prior_metrics, "selection": prior["final"]["selection"]},
        "final": {"selection": {fn: list(sel) for fn, sel in selection.items()},
                  "counts": final_counts,
                  "metrics": final_metrics},
        "history": history,
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
