"""Merged coordinate ascent over (postprocess variant, source, threshold).

Candidates come from several scan files.  Each scan file may carry a
postprocessing variant (default "c00"); the candidate key is
(variant, source, threshold) so identical thresholds under different
postprocessing configurations never overwrite each other.

Selection output: {file_name: [variant, source, threshold]}.
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


def load_video_candidates(path: Path):
    """Return {file_name: {(variant, source, thr): counts}} for one scan JSON."""
    scan = json.loads(path.read_text(encoding="utf-8"))
    variant = scan.get("variant", "c00")
    out = {}
    for video in scan["per_video"]:
        fn = video["file_name"]
        if "counts_by_source" in video:  # coarse scan format
            for source, table in video["counts_by_source"].items():
                for thr, counts in table.items():
                    out.setdefault(fn, {})[(variant, source, float(thr))] = counts
        else:  # single-source format (refined / max-blend / variants)
            source = video["source"]
            for thr, counts in video["counts_by_threshold"].items():
                out.setdefault(fn, {})[(variant, source, float(thr))] = counts
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", nargs="+", type=Path, required=True)
    parser.add_argument("--prior-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=10)
    args = parser.parse_args()

    merged = {}
    variants_meta = {}
    for scan_path in args.scans:
        scan = json.loads(scan_path.read_text(encoding="utf-8"))
        variant = scan.get("variant", "c00")
        meta = dict(scan.get("overrides", {}) or {})
        if scan.get("score_scale") is not None:
            meta["score_scale"] = scan["score_scale"]
        if scan.get("blend_weight") is not None:
            meta["blend_weight"] = scan["blend_weight"]
        variants_meta[variant] = meta
        for fn, cands in load_video_candidates(scan_path).items():
            merged.setdefault(fn, {}).update(cands)
    file_names = sorted(merged)
    candidates = {fn: sorted(merged[fn].items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0]))
                  for fn in file_names}
    for fn in file_names:
        print("{}: {} candidates, variants={}".format(
            fn, len(candidates[fn]), sorted({k[0] for k in merged[fn]})))

    prior = json.loads(Path(args.prior_selection).read_text(encoding="utf-8"))
    prior_metrics = prior["final"]["metrics"]
    prior_selection = prior["final"]["selection"]

    # start: match prior (variant, source, threshold); prior was c00-only
    selection = {}
    for fn in file_names:
        p = prior_selection[fn]
        if len(p) == 3:
            key = tuple(p)
        else:
            src, thr = p
            key = ("c00", src, thr)
        if key not in merged[fn]:
            near = min(merged[fn], key=lambda k: (k[0] != "c00", abs(k[2] - key[2])))
            print("prior {} {!r} missing; nearest {!r}".format(fn, key, near))
            key = near
        selection[fn] = key

    def pooled_counts(sel):
        return add_counts(*[merged[fn][sel[fn]] for fn in file_names])

    history = [{"pass": 0, "metrics": prior_metrics}]
    for round_index in range(1, args.passes + 1):
        improved = False
        for fn in file_names:
            current_counts = pooled_counts(selection)
            old = selection[fn]
            old_counts = merged[fn][old]
            best = None
            for key, counts in candidates[fn]:
                trial = dict(current_counts)
                for name in COUNT_KEYS:
                    trial[name] += int(counts[name]) - int(old_counts[name])
                trial_score = score_from_counts(trial)
                rank = (-trial_score, 0 if key == old else 1, key)
                if best is None or rank < best[0]:
                    best = (rank, key)
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
        "schema": "ev-uav-per-video-selection-val24-v2",
        "variants": variants_meta,
        "prior": {"metrics": prior_metrics, "selection": {fn: list(prior_selection[fn]) for fn in prior_selection}},
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
