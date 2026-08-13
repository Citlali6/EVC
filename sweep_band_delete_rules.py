"""Band-constrained delete-rule sweep for middle and high domains (train-only).

Pure-FP components on middle/high are mid-size short bursts (bbox ~4-18 px,
<=6 bins) while point-like target blips have bbox ~1-3 px and track targets are
much larger.  Sweep delete rules of the form

    bbox_lo <= bbox_diagonal <= bbox_hi AND duration_bins <= dur
    AND log_component_events <= log1p(ev)

replaying the official evaluator, requiring TP >= 0, CO >= 0 and per-family
score >= 0.  All selection happens on train with family isolation.

No validation/test reads, no CUDA.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import analyze_low_editor_rules as rules_mod
import run_low_domain_component_oracle as oracle


def make_rule(bbox_lo, bbox_hi, dur, ev):
    def rule(src):
        rows = src["meta"]["delete_rows"]
        mask = np.zeros(src["del_n"], dtype=bool)
        for r in rows:
            if (bbox_lo <= r["bbox_diagonal"] <= bbox_hi
                    and r["duration_bins"] <= dur
                    and r["log_component_events"] <= math.log1p(ev)):
                mask[r["component_index"]] = True
        return mask
    return rule


def evaluate(srcs, rule):
    per = {}
    for name, src in srcs.items():
        del_mask = rule(src)
        res_mask = np.zeros(src["res_n"], dtype=bool)
        scores = rules_mod.apply_actions(src, del_mask, res_mask)
        per[name] = rules_mod.official_counts(scores, src)
    cand = oracle.sum_counts(per.values())
    base = oracle.sum_counts(s["baseline"] for s in srcs.values())
    cd = oracle.count_delta(cand, base)
    md = oracle.metric_delta(cand, base)
    fam_scores = {}
    fam_ok = True
    for fam in sorted({s["meta"]["family"] for s in srcs.values()}):
        fb = oracle.Counts()
        fc = oracle.Counts()
        for name, src in srcs.items():
            if src["meta"]["family"] != fam:
                continue
            fb += src["baseline"]
            fc += per[name]
        fmd = oracle.metric_delta(fc, fb)
        fam_scores[fam] = fmd["score"]
        if fmd["score"] < 0:
            fam_ok = False
    return cd, md, fam_ok, fam_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bbox-los", default="3,4,5,6")
    parser.add_argument("--bbox-his", default="10,14,18,22")
    parser.add_argument("--durs", default="4,6,8,10")
    parser.add_argument("--evs", default="8,12,16")
    args = parser.parse_args()

    sources = rules_mod.load_cache(args.cache, args.meta)
    for name, src in sources.items():
        src["baseline"] = rules_mod.official_counts(src["final"], src)
    print("loaded {} sources".format(len(sources)), flush=True)

    results = {}
    for bbox_lo, bbox_hi, dur, ev in itertools.product(
        [float(x) for x in args.bbox_los.split(",")],
        [float(x) for x in args.bbox_his.split(",")],
        [int(x) for x in args.durs.split(",")],
        [int(x) for x in args.evs.split(",")],
    ):
        if bbox_lo >= bbox_hi:
            continue
        rule = make_rule(bbox_lo, bbox_hi, dur, ev)
        cd, md, fam_ok, fam_scores = evaluate(sources, rule)
        key = "del_b{}-{}_d{}_e{}".format(bbox_lo, bbox_hi, dur, ev)
        results[key] = {
            "count_delta": cd, "metric_delta": md,
            "fam_ok": fam_ok, "fam_scores": fam_scores,
        }
        flag = "SAFE" if (cd["true_positive_events"] >= 0 and cd["correct_target_frames"] >= 0 and fam_ok) else ""
        print("{}: score {:+.6f} TP {:+.0f} CO {:+.0f} FP {:+.0f} FC {:+.0f} fam_ok {} {}".format(
            key, md["score"], cd["true_positive_events"], cd["correct_target_frames"],
            cd["false_positive_events"], cd["false_components"], fam_ok, flag), flush=True)

    path = args.output.resolve()
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
