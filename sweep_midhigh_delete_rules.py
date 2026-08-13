"""Size/duration delete-rule sweep for middle and high domains (train-only).

The frozen diagnostics show the surviving pure-FP components on middle/high
domains are short bursts (2-6 bins, 4-10 px) while target components are long
tracks (15-19 bins, 31-33 px).  Sweep simple label-free rules over
duration / bbox / event-count thresholds; replay every rule through the
official evaluator; enforce TP / correct-frame (Pd) safety plus per-family
score gates.  Threshold selection happens entirely on train.

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = rules_mod.load_cache(args.cache, args.meta)
    for name, src in sources.items():
        src["baseline"] = rules_mod.official_counts(src["final"], src)
    print("loaded {} sources".format(len(sources)), flush=True)

    def delete_rule(dur, bbox, events, combos):
        """Rule predicate built from thresholds; combos = which conditions apply."""
        def pred(r):
            checks = []
            if "dur" in combos:
                checks.append(r["duration_bins"] <= dur)
            if "bbox" in combos:
                checks.append(r["bbox_diagonal"] <= bbox)
            if "ev" in combos:
                checks.append(r["log_component_events"] <= math.log1p(events))
            return all(checks)
        return rules_mod.make_delete_rule(pred)

    results = {}
    durs = [4, 6, 8, 10]
    bboxes = [10, 14, 18, 22]
    evs = [12]
    combos_list = [
        ("dur_bbox", ("dur", "bbox")),
        ("dur_bbox_ev", ("dur", "bbox", "ev")),
    ]
    for dur, bbox, ev, (combo_name, combo) in itertools.product(durs, bboxes, evs, combos_list):
        rule_name = "del_dur{}_bbox{}_ev{}_{}".format(dur, bbox, ev, combo_name)
        del_fn = delete_rule(dur, bbox, ev, combo)
        res_fn = rules_mod.make_restore_rule(lambda r: False)
        per_source = {}
        for name, src in sources.items():
            del_mask = del_fn(src)
            res_mask = res_fn(src)
            scores = rules_mod.apply_actions(src, del_mask, res_mask)
            per_source[name] = rules_mod.official_counts(scores, src)
        cand = oracle.sum_counts(per_source.values())
        base = oracle.sum_counts(src["baseline"] for src in sources.values())
        cd = oracle.count_delta(cand, base)
        md = oracle.metric_delta(cand, base)
        # per-family
        fam_deltas = {}
        fam_ok = True
        fam_scores = {}
        for fam in sorted({src["meta"]["family"] for src in sources.values()}):
            fam_base = oracle.Counts()
            fam_cand = oracle.Counts()
            for name, src in sources.items():
                if src["meta"]["family"] != fam:
                    continue
                fam_base += src["baseline"]
                fam_cand += per_source[name]
            fcd = oracle.count_delta(fam_cand, fam_base)
            fmd = oracle.metric_delta(fam_cand, fam_base)
            fam_deltas[fam] = {"count_delta": fcd, "metric_delta": fmd}
            fam_scores[fam] = fmd["score"]
            if fmd["score"] < 0 or fcd["true_positive_events"] < 0 or fcd["correct_target_frames"] < 0:
                fam_ok = False
        safe = (
            cd["true_positive_events"] >= 0 and cd["correct_target_frames"] >= 0
            and md["pd"] >= 0.0
        )
        results[rule_name] = {
            "count_delta": cd,
            "metric_delta": md,
            "safe_tp_co": safe,
            "all_families_safe": fam_ok,
            "fam_scores": fam_scores,
        }
        print("{}: score {:+.6f} TP {:+.0f} CO {:+.0f} FP {:+.0f} FC {:+.0f} safe {} fam_ok {}".format(
            rule_name, md["score"], cd["true_positive_events"], cd["correct_target_frames"],
            cd["false_positive_events"], cd["false_components"], safe, fam_ok), flush=True)

    payload = {
        "schema": "ev-uav-midhigh-delete-rule-sweep-train-only-v1",
        "dataset_split": "train",
        "validation_or_test_read": False,
        "results": results,
    }
    path = args.output.resolve()
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
