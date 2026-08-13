"""Fine restore-threshold sweep on low/middle domains (train-only).

P0c retains only components with max raw score >= 0.95; components removed by
P0 with scores just below that are high-confidence.  Sweep restore thresholds
0.85..0.95 to find a deployable label-free restore tier with small FP addition
and per-family safety.  Cache-based; no validation reads, no CUDA.
"""

from __future__ import annotations

import argparse
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

    thresholds = [0.86, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
    variants = {
        "score_max": lambda r, t: r["score_max"] >= t,
        "score_max_2ev": lambda r, t: r["score_max"] >= t and r["log_component_events"] >= math.log1p(2),
        "score_max_2ev_2bin": lambda r, t: (
            r["score_max"] >= t and r["log_component_events"] >= math.log1p(2)
            and r["duration_bins"] >= 2),
        "score_mean": lambda r, t: r["score_mean"] >= t,
    }

    results = {}
    for variant, pred in variants.items():
        for t in thresholds:
            def restore_fn(src, _t=t, _pred=pred):
                rows = src["meta"]["restore_rows"]
                mask = np.zeros(src["res_n"], dtype=bool)
                for r in rows:
                    if _pred(r, _t):
                        mask[r["component_index"]] = True
                return mask
            per_source = {}
            for name, src in sources.items():
                res_mask = restore_fn(src)
                del_mask = np.zeros(src["del_n"], dtype=bool)
                scores = rules_mod.apply_actions(src, del_mask, res_mask)
                per_source[name] = rules_mod.official_counts(scores, src)
            cand = oracle.sum_counts(per_source.values())
            base = oracle.sum_counts(src["baseline"] for src in sources.values())
            cd = oracle.count_delta(cand, base)
            md = oracle.metric_delta(cand, base)
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
                fam_scores[fam] = fmd["score"]
                if fmd["score"] < 0 or fcd["correct_target_frames"] < 0 or fcd["true_positive_events"] < 0:
                    fam_ok = False
            rule_name = "restore_{}_t{:.2f}".format(variant, t)
            results[rule_name] = {
                "count_delta": cd, "metric_delta": md,
                "all_families_safe": fam_ok, "fam_scores": fam_scores,
            }
            print("{}: score {:+.6f} TP {:+.0f} CO {:+.0f} FP {:+.0f} FC {:+.0f} fam_ok {}".format(
                rule_name, md["score"], cd["true_positive_events"], cd["correct_target_frames"],
                cd["false_positive_events"], cd["false_components"], fam_ok), flush=True)

    payload = {
        "schema": "ev-uav-restore-threshold-sweep-train-only-v1",
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
