"""Cutoff grid analysis for the learned editor (train-only, cache-based).

Rebuilds the 5-family LOFO held-out probabilities from the cached component
rows (no partition rebuild), then replays both editor branches over a cutoff
grid, separately and combined, with per-family deltas.

No validation/test reads, no CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import analyze_low_editor_rules as rules_mod
import run_low_domain_component_oracle as oracle


CACHE_DIR = Path(__file__).resolve().parent.parent / "experiments" / "20260812_low_component_editor_lofo_v1"
DEFAULT_OUT = CACHE_DIR / "lofo_cutoff_analysis.json"


def fit_predict(fit_rows, predict_rows, positive_key):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X_fit = np.array([[r[f] for f in rules_mod.FEATURES] for r in fit_rows], dtype=np.float64)
    y_fit = np.array([1 if r[positive_key] else 0 for r in fit_rows], dtype=np.int64)
    scaler = StandardScaler().fit(X_fit)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(
        scaler.transform(X_fit), y_fit
    )
    X_pred = np.array([[r[f] for f in rules_mod.FEATURES] for r in predict_rows], dtype=np.float64)
    return model.predict_proba(scaler.transform(X_pred))[:, 1]


def lofo_probs(sources):
    """Held-out per-component probabilities for both branches."""
    families = sorted({src["meta"]["family"] for src in sources.values()})
    del_probs, res_probs = {}, {}
    for held in families:
        fit_del, fit_res = [], []
        for name, src in sources.items():
            if src["meta"]["family"] == held:
                continue
            fit_del.extend(src["meta"]["delete_rows"])
            fit_res.extend(src["meta"]["restore_rows"])
        for name, src in sources.items():
            if src["meta"]["family"] != held:
                continue
            del_probs[name] = fit_predict(fit_del, src["meta"]["delete_rows"], "pure_fp")
            res_probs[name] = fit_predict(fit_res, src["meta"]["restore_rows"], "has_target")
    return del_probs, res_probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    args = parser.parse_args()

    sources = rules_mod.load_cache(args.cache, args.meta)
    for name, src in sources.items():
        src["baseline"] = rules_mod.official_counts(src["final"], src)

    del_probs, res_probs = lofo_probs(sources)

    def replay_at(del_cut, res_cut):
        per_source = {}
        for name, src in sources.items():
            del_mask = del_probs[name] >= del_cut if del_cut > 0.0 else np.zeros(src["del_n"], dtype=bool)
            res_mask = res_probs[name] >= res_cut if res_cut > 0.0 else np.zeros(src["res_n"], dtype=bool)
            scores = rules_mod.apply_actions(src, del_mask, res_mask)
            per_source[name] = rules_mod.official_counts(scores, src)
        return per_source

    def summarize(per_source):
        cand = oracle.sum_counts(per_source.values())
        base = oracle.sum_counts(src["baseline"] for src in sources.values())
        return oracle.count_delta(cand, base), oracle.metric_delta(cand, base)

    results = {}
    grid = [(0.0, 0.0), (0.5, 0.5), (0.7, 0.5), (0.85, 0.5), (0.95, 0.5),
            (0.0, 0.5), (0.0, 0.7), (0.0, 0.85), (0.0, 0.95),
            (0.85, 0.7), (0.85, 0.85), (0.9, 0.8), (0.95, 0.85)]
    for del_cut, res_cut in grid:
        per_source = replay_at(del_cut, res_cut)
        cd, md = summarize(per_source)
        results["del{}_res{}".format(del_cut, res_cut)] = {
            "count_delta": cd, "metric_delta": md,
        }
        print("del {:.2f} res {:.2f}: score {:+.6f} TP {:+.0f} FP {:+.0f} CO {:+.0f} FC {:+.0f}".format(
            del_cut, res_cut, md["score"],
            cd["true_positive_events"], cd["false_positive_events"],
            cd["correct_target_frames"], cd["false_components"]), flush=True)

    # per-family breakdown of the best combined config
    best_key = max(
        (k for k in results if k != "del0.0_res0.0"),
        key=lambda k: results[k]["metric_delta"]["score"],
    )
    del_cut, res_cut = (float(x) for x in best_key.replace("del", "").split("_res"))
    per_family = {}
    for name, src in sources.items():
        del_mask = del_probs[name] >= del_cut if del_cut > 0.0 else np.zeros(src["del_n"], dtype=bool)
        res_mask = res_probs[name] >= res_cut if res_cut > 0.0 else np.zeros(src["res_n"], dtype=bool)
        counts = rules_mod.official_counts(rules_mod.apply_actions(src, del_mask, res_mask), src)
        fam = src["meta"]["family"]
        entry = per_family.setdefault(fam, {"base": oracle.Counts(), "cand": oracle.Counts()})
        entry["base"] += src["baseline"]
        entry["cand"] += counts
    per_family = {
        fam: {"count_delta": oracle.count_delta(v["cand"], v["base"]),
              "metric_delta": oracle.metric_delta(v["cand"], v["base"])}
        for fam, v in per_family.items()
    }
    print("best:", best_key, "per-family:", {
        fam: round(v["metric_delta"]["score"], 5) for fam, v in per_family.items()}, flush=True)

    payload = {
        "schema": "ev-uav-low-editor-cutoff-grid-train-only-v1",
        "results": results,
        "best_combined": best_key,
        "per_family_best": per_family,
    }
    path = args.output.resolve()
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
