"""CPU-only train-only rule analysis for the low-domain component editor.

Loads the low_editor_cache (built once by dump_low_component_cache.py) and
evaluates a grid of LABEL-FREE deletion/restoration rules through the official
evaluator, plus high-precision logistic operating points, with per-family
breakdowns.  The goal: find a deployable selector that captures a useful share
of the known oracle capacities (delete +0.0036, restore +0.0090) without
damaging TP/Pd.  No validation/test reads, no CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import run_low_domain_component_oracle as oracle
from utils.eval import evalute


DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "experiments" / "20260812_low_component_editor_lofo_v1" / "low_editor_cache.npz"
DEFAULT_OUT = DEFAULT_CACHE.parent / "rule_analysis.json"

FEATURES = [
    "log_video_events", "log_component_events", "score_max", "score_mean",
    "score_min", "score_std", "score_margin_max", "log_unique_cells",
    "bbox_diagonal", "duration_bins", "max_events_per_bin",
    "displacement_per_bin", "max_gap_bins", "t_span",
]


def load_cache(cache_path=None, meta_path=None):
    cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE
    meta_path = Path(meta_path) if meta_path else cache_path.with_suffix(".meta.json")
    data = np.load(cache_path, allow_pickle=False)
    meta = json.load(open(meta_path, encoding="utf-8"))
    sources = {}
    for m in meta:
        name = m["source_name"]
        sources[name] = {
            "meta": m,
            "raw": data["{}_raw".format(name)],
            "final": data["{}_final".format(name)],
            "loc": data["{}_loc".format(name)],
            "labels": data["{}_labels".format(name)],
            "ids": data["{}_ids".format(name)],
            "del_idx": data["{}_del_idx".format(name)],
            "del_pos": data["{}_del_pos".format(name)],
            "res_idx": data["{}_res_idx".format(name)],
            "res_pos": data["{}_res_pos".format(name)],
            "del_n": int(data["{}_del_n".format(name)][0]),
            "res_n": int(data["{}_res_n".format(name)][0]),
        }
        # rebuild component -> event index lists
        sources[name]["del_components"] = [
            sources[name]["del_pos"][sources[name]["del_idx"] == c]
            for c in range(sources[name]["del_n"])
        ]
        sources[name]["res_components"] = [
            sources[name]["res_pos"][sources[name]["res_idx"] == c]
            for c in range(sources[name]["res_n"])
        ]
    return sources


def official_counts(scores, src):
    import torch
    thr = float(src["meta"]["threshold"])
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    truth = np.asarray(src["labels"], dtype=np.uint8).reshape(-1)
    ids = np.asarray(src["ids"]).reshape(-1)
    locations = np.asarray(src["loc"], dtype=np.int64)
    evaluator = evalute(type("Config", (), {"roc": True, "pd_detT": 50, "correct_thresh": 0.0001})())
    evaluator.roc_update(
        torch.from_numpy(locations[:, 3].copy()),
        torch.from_numpy(values.copy()),
        ids,
        torch.from_numpy(truth.astype(np.float32, copy=False)),
        torch.from_numpy(locations.copy()),
        thresh=float(thr),
    )
    predicted = values >= thr
    positive = truth > 0
    return oracle.Counts(
        true_positive_events=int(np.count_nonzero(predicted & positive)),
        false_positive_events=int(np.count_nonzero(predicted & ~positive)),
        false_negative_events=int(np.count_nonzero(~predicted & positive)),
        correct_target_frames=int(evaluator.correct_num),
        target_frames=int(evaluator.obj_num),
        false_components=int(evaluator.false_num),
        frame_count=int(evaluator.frame_num),
        event_count=int(values.size),
    )


def apply_actions(src, delete_mask, restore_mask):
    output = src["final"].copy()
    assigned = np.zeros(output.size, dtype=bool)
    for component_index, indices in enumerate(src["del_components"]):
        if np.any(assigned[indices]):
            raise RuntimeError("delete overlap")
        assigned[indices] = True
        if np.asarray(delete_mask, dtype=bool)[component_index]:
            output[indices] = np.float32(0.0)
    for component_index, indices in enumerate(src["res_components"]):
        if np.any(assigned[indices]):
            raise RuntimeError("restore overlap")
        assigned[indices] = True
        if np.asarray(restore_mask, dtype=bool)[component_index]:
            output[indices] = src["raw"][indices].astype(np.float32, copy=True)
    if not np.array_equal(output[~assigned].view(np.uint32), src["final"][~assigned].view(np.uint32)):
        raise RuntimeError("non-component scores changed")
    return output


def replay(sources, del_fn, res_fn):
    per_source = {}
    for name, src in sources.items():
        del_mask = del_fn(src)
        res_mask = res_fn(src)
        scores = apply_actions(src, del_mask, res_mask)
        per_source[name] = official_counts(scores, src)
    return per_source


def summarize(per_source, sources):
    baseline = oracle.sum_counts(oracle.Counts() for _ in [0])  # placeholder
    base_list = []
    cand_list = []
    for name, src in sources.items():
        # baseline counts = candidate on final scores (identity)
        base_list.append(src.get("baseline"))
        cand_list.append(per_source[name])
    return {
        "count_delta": oracle.count_delta(oracle.sum_counts(cand_list), oracle.sum_counts(base_list)),
        "metric_delta": oracle.metric_delta(oracle.sum_counts(cand_list), oracle.sum_counts(base_list)),
    }


# ---------------- rule builders (label-free) ----------------

def rule_none(src):
    return np.zeros(src["del_n"], dtype=bool), np.zeros(src["res_n"], dtype=bool)


def delete_rule_ge(name, key, op, value):
    def rule(src):
        rows = src["meta"]["delete_rows"]
        mask = np.zeros(src["del_n"], dtype=bool)
        for r in rows:
            if op(r[key], value):
                mask[r["component_index"]] = True
        return mask
    return rule


def make_delete_rule(predicate):
    def rule(src):
        rows = src["meta"]["delete_rows"]
        mask = np.zeros(src["del_n"], dtype=bool)
        for r in rows:
            if predicate(r):
                mask[r["component_index"]] = True
        return mask
    return rule


def make_restore_rule(predicate):
    def rule(src):
        rows = src["meta"]["restore_rows"]
        mask = np.zeros(src["res_n"], dtype=bool)
        for r in rows:
            if predicate(r):
                mask[r["component_index"]] = True
        return mask
    return rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    args = parser.parse_args()

    sources = load_cache(args.cache, args.meta)
    # attach baseline counts (identity replay) once
    for name, src in sources.items():
        src["baseline"] = official_counts(src["final"], src)
    print("loaded {} sources".format(len(sources)), flush=True)

    def attach_baseline(per_source):
        for name, counts in per_source.items():
            pass
        return per_source

    delete_rules = {
        "none": (make_delete_rule(lambda r: False), make_restore_rule(lambda r: False)),
        "D_score_max_lt_0p80": (make_delete_rule(lambda r: r["score_max"] < 0.80), make_restore_rule(lambda r: False)),
        "D_score_max_lt_0p75": (make_delete_rule(lambda r: r["score_max"] < 0.75), make_restore_rule(lambda r: False)),
        "D_score_max_lt_0p70": (make_delete_rule(lambda r: r["score_max"] < 0.70), make_restore_rule(lambda r: False)),
        "D_small_short": (make_delete_rule(
            lambda r: r["score_max"] < 0.85 and r["duration_bins"] <= 4 and r["log_component_events"] <= math.log1p(6)), make_restore_rule(lambda r: False)),
        "D_tiny": (make_delete_rule(
            lambda r: r["score_max"] < 0.85 and r["log_component_events"] <= math.log1p(3) and r["duration_bins"] <= 3), make_restore_rule(lambda r: False)),
        "D_low_mean_short": (make_delete_rule(
            lambda r: r["score_mean"] < 0.72 and r["duration_bins"] <= 5), make_restore_rule(lambda r: False)),
        "D_max_lt_0p85_3ev": (make_delete_rule(
            lambda r: r["score_max"] < 0.85 and r["log_component_events"] <= math.log1p(3)), make_restore_rule(lambda r: False)),
    }
    restore_rules = {
        "R_score_min_ge_0p60": make_restore_rule(lambda r: r["score_min"] >= 0.60),
        "R_score_min_ge_0p65": make_restore_rule(lambda r: r["score_min"] >= 0.65),
        "R_score_min_ge_0p70": make_restore_rule(lambda r: r["score_min"] >= 0.70),
        "R_score_min_ge_0p75": make_restore_rule(lambda r: r["score_min"] >= 0.75),
        "R_score_max_ge_0p80": make_restore_rule(lambda r: r["score_max"] >= 0.80),
        "R_score_max_ge_0p85": make_restore_rule(lambda r: r["score_max"] >= 0.85),
        "R_3ev_max_ge_0p75": make_restore_rule(
            lambda r: r["log_component_events"] >= math.log1p(3) and r["score_max"] >= 0.75),
        "R_2ev_max_ge_0p70": make_restore_rule(
            lambda r: r["log_component_events"] >= math.log1p(2) and r["score_max"] >= 0.70),
    }

    results = {}
    # delete-only rules
    for name, (del_fn, res_fn) in delete_rules.items():
        per_source = replay(sources, del_fn, res_fn)
        results[name] = summarize(per_source, sources)
        print("{}: score {:+.6f} TP {:+.0f} FP {:+.0f} CO {:+.0f} FC {:+.0f}".format(
            name, results[name]["metric_delta"]["score"],
            results[name]["count_delta"]["true_positive_events"],
            results[name]["count_delta"]["false_positive_events"],
            results[name]["count_delta"]["correct_target_frames"],
            results[name]["count_delta"]["false_components"]), flush=True)
    # restore-only rules
    for name, res_fn in restore_rules.items():
        per_source = replay(sources, make_delete_rule(lambda r: False), res_fn)
        results[name] = summarize(per_source, sources)
        print("{}: score {:+.6f} TP {:+.0f} FP {:+.0f} CO {:+.0f} FC {:+.0f}".format(
            name, results[name]["metric_delta"]["score"],
            results[name]["count_delta"]["true_positive_events"],
            results[name]["count_delta"]["false_positive_events"],
            results[name]["count_delta"]["correct_target_frames"],
            results[name]["count_delta"]["false_components"]), flush=True)

    # joined: best delete rule (by pooled score, TP-neutral preferred) + best restore
    best_del = max(
        (k for k in delete_rules if k != "none"),
        key=lambda k: results[k]["metric_delta"]["score"],
    )
    best_res = max(
        restore_rules,
        key=lambda k: results[k]["metric_delta"]["score"],
    )
    joined_del_fn = delete_rules[best_del][0]
    joined_res_fn = restore_rules[best_res]
    per_source = replay(sources, joined_del_fn, joined_res_fn)
    results["joined_{}_plus_{}".format(best_del, best_res)] = summarize(per_source, sources)
    print("JOINED {} + {}: score {:+.6f} TP {:+.0f} FP {:+.0f} CO {:+.0f} FC {:+.0f}".format(
        best_del, best_res,
        results["joined_{}_plus_{}".format(best_del, best_res)]["metric_delta"]["score"],
        results["joined_{}_plus_{}".format(best_del, best_res)]["count_delta"]["true_positive_events"],
        results["joined_{}_plus_{}".format(best_del, best_res)]["count_delta"]["false_positive_events"],
        results["joined_{}_plus_{}".format(best_del, best_res)]["count_delta"]["correct_target_frames"],
        results["joined_{}_plus_{}".format(best_del, best_res)]["count_delta"]["false_components"]), flush=True)

    # per-family breakdown of the joined rule
    family_deltas = {}
    for name, src in sources.items():
        del_mask = joined_del_fn(src)
        res_mask = joined_res_fn(src)
        counts = official_counts(apply_actions(src, del_mask, res_mask), src)
        family_deltas.setdefault(src["meta"]["family"], {"baseline": oracle.Counts(), "candidate": oracle.Counts()})
        family_deltas[src["meta"]["family"]]["baseline"] += src["baseline"]
        family_deltas[src["meta"]["family"]]["candidate"] += counts
    per_family = {
        fam: {
            "count_delta": oracle.count_delta(v["candidate"], v["baseline"]),
            "metric_delta": oracle.metric_delta(v["candidate"], v["baseline"]),
        }
        for fam, v in family_deltas.items()
    }
    print("per-family:", {fam: round(v["metric_delta"]["score"], 5) for fam, v in per_family.items()}, flush=True)

    payload = {
        "schema": "ev-uav-low-editor-label-free-rules-train-only-v1",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "dataset_split": "train",
        "validation_or_test_read": False,
        "rule_results": results,
        "joined_per_family": per_family,
        "best_joined": "joined_{}_plus_{}".format(best_del, best_res),
    }
    path = args.output.resolve()
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
