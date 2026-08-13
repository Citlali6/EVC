"""CPU-only train-only component editor: label-free rule replays + 5-family nested LOFO.

The editor has two label-free action branches over exact post-C00 partitions:
  - DELETE: zero out complete components of FINAL scores (pure-FP candidates).
  - RESTORE: set final = raw for complete components of RAW scores that were
    removed by the C00 chain (target candidates the postprocessor dropped).

Stage 1 (oracles/rules, label-free but truth-evaluated):
  ground check: delete-all-pure-FP must reproduce +0.0035973761.
  label-free restore rules: pure score/size thresholds, no classifier.

Stage 2 (learned, family-isolated nested LOFO):
  outer: 5 frozen continuous families; hold one out, fit LogisticRegression
  (balanced, standardized) on the other four, predict the held-out family.
  restore branch: P(component is target) on removed components, restore iff >= 0.5.
  delete branch:  P(component is pure-FP) on surviving components, delete iff >= 0.5.
  cutoff 0.5 is FIXED in v1 (no validation-driven tuning).
  Every outer fold replays the official evaluator per source; pooling uses
  sufficient statistics exactly like the golden chain.

Gates (pre-registered): pooled Score delta > 0; every outer family delta >= 0;
TP and correct-target-frames (Pd) not lower than baseline.  A failed gate
archives the branch; no Val24 read happens here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import torch

import diagnose_low_c00_recovery_and_separability as diag
import run_low_domain_component_oracle as oracle


EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "20260812_low_component_editor_lofo_v1"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "lofo_report.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_exclusive(path: Path, payload) -> str:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = oracle.sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        "{}  {}\n".format(digest, path.name), encoding="ascii"
    )
    return digest


def pooled_metrics(per_source_counts):
    baseline = oracle.sum_counts(
        per_source_counts[name]["baseline"] for name in per_source_counts
    )
    candidate = oracle.sum_counts(
        per_source_counts[name]["candidate"] for name in per_source_counts
    )
    return {
        "baseline": oracle.record(baseline),
        "candidate": oracle.record(candidate),
        "count_delta": oracle.count_delta(candidate, baseline),
        "metric_delta": oracle.metric_delta(candidate, baseline),
    }


def replay(videos, partitions, delete_fn, restore_fn):
    """Apply an editor to every video; return per-source (baseline, candidate)."""
    per_source = {}
    for video in videos:
        delete_rows, restore_rows, delete_indices, restore_indices = partitions[video.source_name]
        del_mask = delete_fn(video, delete_rows)
        res_mask = restore_fn(video, restore_rows)
        scores = diag.apply_actions(video, delete_indices, restore_indices, del_mask, res_mask)
        per_source[video.source_name] = {
            "baseline": video.baseline,
            "candidate": diag.official_counts_thr(scores, video),
        }
    return per_source


# ---------------- label-free rules (truth only for evaluation) ----------------

def rule_delete_all_pure_fp(video, delete_rows):
    return np.array([r["pure_fp"] for r in delete_rows], dtype=bool)


def rule_none(video, rows):
    return np.zeros(len(rows), dtype=bool)


def make_restore_rule(**criteria):
    def rule(video, restore_rows):
        mask = np.zeros(len(restore_rows), dtype=bool)
        for r in restore_rows:
            ok = True
            for key, (op, value) in criteria.items():
                if op == "ge":
                    ok = ok and r[key] >= value
                elif op == "lt":
                    ok = ok and r[key] < value
                else:
                    raise ValueError(op)
            if ok:
                mask[r["component_index"]] = True
        return mask
    return rule


def restore_rule_score_max(threshold):
    return make_restore_rule(score_max=("ge", threshold))


def restore_rule_events_and_score(min_events, min_score):
    return make_restore_rule(
        log_component_events=("ge", math.log1p(min_events)),
        score_max=("ge", min_score),
    )


# ---------------- learned editor (family-isolated nested LOFO) ----------------

def feature_matrix(rows):
    X = np.array([[r[fname] for fname in diag.FEATURE_NAMES] for r in rows], dtype=np.float64)
    return X


def fit_predict(fit_rows, predict_rows, positive_key):
    """Balanced logistic on standardized features; return predicted probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X_fit = feature_matrix(fit_rows)
    y_fit = np.array([1 if r[positive_key] else 0 for r in fit_rows], dtype=np.int64)
    scaler = StandardScaler().fit(X_fit)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(
        scaler.transform(X_fit), y_fit
    )
    X_pred = feature_matrix(predict_rows)
    return model.predict_proba(scaler.transform(X_pred))[:, 1]


def lofo_editor(videos, partitions, families):
    """Return per-source counts under the learned editor with outer family LOFO.

    Both branches share the same outer loop.  For each held-out family the
    model is fit on the other families' rows only; probabilities on held-out
    rows are thresholded at fixed 0.5 and replayed per source.
    """
    per_source = {}
    for held_family in families:
        fit_videos = [v for v in videos if v.family != held_family]
        held_videos = [v for v in videos if v.family == held_family]
        fit_delete_rows = []
        fit_restore_rows = []
        for v in fit_videos:
            d_rows, r_rows, _, _ = partitions[v.source_name]
            fit_delete_rows.extend(d_rows)
            fit_restore_rows.extend(r_rows)
        held_delete_probs = {}
        held_restore_probs = {}
        for v in held_videos:
            d_rows, r_rows, _, _ = partitions[v.source_name]
            held_delete_probs[v.source_name] = fit_predict(fit_delete_rows, d_rows, "pure_fp")
            held_restore_probs[v.source_name] = fit_predict(fit_restore_rows, r_rows, "has_target")
        for v in held_videos:
            d_rows, r_rows, d_idx, r_idx = partitions[v.source_name]
            del_mask = held_delete_probs[v.source_name] >= 0.5
            res_mask = held_restore_probs[v.source_name] >= 0.5
            scores = diag.apply_actions(v, d_idx, r_idx, del_mask, res_mask)
            per_source[v.source_name] = {
                "baseline": v.baseline,
                "candidate": diag.official_counts_thr(scores, v),
                "delete_probs": held_delete_probs[v.source_name],
                "restore_probs": held_restore_probs[v.source_name],
            }
    return per_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only editor refuses an initialized CUDA process")

    manifest, _ = oracle.validate_inputs()
    records = manifest["records"]
    diag.FAMILY_MAP = diag.build_family_map(records)
    low_records = [r for r in records if r["decision"]["domain"] == "low"]

    videos = []
    partitions = {}
    for index, metadata in enumerate(low_records, start=1):
        video = diag.prepare_video_any(metadata)
        delete_rows, restore_rows, delete_indices, restore_indices = diag.build_partitions(video)
        partitions[video.source_name] = (delete_rows, restore_rows, delete_indices, restore_indices)
        videos.append(video)
        print("prepared {}/{} {}".format(index, len(low_records), video.source_name), flush=True)

    families = sorted({v.family for v in videos})

    # ---- Stage 1: label-free rule replays ----
    rule_results = {}
    rule_specs = {
        "ground_check_delete_all_pure_fp": (rule_delete_all_pure_fp, rule_none),
        "restore_all_removed": (rule_none, make_restore_rule()),
        "restore_score_max_ge_0p80": (rule_none, restore_rule_score_max(0.80)),
        "restore_score_max_ge_0p70": (rule_none, restore_rule_score_max(0.70)),
        "restore_score_max_ge_0p60": (rule_none, restore_rule_score_max(0.60)),
        "restore_2ev_score_ge_0p70": (rule_none, restore_rule_events_and_score(2, 0.70)),
        "restore_3ev_score_ge_0p65": (rule_none, restore_rule_events_and_score(3, 0.65)),
        "joined_delete_pure_fp_restore_3ev_0p65": (
            rule_delete_all_pure_fp, restore_rule_events_and_score(3, 0.65)),
    }
    for name, (del_fn, res_fn) in rule_specs.items():
        rule_results[name] = pooled_metrics(
            replay(videos, partitions, del_fn, res_fn)
        )
        print("rule {}: {}".format(name, rule_results[name]["metric_delta"]["score"]), flush=True)

    # ---- Stage 2: learned editor LOFO ----
    lofo = lofo_editor(videos, partitions, families)
    lofo_result = pooled_metrics(lofo)
    per_family_lofo = {}
    for fam in families:
        fam_videos = [v for v in videos if v.family == fam]
        fam_counts = {v.source_name: lofo[v.source_name] for v in fam_videos}
        per_family_lofo[fam] = pooled_metrics(fam_counts)

    # ---- Gates ----
    md = lofo_result["metric_delta"]
    gates = {
        "pooled_score_delta_positive": md["score"] > 0.0,
        "every_outer_family_score_not_lower": all(
            per_family_lofo[f]["metric_delta"]["score"] >= 0.0 for f in families
        ),
        "pooled_tp_not_lower": lofo_result["count_delta"]["true_positive_events"] >= 0,
        "pooled_correct_frames_not_lower": lofo_result["count_delta"]["correct_target_frames"] >= 0,
        "pooled_pd_not_lower": md["pd"] >= 0.0,
    }
    gates["all_passed"] = all(gates.values())

    payload = {
        "schema": "ev-uav-low-component-editor-lofo-train-only-v1",
        "created_utc": utc_now(),
        "dataset_split": "train",
        "domains": ["low"],
        "validation_or_test_read": False,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "families": families,
        "feature_names": diag.FEATURE_NAMES,
        "cutoff": 0.5,
        "rule_replays": rule_results,
        "lofo_editor": {
            "pooled": lofo_result,
            "per_family": per_family_lofo,
            "per_source": {
                name: {
                    "count_delta": oracle.count_delta(v["candidate"], v["baseline"]),
                    "n_delete_called": int((v["delete_probs"] >= 0.5).sum()),
                    "n_restore_called": int((v["restore_probs"] >= 0.5).sum()),
                    "n_delete_rows": int(len(v["delete_probs"])),
                    "n_restore_rows": int(len(v["restore_probs"])),
                }
                for name, v in lofo.items()
            },
        },
        "gates": gates,
        "inputs": {
            "manifest_path": str(oracle.MANIFEST_PATH),
            "manifest_sha256": oracle.EXPECTED_MANIFEST_SHA256,
            "runner_sha256": oracle.sha256_file(Path(__file__)),
            "diag_sha256": oracle.sha256_file(Path(diag.__file__)),
        },
    }
    digest = write_json_exclusive(args.output, payload)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "sha256": digest,
        "ground_check": rule_results["ground_check_delete_all_pure_fp"]["metric_delta"]["score"],
        "best_label_free_rule": max(
            ((k, v["metric_delta"]["score"]) for k, v in rule_results.items()
             if k != "ground_check_delete_all_pure_fp"),
            key=lambda kv: kv[1],
        ),
        "lofo_score_delta": md["score"],
        "gates": gates,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
