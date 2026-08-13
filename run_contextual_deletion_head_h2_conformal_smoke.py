"""Cross-family OOF-conformal contextual deletion smoke on train F5."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import run_allsize_deletion_head_oof as base
from utils import contextual_deletion_head as contextual


EXPECTED_PROTOCOL_SHA256 = "32a74349c4d2eb059c7f547dc310ae84874c3b42e3ef238dd5315829f687e847"
FAMILIES = ("logistic_l2", "histgb_shallow", "extratrees_shallow")
OUTER_HELD = "block_088_098"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(counts):
    return {"counts": counts.to_dict(), "metrics": base.crossfit.metrics_from_counts(counts)}


def gates(candidate, baseline):
    cm = base.crossfit.metrics_from_counts(candidate); bm = base.crossfit.metrics_from_counts(baseline)
    return {
        "score_not_lower": cm["score"] >= bm["score"],
        "iou_not_lower": cm["iou"] >= bm["iou"],
        "pd_not_lower": cm["pd"] >= bm["pd"],
        "fa_not_higher": cm["fa"] <= bm["fa"],
        "correct_objects_not_lower": candidate.correct_objects >= baseline.correct_objects,
        "tp_not_lower": candidate.true_positive_events >= baseline.true_positive_events,
    }


def essential_components(video):
    """Mark sole component support for a detected target-id/50-bin object."""

    support = defaultdict(set)
    times = video.locations[:, 3].astype(np.int64)
    target_ids = np.asarray(video.target_ids).reshape(-1)
    valid = (times > 0) & ((times % 50) != 0) & (target_ids != 0) & (video.labels > 0)
    frames = np.floor_divide(times - 1, 50)
    for component_index, indices in enumerate(video.event_indices):
        indices = np.asarray(indices, dtype=np.int64)
        component_valid = indices[valid[indices]]
        for key in set(zip(frames[component_valid].tolist(), target_ids[component_valid].tolist())):
            support[key].add(component_index)
    result = np.zeros(len(video.event_indices), dtype=bool)
    for component_indices in support.values():
        if len(component_indices) == 1:
            result[next(iter(component_indices))] = True
    return result


def ensemble_probabilities(models, video, aggregation):
    values = np.stack([base._probabilities(model, video) for model in models])
    if aggregation == "mean":
        return values.mean(axis=0)
    if aggregation == "minimum":
        return values.min(axis=0)
    raise KeyError(aggregation)


def evaluate_videos(videos, probabilities, threshold):
    per_source = []
    candidate_counts = []
    for video in videos:
        candidate = base._candidate_counts(video, probabilities[video.source_name], threshold)
        candidate_counts.append(candidate)
        per_source.append({
            "source_name": video.source_name,
            "score_delta": base.crossfit.metrics_from_counts(candidate)["score"] - base.crossfit.metrics_from_counts(video.baseline_counts)["score"],
        })
    baseline = base._sum_counts(video.baseline_counts for video in videos)
    candidate = base._sum_counts(candidate_counts)
    return baseline, candidate, min(per_source, key=lambda item: item["score_delta"])


def run(args):
    root = Path(__file__).resolve().parent
    protocol_path = Path(args.science_protocol).resolve()
    output = Path(args.output_report).resolve()
    if output.exists():
        raise FileExistsError(output)
    if sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("conformal smoke protocol changed.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if "train-only" not in protocol["scope"] or "no validation" not in protocol["scope"]:
        raise ValueError("protocol is not train-only.")
    base.FEATURE_NAMES = contextual.FEATURE_NAMES
    base.extract_allsize_components = contextual.extract_allsize_components
    cache = Path(args.cache_dir).resolve(); c00 = Path(args.c00_protocol).resolve()
    manifest, _cfg, videos = base.prepare_videos(cache, c00)
    fit_groups = tuple(group for group in base.SOURCE_GROUPS if group != OUTER_HELD)
    outer_videos = [video for video in videos if video.group == OUTER_HELD]
    essential = {video.source_name: essential_components(video) for video in videos}

    candidates = []
    outer_probability_cache = {}
    for family in FAMILIES:
        pair_models = {}
        for training_groups in itertools.combinations(fit_groups, 2):
            pair_models[training_groups] = base._fit_model(
                family, [video for video in videos if video.group in training_groups]
            )
        main_models = []
        for excluded_group in fit_groups:
            training_groups = tuple(group for group in fit_groups if group != excluded_group)
            main_models.append(base._fit_model(
                family, [video for video in videos if video.group in training_groups]
            ))
        for aggregation in ("mean", "minimum"):
            inner_probabilities = {}
            held_labels = []
            held_values = []
            essential_values = []
            for held_group in fit_groups:
                eligible_models = [
                    model for groups, model in pair_models.items() if held_group not in groups
                ]
                if len(eligible_models) != 3:
                    raise RuntimeError("inner held family must have three unseen models.")
                for video in [item for item in videos if item.group == held_group]:
                    probabilities = ensemble_probabilities(eligible_models, video, aggregation)
                    inner_probabilities[video.source_name] = probabilities
                    held_labels.append(video.component_labels)
                    held_values.append(probabilities)
                    essential_values.extend(probabilities[essential[video.source_name]].tolist())
            if not essential_values:
                raise RuntimeError("no essential calibration components.")
            keep_threshold = float(np.min(np.asarray(essential_values, dtype=np.float64)))
            inner_folds = []
            pooled_baseline = base.crossfit.SufficientCounts(); pooled_candidate = base.crossfit.SufficientCounts()
            for held_group in fit_groups:
                held = [video for video in videos if video.group == held_group]
                baseline, candidate, worst = evaluate_videos(held, inner_probabilities, keep_threshold)
                fold_gates = gates(candidate, baseline)
                inner_folds.append({"held_group": held_group, "baseline": record(baseline), "candidate": record(candidate), "score_delta": base.crossfit.metrics_from_counts(candidate)["score"] - base.crossfit.metrics_from_counts(baseline)["score"], "gates": fold_gates, "worst_source": worst})
                pooled_baseline = pooled_baseline + baseline; pooled_candidate = pooled_candidate + candidate
            labels = np.concatenate(held_labels); values = np.concatenate(held_values)
            outer_probabilities = {video.source_name: ensemble_probabilities(main_models, video, aggregation) for video in outer_videos}
            outer_probability_cache[(family, aggregation)] = outer_probabilities
            outer_baseline, outer_candidate, outer_worst = evaluate_videos(outer_videos, outer_probabilities, keep_threshold)
            candidates.append({
                "family": family,
                "aggregation": aggregation,
                "cross_family_oof_essential_min_threshold": keep_threshold,
                "inner_component_roc_auc": float(roc_auc_score(labels, values)),
                "inner_component_average_precision": float(average_precision_score(labels, values)),
                "inner_folds": inner_folds,
                "inner_safety_passed": all(all(fold["gates"].values()) for fold in inner_folds),
                "inner_pooled": {"baseline": record(pooled_baseline), "candidate": record(pooled_candidate), "score_delta": base.crossfit.metrics_from_counts(pooled_candidate)["score"] - base.crossfit.metrics_from_counts(pooled_baseline)["score"]},
                "outer_diagnostic": {"baseline": record(outer_baseline), "candidate": record(outer_candidate), "score_delta": base.crossfit.metrics_from_counts(outer_candidate)["score"] - base.crossfit.metrics_from_counts(outer_baseline)["score"], "gates": gates(outer_candidate, outer_baseline), "worst_source": outer_worst},
            })
    eligible = [item for item in candidates if item["inner_safety_passed"]]
    selected = sorted(eligible, key=lambda item: (-item["inner_pooled"]["candidate"]["metrics"]["score"], item["family"], item["aggregation"]))[0] if eligible else None
    selected_outer = None if selected is None else selected["outer_diagnostic"]
    success = bool(selected is not None and all(selected_outer["gates"].values()) and selected_outer["candidate"]["counts"]["false_positive_events"] < selected_outer["baseline"]["counts"]["false_positive_events"] and selected_outer["candidate"]["counts"]["false_components"] < selected_outer["baseline"]["counts"]["false_components"])
    report = {
        "schema": "ev-uav-contextual-deletion-head-f5-cross-family-conformal-smoke-v2.1",
        "dataset_split": "train",
        "selection_access": "F1-F4 inner grouped predictions only; F5 diagnostic never selects",
        "feature_count": len(contextual.FEATURE_NAMES),
        "source_identity_is_feature": False,
        "inputs": {"science_protocol_sha256": sha256_file(protocol_path), "cache_manifest_sha256": sha256_file(cache / "manifest.json"), "module_sha256": sha256_file(root / "utils" / "contextual_deletion_head.py"), "runner_sha256": sha256_file(Path(__file__).resolve()), "source_count": manifest["selected_video_count"]},
        "candidates": candidates,
        "selected_by_inner": None if selected is None else {"family": selected["family"], "aggregation": selected["aggregation"], "threshold": selected["cross_family_oof_essential_min_threshold"], "inner_score_delta": selected["inner_pooled"]["score_delta"]},
        "selected_outer_diagnostic": selected_outer,
        "success": success,
        "decision": "proceed_to_full_nested_oof" if success else "stop_contextual_conformal_v2_1",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps({"output_report": str(output), "selected_by_inner": report["selected_by_inner"], "selected_outer_diagnostic": selected_outer, "decision": report["decision"]}, indent=2))
    return 0


def parser():
    root = Path(__file__).resolve().parent; experiments = root.parent / "experiments"
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--science-protocol", default=str(root / "protocols" / "contextual_allsize_deletion_head_conformal_smoke_v2_1.json"))
    result.add_argument("--cache-dir", default=str(experiments / "20260810_component_reranker_crosssource_v1" / "train_cache_gt30000"))
    result.add_argument("--c00-protocol", default=str(experiments / "20260810_component_reranker_crosssource_v1" / "crossfit_protocol.json"))
    result.add_argument("--output-report", default=str(experiments / "20260811_contextual_deletion_head_v2_1_conformal_smoke" / "h2_conformal_smoke_report.json"))
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
