"""Strict nested train-only F5 smoke for continuous positive-support guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import run_allsize_deletion_head_oof as base
from utils import positive_support_guard as features_module
from utils.positive_support_distance_guard import RobustPositiveSupport


FIT_FAMILIES = tuple(group for group in base.SOURCE_GROUPS if group != "block_088_098")
OUTER_HELD = "block_088_098"
EXPECTED_PROTOCOL_SHA256 = "187d86eef500b6ae2435dd5419ff63e5cdc77ad03044d91750f4a794656a70e3"
EVIDENCE_CHOICES = (
    "core_raw_relative",
    "relative_anchor",
)
MODE_CHOICES = ("nearest", "robust_box", "robust_rms", "lower_second")
FAMILY_AGGREGATIONS = ("maximum", "second_maximum")
ANCHOR_VETOES = ("none", "core_frame_maximum")
CUTOFF_RULES = ("minimum", "range_extrapolated")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metrics(counts):
    return {"counts": counts.to_dict(), "metrics": base.crossfit.metrics_from_counts(counts)}


def _gates(candidate, baseline):
    cm = base.crossfit.metrics_from_counts(candidate)
    bm = base.crossfit.metrics_from_counts(baseline)
    return {
        "score_not_lower": cm["score"] >= bm["score"],
        "iou_not_lower": cm["iou"] >= bm["iou"],
        "pd_not_lower": cm["pd"] >= bm["pd"],
        "fa_not_higher": cm["fa"] <= bm["fa"],
        "correct_objects_not_lower": candidate.correct_objects >= baseline.correct_objects,
        "zero_tp_loss": candidate.true_positive_events == baseline.true_positive_events,
    }


def _fit_family_models(videos, evidence_names, mode):
    result = {}
    for family in FIT_FAMILIES:
        family_videos = [video for video in videos if video.group == family]
        model = RobustPositiveSupport(evidence_names, mode)
        result[family] = model.fit(
            [video.features for video in family_videos],
            [video.component_labels > 0 for video in family_videos],
        )
    return result


def _support(models, excluded_family, video, family_aggregation, anchor_veto):
    values = np.stack(
        [
            model.predict_support(video.features)
            for family, model in models.items()
            if family != excluded_family
        ],
        axis=0,
    )
    if family_aggregation == "maximum":
        support = values.max(axis=0)
    elif family_aggregation == "second_maximum":
        if values.shape[0] < 2:
            raise RuntimeError("second family support needs at least two models.")
        support = np.partition(values, -2, axis=0)[-2]
    else:
        raise KeyError(family_aggregation)
    if anchor_veto == "core_frame_maximum":
        names = (
            "frame_percentile_score_max",
            "frame_percentile_score_mean",
            "frame_percentile_log_component_events",
            "frame_percentile_track_bin_count",
            "frame_percentile_track_score_max",
        )
        columns = [features_module.FEATURE_NAMES.index(name) for name in names]
        anchors = np.any(video.features[:, columns] >= 1.0, axis=1)
        support = support.copy()
        support[anchors] = np.inf
    elif anchor_veto != "none":
        raise KeyError(anchor_veto)
    return support


def _cutoff(endpoints, rule):
    values = np.asarray(list(endpoints), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("cutoff endpoints must be nonempty and finite.")
    if rule == "minimum":
        return float(values.min())
    if rule == "range_extrapolated":
        return float(values.min() - (values.max() - values.min()))
    raise KeyError(rule)


def _removed_event_counts(video, support, cutoff):
    removed = [
        np.asarray(indices, dtype=np.int64)
        for indices, value in zip(video.event_indices, support)
        if float(value) < float(cutoff)
    ]
    if not removed:
        return 0, 0, 0
    indices = np.concatenate(removed)
    positive = video.labels[indices] > 0
    return int(positive.sum()), int((~positive).sum()), len(removed)


def _official_fold(videos, supports, cutoff):
    baseline = base._sum_counts(video.baseline_counts for video in videos)
    candidates = [
        base._candidate_counts(video, supports[video.source_name], cutoff)
        for video in videos
    ]
    candidate = base._sum_counts(candidates)
    sources = []
    for video, counts in zip(videos, candidates):
        source_gates = _gates(counts, video.baseline_counts)
        sources.append({
            "source_name": video.source_name,
            "score_delta": base.crossfit.metrics_from_counts(counts)["score"]
            - base.crossfit.metrics_from_counts(video.baseline_counts)["score"],
            "gates": source_gates,
        })
    return baseline, candidate, sources


def run(args):
    root = Path(__file__).resolve().parent
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output_report).resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol_hash = _sha256(protocol_path)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(f"distance-guard protocol changed: {protocol_hash}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["scope"]["dataset_split"] != "train":
        raise ValueError("distance guard smoke must remain train-only.")

    base.FEATURE_NAMES = features_module.FEATURE_NAMES
    base.extract_allsize_components = features_module.extract_positive_support_components
    cache = Path(args.cache_dir).resolve()
    c00_protocol = Path(args.c00_protocol).resolve()
    manifest, _cfg, videos = base.prepare_videos(cache, c00_protocol)

    model_cache = {}
    support_cache = {}
    candidate_records = []
    for evidence_choice in EVIDENCE_CHOICES:
        evidence_names = features_module.EVIDENCE_SETS[evidence_choice]
        for mode in MODE_CHOICES:
            model_key = (evidence_choice, mode)
            model_cache[model_key] = _fit_family_models(videos, evidence_names, mode)
            for family_aggregation in FAMILY_AGGREGATIONS:
                for anchor_veto in ANCHOR_VETOES:
                    support_key = (evidence_choice, mode, family_aggregation, anchor_veto)
                    supports = {}
                    endpoints = {}
                    for held_family in FIT_FAMILIES:
                        for video in [item for item in videos if item.group == held_family]:
                            supports[video.source_name] = _support(
                                model_cache[model_key], held_family, video,
                                family_aggregation, anchor_veto,
                            )
                        family_positive = np.concatenate([
                            supports[video.source_name][video.component_labels > 0]
                            for video in videos if video.group == held_family
                        ])
                        endpoints[held_family] = float(family_positive.min())
                    support_cache[support_key] = supports
                    for cutoff_rule in CUTOFF_RULES:
                        fold_cutoffs = {
                            held_family: _cutoff(
                                [value for family, value in endpoints.items() if family != held_family],
                                cutoff_rule,
                            )
                            for held_family in FIT_FAMILIES
                        }
                        quick_folds = []
                        exact_positive_loss = 0
                        removed_false_events = 0
                        removed_components = 0
                        for held_family in FIT_FAMILIES:
                            fold_positive_loss = 0
                            fold_false_events = 0
                            fold_components = 0
                            for video in [item for item in videos if item.group == held_family]:
                                positive_loss, false_events, components = _removed_event_counts(
                                    video, supports[video.source_name], fold_cutoffs[held_family]
                                )
                                fold_positive_loss += positive_loss
                                fold_false_events += false_events
                                fold_components += components
                            exact_positive_loss += fold_positive_loss
                            removed_false_events += fold_false_events
                            removed_components += fold_components
                            quick_folds.append({
                                "held_family": held_family,
                                "transferred_cutoff": fold_cutoffs[held_family],
                                "positive_event_loss": fold_positive_loss,
                                "removed_false_positive_events": fold_false_events,
                                "removed_components": fold_components,
                            })
                        candidate_records.append({
                            "evidence_choice": evidence_choice,
                            "mode": mode,
                            "family_aggregation": family_aggregation,
                            "anchor_veto": anchor_veto,
                            "cutoff_rule": cutoff_rule,
                            "positive_support_endpoints": endpoints,
                            "quick_folds": quick_folds,
                            "inner_exact_positive_event_loss": exact_positive_loss,
                            "inner_removed_false_positive_events": removed_false_events,
                            "inner_removed_components": removed_components,
                            "inner_safety_passed": False,
                            "inner_pooled": None,
                        })

    safe_quick = [
        item for item in candidate_records
        if item["inner_exact_positive_event_loss"] == 0
        and item["inner_removed_false_positive_events"] > 0
    ]
    for item in safe_quick:
        supports = support_cache[(
            item["evidence_choice"], item["mode"],
            item["family_aggregation"], item["anchor_veto"],
        )]
        pooled_baseline = base.crossfit.SufficientCounts()
        pooled_candidate = base.crossfit.SufficientCounts()
        official_folds = []
        for quick in item["quick_folds"]:
            held_family = quick["held_family"]
            held_videos = [video for video in videos if video.group == held_family]
            baseline, candidate, sources = _official_fold(
                held_videos, supports, quick["transferred_cutoff"]
            )
            gates = _gates(candidate, baseline)
            gates["every_source_safe"] = all(all(source["gates"].values()) for source in sources)
            official_folds.append({
                **quick,
                "baseline": _metrics(baseline),
                "candidate": _metrics(candidate),
                "score_delta": base.crossfit.metrics_from_counts(candidate)["score"]
                - base.crossfit.metrics_from_counts(baseline)["score"],
                "gates": gates,
                "worst_source": sorted(sources, key=lambda source: source["score_delta"])[0],
            })
            pooled_baseline = pooled_baseline + baseline
            pooled_candidate = pooled_candidate + candidate
        item["official_folds"] = official_folds
        item["inner_safety_passed"] = all(
            all(fold["gates"].values()) for fold in official_folds
        )
        item["inner_pooled"] = {
            "baseline": _metrics(pooled_baseline),
            "candidate": _metrics(pooled_candidate),
            "score_delta": base.crossfit.metrics_from_counts(pooled_candidate)["score"]
            - base.crossfit.metrics_from_counts(pooled_baseline)["score"],
        }
        print(
            f"safe-candidate {item['evidence_choice']}/{item['mode']}/"
            f"{item['family_aggregation']}/{item['anchor_veto']}/{item['cutoff_rule']}: "
            f"FP={item['inner_removed_false_positive_events']} "
            f"delta={item['inner_pooled']['score_delta']:+.9f} "
            f"safe={item['inner_safety_passed']}", flush=True,
        )

    eligible = [item for item in candidate_records if item["inner_safety_passed"]]
    selected = sorted(
        eligible,
        key=lambda item: (
            -item["inner_pooled"]["candidate"]["metrics"]["score"],
            item["evidence_choice"], item["mode"], item["family_aggregation"],
            item["anchor_veto"], item["cutoff_rule"],
        ),
    )[0] if eligible else None

    selected_outer = None
    if selected is not None and selected["inner_pooled"]["score_delta"] > 0.0:
        models = model_cache[(selected["evidence_choice"], selected["mode"])]
        outer_videos = [video for video in videos if video.group == OUTER_HELD]
        outer_supports = {
            video.source_name: _support(
                models, None, video, selected["family_aggregation"], selected["anchor_veto"]
            )
            for video in outer_videos
        }
        outer_cutoff = _cutoff(
            selected["positive_support_endpoints"].values(), selected["cutoff_rule"]
        )
        baseline, candidate, sources = _official_fold(
            outer_videos, outer_supports, outer_cutoff
        )
        outer_gates = _gates(candidate, baseline)
        outer_gates["every_source_safe"] = all(
            all(source["gates"].values()) for source in sources
        )
        selected_outer = {
            "fit_derived_cutoff": outer_cutoff,
            "baseline": _metrics(baseline),
            "candidate": _metrics(candidate),
            "score_delta": base.crossfit.metrics_from_counts(candidate)["score"]
            - base.crossfit.metrics_from_counts(baseline)["score"],
            "gates": outer_gates,
            "worst_source": sorted(sources, key=lambda source: source["score_delta"])[0],
        }

    success = bool(
        selected_outer is not None
        and all(selected_outer["gates"].values())
        and selected_outer["candidate"]["counts"]["false_positive_events"]
        < selected_outer["baseline"]["counts"]["false_positive_events"]
        and selected_outer["candidate"]["counts"]["false_components"]
        < selected_outer["baseline"]["counts"]["false_components"]
    )
    report = {
        "schema": "ev-uav-positive-support-distance-guard-f5-smoke-v2",
        "dataset_split": "train",
        "selection_access": "strict F1-F4 transferred-cutoff OOF; only selected winner opened on F5",
        "no_validation_test_or_gpu_access": True,
        "source_identity_is_feature": False,
        "feature_count": len(features_module.FEATURE_NAMES),
        "inputs": {
            "protocol_sha256": protocol_hash,
            "cache_manifest_sha256": _sha256(cache / "manifest.json"),
            "feature_module_sha256": _sha256(root / "utils" / "positive_support_guard.py"),
            "support_module_sha256": _sha256(root / "utils" / "positive_support_distance_guard.py"),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "source_count": manifest["selected_video_count"],
        },
        "candidate_count": len(candidate_records),
        "quick_safe_nonidentity_count": len(safe_quick),
        "candidates": candidate_records,
        "selected_by_inner": None if selected is None else {
            key: selected[key] for key in (
                "evidence_choice", "mode", "family_aggregation", "anchor_veto", "cutoff_rule"
            )
        } | {
            "inner_score_delta": selected["inner_pooled"]["score_delta"],
            "inner_removed_false_positive_events": selected["inner_removed_false_positive_events"],
        },
        "selected_outer_diagnostic": selected_outer,
        "success": success,
        "decision": "proceed_to_full_nested_oof" if success else "stop_positive_support_distance_v2",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "output_report": str(output),
        "selected_by_inner": report["selected_by_inner"],
        "selected_outer_diagnostic": selected_outer,
        "decision": report["decision"],
    }, indent=2))
    return 0


def parser():
    root = Path(__file__).resolve().parent
    experiments = root.parent / "experiments"
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", default=str(root / "protocols" / "positive_support_distance_guard_f5_smoke_v2.json"))
    result.add_argument("--cache-dir", default=str(experiments / "20260810_component_reranker_crosssource_v1" / "train_cache_gt30000"))
    result.add_argument("--c00-protocol", default=str(experiments / "20260810_component_reranker_crosssource_v1" / "crossfit_protocol.json"))
    result.add_argument("--output-report", default=str(experiments / "20260811_positive_support_distance_guard_v2_f5_smoke" / "f5_smoke_report.json"))
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
