"""Train-only held-F5 smoke for a source-free positive-support deletion guard."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import run_allsize_deletion_head_oof as base
from utils import positive_support_guard as guard


FAMILIES = tuple(group for group in base.SOURCE_GROUPS if group != "block_088_098")
OUTER_HELD = "block_088_098"
EXPECTED_PROTOCOL_SHA256 = "2667e401e99e4ca174ee8ce284d2438ca62318599685b4df85b368a6f2b164c7"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(counts):
    return {"counts": counts.to_dict(), "metrics": base.crossfit.metrics_from_counts(counts)}


def _gates(candidate, baseline):
    candidate_metrics = base.crossfit.metrics_from_counts(candidate)
    baseline_metrics = base.crossfit.metrics_from_counts(baseline)
    return {
        "score_not_lower": candidate_metrics["score"] >= baseline_metrics["score"],
        "iou_not_lower": candidate_metrics["iou"] >= baseline_metrics["iou"],
        "pd_not_lower": candidate_metrics["pd"] >= baseline_metrics["pd"],
        "fa_not_higher": candidate_metrics["fa"] <= baseline_metrics["fa"],
        "correct_objects_not_lower": candidate.correct_objects >= baseline.correct_objects,
        "tp_not_lower": candidate.true_positive_events >= baseline.true_positive_events,
    }


def _essential_components(video):
    support = defaultdict(set)
    times = video.locations[:, 3].astype(np.int64)
    target_ids = np.asarray(video.target_ids).reshape(-1)
    valid = (times > 0) & ((times % 50) != 0) & (target_ids != 0) & (video.labels > 0)
    frames = np.floor_divide(times - 1, 50)
    for component_index, indices in enumerate(video.event_indices):
        indices = np.asarray(indices, dtype=np.int64)
        selected = indices[valid[indices]]
        for key in set(zip(frames[selected].tolist(), target_ids[selected].tolist())):
            support[key].add(component_index)
    result = np.zeros(len(video.event_indices), dtype=bool)
    for component_indices in support.values():
        if len(component_indices) == 1:
            result[next(iter(component_indices))] = True
    return result


def _fit_family_guard(videos, evidence_set, aggregation, positive_scope, essential):
    model = guard.WeightedMarginalSupport(
        guard.EVIDENCE_SETS[evidence_set], aggregation=aggregation
    )
    masks = []
    for video in videos:
        all_positive = video.component_labels > 0
        if positive_scope == "all_positive":
            masks.append(all_positive)
        elif positive_scope == "essential_or_all_positive":
            # The essential target-support model is a conservative extra guard:
            # unioning it with the all-positive model occurs in _fit_models.
            masks.append(essential[video.source_name])
        else:
            raise KeyError(positive_scope)
    return model.fit([video.features for video in videos], masks)


def _fit_models(videos, evidence_set, aggregation, positive_scope, essential):
    fitted = []
    for family in sorted({video.group for video in videos}):
        family_videos = [video for video in videos if video.group == family]
        all_model = _fit_family_guard(
            family_videos, evidence_set, aggregation, "all_positive", essential
        )
        if positive_scope == "all_positive":
            fitted.append((all_model,))
        else:
            essential_model = _fit_family_guard(
                family_videos, evidence_set, aggregation,
                "essential_or_all_positive", essential
            )
            fitted.append((all_model, essential_model))
    return fitted


def _ensemble_support(models, video):
    # A component is retained when any fitted positive population supports it.
    values = [model.predict_support(video.features) for pair in models for model in pair]
    return np.stack(values, axis=0).max(axis=0)


def _evaluate(videos, probabilities, threshold):
    baseline = base._sum_counts(video.baseline_counts for video in videos)
    candidates = [
        base._candidate_counts(video, probabilities[video.source_name], threshold)
        for video in videos
    ]
    candidate = base._sum_counts(candidates)
    per_source = []
    for video, counts in zip(videos, candidates):
        per_source.append(
            {
                "source_name": video.source_name,
                "score_delta": base.crossfit.metrics_from_counts(counts)["score"]
                - base.crossfit.metrics_from_counts(video.baseline_counts)["score"],
                "tp_delta": counts.true_positive_events - video.baseline_counts.true_positive_events,
                "fp_delta": counts.false_positive_events - video.baseline_counts.false_positive_events,
                "false_components_delta": counts.false_components - video.baseline_counts.false_components,
            }
        )
    return baseline, candidate, sorted(per_source, key=lambda item: item["score_delta"])[0]


def run(args):
    root = Path(__file__).resolve().parent
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output_report).resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol_hash = sha256_file(protocol_path)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(f"positive-support protocol changed: {protocol_hash}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["scope"]["dataset_split"] != "train":
        raise ValueError("positive-support smoke must remain train-only.")

    base.FEATURE_NAMES = guard.FEATURE_NAMES
    base.extract_allsize_components = guard.extract_positive_support_components
    cache = Path(args.cache_dir).resolve()
    c00_protocol = Path(args.c00_protocol).resolve()
    manifest, _cfg, videos = base.prepare_videos(cache, c00_protocol)
    essential = {video.source_name: _essential_components(video) for video in videos}

    configurations = [
        (evidence_set, aggregation, positive_scope)
        for evidence_set in sorted(guard.EVIDENCE_SETS)
        for aggregation in ("maximum", "second_maximum")
        for positive_scope in ("all_positive", "essential_or_all_positive")
    ]
    candidate_records = []
    outer_cache = {}
    for evidence_set, aggregation, positive_scope in configurations:
        inner_probabilities = {}
        for held_family in FAMILIES:
            fit_videos = [
                video for video in videos
                if video.group in FAMILIES and video.group != held_family
            ]
            models = _fit_models(
                fit_videos, evidence_set, aggregation, positive_scope, essential
            )
            for video in [item for item in videos if item.group == held_family]:
                inner_probabilities[video.source_name] = _ensemble_support(models, video)

        all_positive_support = np.concatenate(
            [
                inner_probabilities[video.source_name][video.component_labels > 0]
                for video in videos if video.group in FAMILIES
            ]
        )
        if all_positive_support.size == 0:
            raise RuntimeError("inner OOF calibration has no positive components.")
        keep_threshold = float(all_positive_support.min())
        folds = []
        pooled_baseline = base.crossfit.SufficientCounts()
        pooled_candidate = base.crossfit.SufficientCounts()
        for held_family in FAMILIES:
            held = [video for video in videos if video.group == held_family]
            baseline, candidate, worst = _evaluate(
                held, inner_probabilities, keep_threshold
            )
            folds.append(
                {
                    "held_family": held_family,
                    "baseline": _record(baseline),
                    "candidate": _record(candidate),
                    "score_delta": base.crossfit.metrics_from_counts(candidate)["score"]
                    - base.crossfit.metrics_from_counts(baseline)["score"],
                    "gates": _gates(candidate, baseline),
                    "worst_source": worst,
                }
            )
            pooled_baseline = pooled_baseline + baseline
            pooled_candidate = pooled_candidate + candidate

        outer_fit = [video for video in videos if video.group in FAMILIES]
        outer_models = _fit_models(
            outer_fit, evidence_set, aggregation, positive_scope, essential
        )
        outer_videos = [video for video in videos if video.group == OUTER_HELD]
        outer_probabilities = {
            video.source_name: _ensemble_support(outer_models, video)
            for video in outer_videos
        }
        outer_cache[(evidence_set, aggregation, positive_scope)] = outer_probabilities
        outer_baseline, outer_candidate, outer_worst = _evaluate(
            outer_videos, outer_probabilities, keep_threshold
        )
        inner_safe = all(all(fold["gates"].values()) for fold in folds)
        candidate_records.append(
            {
                "evidence_set": evidence_set,
                "aggregation": aggregation,
                "positive_scope": positive_scope,
                "inner_oof_positive_min_threshold": keep_threshold,
                "inner_folds": folds,
                "inner_safety_passed": inner_safe,
                "inner_pooled": {
                    "baseline": _record(pooled_baseline),
                    "candidate": _record(pooled_candidate),
                    "score_delta": base.crossfit.metrics_from_counts(pooled_candidate)["score"]
                    - base.crossfit.metrics_from_counts(pooled_baseline)["score"],
                },
                "outer_diagnostic": {
                    "baseline": _record(outer_baseline),
                    "candidate": _record(outer_candidate),
                    "score_delta": base.crossfit.metrics_from_counts(outer_candidate)["score"]
                    - base.crossfit.metrics_from_counts(outer_baseline)["score"],
                    "gates": _gates(outer_candidate, outer_baseline),
                    "worst_source": outer_worst,
                },
            }
        )
        print(
            f"candidate {evidence_set}/{aggregation}/{positive_scope}: "
            f"inner={candidate_records[-1]['inner_pooled']['score_delta']:+.9f} "
            f"safe={inner_safe} outer={candidate_records[-1]['outer_diagnostic']['score_delta']:+.9f}",
            flush=True,
        )

    eligible = [item for item in candidate_records if item["inner_safety_passed"]]
    selected = sorted(
        eligible,
        key=lambda item: (
            -item["inner_pooled"]["candidate"]["metrics"]["score"],
            item["evidence_set"], item["aggregation"], item["positive_scope"],
        ),
    )[0] if eligible else None
    selected_outer = None if selected is None else selected["outer_diagnostic"]
    nonidentity = bool(
        selected is not None
        and selected["inner_pooled"]["candidate"]["counts"]
        != selected["inner_pooled"]["baseline"]["counts"]
    )
    success = bool(
        nonidentity
        and all(selected_outer["gates"].values())
        and selected_outer["candidate"]["counts"]["false_positive_events"]
        < selected_outer["baseline"]["counts"]["false_positive_events"]
        and selected_outer["candidate"]["counts"]["false_components"]
        < selected_outer["baseline"]["counts"]["false_components"]
    )
    report = {
        "schema": "ev-uav-positive-support-guard-f5-smoke-v1",
        "dataset_split": "train",
        "selection_access": "F1-F4 grouped OOF only; F5 diagnostic does not select",
        "no_validation_test_or_gpu_access": True,
        "source_identity_is_feature": False,
        "feature_count": len(guard.FEATURE_NAMES),
        "feature_names": list(guard.FEATURE_NAMES),
        "inputs": {
            "protocol_sha256": protocol_hash,
            "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
            "module_sha256": sha256_file(root / "utils" / "positive_support_guard.py"),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "source_count": manifest["selected_video_count"],
        },
        "candidates": candidate_records,
        "selected_by_inner": None if selected is None else {
            "evidence_set": selected["evidence_set"],
            "aggregation": selected["aggregation"],
            "positive_scope": selected["positive_scope"],
            "threshold": selected["inner_oof_positive_min_threshold"],
            "inner_score_delta": selected["inner_pooled"]["score_delta"],
        },
        "selected_outer_diagnostic": selected_outer,
        "success": success,
        "decision": "proceed_to_full_nested_oof" if success else "stop_positive_support_guard_v1",
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
    result.add_argument(
        "--protocol",
        default=str(root / "protocols" / "positive_support_guard_f5_smoke_v1.json"),
    )
    result.add_argument(
        "--cache-dir",
        default=str(experiments / "20260810_component_reranker_crosssource_v1" / "train_cache_gt30000"),
    )
    result.add_argument(
        "--c00-protocol",
        default=str(experiments / "20260810_component_reranker_crosssource_v1" / "crossfit_protocol.json"),
    )
    result.add_argument(
        "--output-report",
        default=str(experiments / "20260811_positive_support_guard_v1_f5_smoke" / "f5_smoke_report.json"),
    )
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
