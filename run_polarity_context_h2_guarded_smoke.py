"""Train-only F5 smoke for polarity-context deletion with frame guards.

The numeric deletion threshold and the frame guard are selected exclusively
from cross-family predictions on F1--F4.  F5/H2 labels are opened only after
that decision is fixed.  This is a bounded diagnostic, not a deployment
artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import run_allsize_deletion_head_oof as base
from run_contextual_deletion_head_h2_conformal_smoke import essential_components
from utils.polarity_context_deletion_head import (
    FEATURE_NAMES,
    extract_polarity_context_components,
)


OUTER_HELD = "block_088_098"
MODEL_FAMILIES = ("histgb_shallow", "extratrees_shallow")
AGGREGATIONS = ("raw_mean", "rank_mean")
GUARD_TOP_K = (0, 1, 2)
QUANTILE_GRID = np.linspace(0.0, 0.30, 61, dtype=np.float64)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fractional_ranks(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return np.ones(values.size, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) / (values.size - 1)
        start = end
    return ranks


def _load_observable_inputs(dataset_root, video):
    path = Path(dataset_root) / video.source_name
    with np.load(path, allow_pickle=False) as payload:
        raw_locations = np.asarray(payload["ev_loc"])
        events = np.asarray(payload["ev"])
        polarities = np.asarray(events["p"]).reshape(-1).copy()
        fine_timestamps = np.asarray(events["t"], dtype=np.float64).reshape(-1).copy()
    if raw_locations.shape != video.locations[:, 1:4].shape:
        raise ValueError(f"raw/cache shape mismatch: {video.source_name}")
    if not np.array_equal(raw_locations, video.locations[:, 1:4]):
        raise ValueError(f"raw/cache location mismatch: {video.source_name}")
    return polarities, fine_timestamps


def prepare_polarity_videos(cache_dir, c00_protocol, dataset_root):
    manifest, cfg, videos = base.prepare_videos(cache_dir, c00_protocol)
    rebuilt = []
    for index, video in enumerate(videos, start=1):
        polarities, fine_timestamps = _load_observable_inputs(dataset_root, video)
        components = extract_polarity_context_components(
            video.scores,
            video.locations,
            base.THRESHOLD,
            base.TOPOLOGY,
            video.event_count,
            polarities,
            fine_timestamps,
            context_scores=video.scores,
        )
        # Labels are attached only after the label-free features are complete.
        component_labels = np.asarray(
            [int(np.any(video.labels[np.asarray(item, dtype=np.int64)] > 0)) for item in components.event_indices],
            dtype=np.uint8,
        )
        if len(components.event_indices) != len(video.event_indices):
            raise RuntimeError("polarity representation changed frozen components.")
        for old, new in zip(video.event_indices, components.event_indices):
            if not np.array_equal(old, new):
                raise RuntimeError("polarity representation changed component membership.")
        rebuilt.append(
            replace(
                video,
                event_indices=components.event_indices,
                features=components.features,
                component_labels=component_labels,
            )
        )
        print(
            f"polarity {index:02d}/54 {video.source_name}: "
            f"{components.features.shape[0]}x{components.features.shape[1]}",
            flush=True,
        )
    return manifest, cfg, rebuilt


def _video_ranks(values, video):
    output = np.empty_like(values, dtype=np.float64)
    bins = np.asarray(
        [video.locations[np.asarray(indices, dtype=np.int64)[0], 3] // 50 for indices in video.event_indices],
        dtype=np.int64,
    )
    for temporal_bin in np.unique(bins):
        members = np.flatnonzero(bins == temporal_bin)
        output[members] = _fractional_ranks(values[members])
    return output


def ensemble_probabilities(models, video, aggregation):
    raw = np.stack([base._probabilities(model, video) for model in models])
    if aggregation == "raw_mean":
        return raw.mean(axis=0)
    if aggregation == "rank_mean":
        return np.stack([_video_ranks(row, video) for row in raw]).mean(axis=0)
    raise KeyError(aggregation)


def deletion_mask(video, probabilities, threshold, guard_top_k):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    delete = probabilities < float(threshold)
    if guard_top_k > 0 and delete.any():
        bins = np.asarray(
            [video.locations[np.asarray(indices, dtype=np.int64)[0], 3] // 50 for indices in video.event_indices],
            dtype=np.int64,
        )
        for temporal_bin in np.unique(bins):
            members = np.flatnonzero(bins == temporal_bin)
            count = min(int(guard_top_k), members.size)
            keep = members[np.argsort(probabilities[members], kind="mergesort")[-count:]]
            delete[keep] = False
    return delete


def candidate_counts(video, probabilities, threshold, guard_top_k):
    delete = deletion_mask(video, probabilities, threshold, guard_top_k)
    scores = video.scores.copy()
    for component_index in np.flatnonzero(delete):
        scores[np.asarray(video.event_indices[int(component_index)], dtype=np.int64)] = np.float32(0.0)
    return base.official_counts(scores, video.labels, video.target_ids, video.locations)


def _counts_record(counts):
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
    }


def approximate_counts(video, delete):
    tp_removed = 0
    fp_removed = 0
    false_components_removed = 0
    for component_index in np.flatnonzero(delete):
        indices = np.asarray(video.event_indices[int(component_index)], dtype=np.int64)
        labels = video.labels[indices] > 0
        tp_removed += int(labels.sum())
        fp_removed += int((~labels).sum())
        false_components_removed += int(not labels.any())
    return base.crossfit.SufficientCounts(
        true_positive_events=video.baseline_counts.true_positive_events - tp_removed,
        false_positive_events=video.baseline_counts.false_positive_events - fp_removed,
        false_negative_events=video.baseline_counts.false_negative_events + tp_removed,
        correct_objects=video.baseline_counts.correct_objects,
        object_count=video.baseline_counts.object_count,
        false_components=max(0, video.baseline_counts.false_components - false_components_removed),
        frame_count=video.baseline_counts.frame_count,
        event_count=video.baseline_counts.event_count,
    )


def select_from_inner(videos, probabilities):
    fit_groups = tuple(group for group in base.SOURCE_GROUPS if group != OUTER_HELD)
    all_values = np.concatenate([probabilities[video.source_name] for video in videos if video.group in fit_groups])
    thresholds = np.unique(np.quantile(all_values, QUANTILE_GRID))
    essentials = {video.source_name: essential_components(video) for video in videos if video.group in fit_groups}
    candidates = []
    for guard_top_k in GUARD_TOP_K:
        for threshold in thresholds:
            fold_records = []
            pooled_baseline = base.crossfit.SufficientCounts()
            pooled_candidate = base.crossfit.SufficientCounts()
            safe = True
            for held_group in fit_groups:
                held = [video for video in videos if video.group == held_group]
                baseline = base._sum_counts(video.baseline_counts for video in held)
                approximate = []
                for video in held:
                    delete = deletion_mask(video, probabilities[video.source_name], threshold, guard_top_k)
                    if np.any(delete & essentials[video.source_name]):
                        safe = False
                    approximate.append(approximate_counts(video, delete))
                candidate = base._sum_counts(approximate)
                fold_gates = _gates(candidate, baseline)
                safe = safe and all(fold_gates.values())
                fold_records.append((held_group, baseline, candidate, fold_gates))
                pooled_baseline = pooled_baseline + baseline
                pooled_candidate = pooled_candidate + candidate
            score_delta = base.crossfit.metrics_from_counts(pooled_candidate)["score"] - base.crossfit.metrics_from_counts(pooled_baseline)["score"]
            candidates.append((safe, score_delta, int(guard_top_k), float(threshold), fold_records))
    eligible = [candidate for candidate in candidates if candidate[0]]
    if not eligible:
        return None, candidates
    return sorted(eligible, key=lambda item: (-item[1], item[2], item[3]))[0], candidates


def exact_evaluate(videos, probabilities, threshold, guard_top_k):
    per_source = []
    baseline = base.crossfit.SufficientCounts()
    candidate = base.crossfit.SufficientCounts()
    for video in videos:
        value = candidate_counts(video, probabilities[video.source_name], threshold, guard_top_k)
        baseline = baseline + video.baseline_counts
        candidate = candidate + value
        per_source.append(
            {
                "source_name": video.source_name,
                "score_delta": base.crossfit.metrics_from_counts(value)["score"]
                - base.crossfit.metrics_from_counts(video.baseline_counts)["score"],
            }
        )
    return baseline, candidate, sorted(per_source, key=lambda item: item["score_delta"])[0]


def run(args):
    cache_dir = Path(args.cache_dir).resolve()
    c00_protocol = Path(args.c00_protocol).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output = Path(args.output_report).resolve()
    if output.exists():
        raise FileExistsError(output)
    base.FEATURE_NAMES = FEATURE_NAMES
    manifest, _cfg, videos = prepare_polarity_videos(cache_dir, c00_protocol, dataset_root)
    fit_groups = tuple(group for group in base.SOURCE_GROUPS if group != OUTER_HELD)
    outer_videos = [video for video in videos if video.group == OUTER_HELD]
    results = []
    for family in MODEL_FAMILIES:
        pair_models = {}
        for left_index, left in enumerate(fit_groups):
            for right in fit_groups[left_index + 1 :]:
                pair_models[(left, right)] = base._fit_model(
                    family, [video for video in videos if video.group in (left, right)]
                )
        for aggregation in AGGREGATIONS:
            inner_probabilities = {}
            inner_labels = []
            inner_values = []
            for held_group in fit_groups:
                models = [model for groups, model in pair_models.items() if held_group not in groups]
                if len(models) != 3:
                    raise RuntimeError("each inner family requires three unseen pair models.")
                for video in [item for item in videos if item.group == held_group]:
                    values = ensemble_probabilities(models, video, aggregation)
                    inner_probabilities[video.source_name] = values
                    inner_labels.append(video.component_labels)
                    inner_values.append(values)
            selected, search = select_from_inner(videos, inner_probabilities)
            if selected is None:
                results.append({"family": family, "aggregation": aggregation, "selected": None})
                continue
            _, approximate_delta, guard_top_k, threshold, _ = selected
            inner_videos = [video for video in videos if video.group in fit_groups]
            inner_baseline, inner_candidate, inner_worst = exact_evaluate(
                inner_videos, inner_probabilities, threshold, guard_top_k
            )
            outer_probabilities = {
                video.source_name: ensemble_probabilities(list(pair_models.values()), video, aggregation)
                for video in outer_videos
            }
            outer_baseline, outer_candidate, outer_worst = exact_evaluate(
                outer_videos, outer_probabilities, threshold, guard_top_k
            )
            results.append(
                {
                    "family": family,
                    "aggregation": aggregation,
                    "selected": {
                        "threshold_from_inner_quantiles": threshold,
                        "guard_top_k_from_inner": guard_top_k,
                        "approximate_inner_score_delta": approximate_delta,
                    },
                    "inner_exact": {
                        "baseline": _counts_record(inner_baseline),
                        "candidate": _counts_record(inner_candidate),
                        "score_delta": base.crossfit.metrics_from_counts(inner_candidate)["score"] - base.crossfit.metrics_from_counts(inner_baseline)["score"],
                        "gates": _gates(inner_candidate, inner_baseline),
                        "worst_source": inner_worst,
                    },
                    "outer_h2_diagnostic": {
                        "baseline": _counts_record(outer_baseline),
                        "candidate": _counts_record(outer_candidate),
                        "score_delta": base.crossfit.metrics_from_counts(outer_candidate)["score"] - base.crossfit.metrics_from_counts(outer_baseline)["score"],
                        "gates": _gates(outer_candidate, outer_baseline),
                        "worst_source": outer_worst,
                    },
                    "inner_component_positive_mean": float(np.concatenate(inner_values)[np.concatenate(inner_labels) > 0].mean()),
                    "inner_component_negative_mean": float(np.concatenate(inner_values)[np.concatenate(inner_labels) == 0].mean()),
                }
            )
            print(
                f"{family}/{aggregation}: q-threshold={threshold:.8g} top{guard_top_k} "
                f"inner={results[-1]['inner_exact']['score_delta']:+.9f} "
                f"H2={results[-1]['outer_h2_diagnostic']['score_delta']:+.9f}",
                flush=True,
            )
    eligible = [
        result for result in results
        if result.get("selected") is not None
        and all(result["inner_exact"]["gates"].values())
    ]
    selected = sorted(
        eligible,
        key=lambda item: (-item["inner_exact"]["candidate"]["metrics"]["score"], item["family"], item["aggregation"]),
    )[0] if eligible else None
    outer = None if selected is None else selected["outer_h2_diagnostic"]
    success = bool(
        outer
        and all(outer["gates"].values())
        and outer["candidate"]["counts"]["false_positive_events"] < outer["baseline"]["counts"]["false_positive_events"]
        and outer["candidate"]["counts"]["false_components"] < outer["baseline"]["counts"]["false_components"]
    )
    report = {
        "schema": "ev-uav-polarity-context-guarded-f5-smoke-v1",
        "dataset_split": "train",
        "selection_access": "F1-F4 only; F5/H2 diagnostic labels opened after selection",
        "feature_count": len(FEATURE_NAMES),
        "forbidden_features": ["source_name", "source_index", "path", "hash", "fold", "label", "target_id"],
        "threshold_selection": "inner cross-family OOF quantile search with official safety gates; no fixed numeric deployment threshold",
        "results": results,
        "selected_by_inner": None if selected is None else {key: selected[key] for key in ("family", "aggregation", "selected")},
        "selected_outer_h2_diagnostic": outer,
        "success": success,
        "decision": "proceed_to_full_nested_oof" if success else "stop_polarity_context_v1",
        "inputs": {
            "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            "c00_protocol_sha256": sha256_file(c00_protocol),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "module_sha256": sha256_file(Path(__file__).resolve().parent / "utils" / "polarity_context_deletion_head.py"),
            "train_source_count": manifest["selected_video_count"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "decision": report["decision"], "selected": report["selected_by_inner"], "outer": outer}, indent=2))
    return 0


def parser():
    root = Path(__file__).resolve().parent
    experiments = root.parent / "experiments"
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache-dir", default=str(experiments / "20260810_component_reranker_crosssource_v1" / "train_cache_gt30000"))
    result.add_argument("--c00-protocol", default=str(experiments / "20260810_component_reranker_crosssource_v1" / "crossfit_protocol.json"))
    result.add_argument("--dataset-root", default=str(root.parent / "datasets" / "EV-UAV-Challenge2" / "train"))
    result.add_argument("--output-report", default=str(experiments / "20260811_polarity_context_h2_guarded_smoke_v1" / "report.json"))
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
