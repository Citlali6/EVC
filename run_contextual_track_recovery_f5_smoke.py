"""Train-only nested-LOFO smoke for contextual P18 track recovery on F5.

F5 scores/locations are observable and may be transformed before selection.
F5 labels and target IDs are deliberately not opened until an inner-OOF
winner (or a frozen stop decision) has been serialized and hashed in memory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import crossfit_component_reranker as crossfit
import replay_temporal_memory_validation as replay
from train_component_reranker import _load_cache_record
from utils.contextual_track_recovery import (
    FEATURE_NAMES,
    FEATURE_SEMANTICS_VERSION,
    extract_contextual_track_edge_candidates,
)
from utils.postprocess import ChallengePostprocessor
from utils.track_edge_recovery import (
    FROZEN_TOPOLOGY,
    attach_training_targets,
)


EXPECTED_PROTOCOL_SHA256 = "1f5d36d78215d664f0eed00804862a10667a98987258b34432f02a09ddc1b0ad"
SEED = 20260811
THRESHOLD = 0.719
RECOVERY_SCORE = np.nextafter(np.float32(THRESHOLD), np.float32(np.inf))
MODEL_FAMILIES = ("logistic_l2", "histgb_shallow", "extratrees_shallow")
TARGETS = ("event_positive", "pd_group_recovery")
AGGREGATIONS = ("mean", "median", "minimum")
REPRESENTATIONS = ("raw_probability", "within_video_percentile")
OUTER_HELD_GROUP = "block_088_098"
SOURCE_GROUPS = {
    "block_000_014": tuple(f"train_{index:03d}.npz" for index in range(0, 15)),
    "block_028_032": tuple(f"train_{index:03d}.npz" for index in range(28, 33)),
    "block_040_047": tuple(f"train_{index:03d}.npz" for index in range(40, 48)),
    "block_059_074": tuple(
        f"train_{index:03d}.npz"
        for index in tuple(range(59, 66)) + tuple(range(67, 75))
    ),
    OUTER_HELD_GROUP: tuple(f"train_{index:03d}.npz" for index in range(88, 99)),
}


@dataclass
class ObservableVideo:
    source_name: str
    group: str
    metadata: dict
    event_count: int
    raw_scores: np.ndarray
    baseline_scores: np.ndarray
    locations: np.ndarray
    candidates: tuple
    features: np.ndarray
    feature_sha256: str


@dataclass
class SupervisedVideo:
    observable: ObservableVideo
    labels: np.ndarray
    target_ids: np.ndarray
    targets: tuple
    baseline_counts: crossfit.SufficientCounts
    event_positive: np.ndarray
    pd_group_recovery: np.ndarray
    pd_group_keys: tuple
    false_component_deltas: np.ndarray


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def group_for_source(source_name):
    matches = [group for group, members in SOURCE_GROUPS.items() if source_name in members]
    if len(matches) != 1:
        raise ValueError(f"source must belong to one frozen family: {source_name}")
    return matches[0]


def record_path(cache_dir, metadata):
    relative = Path(str(metadata["record"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("cache record escaped cache directory.")
    path = (cache_dir / relative).resolve()
    path.relative_to(cache_dir)
    if sha256_file(path) != metadata["record_sha256"]:
        raise ValueError(f"cache record SHA mismatch: {path}")
    return path


def load_observable_fields(cache_dir, metadata):
    path = record_path(cache_dir, metadata)
    with np.load(path, allow_pickle=False) as record:
        if set(record.files) != {"scores", "locs", "labels", "target_ids"}:
            raise ValueError(f"unexpected cache fields: {path}")
        scores = np.ascontiguousarray(record["scores"])
        locs = np.ascontiguousarray(record["locs"])
    return scores, locs


def load_supervision_fields(cache_dir, metadata):
    # The record SHA was already checked before observable preparation.  This
    # second open occurs only for fit families, or after the winner freeze for F5.
    path = (cache_dir / Path(str(metadata["record"]))).resolve()
    with np.load(path, allow_pickle=False) as record:
        labels = np.ascontiguousarray(record["labels"])
        target_ids = np.ascontiguousarray(record["target_ids"])
    return labels, target_ids


def prepare_observables(cache_dir, c00_protocol):
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_split") != "train" or len(manifest.get("records", ())) != 54:
        raise ValueError("smoke requires the immutable 54-source train cache.")
    expected = set().union(*(set(value) for value in SOURCE_GROUPS.values()))
    actual = {item["source_name"] for item in manifest["records"]}
    if expected != actual:
        raise ValueError("cache sources differ from frozen contiguous families.")
    c00 = json.loads(c00_protocol.read_text(encoding="utf-8"))
    overrides = c00["definition"]["config"]["overrides"]
    cfg = replay.load_flat_config(
        Path(__file__).resolve().parent / "configs" / "evisseg_evuav.yaml",
        overrides,
    )
    crossfit.validate_c00_config(cfg)
    videos = []
    for ordinal, metadata in enumerate(manifest["records"], start=1):
        raw, locs = load_observable_fields(cache_dir, metadata)
        event_count = int(metadata["event_count"])
        raw = raw.reshape(-1).astype(np.float32, copy=False)
        locations = np.column_stack(
            (np.zeros(event_count, dtype=np.int64), locs.astype(np.int64, copy=False))
        )
        baseline_tensor, _ = ChallengePostprocessor.from_cfg(
            cfg, THRESHOLD, event_count=event_count
        ).apply(
            torch.from_numpy(raw.copy()),
            torch.from_numpy(locations).to(torch.int64).contiguous(),
        )
        baseline = baseline_tensor.numpy().astype(np.float32, copy=True)
        candidates = extract_contextual_track_edge_candidates(
            raw, baseline, locations, event_count, FROZEN_TOPOLOGY
        )
        features = (
            np.stack([candidate.features for candidate in candidates]).astype(
                np.float64, copy=False
            )
            if candidates
            else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
        )
        videos.append(
            ObservableVideo(
                source_name=metadata["source_name"],
                group=group_for_source(metadata["source_name"]),
                metadata=dict(metadata),
                event_count=event_count,
                raw_scores=raw.copy(),
                baseline_scores=baseline,
                locations=locations,
                candidates=candidates,
                features=features,
                feature_sha256=sha256_array(features),
            )
        )
        print(
            f"observable {ordinal:02d}/54 {metadata['source_name']} "
            f"[{videos[-1].group}] candidates={len(candidates)}",
            flush=True,
        )
    return manifest, videos


def official_frame(timestamp):
    timestamp = int(timestamp)
    if timestamp <= 0 or timestamp % FROZEN_TOPOLOGY.temporal_bin_size == 0:
        return None
    return timestamp // FROZEN_TOPOLOGY.temporal_bin_size


def attach_supervision(cache_dir, observable):
    before = observable.feature_sha256
    labels, target_ids = load_supervision_fields(cache_dir, observable.metadata)
    labels = labels.reshape(-1).astype(np.uint8, copy=True)
    target_ids = target_ids.reshape(-1).copy()
    targets = attach_training_targets(
        observable.candidates,
        labels,
        target_ids,
        observable.baseline_scores,
        observable.locations,
        FROZEN_TOPOLOGY,
        crossfit.CORRECT_THRESHOLD,
    )
    if before != sha256_array(observable.features):
        raise RuntimeError("feature bytes changed after supervision was attached.")
    event_positive = np.asarray([target.label for target in targets], dtype=np.uint8)
    pd_recovery = np.asarray(
        [target.recovers_target_group for target in targets], dtype=np.uint8
    )
    group_keys = []
    for candidate, target in zip(observable.candidates, targets):
        if not target.recovers_target_group:
            group_keys.append(None)
            continue
        index = int(candidate.event_index)
        frame = official_frame(observable.locations[index, 3])
        target_id = int(target_ids[index])
        if frame is None or target_id <= 0:
            raise RuntimeError("Pd recovery target lacks a valid group key.")
        group_keys.append((frame, target_id))
    baseline_counts = crossfit.sufficient_counts_for_video(
        observable.baseline_scores,
        labels,
        target_ids,
        observable.locations,
    )
    return SupervisedVideo(
        observable=observable,
        labels=labels,
        target_ids=target_ids,
        targets=targets,
        baseline_counts=baseline_counts,
        event_positive=event_positive,
        pd_group_recovery=pd_recovery,
        pd_group_keys=tuple(group_keys),
        false_component_deltas=np.asarray(
            [target.false_component_delta for target in targets], dtype=np.int64
        ),
    )


def sum_counts(values):
    total = crossfit.SufficientCounts()
    for value in values:
        total = total + value
    return total


def metric_record(counts):
    return {"counts": counts.to_dict(), "metrics": crossfit.metrics_from_counts(counts)}


def training_arrays(videos, target_name):
    bearing = [video for video in videos if len(video.observable.candidates)]
    features = np.concatenate([video.observable.features for video in bearing], axis=0)
    labels = np.concatenate([getattr(video, target_name) for video in bearing]).astype(np.uint8)
    if np.unique(labels).size != 2:
        raise ValueError("fit target lacks both classes.")
    weights = []
    for video in bearing:
        count = len(video.observable.candidates)
        weights.append(np.full(count, 1.0 / (len(bearing) * count), dtype=np.float64))
    weights = np.concatenate(weights)
    positive_mass = float(weights[labels > 0].sum())
    negative_mass = float(weights[labels == 0].sum())
    weights[labels > 0] *= negative_mass / positive_mass
    weights *= weights.size / weights.sum()
    return features, labels, weights


def new_model(family):
    if family == "logistic_l2":
        return Pipeline(
            (
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=500,
                        random_state=SEED,
                    ),
                ),
            )
        )
    if family == "histgb_shallow":
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=15,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=3.0,
            random_state=SEED,
        )
    if family == "extratrees_shallow":
        return ExtraTreesClassifier(
            n_estimators=128,
            max_depth=8,
            min_samples_leaf=8,
            max_features="sqrt",
            bootstrap=False,
            n_jobs=1,
            random_state=SEED,
        )
    raise KeyError(family)


def fit_model(family, target_name, videos):
    features, labels, weights = training_arrays(videos, target_name)
    model = new_model(family)
    if family == "logistic_l2":
        model.fit(features, labels, model__sample_weight=weights)
    else:
        model.fit(features, labels, sample_weight=weights)
    return model


def probabilities(model, observable):
    if observable.features.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    return model.predict_proba(observable.features)[:, 1].astype(np.float64, copy=False)


def aggregate(values, method):
    stack = np.stack(values)
    if method == "mean":
        return stack.mean(axis=0)
    if method == "median":
        return np.median(stack, axis=0)
    if method == "minimum":
        return stack.min(axis=0)
    raise KeyError(method)


def represent(values, method):
    values = np.asarray(values, dtype=np.float64)
    if method == "raw_probability" or values.size == 0:
        return values
    if method == "within_video_percentile":
        unique = np.unique(values)
        return np.searchsorted(unique, values, side="right").astype(np.float64) / unique.size
    raise KeyError(method)


def select_positions(observable, values, threshold):
    by_endpoint = {}
    for position, (candidate, value) in enumerate(zip(observable.candidates, values)):
        if float(value) < float(threshold):
            continue
        ordering = (float(value), float(candidate.raw_score), -int(candidate.event_index))
        previous = by_endpoint.get(candidate.endpoint_key)
        if previous is None or ordering > previous[0]:
            by_endpoint[candidate.endpoint_key] = (ordering, position)
    return np.asarray(sorted(item[1] for item in by_endpoint.values()), dtype=np.int64)


def conservative_counts(video, positions):
    values = video.baseline_counts.to_dict()
    recovered_groups = set()
    for position in np.asarray(positions, dtype=np.int64).tolist():
        if video.event_positive[position]:
            values["true_positive_events"] += 1
            values["false_negative_events"] -= 1
            key = video.pd_group_keys[position]
            if key is not None:
                recovered_groups.add(key)
        else:
            values["false_positive_events"] += 1
            values["false_components"] += max(
                0, int(video.false_component_deltas[position])
            )
    values["correct_objects"] += len(recovered_groups)
    return crossfit.SufficientCounts(**values), {
        "selected": int(len(positions)),
        "selected_true": int(video.event_positive[positions].sum()) if len(positions) else 0,
        "selected_false": int(len(positions) - video.event_positive[positions].sum()) if len(positions) else 0,
        "new_correct_objects": int(len(recovered_groups)),
        "false_component_delta_upper_bound": int(
            sum(
                max(0, int(video.false_component_deltas[position]))
                for position in positions
                if not video.event_positive[position]
            )
        ),
    }


def exact_counts(video, positions):
    recovered = video.observable.baseline_scores.copy()
    event_indices = [
        video.observable.candidates[int(position)].event_index for position in positions
    ]
    recovered[np.asarray(event_indices, dtype=np.int64)] = RECOVERY_SCORE
    return crossfit.sufficient_counts_for_video(
        recovered, video.labels, video.target_ids, video.observable.locations
    )


def gates(candidate, baseline, pooled=False):
    cm = crossfit.metrics_from_counts(candidate)
    bm = crossfit.metrics_from_counts(baseline)
    result = {
        "score_not_lower": cm["score"] >= bm["score"],
        "iou_not_lower": cm["iou"] >= bm["iou"],
        "pd_not_lower": cm["pd"] >= bm["pd"],
        "correct_objects_not_lower": candidate.correct_objects >= baseline.correct_objects,
    }
    if pooled:
        result.update(
            {
                "score_strictly_higher": cm["score"] > bm["score"],
                "pd_strictly_higher": cm["pd"] > bm["pd"],
                "new_correct_object": candidate.correct_objects > baseline.correct_objects,
            }
        )
    return result


def evaluate_threshold(videos, scores, threshold):
    folds = []
    pooled_baseline = crossfit.SufficientCounts()
    pooled_candidate = crossfit.SufficientCounts()
    total_actions = {"selected": 0, "selected_true": 0, "selected_false": 0, "new_correct_objects": 0, "false_component_delta_upper_bound": 0}
    for group in sorted({video.observable.group for video in videos}):
        held = [video for video in videos if video.observable.group == group]
        baseline = sum_counts(video.baseline_counts for video in held)
        candidates = []
        group_actions = {name: 0 for name in total_actions}
        worst_source = None
        for video in held:
            positions = select_positions(
                video.observable, scores[video.observable.source_name], threshold
            )
            candidate, actions = conservative_counts(video, positions)
            candidates.append(candidate)
            for name in group_actions:
                group_actions[name] += actions[name]
            delta = crossfit.metrics_from_counts(candidate)["score"] - crossfit.metrics_from_counts(video.baseline_counts)["score"]
            if worst_source is None or delta < worst_source["score_delta"]:
                worst_source = {"source_name": video.observable.source_name, "score_delta": delta}
        candidate = sum_counts(candidates)
        fold_gate = gates(candidate, baseline)
        folds.append(
            {
                "held_group": group,
                "baseline": metric_record(baseline),
                "candidate_conservative": metric_record(candidate),
                "score_delta": crossfit.metrics_from_counts(candidate)["score"] - crossfit.metrics_from_counts(baseline)["score"],
                "gates": fold_gate,
                "actions": group_actions,
                "worst_source": worst_source,
            }
        )
        pooled_baseline = pooled_baseline + baseline
        pooled_candidate = pooled_candidate + candidate
        for name in total_actions:
            total_actions[name] += group_actions[name]
    pooled_gates = gates(pooled_candidate, pooled_baseline, pooled=True)
    return {
        "threshold": float(threshold),
        "folds": folds,
        "all_fold_gates_passed": all(all(item["gates"].values()) for item in folds),
        "pooled_baseline": metric_record(pooled_baseline),
        "pooled_candidate_conservative": metric_record(pooled_candidate),
        "pooled_score_delta": crossfit.metrics_from_counts(pooled_candidate)["score"] - crossfit.metrics_from_counts(pooled_baseline)["score"],
        "pooled_gates": pooled_gates,
        "actions": total_actions,
    }


def choose_threshold(videos, scores):
    values = np.concatenate([scores[video.observable.source_name] for video in videos])
    candidates = []
    for threshold in np.unique(values)[::-1]:
        result = evaluate_threshold(videos, scores, float(threshold))
        if result["all_fold_gates_passed"] and all(result["pooled_gates"].values()):
            candidates.append(result)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            -item["pooled_candidate_conservative"]["metrics"]["score"],
            -min(fold["score_delta"] for fold in item["folds"]),
            item["actions"]["selected"],
            -item["threshold"],
        ),
    )[0]


def exact_evaluation(videos, scores, threshold):
    per_group = []
    pooled_baseline = crossfit.SufficientCounts()
    pooled_candidate = crossfit.SufficientCounts()
    all_positions = {}
    for group in sorted({video.observable.group for video in videos}):
        held = [video for video in videos if video.observable.group == group]
        baseline = sum_counts(video.baseline_counts for video in held)
        candidates = []
        for video in held:
            positions = select_positions(
                video.observable, scores[video.observable.source_name], threshold
            )
            all_positions[video.observable.source_name] = positions
            candidates.append(exact_counts(video, positions))
        candidate = sum_counts(candidates)
        per_group.append(
            {
                "held_group": group,
                "baseline": metric_record(baseline),
                "candidate": metric_record(candidate),
                "score_delta": crossfit.metrics_from_counts(candidate)["score"] - crossfit.metrics_from_counts(baseline)["score"],
                "gates": gates(candidate, baseline),
            }
        )
        pooled_baseline = pooled_baseline + baseline
        pooled_candidate = pooled_candidate + candidate
    return {
        "groups": per_group,
        "baseline": metric_record(pooled_baseline),
        "candidate": metric_record(pooled_candidate),
        "score_delta": crossfit.metrics_from_counts(pooled_candidate)["score"] - crossfit.metrics_from_counts(pooled_baseline)["score"],
        "gates": gates(pooled_candidate, pooled_baseline, pooled=True),
    }, all_positions


def maximum_pd_oracle_positions(video):
    edges = {}
    for position, (candidate, key) in enumerate(
        zip(video.observable.candidates, video.pd_group_keys)
    ):
        if key is not None:
            edges.setdefault(candidate.endpoint_key, []).append((key, position))
    match_group = {}
    match_endpoint = {}

    def augment(endpoint, visited):
        for key, position in sorted(edges[endpoint], key=lambda item: (item[0], item[1])):
            if key in visited:
                continue
            visited.add(key)
            previous = match_group.get(key)
            if previous is None or augment(previous[0], visited):
                match_group[key] = (endpoint, position)
                match_endpoint[endpoint] = (key, position)
                return True
        return False

    for endpoint in sorted(edges):
        augment(endpoint, set())
    return np.asarray(
        sorted(value[1] for value in match_endpoint.values()), dtype=np.int64
    )


def oracle_evaluation(videos):
    baseline = sum_counts(video.baseline_counts for video in videos)
    candidates = []
    per_source = []
    for video in videos:
        positions = maximum_pd_oracle_positions(video)
        candidate = exact_counts(video, positions)
        candidates.append(candidate)
        per_source.append(
            {
                "source_name": video.observable.source_name,
                "candidate_count": len(video.observable.candidates),
                "positive_candidate_count": int(video.event_positive.sum()),
                "pd_recovery_candidate_count": int(video.pd_group_recovery.sum()),
                "oracle_selected": int(len(positions)),
                "new_correct_objects": candidate.correct_objects - video.baseline_counts.correct_objects,
            }
        )
    candidate = sum_counts(candidates)
    return {
        "semantics": "maximum bipartite matching of endpoint to missed target-frame group; no false event selected",
        "baseline": metric_record(baseline),
        "candidate": metric_record(candidate),
        "score_delta": crossfit.metrics_from_counts(candidate)["score"] - crossfit.metrics_from_counts(baseline)["score"],
        "new_correct_objects": candidate.correct_objects - baseline.correct_objects,
        "new_true_positive_events": candidate.true_positive_events - baseline.true_positive_events,
        "per_source": per_source,
    }


def report_auc(videos, scores, target_name):
    labels = np.concatenate([getattr(video, target_name) for video in videos])
    values = np.concatenate([scores[video.observable.source_name] for video in videos])
    if np.unique(labels).size != 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(labels, values)),
        "average_precision": float(average_precision_score(labels, values)),
    }


def run(args):
    root = Path(__file__).resolve().parent
    cache_dir = Path(args.cache_dir).resolve()
    protocol_path = Path(args.science_protocol).resolve()
    c00_protocol = Path(args.c00_protocol).resolve()
    output_path = Path(args.output_report).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    if sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("science protocol changed after freeze.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if "train-only" not in protocol["scope"] or "no validation" not in protocol["scope"]:
        raise ValueError("protocol is not train-only.")

    manifest, observable_videos = prepare_observables(cache_dir, c00_protocol)
    fit_observables = [video for video in observable_videos if video.group != OUTER_HELD_GROUP]
    outer_observables = [video for video in observable_videos if video.group == OUTER_HELD_GROUP]
    supervised_fit = [attach_supervision(cache_dir, video) for video in fit_observables]
    print("fit supervision attached; F5 labels remain unopened", flush=True)

    fit_groups = tuple(group for group in SOURCE_GROUPS if group != OUTER_HELD_GROUP)
    pair_models = {}
    fit_failures = []
    for family in MODEL_FAMILIES:
        for target_name in TARGETS:
            models = {}
            try:
                for pair in itertools.combinations(fit_groups, 2):
                    pair_videos = [
                        video for video in supervised_fit if video.observable.group in pair
                    ]
                    models[pair] = fit_model(family, target_name, pair_videos)
            except ValueError as error:
                fit_failures.append(
                    {"family": family, "target": target_name, "reason": str(error)}
                )
                continue
            pair_models[(family, target_name)] = models

    configurations = []
    score_cache = {}
    for (family, target_name), models in pair_models.items():
        for aggregation in AGGREGATIONS:
            raw_inner = {}
            for group in fit_groups:
                eligible = [model for pair, model in models.items() if group not in pair]
                if len(eligible) != 3:
                    raise RuntimeError("each inner held family requires three unseen pair models.")
                for video in [item for item in fit_observables if item.group == group]:
                    raw_inner[video.source_name] = aggregate(
                        [probabilities(model, video) for model in eligible], aggregation
                    )
            for representation in REPRESENTATIONS:
                scores = {
                    name: represent(values, representation)
                    for name, values in raw_inner.items()
                }
                selection = choose_threshold(supervised_fit, scores)
                key = (family, target_name, aggregation, representation)
                score_cache[key] = scores
                configurations.append(
                    {
                        "family": family,
                        "target": target_name,
                        "aggregation": aggregation,
                        "representation": representation,
                        "inner_discrimination": report_auc(
                            supervised_fit, scores, target_name
                        ),
                        "eligible": selection is not None,
                        "selection": selection,
                    }
                )

    eligible = [item for item in configurations if item["eligible"]]
    if eligible:
        winner = sorted(
            eligible,
            key=lambda item: (
                -item["selection"]["pooled_candidate_conservative"]["metrics"]["score"],
                item["family"],
                item["target"],
                item["aggregation"],
                item["representation"],
            ),
        )[0]
        winner_key = (
            winner["family"],
            winner["target"],
            winner["aggregation"],
            winner["representation"],
        )
        winner_scores = score_cache[winner_key]
        inner_exact, _ = exact_evaluation(
            supervised_fit, winner_scores, winner["selection"]["threshold"]
        )
        frozen_decision = {
            "action": "evaluate_selected_model_on_F5",
            "family": winner["family"],
            "target": winner["target"],
            "aggregation": winner["aggregation"],
            "representation": winner["representation"],
            "threshold": winner["selection"]["threshold"],
            "inner_conservative_score_delta": winner["selection"]["pooled_score_delta"],
            "inner_exact_score_delta": inner_exact["score_delta"],
        }
        selected_models = pair_models[(winner["family"], winner["target"])]
        outer_scores = {}
        for video in outer_observables:
            raw_values = aggregate(
                [probabilities(model, video) for model in selected_models.values()],
                winner["aggregation"],
            )
            outer_scores[video.source_name] = represent(
                raw_values, winner["representation"]
            )
    else:
        winner = None
        inner_exact = None
        frozen_decision = {
            "action": "stop_no_inner_safe_candidate",
            "identity_only": True,
        }
        outer_scores = {
            video.source_name: np.zeros(len(video.candidates), dtype=np.float64)
            for video in outer_observables
        }

    freeze_receipt = {
        "decision": frozen_decision,
        "decision_sha256": sha256_json(frozen_decision),
        "F5_labels_opened": False,
        "F5_target_ids_opened": False,
    }
    print(
        "winner frozen before F5 labels: " + freeze_receipt["decision_sha256"],
        flush=True,
    )

    supervised_outer = [attach_supervision(cache_dir, video) for video in outer_observables]
    outer_oracle = oracle_evaluation(supervised_outer)
    if winner is not None:
        outer_exact, outer_positions = exact_evaluation(
            supervised_outer, outer_scores, winner["selection"]["threshold"]
        )
        outer_baseline = crossfit.SufficientCounts(**outer_exact["baseline"]["counts"])
        outer_candidate = crossfit.SufficientCounts(**outer_exact["candidate"]["counts"])
        new_objects = outer_candidate.correct_objects - outer_baseline.correct_objects
        false_delta = outer_candidate.false_components - outer_baseline.false_components
        inner_base = crossfit.SufficientCounts(**inner_exact["baseline"]["counts"])
        inner_candidate = crossfit.SufficientCounts(**inner_exact["candidate"]["counts"])
        inner_objects = inner_candidate.correct_objects - inner_base.correct_objects
        inner_false_delta = inner_candidate.false_components - inner_base.false_components
        inner_tradeoff = (
            max(0, inner_false_delta) / inner_objects if inner_objects > 0 else None
        )
        outer_tradeoff = max(0, false_delta) / new_objects if new_objects > 0 else None
        success_gates = {
            "outer_score_strictly_higher": outer_exact["score_delta"] > 0.0,
            "outer_iou_not_lower": outer_exact["candidate"]["metrics"]["iou"] >= outer_exact["baseline"]["metrics"]["iou"],
            "outer_pd_strictly_higher": outer_exact["candidate"]["metrics"]["pd"] > outer_exact["baseline"]["metrics"]["pd"],
            "outer_correct_objects_increase": new_objects >= 1,
            "outer_false_components_per_object_not_above_inner": (
                outer_tradeoff is not None
                and inner_tradeoff is not None
                and outer_tradeoff <= inner_tradeoff
            ),
        }
        outer_model = {
            "evaluation": outer_exact,
            "discrimination": report_auc(
                supervised_outer, outer_scores, winner["target"]
            ),
            "selected_action_count": int(
                sum(len(value) for value in outer_positions.values())
            ),
            "inner_false_components_per_new_object": inner_tradeoff,
            "outer_false_components_per_new_object": outer_tradeoff,
            "success_gates": success_gates,
            "success": all(success_gates.values()),
        }
    else:
        outer_model = {
            "evaluation": None,
            "success_gates": {"inner_safe_candidate_exists": False},
            "success": False,
        }

    report = {
        "schema": "ev-uav-contextual-track-recovery-f5-smoke-report-v2",
        "dataset_split": "train",
        "no_validation_or_test_access": True,
        "device": "cpu",
        "source_or_file_identity_is_feature": False,
        "absolute_coordinate_or_time_is_feature": False,
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "topology": asdict(FROZEN_TOPOLOGY),
        "source_groups": {name: list(values) for name, values in SOURCE_GROUPS.items()},
        "inputs": {
            "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            "cache_source_count": manifest["selected_video_count"],
            "cache_event_count": manifest["selected_event_count"],
            "science_protocol_sha256": sha256_file(protocol_path),
            "c00_protocol_sha256": sha256_file(c00_protocol),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "module_sha256": sha256_file(root / "utils" / "contextual_track_recovery.py"),
        },
        "feature_firewall": {
            "extractor_signature_has_no_labels_or_target_ids": True,
            "all_feature_hashes_unchanged_after_supervision_attachment": True,
        },
        "fit_failures": fit_failures,
        "inner_configurations": configurations,
        "selected_by_inner": None if winner is None else {
            name: winner[name]
            for name in ("family", "target", "aggregation", "representation")
        } | {"threshold": winner["selection"]["threshold"]},
        "inner_selected_exact": inner_exact,
        "winner_freeze_receipt": freeze_receipt,
        "outer_F5_proposal_oracle": outer_oracle,
        "outer_F5_model": outer_model,
        "decision": (
            "proceed_to_full_five_family_nested_oof"
            if outer_model["success"]
            else "stop_recovery_v2_after_F5_smoke"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(output_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output_report": str(output_path),
                "report_sha256": sha256_file(output_path),
                "selected_by_inner": report["selected_by_inner"],
                "F5_oracle_score_delta": outer_oracle["score_delta"],
                "F5_oracle_new_correct_objects": outer_oracle["new_correct_objects"],
                "F5_model": outer_model,
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


def parser():
    root = Path(__file__).resolve().parent
    experiments = root.parent / "experiments"
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--science-protocol",
        default=str(root / "protocols" / "contextual_track_recovery_f5_smoke_v2.json"),
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
        default=str(experiments / "20260811_contextual_track_recovery_f5_smoke_v2" / "report.json"),
    )
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))

