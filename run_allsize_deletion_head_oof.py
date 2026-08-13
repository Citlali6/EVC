"""Train-only nested source-group OOF prototype for an all-size deletion head."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import crossfit_component_reranker as crossfit
import replay_temporal_memory_validation as replay
from utils.allsize_deletion_head import (
    FEATURE_NAMES,
    extract_allsize_components,
    suppress_components,
)
from utils.component_reranker import ComponentTopology
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor


SEED = 20260811
THRESHOLD = 0.719
EFFECTIVE_THRESHOLD = np.float32(THRESHOLD)
RISK_RULES = (
    "identity",
    "zero_positive_event_loss",
)
FULL_TARGET_GAP = 0.9700 - 0.9628776541559201
TOPOLOGY = ComponentTopology(
    spatial_radius=1,
    temporal_bin_size=50,
    max_link_distance=6.0,
    max_gap_bins=1,
    max_component_events=2**31 - 1,
)
SOURCE_GROUPS = {
    "block_000_014": tuple(f"train_{index:03d}.npz" for index in range(0, 15)),
    "block_028_032": tuple(f"train_{index:03d}.npz" for index in range(28, 33)),
    "block_040_047": tuple(f"train_{index:03d}.npz" for index in range(40, 48)),
    "block_059_074": tuple(
        f"train_{index:03d}.npz"
        for index in tuple(range(59, 66)) + tuple(range(67, 75))
    ),
    "block_088_098": tuple(f"train_{index:03d}.npz" for index in range(88, 99)),
}


@dataclass
class PreparedVideo:
    source_name: str
    group: str
    event_count: int
    scores: np.ndarray
    locations: np.ndarray
    labels: np.ndarray
    target_ids: np.ndarray
    event_indices: tuple[np.ndarray, ...]
    features: np.ndarray
    component_labels: np.ndarray
    baseline_counts: crossfit.SufficientCounts


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_for_source(source_name):
    groups = [name for name, members in SOURCE_GROUPS.items() if source_name in members]
    if len(groups) != 1:
        raise ValueError(f"source must belong to exactly one frozen group: {source_name}")
    return groups[0]


def _sum_counts(values):
    total = crossfit.SufficientCounts()
    for value in values:
        total = total + value
    return total


def official_counts(scores, labels, target_ids, locations):
    """Mirror official float32 comparison, including exact-threshold recoveries."""

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    locations = np.asarray(locations, dtype=np.int64)
    evaluator = evalute(SimpleNamespace(roc=True, pd_detT=50, correct_thresh=0.0001))
    score_tensor = torch.from_numpy(scores.copy())
    label_tensor = torch.from_numpy(labels.astype(np.float32, copy=False))
    location_tensor = torch.from_numpy(locations)
    evaluator.roc_update(
        location_tensor[:, 3],
        score_tensor,
        np.asarray(target_ids).reshape(-1),
        label_tensor,
        location_tensor,
        thresh=THRESHOLD,
    )
    predicted = scores >= EFFECTIVE_THRESHOLD
    positive = labels > 0
    return crossfit.SufficientCounts(
        true_positive_events=int(np.sum(predicted & positive)),
        false_positive_events=int(np.sum(predicted & ~positive)),
        false_negative_events=int(np.sum(~predicted & positive)),
        correct_objects=int(evaluator.correct_num),
        object_count=int(evaluator.obj_num),
        false_components=int(evaluator.false_num),
        frame_count=int(evaluator.frame_num),
        event_count=int(scores.size),
    )


def prepare_videos(cache_dir, protocol_path):
    with (cache_dir / "manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != "ev-uav-component-reranker-train-cache-v1":
        raise ValueError("unsupported train-cache schema.")
    if manifest.get("dataset_split") != "train" or len(manifest.get("records", ())) != 54:
        raise ValueError("prototype requires the exact 54-source train cache.")
    with protocol_path.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    overrides = protocol["definition"]["config"]["overrides"]
    cfg = replay.load_flat_config(Path(__file__).resolve().parent / "configs" / "evisseg_evuav.yaml", overrides)
    crossfit.validate_c00_config(cfg)

    expected_names = set().union(*map(set, SOURCE_GROUPS.values()))
    actual_names = {item["source_name"] for item in manifest["records"]}
    if actual_names != expected_names:
        raise ValueError("train-cache sources differ from frozen five source groups.")

    videos = []
    for index, metadata in enumerate(manifest["records"], start=1):
        record = crossfit._load_cache_record(cache_dir, metadata)
        event_count = int(metadata["event_count"])
        raw_scores = torch.from_numpy(
            record["scores"].reshape(-1).astype(np.float32, copy=False)
        )
        locations = np.column_stack(
            (
                np.zeros(event_count, dtype=np.int64),
                record["locs"].astype(np.int64, copy=False),
            )
        )
        location_tensor = torch.from_numpy(locations)
        scores, _ = ChallengePostprocessor.from_cfg(
            cfg, THRESHOLD, event_count=event_count
        ).apply(raw_scores, location_tensor)
        score_values = scores.numpy().astype(np.float32, copy=True)
        components = extract_allsize_components(
            score_values,
            locations,
            THRESHOLD,
            TOPOLOGY,
            event_count,
            labels=record["labels"],
            context_scores=record["scores"],
        )
        if components.features.shape[0] == 0:
            raise RuntimeError(f"source has no retained components: {metadata['source_name']}")
        labels = record["labels"].reshape(-1).astype(np.uint8, copy=True)
        videos.append(
            PreparedVideo(
                source_name=metadata["source_name"],
                group=_group_for_source(metadata["source_name"]),
                event_count=event_count,
                scores=score_values,
                locations=locations,
                labels=labels,
                target_ids=record["target_ids"].reshape(-1).copy(),
                event_indices=components.event_indices,
                features=components.features,
                component_labels=components.labels,
                baseline_counts=official_counts(
                    score_values, labels, record["target_ids"], locations
                ),
            )
        )
        print(
            f"prepare {index:02d}/54 {metadata['source_name']} -> "
            f"{components.features.shape[0]} components",
            flush=True,
        )
    return manifest, cfg, videos


def _training_arrays(videos):
    features = np.concatenate([video.features for video in videos], axis=0)
    labels = np.concatenate([video.component_labels for video in videos], axis=0)
    weights = []
    for video in videos:
        per_source = np.full(video.component_labels.size, 1.0 / video.component_labels.size)
        weights.append(per_source)
    sample_weight = np.concatenate(weights).astype(np.float64, copy=False)
    positive_mass = float(sample_weight[labels > 0].sum())
    negative_mass = float(sample_weight[labels == 0].sum())
    if positive_mass <= 0.0 or negative_mass <= 0.0:
        raise RuntimeError("fit components must contain both classes.")
    sample_weight[labels > 0] *= negative_mass / positive_mass
    sample_weight *= sample_weight.size / sample_weight.sum()
    return features, labels, sample_weight


def _new_model(family):
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


def _fit_model(family, videos):
    features, labels, sample_weight = _training_arrays(videos)
    model = _new_model(family)
    if family == "logistic_l2":
        model.fit(features, labels, model__sample_weight=sample_weight)
    else:
        model.fit(features, labels, sample_weight=sample_weight)
    return model


def _probabilities(model, video):
    return model.predict_proba(video.features)[:, 1].astype(np.float64, copy=False)


def _fit_derived_thresholds(model, videos):
    """Derive identity/safe thresholds from fit-positive probabilities."""

    positive_probabilities = []
    for video in videos:
        probabilities = _probabilities(model, video)
        mask = video.component_labels > 0
        positive_probabilities.extend(probabilities[mask].tolist())
    probabilities = np.asarray(positive_probabilities, dtype=np.float64)
    if probabilities.size == 0:
        raise RuntimeError("fit data contain no valid positive components.")
    return {
        "identity": 0.0,
        "zero_positive_event_loss": float(probabilities.min()),
    }


def _candidate_counts(video, probabilities, keep_threshold):
    scores = suppress_components(
        video.scores, video.event_indices, probabilities, keep_threshold
    )
    return official_counts(scores, video.labels, video.target_ids, video.locations)


def _metrics_record(counts):
    return {"counts": counts.to_dict(), "metrics": crossfit.metrics_from_counts(counts)}


def _safety_gates(candidate, baseline):
    return {
        "correct_objects_not_lower": candidate.correct_objects >= baseline.correct_objects,
        "pd_not_lower": crossfit.metrics_from_counts(candidate)["pd"]
        >= crossfit.metrics_from_counts(baseline)["pd"],
        "tp_not_lower": candidate.true_positive_events >= baseline.true_positive_events,
    }


def _nested_select(outer_group, videos, families):
    fit_groups = tuple(group for group in SOURCE_GROUPS if group != outer_group)
    probability_cache = {family: {} for family in families}
    threshold_cache = {family: {} for family in families}
    for inner_group in fit_groups:
        inner_fit = [v for v in videos if v.group in fit_groups and v.group != inner_group]
        inner_held = [v for v in videos if v.group == inner_group]
        for family in families:
            model = _fit_model(family, inner_fit)
            threshold_cache[family][inner_group] = _fit_derived_thresholds(
                model, inner_fit
            )
            for video in inner_held:
                probability_cache[family][video.source_name] = _probabilities(model, video)

    candidates = []
    for family in families:
        for risk_rule in RISK_RULES:
            fold_records = []
            pooled_baseline = crossfit.SufficientCounts()
            pooled_candidate = crossfit.SufficientCounts()
            for inner_group in fit_groups:
                held = [v for v in videos if v.group == inner_group]
                keep_threshold = threshold_cache[family][inner_group][risk_rule]
                baseline = _sum_counts(v.baseline_counts for v in held)
                candidate = _sum_counts(
                    _candidate_counts(
                        v, probability_cache[family][v.source_name], keep_threshold
                    )
                    for v in held
                )
                gates = _safety_gates(candidate, baseline)
                fold_records.append(
                    {
                        "inner_group": inner_group,
                        "fit_derived_keep_threshold": keep_threshold,
                        "baseline": _metrics_record(baseline),
                        "candidate": _metrics_record(candidate),
                        "score_delta": crossfit.metrics_from_counts(candidate)["score"]
                        - crossfit.metrics_from_counts(baseline)["score"],
                        "gates": gates,
                    }
                )
                pooled_baseline = pooled_baseline + baseline
                pooled_candidate = pooled_candidate + candidate
            eligible = all(
                all(fold["gates"].values())
                for fold in fold_records
            )
            candidates.append(
                {
                    "family": family,
                    "risk_rule": risk_rule,
                    "eligible": eligible,
                    "inner_folds": fold_records,
                    "pooled_baseline": _metrics_record(pooled_baseline),
                    "pooled_candidate": _metrics_record(pooled_candidate),
                    "pooled_score_delta": crossfit.metrics_from_counts(pooled_candidate)["score"]
                    - crossfit.metrics_from_counts(pooled_baseline)["score"],
                }
            )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError("nested selection produced no safety-eligible candidate.")
    winner = sorted(
        eligible,
        key=lambda item: (
            -item["pooled_candidate"]["metrics"]["score"],
            item["family"],
            item["risk_rule"],
        ),
    )[0]
    return winner, candidates


def run(args):
    project_root = Path(__file__).resolve().parent
    cache_dir = Path(args.cache_dir).resolve()
    protocol_path = Path(args.protocol).resolve()
    output_report = Path(args.output_report).resolve()
    if output_report.exists():
        raise FileExistsError(f"report already exists: {output_report}")
    manifest, _cfg, videos = prepare_videos(cache_dir, protocol_path)
    families = ("logistic_l2", "histgb_shallow", "extratrees_shallow")
    outer_results = []
    outer_predictions = {}
    for outer_group in SOURCE_GROUPS:
        winner, inner_candidates = _nested_select(outer_group, videos, families)
        fit_videos = [video for video in videos if video.group != outer_group]
        held_videos = [video for video in videos if video.group == outer_group]
        model = _fit_model(winner["family"], fit_videos)
        fit_derived_thresholds = _fit_derived_thresholds(model, fit_videos)
        keep_threshold = fit_derived_thresholds[winner["risk_rule"]]
        per_source = []
        for video in held_videos:
            probabilities = _probabilities(model, video)
            outer_predictions[video.source_name] = _candidate_counts(
                video, probabilities, keep_threshold
            )
            candidate = outer_predictions[video.source_name]
            baseline_metrics = crossfit.metrics_from_counts(video.baseline_counts)
            candidate_metrics = crossfit.metrics_from_counts(candidate)
            per_source.append(
                {
                    "source_name": video.source_name,
                    "baseline": _metrics_record(video.baseline_counts),
                    "candidate": _metrics_record(candidate),
                    "score_delta": candidate_metrics["score"] - baseline_metrics["score"],
                }
            )
        baseline = _sum_counts(video.baseline_counts for video in held_videos)
        candidate = _sum_counts(outer_predictions[video.source_name] for video in held_videos)
        outer_results.append(
            {
                "outer_group": outer_group,
                "winner": {
                    "family": winner["family"],
                    "risk_rule": winner["risk_rule"],
                    "fit_derived_keep_threshold": keep_threshold,
                    "all_fit_derived_thresholds": fit_derived_thresholds,
                    "inner_pooled_score_delta": winner["pooled_score_delta"],
                },
                "inner_candidates": inner_candidates,
                "baseline": _metrics_record(baseline),
                "candidate": _metrics_record(candidate),
                "score_delta": crossfit.metrics_from_counts(candidate)["score"]
                - crossfit.metrics_from_counts(baseline)["score"],
                "gates": _safety_gates(candidate, baseline),
                "worst_source": sorted(per_source, key=lambda item: item["score_delta"])[0],
                "sources": per_source,
            }
        )
        print(
            f"outer {outer_group}: {winner['family']}[{winner['risk_rule']}] "
            f"threshold={keep_threshold:.8g} "
            f"delta={outer_results[-1]['score_delta']:+.9f}",
            flush=True,
        )

    pooled_baseline = _sum_counts(video.baseline_counts for video in videos)
    pooled_candidate = _sum_counts(outer_predictions[video.source_name] for video in videos)
    high_names = set(crossfit.H1_NAMES) | set(crossfit.H2_NAMES)
    high_videos = [video for video in videos if video.source_name in high_names]
    high_baseline = _sum_counts(video.baseline_counts for video in high_videos)
    high_candidate = _sum_counts(outer_predictions[video.source_name] for video in high_videos)
    pooled_delta = crossfit.metrics_from_counts(pooled_candidate)["score"] - crossfit.metrics_from_counts(pooled_baseline)["score"]
    high_delta = crossfit.metrics_from_counts(high_candidate)["score"] - crossfit.metrics_from_counts(high_baseline)["score"]
    promotion_gates = {
        "every_outer_correct_objects_not_lower": all(
            result["gates"]["correct_objects_not_lower"] for result in outer_results
        ),
        "every_outer_pd_not_lower": all(result["gates"]["pd_not_lower"] for result in outer_results),
        "every_outer_tp_not_lower": all(
            result["gates"]["tp_not_lower"] for result in outer_results
        ),
        "high_domain_score_gain_at_least_full_target_gap": high_delta >= FULL_TARGET_GAP,
        "pooled_score_strictly_improves": pooled_delta > 0.0,
    }
    report = {
        "schema": "ev-uav-allsize-deletion-head-source-group-oof-v1",
        "dataset_split": "train",
        "evidence_class": "strict_nested_five_source_group_outer_oof",
        "no_validation_or_test_access": True,
        "source_name_or_index_is_feature": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "topology": asdict(TOPOLOGY),
        "model_families": list(families),
        "threshold_derivation": {
            "risk_rules": list(RISK_RULES),
            "source": "each fitted model's positive-component probability distribution",
            "safety_rule": "identity or minimum fit-positive keep probability; no fitted positive component is deleted",
            "numeric_probability_thresholds_are_not_hardcoded": True,
        },
        "sample_weighting": {
            "source_weighting": "each fit source has equal component mass",
            "class_weighting": "fit-derived balance of effective positive and negative component mass",
            "manual_class_multiplier": None,
        },
        "full_target_gap": FULL_TARGET_GAP,
        "source_groups": {name: list(members) for name, members in SOURCE_GROUPS.items()},
        "inputs": {
            "cache_manifest_path": str((cache_dir / "manifest.json").resolve()),
            "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            "cache_selected_video_count": manifest["selected_video_count"],
            "cache_selected_event_count": manifest["selected_event_count"],
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "module_sha256": sha256_file(project_root / "utils" / "allsize_deletion_head.py"),
        },
        "outer_folds": outer_results,
        "pooled": {
            "baseline": _metrics_record(pooled_baseline),
            "candidate": _metrics_record(pooled_candidate),
            "score_delta": pooled_delta,
        },
        "high_domain_pooled": {
            "sources": sorted(high_names),
            "baseline": _metrics_record(high_baseline),
            "candidate": _metrics_record(high_candidate),
            "score_delta": high_delta,
        },
        "promotion_gates": promotion_gates,
        "promotion_passed": all(promotion_gates.values()),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(output_report), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output_report": str(output_report), "promotion_passed": report["promotion_passed"], "pooled_score_delta": pooled_delta, "high_domain_score_delta": high_delta}, indent=2))
    return 0


def build_parser():
    root = Path(__file__).resolve().parent
    experiment = root.parent / "experiments" / "20260811_allsize_deletion_head_oof_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(root.parent / "experiments" / "20260810_component_reranker_crosssource_v1" / "train_cache_gt30000"))
    parser.add_argument("--protocol", default=str(root.parent / "experiments" / "20260810_component_reranker_crosssource_v1" / "crossfit_protocol.json"))
    parser.add_argument("--output-report", default=str(experiment / "deletion_head_oof_report.json"))
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
