"""CPU-only H2 temporal-track oracle and fixed linear separability gate.

This diagnostic consumes only the immutable released-M20 train cache and the
official train_088..098 input arrays.  It cannot accept validation/test paths,
does not initialize CUDA, and performs no model or threshold grid search.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import crossfit_component_reranker as crossfit
import replay_temporal_memory_validation as replay
import run_allsize_deletion_head_oof as base
from train_component_reranker import _load_cache_record
from utils.h2_temporal_track_graph import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    TRACK_FEATURE_NAMES,
    aggregate_track_node_features,
    atomic_delete_from_graph,
    derive_zero_observed_target_loss_cutoff,
    extract_temporal_track_graph,
    pure_false_positive_component_targets,
    pure_false_positive_track_targets,
)
from utils.postprocess import ChallengePostprocessor


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CACHE_DIR = (
    WORKSPACE
    / "experiments"
    / "20260810_component_reranker_crosssource_v1"
    / "train_cache_gt30000"
)
CACHE_PROTOCOL = (
    WORKSPACE
    / "experiments"
    / "20260810_component_reranker_crosssource_v1"
    / "crossfit_protocol.json"
)
TRAIN_ROOT = WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "train"
OUTPUT_PATH = (
    WORKSPACE
    / "experiments"
    / "20260811_h2_temporal_track_graph_expert_v1"
    / "cpu_oracle_separability"
    / "report.json"
)

H2_GROUPS = {
    "g1_088_091": tuple("train_{:03d}.npz".format(index) for index in range(88, 92)),
    "g2_092_094": tuple("train_{:03d}.npz".format(index) for index in range(92, 95)),
    "g3_095_098": tuple("train_{:03d}.npz".format(index) for index in range(95, 99)),
}
H2_NAMES = tuple(name for members in H2_GROUPS.values() for name in members)
FIXED_MODEL_SEED = 20260811


@dataclass
class PreparedGraphVideo:
    name: str
    group: str
    scores: np.ndarray
    locations: np.ndarray
    labels: np.ndarray
    target_ids: np.ndarray
    graph: object
    component_targets: np.ndarray
    track_targets: np.ndarray
    track_design: np.ndarray
    baseline_counts: object


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def group_for_name(name):
    matches = [group for group, members in H2_GROUPS.items() if name in members]
    if len(matches) != 1:
        raise ValueError("H2 source must belong to exactly one frozen group")
    return matches[0]


def metric_record(counts):
    return {
        "counts": counts.to_dict(),
        "metrics": crossfit.metrics_from_counts(counts),
    }


def safety_gates(candidate, baseline):
    candidate_metrics = crossfit.metrics_from_counts(candidate)
    baseline_metrics = crossfit.metrics_from_counts(baseline)
    return {
        "score_not_lower": candidate_metrics["score"] >= baseline_metrics["score"],
        "iou_not_lower": candidate_metrics["iou"] >= baseline_metrics["iou"],
        "pd_not_lower": candidate_metrics["pd"] >= baseline_metrics["pd"],
        "fa_not_higher": candidate_metrics["fa"] <= baseline_metrics["fa"],
        "true_positive_events_not_lower": (
            candidate.true_positive_events >= baseline.true_positive_events
        ),
        "correct_objects_not_lower": candidate.correct_objects >= baseline.correct_objects,
        "false_positive_events_not_higher": (
            candidate.false_positive_events <= baseline.false_positive_events
        ),
        "false_components_not_higher": candidate.false_components <= baseline.false_components,
    }


def _sum_counts(values):
    return base._sum_counts(values)


def _load_config():
    with CACHE_PROTOCOL.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    overrides = protocol["definition"]["config"]["overrides"]
    cfg = replay.load_flat_config(ROOT / "configs" / "evisseg_evuav.yaml", overrides)
    effective_c00 = crossfit.validate_c00_config(cfg, base.THRESHOLD)
    return cfg, effective_c00


def prepare_h2_graphs():
    with (CACHE_DIR / "manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("schema") != "ev-uav-component-reranker-train-cache-v1"
        or manifest.get("dataset_split") != "train"
        or manifest.get("base_checkpoint_sha256")
        != "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
    ):
        raise RuntimeError("released-M20 train cache contract changed")
    metadata_by_name = {record["source_name"]: record for record in manifest["records"]}
    if not set(H2_NAMES).issubset(metadata_by_name):
        raise RuntimeError("train cache lacks a frozen H2 source")
    cfg, effective_c00 = _load_config()
    videos = []
    for source_number, name in enumerate(H2_NAMES, start=1):
        metadata = metadata_by_name[name]
        record = _load_cache_record(CACHE_DIR, metadata)
        event_count = int(metadata["event_count"])
        locations = np.column_stack(
            (
                np.zeros(event_count, dtype=np.int64),
                record["locs"].astype(np.int64, copy=False),
            )
        )
        processed, _ = ChallengePostprocessor.from_cfg(
            cfg, base.THRESHOLD, event_count=event_count
        ).apply(
            torch.from_numpy(record["scores"].astype(np.float32, copy=True)),
            torch.from_numpy(locations),
        )
        scores = processed.numpy().astype(np.float32, copy=True)
        source_path = TRAIN_ROOT / name
        with np.load(source_path, allow_pickle=False) as archive:
            input_locations = np.asarray(archive["ev_loc"]).astype(np.int64, copy=False)
            input_values = np.asarray(archive["evs_norm"])
            polarities = input_values[:, 3].astype(np.float32, copy=True)
        if not np.array_equal(input_locations, locations[:, 1:4]):
            raise RuntimeError("input/cache locations differ for {}".format(name))

        # The graph exists before any train-only target is derived.  Its public
        # extractor has no label, target-id, source-name or path argument.
        graph = extract_temporal_track_graph(
            scores,
            locations,
            polarities,
            base.THRESHOLD,
            event_count,
            decoder_event_features=None,
        )
        labels = record["labels"].reshape(-1).astype(np.uint8, copy=True)
        target_ids = record["target_ids"].reshape(-1).copy()
        component_targets = pure_false_positive_component_targets(
            graph.event_indices, labels
        )
        track_targets = pure_false_positive_track_targets(graph, component_targets)
        videos.append(
            PreparedGraphVideo(
                name=name,
                group=group_for_name(name),
                scores=scores,
                locations=locations,
                labels=labels,
                target_ids=target_ids,
                graph=graph,
                component_targets=component_targets,
                track_targets=track_targets,
                track_design=aggregate_track_node_features(graph),
                baseline_counts=base.official_counts(
                    scores, labels, target_ids, locations
                ),
            )
        )
        print(
            "CPU graph {:02d}/11 {}: {} components, {} tracks".format(
                source_number,
                name,
                len(graph.event_indices),
                len(graph.track_component_rows),
            ),
            flush=True,
        )
    return manifest, effective_c00, videos


def _oracle(videos, mode):
    baseline_parts = []
    candidate_parts = []
    per_source = []
    for video in videos:
        if mode == "track":
            candidate_scores, receipt = atomic_delete_from_graph(
                video.scores,
                video.graph,
                cutoff=0.5,
                track_pure_fp_probabilities=video.track_targets.astype(np.float64),
                mode="track",
            )
        elif mode == "component":
            candidate_scores, receipt = atomic_delete_from_graph(
                video.scores,
                video.graph,
                cutoff=0.5,
                component_pure_fp_probabilities=video.component_targets.astype(np.float64),
                mode="component",
            )
        else:
            raise ValueError("unknown oracle mode")
        candidate = base.official_counts(
            candidate_scores, video.labels, video.target_ids, video.locations
        )
        baseline_parts.append(video.baseline_counts)
        candidate_parts.append(candidate)
        per_source.append(
            {
                "name": video.name,
                "group": video.group,
                "component_count": len(video.graph.event_indices),
                "track_count": len(video.graph.track_component_rows),
                "pure_fp_component_count": int(video.component_targets.sum()),
                "pure_fp_track_count": int(video.track_targets.sum()),
                "atomic_edit": asdict(receipt),
                "baseline": metric_record(video.baseline_counts),
                "candidate": metric_record(candidate),
                "gates": safety_gates(candidate, video.baseline_counts),
            }
        )
    baseline = _sum_counts(baseline_parts)
    candidate = _sum_counts(candidate_parts)
    baseline_metrics = crossfit.metrics_from_counts(baseline)
    candidate_metrics = crossfit.metrics_from_counts(candidate)
    return {
        "mode": mode,
        "baseline": metric_record(baseline),
        "candidate": metric_record(candidate),
        "score_delta": candidate_metrics["score"] - baseline_metrics["score"],
        "count_delta": {
            key: int(value - getattr(baseline, key))
            for key, value in candidate.to_dict().items()
        },
        "gates": safety_gates(candidate, baseline),
        "per_source": per_source,
    }


def _training_arrays(videos):
    design = np.concatenate([video.track_design for video in videos], axis=0)
    targets = np.concatenate([video.track_targets for video in videos], axis=0)
    base_weights = np.concatenate(
        [
            np.full(video.track_targets.size, 1.0 / video.track_targets.size)
            for video in videos
        ]
    ).astype(np.float64, copy=False)
    negative_mass = float(base_weights[targets == 0].sum())
    positive_mass = float(base_weights[targets == 1].sum())
    if min(negative_mass, positive_mass) <= 0:
        raise RuntimeError("fixed linear diagnostic needs both track classes")
    base_weights[targets == 1] *= negative_mass / positive_mass
    base_weights *= base_weights.size / base_weights.sum()
    return design, targets, base_weights


def _fit_fixed_linear(videos):
    design, targets, weights = _training_arrays(videos)
    model = Pipeline(
        (
            ("scale", StandardScaler()),
            (
                "linear",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=400,
                    random_state=FIXED_MODEL_SEED,
                ),
            ),
        )
    )
    model.fit(design, targets, linear__sample_weight=weights)
    return model


def _probability(model, video):
    probability = model.predict_proba(video.track_design)[:, 1]
    if probability.shape != video.track_targets.shape or not np.isfinite(probability).all():
        raise RuntimeError("fixed linear diagnostic produced invalid probability")
    return probability


def _evaluate_probability(videos, probabilities, cutoff, enabled):
    baseline_parts = []
    candidate_parts = []
    deleted_tracks = 0
    per_source = []
    for video in videos:
        probability = probabilities[video.name]
        candidate_scores, receipt = atomic_delete_from_graph(
            video.scores,
            video.graph,
            cutoff=cutoff,
            track_pure_fp_probabilities=probability,
            mode="track",
            enabled=enabled,
        )
        candidate = base.official_counts(
            candidate_scores, video.labels, video.target_ids, video.locations
        )
        baseline_parts.append(video.baseline_counts)
        candidate_parts.append(candidate)
        deleted_tracks += receipt.deleted_track_count
        per_source.append(
            {
                "name": video.name,
                "atomic_edit": asdict(receipt),
                "gates": safety_gates(candidate, video.baseline_counts),
            }
        )
    baseline = _sum_counts(baseline_parts)
    candidate = _sum_counts(candidate_parts)
    baseline_metrics = crossfit.metrics_from_counts(baseline)
    candidate_metrics = crossfit.metrics_from_counts(candidate)
    return {
        "baseline": metric_record(baseline),
        "candidate": metric_record(candidate),
        "score_delta": candidate_metrics["score"] - baseline_metrics["score"],
        "gates": safety_gates(candidate, baseline),
        "deleted_track_count": int(deleted_tracks),
        "per_source": per_source,
    }


def fixed_linear_nested_oof(videos):
    folds = []
    pooled_targets = []
    pooled_probabilities = []
    pooled_baselines = []
    pooled_candidates = []
    for held_group in H2_GROUPS:
        fit_groups = [group for group in H2_GROUPS if group != held_group]
        fit_videos = [video for video in videos if video.group in fit_groups]
        held_videos = [video for video in videos if video.group == held_group]
        inner_targets = []
        inner_probabilities = []
        inner_records = []
        for inner_held_group in fit_groups:
            inner_fit = [video for video in fit_videos if video.group != inner_held_group]
            inner_held = [video for video in fit_videos if video.group == inner_held_group]
            inner_model = _fit_fixed_linear(inner_fit)
            for video in inner_held:
                probability = _probability(inner_model, video)
                inner_targets.append(video.track_targets)
                inner_probabilities.append(probability)
                inner_records.append(
                    {
                        "name": video.name,
                        "inner_held_group": inner_held_group,
                        "track_count": int(video.track_targets.size),
                    }
                )
        inner_targets = np.concatenate(inner_targets)
        inner_probabilities = np.concatenate(inner_probabilities)
        cutoff, enabled = derive_zero_observed_target_loss_cutoff(
            inner_probabilities, inner_targets
        )
        outer_model = _fit_fixed_linear(fit_videos)
        probability_by_name = {
            video.name: _probability(outer_model, video) for video in held_videos
        }
        outer_targets = np.concatenate([video.track_targets for video in held_videos])
        outer_probabilities = np.concatenate(
            [probability_by_name[video.name] for video in held_videos]
        )
        evaluation = _evaluate_probability(
            held_videos, probability_by_name, cutoff, enabled
        )
        folds.append(
            {
                "held_group": held_group,
                "fit_groups": fit_groups,
                "inner_oof": {
                    "record_count": len(inner_records),
                    "track_count": int(inner_targets.size),
                    "target_bearing_track_count": int(np.sum(inner_targets == 0)),
                    "pure_fp_track_count": int(np.sum(inner_targets == 1)),
                    "strict_zero_observed_target_loss_cutoff": cutoff,
                    "deletion_enabled": enabled,
                },
                "held_threshold_free": {
                    "roc_auc": float(roc_auc_score(outer_targets, outer_probabilities)),
                    "average_precision": float(
                        average_precision_score(outer_targets, outer_probabilities)
                    ),
                },
                "held_atomic_evaluation": evaluation,
            }
        )
        pooled_targets.append(outer_targets)
        pooled_probabilities.append(outer_probabilities)
        pooled_baselines.append(
            _sum_counts(video.baseline_counts for video in held_videos)
        )
        pooled_candidates.append(
            crossfit.SufficientCounts(**evaluation["candidate"]["counts"])
        )
    targets = np.concatenate(pooled_targets)
    probabilities = np.concatenate(pooled_probabilities)
    baseline = _sum_counts(pooled_baselines)
    candidate = _sum_counts(pooled_candidates)
    baseline_metrics = crossfit.metrics_from_counts(baseline)
    candidate_metrics = crossfit.metrics_from_counts(candidate)
    all_fold_safety = all(
        all(fold["held_atomic_evaluation"]["gates"].values()) for fold in folds
    )
    deleted_tracks = sum(
        fold["held_atomic_evaluation"]["deleted_track_count"] for fold in folds
    )
    return {
        "model": "fixed_standardized_logistic_C1_liblinear_diagnostic_only",
        "model_or_threshold_grid": False,
        "folds": folds,
        "pooled_threshold_free": {
            "roc_auc": float(roc_auc_score(targets, probabilities)),
            "average_precision": float(average_precision_score(targets, probabilities)),
        },
        "pooled_atomic": {
            "baseline": metric_record(baseline),
            "candidate": metric_record(candidate),
            "score_delta": candidate_metrics["score"] - baseline_metrics["score"],
            "gates": safety_gates(candidate, baseline),
            "deleted_track_count": int(deleted_tracks),
        },
        "all_fold_safety_gates_passed": all_fold_safety,
        "non_identity_all_folds": all(
            fold["held_atomic_evaluation"]["deleted_track_count"] > 0 for fold in folds
        ),
    }


def run(args):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU gate refuses a process that initialized CUDA")
    manifest, effective_c00, videos = prepare_h2_graphs()
    component_oracle = _oracle(videos, "component")
    track_oracle = _oracle(videos, "track")
    separability = fixed_linear_nested_oof(videos)
    oracle_gate = bool(
        all(component_oracle["gates"].values())
        and all(track_oracle["gates"].values())
        and track_oracle["score_delta"] >= 0.02
    )
    basic_separability_gate = bool(
        separability["pooled_threshold_free"]["roc_auc"] >= 0.75
        and separability["all_fold_safety_gates_passed"]
        and separability["non_identity_all_folds"]
        and separability["pooled_atomic"]["score_delta"] > 0.0
    )
    decision = (
        "eligible_for_root_authorized_gpu_probe"
        if oracle_gate and basic_separability_gate
        else "stop_before_gpu_insufficient_or_unsafe_cpu_evidence"
    )
    payload = {
        "schema": "ev-uav-h2-temporal-track-graph-cpu-gate-v1",
        "created_utc": utc_now(),
        "dataset_split": "train",
        "sources": list(H2_NAMES),
        "groups": {key: list(value) for key, value in H2_GROUPS.items()},
        "released_m20_cache_manifest_sha256": sha256_file(CACHE_DIR / "manifest.json"),
        "released_m20_sha256": manifest["base_checkpoint_sha256"],
        "cache_protocol_sha256": sha256_file(CACHE_PROTOCOL),
        "effective_c00": effective_c00,
        "prediction_threshold": base.THRESHOLD,
        "graph_contract": {
            "node_feature_count": len(NODE_FEATURE_NAMES),
            "edge_feature_count": len(EDGE_FEATURE_NAMES),
            "track_feature_count": len(TRACK_FEATURE_NAMES),
            "decoder_features_available_in_cpu_gate": False,
            "source_name_path_hash_fold_index_label_or_target_id_is_inference_feature": False,
            "labels_attached_only_after_input_graph": True,
            "atomic_edits_only": True,
            "local_score_attenuation_allowed": False,
        },
        "component_oracle": component_oracle,
        "track_oracle": track_oracle,
        "fixed_linear_nested_grouped_oof": separability,
        "gates": {
            "track_oracle_safe_and_score_gain_at_least_0_02": oracle_gate,
            "fixed_linear_basic_separability_safe_nonidentity_all_folds": basic_separability_gate,
        },
        "decision": decision,
        "validation_or_test_read": False,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    output = Path(args.output)
    if output.exists() or output.parent.exists():
        raise FileExistsError("refusing to overwrite CPU track-graph evidence")
    output.parent.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "component_oracle_score_delta": component_oracle["score_delta"],
                "track_oracle_score_delta": track_oracle["score_delta"],
                "linear_oof_auc": separability["pooled_threshold_free"]["roc_auc"],
                "linear_oof_score_delta": separability["pooled_atomic"]["score_delta"],
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", default=str(OUTPUT_PATH))
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
