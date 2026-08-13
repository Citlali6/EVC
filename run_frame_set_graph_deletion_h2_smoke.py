"""Train-only nested frame-set graph deletion smoke on held family F5/H2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

import run_allsize_deletion_head_oof as base
from utils.frame_set_graph_deletion_head import (
    FRAME_FEATURE_NAMES,
    broadcast_frame_probabilities,
    extract_frame_set_graph_features,
)


EXPECTED_PROTOCOL_SHA256 = "719c696cbb2bd0afb66be95e44c3cd36e3460dca7dbdceeecefa7ab019c300d2"
OUTER_HELD_GROUP = "block_088_098"
MODEL_SEED = 20260811


@dataclass(frozen=True)
class FrameVideo:
    video: object
    frame_batch: object
    frame_labels: np.ndarray


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(counts):
    return {
        "counts": counts.to_dict(),
        "metrics": base.crossfit.metrics_from_counts(counts),
    }


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


def _prepare_frame_videos(videos):
    result = []
    for video in videos:
        batch = extract_frame_set_graph_features(
            video.scores,
            video.locations,
            video.event_indices,
            video.event_count,
            temporal_bin_size=50,
            link_distance=6.0,
            context_distance=12.0,
        )
        labels = np.asarray(
            [
                int(np.any(video.component_labels[np.asarray(rows, dtype=np.int64)] > 0))
                for rows in batch.component_rows
            ],
            dtype=np.uint8,
        )
        if batch.features.shape[0] == 0 or labels.size != batch.features.shape[0]:
            raise RuntimeError(f"invalid frame-set rows for {video.source_name}")
        result.append(FrameVideo(video, batch, labels))
    return result


def _training_arrays(frame_videos):
    features = np.concatenate([item.frame_batch.features for item in frame_videos], axis=0)
    labels = np.concatenate([item.frame_labels for item in frame_videos], axis=0)
    if np.unique(labels).tolist() != [0, 1]:
        raise RuntimeError("frame model requires both target-support and pure-FP frames.")
    base_weights = []
    for item in frame_videos:
        base_weights.append(np.full(item.frame_labels.size, 1.0 / item.frame_labels.size))
    initial = np.concatenate(base_weights).astype(np.float64, copy=False)
    positive_mass = float(initial[labels > 0].sum())
    negative_mass = float(initial[labels == 0].sum())
    class_ratio = negative_mass / positive_mass
    weights = []
    for item in frame_videos:
        values = np.ones(item.frame_labels.size, dtype=np.float64)
        values[item.frame_labels > 0] *= class_ratio
        values /= values.sum()
        weights.append(values)
    sample_weight = np.concatenate(weights)
    sample_weight *= sample_weight.size / sample_weight.sum()
    return features, labels, sample_weight


def _fit_model(frame_videos):
    features, labels, sample_weight = _training_arrays(frame_videos)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=120,
        max_leaf_nodes=15,
        max_depth=3,
        min_samples_leaf=24,
        l2_regularization=4.0,
        random_state=MODEL_SEED,
    )
    model.fit(features, labels, sample_weight=sample_weight)
    return model


def _probabilities(model, frame_video):
    values = model.predict_proba(frame_video.frame_batch.features)[:, 1]
    values = np.asarray(values, dtype=np.float64)
    if values.shape != frame_video.frame_labels.shape or not np.isfinite(values).all():
        raise RuntimeError("invalid frame keep probabilities.")
    return values


def _candidate_counts(frame_video, frame_probabilities, threshold):
    component_probabilities = broadcast_frame_probabilities(
        frame_video.frame_batch,
        frame_probabilities,
        len(frame_video.video.event_indices),
    )
    return base._candidate_counts(frame_video.video, component_probabilities, threshold)


def _evaluate(frame_videos, probability_by_source, threshold):
    baseline = base._sum_counts(item.video.baseline_counts for item in frame_videos)
    candidate_parts = []
    per_source = []
    deleted_frames = 0
    for item in frame_videos:
        probabilities = probability_by_source[item.video.source_name]
        candidate = _candidate_counts(item, probabilities, threshold)
        candidate_parts.append(candidate)
        deleted_frames += int(np.sum(probabilities < float(threshold)))
        baseline_metrics = base.crossfit.metrics_from_counts(item.video.baseline_counts)
        candidate_metrics = base.crossfit.metrics_from_counts(candidate)
        per_source.append(
            {
                "source_name": item.video.source_name,
                "frame_set_count": int(item.frame_labels.size),
                "deleted_frame_set_count": int(np.sum(probabilities < float(threshold))),
                "baseline": _record(item.video.baseline_counts),
                "candidate": _record(candidate),
                "score_delta": candidate_metrics["score"] - baseline_metrics["score"],
            }
        )
    candidate = base._sum_counts(candidate_parts)
    return {
        "baseline": _record(baseline),
        "candidate": _record(candidate),
        "score_delta": base.crossfit.metrics_from_counts(candidate)["score"]
        - base.crossfit.metrics_from_counts(baseline)["score"],
        "gates": _gates(candidate, baseline),
        "false_positive_events_deleted": int(
            baseline.false_positive_events - candidate.false_positive_events
        ),
        "false_components_deleted": int(
            baseline.false_components - candidate.false_components
        ),
        "true_positive_events_delta": int(
            candidate.true_positive_events - baseline.true_positive_events
        ),
        "deleted_frame_set_count": int(deleted_frames),
        "per_source": per_source,
    }


def _validate_protocol(protocol, cache_manifest_sha256, c00_protocol_sha256):
    expected_groups = {
        name: list(members) for name, members in base.SOURCE_GROUPS.items()
    }
    if protocol.get("schema") != "ev-uav-frame-set-graph-deletion-f5-smoke-science-v1":
        raise ValueError("unexpected frame-set graph science schema.")
    if protocol.get("dataset_scope") != "train-only 54 sources with event_count > 30000":
        raise ValueError("science protocol is not the frozen train-only population.")
    if protocol.get("source_groups") != expected_groups:
        raise ValueError("science protocol source groups changed.")
    if protocol.get("outer_held_group") != OUTER_HELD_GROUP:
        raise ValueError("science protocol outer held family changed.")
    if protocol.get("feature_names") != list(FRAME_FEATURE_NAMES):
        raise ValueError("science protocol feature order changed.")
    if protocol["inputs"]["cache_manifest_sha256"] != cache_manifest_sha256:
        raise ValueError("train cache manifest hash changed.")
    if protocol["inputs"]["c00_protocol_sha256"] != c00_protocol_sha256:
        raise ValueError("C00 protocol hash changed.")
    if protocol.get("selection_rule") != (
        "minimum inner-family OOF probability among target-support frames; "
        "nonidentity selected only if every inner fold passes all official safety gates "
        "and pooled FP plus false-components strictly decrease"
    ):
        raise ValueError("science protocol selection rule changed.")


def run(args):
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must remain uninitialized for the CPU smoke.")
    root = Path(__file__).resolve().parent
    protocol_path = Path(args.science_protocol).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    c00_protocol = Path(args.c00_protocol).resolve()
    output_report = Path(args.output_report).resolve()
    if output_report.exists():
        raise FileExistsError(output_report)
    protocol_sha256 = sha256_file(protocol_path)
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("frame-set graph science protocol changed.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cache_manifest_sha256 = sha256_file(cache_dir / "manifest.json")
    c00_protocol_sha256 = sha256_file(c00_protocol)
    _validate_protocol(protocol, cache_manifest_sha256, c00_protocol_sha256)
    input_hashes_before = {
        "cache_manifest_sha256": cache_manifest_sha256,
        "c00_protocol_sha256": c00_protocol_sha256,
        "science_protocol_sha256": protocol_sha256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "feature_module_sha256": sha256_file(
            root / "utils" / "frame_set_graph_deletion_head.py"
        ),
    }

    manifest, _cfg, videos = base.prepare_videos(cache_dir, c00_protocol)
    frame_videos = _prepare_frame_videos(videos)
    fit_groups = tuple(
        group for group in base.SOURCE_GROUPS if group != OUTER_HELD_GROUP
    )
    outer_videos = [item for item in frame_videos if item.video.group == OUTER_HELD_GROUP]
    if len(outer_videos) != 11:
        raise RuntimeError("F5/H2 outer smoke must contain exactly eleven sources.")

    inner_probabilities = {}
    inner_folds = []
    positive_probabilities = []
    for held_group in fit_groups:
        inner_fit = [
            item
            for item in frame_videos
            if item.video.group in fit_groups and item.video.group != held_group
        ]
        inner_held = [item for item in frame_videos if item.video.group == held_group]
        model = _fit_model(inner_fit)
        held_probability_map = {}
        for item in inner_held:
            values = _probabilities(model, item)
            held_probability_map[item.video.source_name] = values
            inner_probabilities[item.video.source_name] = values
            positive_probabilities.extend(values[item.frame_labels > 0].tolist())
        inner_folds.append((held_group, inner_held, held_probability_map))
    if not positive_probabilities:
        raise RuntimeError("inner OOF contains no target-support frames.")
    learned_threshold = float(np.min(np.asarray(positive_probabilities, dtype=np.float64)))

    inner_records = []
    pooled_baseline = base.crossfit.SufficientCounts()
    pooled_candidate = base.crossfit.SufficientCounts()
    for held_group, inner_held, held_probability_map in inner_folds:
        result = _evaluate(inner_held, held_probability_map, learned_threshold)
        result["held_group"] = held_group
        inner_records.append(result)
        pooled_baseline = pooled_baseline + base.crossfit.SufficientCounts(
            **result["baseline"]["counts"]
        )
        pooled_candidate = pooled_candidate + base.crossfit.SufficientCounts(
            **result["candidate"]["counts"]
        )
    inner_pooled = {
        "baseline": _record(pooled_baseline),
        "candidate": _record(pooled_candidate),
        "score_delta": base.crossfit.metrics_from_counts(pooled_candidate)["score"]
        - base.crossfit.metrics_from_counts(pooled_baseline)["score"],
        "false_positive_events_deleted": int(
            pooled_baseline.false_positive_events - pooled_candidate.false_positive_events
        ),
        "false_components_deleted": int(
            pooled_baseline.false_components - pooled_candidate.false_components
        ),
    }
    inner_safe = all(all(item["gates"].values()) for item in inner_records)
    selected_nonidentity = bool(
        inner_safe
        and inner_pooled["score_delta"] > 0.0
        and inner_pooled["false_positive_events_deleted"] > 0
        and inner_pooled["false_components_deleted"] > 0
    )
    selected_threshold = learned_threshold if selected_nonidentity else 0.0

    final_model = _fit_model(
        [item for item in frame_videos if item.video.group in fit_groups]
    )
    outer_probability_map = {
        item.video.source_name: _probabilities(final_model, item)
        for item in outer_videos
    }
    outer_result = _evaluate(outer_videos, outer_probability_map, selected_threshold)
    outer_success = bool(
        selected_nonidentity
        and all(outer_result["gates"].values())
        and outer_result["false_positive_events_deleted"] > 0
        and outer_result["false_components_deleted"] > 0
    )

    input_hashes_after = {
        "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "c00_protocol_sha256": sha256_file(c00_protocol),
        "science_protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "feature_module_sha256": sha256_file(
            root / "utils" / "frame_set_graph_deletion_head.py"
        ),
    }
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("bound CPU smoke inputs changed during execution.")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU smoke initialized CUDA unexpectedly.")

    report = {
        "schema": "ev-uav-frame-set-graph-deletion-f5-smoke-result-v1",
        "dataset_split": "train",
        "evidence_class": "nested_contiguous_family_f5_smoke_only",
        "no_validation_or_test_access": True,
        "cuda_initialized": False,
        "source_identity_path_hash_fold_is_feature": False,
        "prediction_unit": "one joint keep/delete decision per nonempty 50-bin frame set",
        "feature_names": list(FRAME_FEATURE_NAMES),
        "feature_count": len(FRAME_FEATURE_NAMES),
        "source_groups": {
            name: list(members) for name, members in base.SOURCE_GROUPS.items()
        },
        "outer_held_group": OUTER_HELD_GROUP,
        "inputs_before": input_hashes_before,
        "inputs_after": input_hashes_after,
        "cache_selected_video_count": int(manifest["selected_video_count"]),
        "selection": {
            "access": "F1-F4 inner grouped OOF only; F5 never selects",
            "model": "single frozen HistGradientBoostingClassifier",
            "threshold_rule": "minimum inner OOF probability among target-support frames",
            "learned_threshold": learned_threshold,
            "selected_nonidentity": selected_nonidentity,
            "deployed_threshold_for_smoke": selected_threshold,
        },
        "inner_folds": inner_records,
        "inner_pooled": inner_pooled,
        "inner_all_safety_gates_passed": inner_safe,
        "outer_f5": outer_result,
        "success": outer_success,
        "decision": (
            "proceed_to_full_nested_frame_set_oof"
            if outer_success
            else "stop_frame_set_graph_v1_no_safe_f5_signal"
        ),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(output_report), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output_report": str(output_report),
                "selected_nonidentity": selected_nonidentity,
                "inner_pooled_score_delta": inner_pooled["score_delta"],
                "outer_f5_score_delta": outer_result["score_delta"],
                "outer_f5_fp_deleted": outer_result["false_positive_events_deleted"],
                "outer_f5_false_components_deleted": outer_result["false_components_deleted"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )
    return 0


def build_parser():
    root = Path(__file__).resolve().parent
    experiments = root.parent / "experiments"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--science-protocol",
        default=str(root / "protocols" / "frame_set_graph_deletion_f5_smoke_v1.json"),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(
            experiments
            / "20260810_component_reranker_crosssource_v1"
            / "train_cache_gt30000"
        ),
    )
    parser.add_argument(
        "--c00-protocol",
        default=str(
            experiments
            / "20260810_component_reranker_crosssource_v1"
            / "crossfit_protocol.json"
        ),
    )
    parser.add_argument(
        "--output-report",
        default=str(
            experiments
            / "20260811_frame_set_graph_deletion_f5_smoke_v1"
            / "f5_smoke_report.json"
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
