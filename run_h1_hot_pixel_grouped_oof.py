"""Train-only two-fold OOF audit for conservative H1 hot-pixel suppression.

The entry point is deliberately limited to official train_044..047.  Candidate
decisions use complete-video x/y/t/p statistics and frozen M20 baseline scores.
Held labels are touched only after a candidate score vector has been built.
No model inference or GPU is required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

import crossfit_component_reranker as component_crossfit
import crossfit_persistent_pixel_prior as persistence_crossfit
import replay_temporal_memory_validation as replay
from train_component_reranker import load_train_cache
from utils.component_reranker import sha256_file, sha256_json
from utils.postprocess import P18ScoreTrackRecovery


REPORT_SCHEMA = "ev-uav-h1-hot-pixel-grouped-oof-report-v1"
PROTOCOL_SCHEMA = "ev-uav-h1-hot-pixel-grouped-oof-science-v1"
H1_NAMES = tuple("train_{:03d}.npz".format(index) for index in range(44, 48))
WIDTH = 346
HEIGHT = 260
PIXEL_COUNT = WIDTH * HEIGHT
PREDICTION_THRESHOLD = 0.719
EPSILON = 1e-15


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream), path


def atomic_json(path, payload):
    path = Path(path).resolve()
    sidecar = path.with_name(path.name + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError("Refusing to overwrite immutable output: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = sha256_file(path)
    sidecar.write_text("{}  {}\n".format(digest, path.name), encoding="ascii")
    return digest


def validate_train_path(path, label, require_name=None):
    path = Path(path).resolve()
    if require_name is not None and path.name.lower() != require_name:
        raise ValueError("{} must be named {}.".format(label, require_name))
    lowered = {part.lower() for part in path.parts}
    if require_name is not None:
        lowered.discard(require_name)
    forbidden = {"val", "validation", "test"}
    if lowered & forbidden:
        raise ValueError("{} contains a forbidden split token.".format(label))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def validate_protocol(payload):
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unexpected science protocol schema.")
    names = tuple(payload["source_population"]["names"])
    if names != H1_NAMES:
        raise ValueError("Science protocol source population changed.")
    if float(payload["baseline"]["prediction_threshold"]) != PREDICTION_THRESHOLD:
        raise ValueError("Science protocol threshold changed.")
    fold_plan = payload.get("fold_plan")
    expected = (
        {
            "fold_id": "holdout_044_045",
            "fit_names": ["train_046.npz", "train_047.npz"],
            "held_names": ["train_044.npz", "train_045.npz"],
        },
        {
            "fold_id": "holdout_046_047",
            "fit_names": ["train_044.npz", "train_045.npz"],
            "held_names": ["train_046.npz", "train_047.npz"],
        },
    )
    if tuple(fold_plan) != expected:
        raise ValueError("Science protocol fold plan changed.")
    candidates = payload.get("candidates", [])
    candidate_ids = [value.get("candidate_id") for value in candidates]
    if len(candidates) != 15 or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Science protocol candidate grid is incomplete or duplicated.")
    allowed_rules = set(payload["hot_rules"])
    allowed_modes = set(payload["coordinate_modes"])
    for candidate in candidates:
        if candidate["hot_rule"] not in allowed_rules:
            raise ValueError("Candidate references an unknown hot rule.")
        if candidate["coordinate_mode"] not in allowed_modes:
            raise ValueError("Candidate references an unknown coordinate mode.")
        if not 1 <= int(candidate["max_component_events"]) <= 3:
            raise ValueError("Candidate component-event cap is invalid.")
        if int(candidate["max_track_bins"]) <= 0:
            raise ValueError("Candidate track-bin cap is invalid.")
        if not PREDICTION_THRESHOLD <= float(candidate["max_score"]) <= 1.0:
            raise ValueError("Candidate score cap is invalid.")
        if candidate["coordinate_mode"] == "fit_union_zero_positive":
            if int(candidate["min_fit_false_components"]) <= 0:
                raise ValueError("Label-vetted coordinate candidate lacks false evidence.")
        elif int(candidate["min_fit_false_components"]) != 0:
            raise ValueError("Input-only coordinate mode must not use label evidence.")
    gates = payload["promotion_gates"]
    expected_gates = {
        "each_fold_score_delta_min": 0.0,
        "each_fold_iou_delta_min": 0.0,
        "each_fold_pd_delta_min": 0.0,
        "each_fold_fa_delta_max": 0.0,
        "each_fold_true_positive_event_delta": 0,
        "each_fold_correct_object_delta": 0,
        "pooled_score_delta_min": 0.0002,
        "pooled_iou_delta_min": 0.0,
        "pooled_pd_delta_min": 0.0,
        "pooled_fa_delta_max": 0.0,
        "pooled_true_positive_event_delta": 0,
        "pooled_correct_object_delta": 0,
        "pooled_false_positive_event_delta_max": -1,
        "pooled_false_component_delta_max": -1,
    }
    if gates != expected_gates:
        raise ValueError("Science protocol promotion gates changed.")
    return payload


@dataclass
class H1Video:
    prepared: component_crossfit.PreparedVideo
    prior: persistence_crossfit.PixelPrior
    persistence_features: np.ndarray
    source_sha256: str

    @property
    def source_name(self):
        return self.prepared.source_name


def hot_mask(prior, rule_name):
    if rule_name == "full_life":
        result = (prior.active_fraction >= 1.0) & (
            prior.longest_run_fraction >= 1.0
        )
    elif rule_name == "persistent_polar":
        result = (
            (prior.active_fraction >= 0.90)
            & (prior.longest_run_fraction >= 0.50)
            & (prior.polarity_dominance >= 0.90)
        )
    elif rule_name == "saturated":
        result = (
            (prior.log_max_bin_events >= math.log1p(54.0))
            & (prior.collision_fraction >= 0.50)
            & (prior.polarity_dominance >= 0.90)
        )
    else:
        raise ValueError("Unknown hot rule: {}".format(rule_name))
    result = np.asarray(result, dtype=bool).reshape(-1)
    if result.size != PIXEL_COUNT:
        raise RuntimeError("Hot mask does not cover the frozen sensor grid.")
    return result


def component_descriptor(video, component_index):
    indices = video.prepared.event_indices[component_index]
    pixels = np.unique(video.prior.event_pixel_ids[indices])
    features = video.prepared.features[component_index]
    return {
        "indices": indices,
        "pixels": pixels,
        "event_count": int(indices.size),
        "score_max": float(features[2]),
        "track_bins": int(round(float(features[9]))),
    }


def component_passes_gates(descriptor, candidate):
    return (
        descriptor["pixels"].size == 1
        and descriptor["event_count"] <= int(candidate["max_component_events"])
        and descriptor["track_bins"] <= int(candidate["max_track_bins"])
        and descriptor["score_max"] <= float(candidate["max_score"]) + EPSILON
    )


def learn_coordinates(fit_videos, candidate):
    masks = [hot_mask(video.prior, candidate["hot_rule"]) for video in fit_videos]
    mode = candidate["coordinate_mode"]
    if mode == "fit_intersection_input":
        learned = np.logical_and.reduce(masks)
    else:
        learned = np.logical_or.reduce(masks)
    label_evidence = {
        "fit_positive_components_on_coordinates": 0,
        "fit_negative_components_on_coordinates": 0,
    }
    if mode != "fit_union_zero_positive":
        return learned, label_evidence

    positive = np.zeros(PIXEL_COUNT, dtype=np.int64)
    negative = np.zeros(PIXEL_COUNT, dtype=np.int64)
    for video, local_hot in zip(fit_videos, masks):
        for component_index in range(len(video.prepared.event_indices)):
            descriptor = component_descriptor(video, component_index)
            if descriptor["pixels"].size != 1:
                continue
            pixel = int(descriptor["pixels"][0])
            # Any known positive component at a learned coordinate is a veto,
            # even if this particular fit video does not call the coordinate hot.
            if learned[pixel] and int(video.prepared.component_labels[component_index]) > 0:
                positive[pixel] += 1
            if (
                learned[pixel]
                and local_hot[pixel]
                and component_passes_gates(descriptor, candidate)
                and int(video.prepared.component_labels[component_index]) == 0
            ):
                negative[pixel] += 1
    label_evidence = {
        "fit_positive_components_on_coordinates": int(positive[learned].sum()),
        "fit_negative_components_on_coordinates": int(negative[learned].sum()),
    }
    learned &= positive == 0
    learned &= negative >= int(candidate["min_fit_false_components"])
    return learned, label_evidence


def build_candidate_scores(video, learned_coordinates, candidate, cfg):
    """Build held scores without reading held labels or target ids."""

    held_hot = hot_mask(video.prior, candidate["hot_rule"])
    allowed = learned_coordinates & held_hot
    scores = video.prepared.p0_scores.copy()
    removed_components = 0
    removed_events = 0
    removed_coordinates = set()
    for component_index in range(len(video.prepared.event_indices)):
        descriptor = component_descriptor(video, component_index)
        if not component_passes_gates(descriptor, candidate):
            continue
        pixel = int(descriptor["pixels"][0])
        if not allowed[pixel]:
            continue
        scores[descriptor["indices"]] = 0.0
        removed_components += 1
        removed_events += descriptor["event_count"]
        removed_coordinates.add(pixel)
    recovery = P18ScoreTrackRecovery.from_cfg(cfg, PREDICTION_THRESHOLD)
    score_tensor, _ = recovery.apply(
        torch.from_numpy(scores),
        torch.from_numpy(video.prepared.locations.astype(np.int64, copy=False)),
    )
    result = score_tensor.numpy().astype(np.float32, copy=False)
    if result.shape != (video.prepared.event_count,) or not np.isfinite(result).all():
        raise RuntimeError("Candidate score vector is invalid.")
    return result, {
        "held_hot_coordinate_count": int(held_hot.sum()),
        "learned_and_held_hot_coordinate_count": int(allowed.sum()),
        "removed_components": removed_components,
        "removed_events": removed_events,
        "removed_coordinate_count": len(removed_coordinates),
    }


def add_counts(values):
    result = component_crossfit.SufficientCounts()
    for value in values:
        result = result + value
    return result


def count_delta(candidate, baseline):
    return {
        field: int(getattr(candidate, field) - getattr(baseline, field))
        for field in baseline.__dataclass_fields__
    }


def metric_delta(candidate, baseline):
    return {key: float(candidate[key] - baseline[key]) for key in baseline}


def evaluate_fold(fold, videos_by_name, candidates, cfg):
    fit = [videos_by_name[name] for name in fold["fit_names"]]
    held = [videos_by_name[name] for name in fold["held_names"]]
    if {video.source_name for video in fit} & {video.source_name for video in held}:
        raise RuntimeError("Grouped OOF source leakage.")
    baseline_counts = add_counts(video.prepared.baseline_counts for video in held)
    baseline_metrics = component_crossfit.metrics_from_counts(baseline_counts)
    results = []
    for candidate in candidates:
        learned, evidence = learn_coordinates(fit, candidate)
        # Candidate score vectors are constructed before the evaluation loop
        # receives held labels or target ids.
        built = []
        for video in held:
            scores, changes = build_candidate_scores(video, learned, candidate, cfg)
            built.append((video, scores, changes))
        per_video = []
        candidate_counts = component_crossfit.SufficientCounts()
        for video, scores, changes in built:
            counts = component_crossfit.sufficient_counts_for_video(
                scores,
                video.prepared.event_labels,
                video.prepared.target_ids,
                video.prepared.locations,
            )
            candidate_counts = candidate_counts + counts
            per_video.append(
                {
                    "source_name": video.source_name,
                    "baseline_counts": video.prepared.baseline_counts.to_dict(),
                    "candidate_counts": counts.to_dict(),
                    "count_delta": count_delta(
                        counts, video.prepared.baseline_counts
                    ),
                    **changes,
                }
            )
        metrics = component_crossfit.metrics_from_counts(candidate_counts)
        results.append(
            {
                **candidate,
                "learned_coordinate_count": int(learned.sum()),
                "fit_label_evidence": evidence,
                "counts": candidate_counts.to_dict(),
                "metrics": metrics,
                "count_delta": count_delta(candidate_counts, baseline_counts),
                "metric_delta": metric_delta(metrics, baseline_metrics),
                "per_video": per_video,
            }
        )
    return {
        "fold_id": fold["fold_id"],
        "fit_names": list(fold["fit_names"]),
        "held_names": list(fold["held_names"]),
        "baseline": {
            "counts": baseline_counts.to_dict(),
            "metrics": baseline_metrics,
        },
        "candidate_results": results,
    }


def result_by_id(fold, candidate_id):
    values = [
        value
        for value in fold["candidate_results"]
        if value["candidate_id"] == candidate_id
    ]
    if len(values) != 1:
        raise RuntimeError("Candidate result lookup is not unique.")
    return values[0]


def promotion_checks(result, fold_results, gates):
    fold_checks = []
    for fold in fold_results:
        value = result_by_id(fold, result["candidate_id"])
        delta_m = value["metric_delta"]
        delta_c = value["count_delta"]
        checks = {
            "score_nonnegative": delta_m["score"] + EPSILON
            >= float(gates["each_fold_score_delta_min"]),
            "iou_nonnegative": delta_m["iou"] + EPSILON
            >= float(gates["each_fold_iou_delta_min"]),
            "pd_nonnegative": delta_m["pd"] + EPSILON
            >= float(gates["each_fold_pd_delta_min"]),
            "fa_nonincreasing": delta_m["fa"]
            <= float(gates["each_fold_fa_delta_max"]) + EPSILON,
            "true_positive_events_preserved": delta_c["true_positive_events"]
            == int(gates["each_fold_true_positive_event_delta"]),
            "correct_objects_preserved": delta_c["correct_objects"]
            == int(gates["each_fold_correct_object_delta"]),
        }
        fold_checks.append(
            {
                "fold_id": fold["fold_id"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    delta_m = result["metric_delta"]
    delta_c = result["count_delta"]
    pooled = {
        "score_clearly_positive": delta_m["score"] + EPSILON
        >= float(gates["pooled_score_delta_min"]),
        "iou_nonnegative": delta_m["iou"] + EPSILON
        >= float(gates["pooled_iou_delta_min"]),
        "pd_nonnegative": delta_m["pd"] + EPSILON
        >= float(gates["pooled_pd_delta_min"]),
        "fa_nonincreasing": delta_m["fa"]
        <= float(gates["pooled_fa_delta_max"]) + EPSILON,
        "true_positive_events_preserved": delta_c["true_positive_events"]
        == int(gates["pooled_true_positive_event_delta"]),
        "correct_objects_preserved": delta_c["correct_objects"]
        == int(gates["pooled_correct_object_delta"]),
        "false_positive_events_reduced": delta_c["false_positive_events"]
        <= int(gates["pooled_false_positive_event_delta_max"]),
        "false_components_reduced": delta_c["false_components"]
        <= int(gates["pooled_false_component_delta_max"]),
    }
    return {
        "folds": fold_checks,
        "pooled": pooled,
        "passed": all(item["passed"] for item in fold_checks)
        and all(pooled.values()),
    }


def pooled_results(fold_results, candidates, gates):
    baseline_counts = add_counts(
        component_crossfit.SufficientCounts(**fold["baseline"]["counts"])
        for fold in fold_results
    )
    baseline_metrics = component_crossfit.metrics_from_counts(baseline_counts)
    results = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        counts = add_counts(
            component_crossfit.SufficientCounts(
                **result_by_id(fold, candidate_id)["counts"]
            )
            for fold in fold_results
        )
        metrics = component_crossfit.metrics_from_counts(counts)
        result = {
            **candidate,
            "counts": counts.to_dict(),
            "metrics": metrics,
            "count_delta": count_delta(counts, baseline_counts),
            "metric_delta": metric_delta(metrics, baseline_metrics),
        }
        result["promotion"] = promotion_checks(result, fold_results, gates)
        results.append(result)
    eligible = [value for value in results if value["promotion"]["passed"]]
    winner = None
    if eligible:
        winner = sorted(
            eligible,
            key=lambda value: (-value["metrics"]["score"], value["candidate_id"]),
        )[0]["candidate_id"]
    return {
        "baseline": {
            "counts": baseline_counts.to_dict(),
            "metrics": baseline_metrics,
        },
        "candidates": results,
        "promotion_candidate_ids": [value["candidate_id"] for value in eligible],
        "winner_candidate_id": winner,
        "family_status": "promoted_train_oof_candidate" if winner else "eliminated",
    }


def audit_prior_identity(prior_report, fold_results, videos_by_name):
    audits = []
    for fold in fold_results:
        prior_folds = [
            value
            for value in prior_report["fold_results"]
            if tuple(value["held_video_names"]) == tuple(fold["held_names"])
        ]
        if len(prior_folds) != 1:
            raise RuntimeError("Prior H1 fold lookup failed.")
        prior_fold = prior_folds[0]
        candidates = [
            value
            for value in prior_fold["candidate_results"]
            if value["candidate_id"] == "persistence_pw08_kp050"
        ]
        if len(candidates) != 1:
            raise RuntimeError("Prior persistence control lookup failed.")
        control = candidates[0]
        if any(value != 0 for value in control["count_delta"].values()):
            raise RuntimeError("Prior H1 persistence control is no longer identity.")
        model = control["fit_model"]
        fitted = {
            "feature_mean": np.asarray(model["feature_mean"], dtype=np.float64),
            "feature_scale": np.asarray(model["feature_scale"], dtype=np.float64),
            "coefficients": np.asarray(model["coefficients"], dtype=np.float64),
            "intercept": float(model["intercept"]),
        }
        probabilities = np.concatenate(
            [
                component_crossfit._predict_probabilities(
                    videos_by_name[name].persistence_features, fitted
                )
                for name in fold["held_names"]
            ]
        )
        below = int(np.sum(probabilities < 0.50))
        if below != 0 or int(control["removed_candidate_components"]) != 0:
            raise RuntimeError("Prior identity probability audit disagrees with report.")
        audits.append(
            {
                "fold_id": fold["fold_id"],
                "held_component_count": int(probabilities.size),
                "keep_probability_threshold": 0.50,
                "keep_probability_quantiles": {
                    key: float(value)
                    for key, value in zip(
                        ("min", "q01", "q50", "q99", "max"),
                        np.quantile(probabilities, (0.0, 0.01, 0.50, 0.99, 1.0)),
                    )
                },
                "components_below_keep_threshold": below,
                "fit_model_intercept": float(model["intercept"]),
                "reported_removed_components": int(
                    control["removed_candidate_components"]
                ),
                "reported_removed_events": int(control["removed_candidate_events"]),
            }
        )
    return {
        "candidate_id": "persistence_pw08_kp050",
        "folds": audits,
        "reason": (
            "The positive-weighted logistic control reduced each candidate to "
            "component-level mean/max persistence features and assigned every held "
            "H1 component keep_probability>=0.50. It therefore had no fixed-coordinate "
            "blacklist and removed nothing."
        ),
    }


def run(args):
    protocol, protocol_path = load_json(args.protocol)
    validate_protocol(protocol)
    protocol_sha = sha256_file(protocol_path)
    raw_train_dir = validate_train_path(
        args.raw_train_dir, "raw train", require_name="train"
    )
    cache_dir = validate_train_path(args.cache_dir, "train score cache")
    output_path = Path(args.output).resolve()
    validate_train_path(output_path.parent, "output directory")
    config_path = validate_train_path(args.config, "config")
    reference_protocol_path = validate_train_path(
        args.reference_protocol, "reference train protocol"
    )
    prior_report, prior_report_path = load_json(args.prior_persistence_report)
    validate_train_path(prior_report_path, "prior train persistence report")
    expected_prior_sha = protocol["prior_persistence_control"]["report_sha256"]
    if sha256_file(prior_report_path) != expected_prior_sha:
        raise ValueError("Prior persistence report SHA-256 mismatch.")

    reference_payload, overrides = persistence_crossfit._read_reference_protocol(
        reference_protocol_path
    )
    cfg = replay.load_flat_config(config_path, overrides)
    component_crossfit.validate_c00_config(cfg)
    cache_dir, manifest_path, manifest_sha, manifest = load_train_cache(cache_dir)
    if manifest_sha != protocol["baseline"]["score_cache_manifest_sha256"]:
        raise ValueError("Train score-cache manifest SHA-256 mismatch.")
    if manifest.get("dataset_split") != "train":
        raise ValueError("Score cache is not train-only.")
    if manifest.get("base_checkpoint_sha256") != protocol["baseline"][
        "checkpoint_sha256"
    ]:
        raise ValueError("M20 checkpoint identity mismatch.")

    records_by_name = {record["source_name"]: record for record in manifest["records"]}
    if not set(H1_NAMES).issubset(records_by_name):
        raise ValueError("Train cache lacks the frozen H1 population.")
    selected_manifest = dict(manifest)
    selected_manifest["records"] = [records_by_name[name] for name in H1_NAMES]
    prepared = component_crossfit._prepare_videos(
        cache_dir, selected_manifest, cfg, middle_names=set()
    )
    prepared_by_name = {video.source_name: video for video in prepared}

    videos = []
    source_audit = []
    for name in H1_NAMES:
        expected_sha = protocol["source_population"]["source_sha256"][name]
        raw_path = raw_train_dir / name
        if sha256_file(raw_path) != expected_sha:
            raise ValueError("Raw train source SHA-256 mismatch: {}".format(name))
        video = prepared_by_name[name]
        prior = persistence_crossfit.derive_pixel_prior(
            raw_path, video.locations[:, 1:4]
        )
        if prior.summary["observable_domain"] != "h1":
            raise RuntimeError("Complete-input polarity route no longer selects H1.")
        persistence_features = persistence_crossfit.component_persistence_features(
            prior, video.event_indices
        )
        videos.append(H1Video(video, prior, persistence_features, expected_sha))
        source_audit.append(
            {
                "source_name": name,
                "source_sha256": expected_sha,
                **prior.summary,
                "hot_coordinate_counts": {
                    rule: int(hot_mask(prior, rule).sum())
                    for rule in protocol["hot_rules"]
                },
            }
        )
    videos_by_name = {video.source_name: video for video in videos}

    prior_baseline_by_held = {
        tuple(value["held_video_names"]): value["baseline"]["counts"]
        for value in prior_report["fold_results"]
        if value["domain"] == "h1"
    }
    fold_results = []
    for fold in protocol["fold_plan"]:
        result = evaluate_fold(fold, videos_by_name, protocol["candidates"], cfg)
        expected = prior_baseline_by_held.get(tuple(fold["held_names"]))
        if expected != result["baseline"]["counts"]:
            raise RuntimeError("Current H1 baseline differs from prior immutable result.")
        fold_results.append(result)
    pooled = pooled_results(
        fold_results, protocol["candidates"], protocol["promotion_gates"]
    )
    prior_identity = audit_prior_identity(
        prior_report, fold_results, videos_by_name
    )

    project_root = Path(__file__).resolve().parent
    code_paths = {
        "run_h1_hot_pixel_grouped_oof.py": Path(__file__).resolve(),
        "crossfit_persistent_pixel_prior.py": Path(
            persistence_crossfit.__file__
        ).resolve(),
        "crossfit_component_reranker.py": Path(component_crossfit.__file__).resolve(),
        "utils/component_reranker.py": project_root / "utils/component_reranker.py",
        "utils/postprocess.py": project_root / "utils/postprocess.py",
        "utils/eval.py": project_root / "utils/eval.py",
        "utils/challenge_eval.py": project_root / "utils/challenge_eval.py",
    }
    payload = {
        "schema": REPORT_SCHEMA,
        "created_utc": utc_now(),
        "status": "complete",
        "evidence_class": protocol["evidence_class"],
        "split_access": {
            "dataset_split": "train",
            "consumed_sources": list(H1_NAMES),
            "validation_or_test_read": False,
            "candidate_runtime_uses_source_name": False,
            "held_labels_used_after_candidate_score_construction_only": True,
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha,
        },
        "baseline": protocol["baseline"],
        "source_audit": source_audit,
        "prior_persistence_identity_audit": prior_identity,
        "fold_results": fold_results,
        "pooled_oof": pooled,
        "conclusion": {
            "promotion_candidate_found": pooled["winner_candidate_id"] is not None,
            "winner_candidate_id": pooled["winner_candidate_id"],
            "family_status": pooled["family_status"],
            "no_pass_action": protocol["selection"]["no_pass_action"],
            "independent_final_estimate": False,
        },
        "provenance": {
            "cache_manifest_path": str(manifest_path),
            "cache_manifest_sha256": manifest_sha,
            "reference_train_protocol_path": str(reference_protocol_path),
            "reference_train_protocol_sha256": sha256_file(
                reference_protocol_path
            ),
            "prior_persistence_report_path": str(prior_report_path),
            "prior_persistence_report_sha256": sha256_file(prior_report_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "reference_config_sha256": reference_payload["definition"]["config"].get(
                "sha256"
            ),
            "code_sha256": {
                name: sha256_file(path) for name, path in code_paths.items()
            },
        },
    }
    digest = atomic_json(output_path, payload)
    print("wrote:", output_path)
    print("sha256:", digest)
    print("family_status:", pooled["family_status"])
    print("winner:", pooled["winner_candidate_id"])
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--raw-train-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-protocol", type=Path, required=True)
    parser.add_argument("--prior-persistence-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
