"""Train-only grouped OOF audit for an input-derived persistent-pixel prior.

The script is deliberately unable to consume validation or test records.  It
joins the immutable official-train M20 score cache to the matching raw train
sources, derives label-free per-pixel lifetime statistics, and evaluates small
component suppression on five source-group holdouts.  Labels are used only for
fitting the train-only component model and for official-metric OOF scoring.

This is an experiment/audit entry point, not a deployment postprocessor.  It
does not modify the core inference chain and emits no runtime artifact.
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

import cv2
import numpy as np
import torch

import crossfit_component_reranker as component_crossfit
import replay_temporal_memory_validation as replay
from train_component_reranker import load_train_cache
from utils.component_reranker import sha256_file, sha256_json
from utils.postprocess import P18ScoreTrackRecovery


REPORT_SCHEMA = "ev-uav-persistent-pixel-prior-grouped-oof-v1"
DATASET_SPLIT = "train"
PREDICTION_THRESHOLD = 0.719
WIDTH = 346
HEIGHT = 260
TEMPORAL_BIN_SIZE = 50
VIDEO_DURATION = 8000
TEMPORAL_BIN_COUNT = VIDEO_DURATION // TEMPORAL_BIN_SIZE
DOMAIN_POLARITY_MINORITY_CUTOFF = 0.20
LOG_COUNT_CLIP = 4.0

H1_NAMES = tuple("train_{:03d}.npz".format(index) for index in range(44, 48))
H2_NAMES = tuple("train_{:03d}.npz".format(index) for index in range(88, 99))
HIGH_NAMES = H1_NAMES + H2_NAMES

# Adjacent captures remain together wherever sample count permits.  Every high
# source appears in exactly one held partition and never in that fold's fit set.
FOLD_PLAN = (
    {
        "fold_id": "h1_holdout_044_045",
        "domain": "h1",
        "held_names": H1_NAMES[:2],
    },
    {
        "fold_id": "h1_holdout_046_047",
        "domain": "h1",
        "held_names": H1_NAMES[2:],
    },
    {
        "fold_id": "h2_holdout_088_091",
        "domain": "h2",
        "held_names": H2_NAMES[:4],
    },
    {
        "fold_id": "h2_holdout_092_094",
        "domain": "h2",
        "held_names": H2_NAMES[4:7],
    },
    {
        "fold_id": "h2_holdout_095_098",
        "domain": "h2",
        "held_names": H2_NAMES[7:],
    },
)

PERSISTENCE_FEATURE_NAMES = (
    "pixel_log_events_mean",
    "pixel_log_events_max",
    "pixel_active_fraction_mean",
    "pixel_active_fraction_max",
    "pixel_longest_run_fraction_mean",
    "pixel_longest_run_fraction_max",
    "pixel_collision_fraction_mean",
    "pixel_collision_fraction_max",
    "pixel_log_max_bin_events_mean",
    "pixel_log_max_bin_events_max",
    "pixel_polarity_dominance_mean",
    "pixel_polarity_dominance_max",
    "neighbor_active_fraction_mean",
    "neighbor_active_fraction_max",
)

# This compact, frozen grid tests whether lifetime features add information
# beyond the existing component features.  Candidate selection is reported but
# no candidate is promoted from this script.
CANDIDATES = (
    {
        "candidate_id": "legacy_pw08_kp040_control",
        "family": "legacy",
        "positive_weight": 8.0,
        "keep_probability": 0.40,
    },
    {
        "candidate_id": "persistence_pw08_kp050",
        "family": "persistence",
        "positive_weight": 8.0,
        "keep_probability": 0.50,
    },
    {
        "candidate_id": "persistence_pw16_kp050",
        "family": "persistence",
        "positive_weight": 16.0,
        "keep_probability": 0.50,
    },
    {
        "candidate_id": "combined_pw08_kp050",
        "family": "combined",
        "positive_weight": 8.0,
        "keep_probability": 0.50,
    },
    {
        "candidate_id": "combined_pw16_kp050",
        "family": "combined",
        "positive_weight": 16.0,
        "keep_probability": 0.50,
    },
    {
        "candidate_id": "combined_pw16_kp060",
        "family": "combined",
        "positive_weight": 16.0,
        "keep_probability": 0.60,
    },
    {
        "candidate_id": "combined_pw32_kp050",
        "family": "combined",
        "positive_weight": 32.0,
        "keep_probability": 0.50,
    },
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Output exists; refusing to overwrite: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_mapping(paths):
    return {str(name): sha256_file(path) for name, path in paths.items()}


def _read_reference_protocol(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    definition = payload.get("definition")
    if not isinstance(definition, dict):
        raise ValueError("Reference protocol has no definition object.")
    config = definition.get("config")
    dataset = definition.get("dataset")
    if not isinstance(config, dict) or not isinstance(dataset, dict):
        raise ValueError("Reference protocol lacks config/dataset bindings.")
    overrides = config.get("overrides")
    if not isinstance(overrides, list) or not all(isinstance(x, str) for x in overrides):
        raise ValueError("Reference config overrides are invalid.")
    if dataset.get("dataset_split") != DATASET_SPLIT:
        raise ValueError("Reference protocol is not train-only.")
    return payload, overrides


def _validate_train_paths(raw_train_dir, cache_dir):
    raw_train_dir = Path(raw_train_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    if raw_train_dir.name.lower() != DATASET_SPLIT:
        raise ValueError("--raw-train-dir must name the official train directory.")
    forbidden = {"val", "validation", "test"}
    for label, path in (("raw train", raw_train_dir), ("cache", cache_dir)):
        lowered = {part.lower() for part in path.parts}
        if label == "raw train":
            lowered.discard(DATASET_SPLIT)
        if lowered & forbidden:
            raise ValueError("{} path contains a forbidden split token: {}".format(label, path))
    if not raw_train_dir.is_dir():
        raise NotADirectoryError(raw_train_dir)
    if not cache_dir.is_dir():
        raise NotADirectoryError(cache_dir)
    return raw_train_dir, cache_dir


def _domain_for_name(name):
    if name in H1_NAMES:
        return "h1"
    if name in H2_NAMES:
        return "h2"
    raise ValueError("Source is outside the frozen high-density population: {}".format(name))


def _observable_domain(polarity_minority_fraction):
    return (
        "h1"
        if float(polarity_minority_fraction) < DOMAIN_POLARITY_MINORITY_CUTOFF
        else "h2"
    )


def _longest_active_runs(unique_pair_ids, pixel_count):
    pair_ids = np.asarray(unique_pair_ids, dtype=np.int64).reshape(-1)
    if pair_ids.size == 0:
        return np.zeros(pixel_count, dtype=np.int16)
    pixels = np.floor_divide(pair_ids, TEMPORAL_BIN_COUNT)
    bins = np.remainder(pair_ids, TEMPORAL_BIN_COUNT)
    starts = np.ones(pair_ids.size, dtype=bool)
    starts[1:] = (pixels[1:] != pixels[:-1]) | (bins[1:] != bins[:-1] + 1)
    run_ids = np.cumsum(starts, dtype=np.int64) - 1
    run_lengths = np.bincount(run_ids)
    run_pixels = pixels[starts]
    longest = np.zeros(pixel_count, dtype=np.int16)
    np.maximum.at(longest, run_pixels, run_lengths.astype(np.int16, copy=False))
    return longest


@dataclass(frozen=True)
class PixelPrior:
    event_pixel_ids: np.ndarray
    log_events: np.ndarray
    active_fraction: np.ndarray
    longest_run_fraction: np.ndarray
    collision_fraction: np.ndarray
    log_max_bin_events: np.ndarray
    polarity_dominance: np.ndarray
    neighbor_active_fraction: np.ndarray
    summary: dict


def derive_pixel_prior(raw_path, expected_locations):
    """Derive label-free per-pixel statistics from x/y/t/p only."""
    raw_path = Path(raw_path).resolve()
    with np.load(raw_path, allow_pickle=False) as source:
        if "ev_loc" not in source.files or "ev" not in source.files:
            raise ValueError("Raw source lacks ev_loc/ev: {}".format(raw_path))
        locations = np.ascontiguousarray(source["ev_loc"], dtype=np.int64)
        events = source["ev"]
        if events.dtype.names is None or "p" not in events.dtype.names:
            raise ValueError("Raw ev array lacks polarity field: {}".format(raw_path))
        polarity = np.ascontiguousarray(events["p"] > 0, dtype=np.uint8)
    expected_locations = np.asarray(expected_locations, dtype=np.int64)
    if locations.shape != expected_locations.shape or not np.array_equal(
        locations, expected_locations
    ):
        raise ValueError("Raw/cache location mismatch: {}".format(raw_path.name))
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("ev_loc must have shape [N,3].")
    if locations.size and (
        locations[:, 0].min() < 0
        or locations[:, 0].max() >= WIDTH
        or locations[:, 1].min() < 0
        or locations[:, 1].max() >= HEIGHT
        or locations[:, 2].min() < 0
        or locations[:, 2].max() >= VIDEO_DURATION
    ):
        raise ValueError("Raw locations exceed the frozen resolution/duration.")

    total_pixels = WIDTH * HEIGHT
    pixel_ids = locations[:, 1] * WIDTH + locations[:, 0]
    temporal_bins = np.floor_divide(locations[:, 2], TEMPORAL_BIN_SIZE)
    pair_ids = pixel_ids * TEMPORAL_BIN_COUNT + temporal_bins
    unique_pairs, pair_event_counts = np.unique(pair_ids, return_counts=True)
    pair_pixels = np.floor_divide(unique_pairs, TEMPORAL_BIN_COUNT)

    event_counts = np.bincount(pixel_ids, minlength=total_pixels).astype(np.int64)
    active_bins = np.bincount(pair_pixels, minlength=total_pixels).astype(np.int64)
    max_bin_events = np.zeros(total_pixels, dtype=np.int64)
    np.maximum.at(max_bin_events, pair_pixels, pair_event_counts)
    longest_runs = _longest_active_runs(unique_pairs, total_pixels)
    positive_counts = np.bincount(
        pixel_ids, weights=polarity.astype(np.float64), minlength=total_pixels
    )
    nonzero = event_counts > 0

    active_fraction = active_bins.astype(np.float64) / TEMPORAL_BIN_COUNT
    longest_fraction = longest_runs.astype(np.float64) / TEMPORAL_BIN_COUNT
    collision_fraction = np.zeros(total_pixels, dtype=np.float64)
    collision_fraction[nonzero] = 1.0 - (
        active_bins[nonzero].astype(np.float64) / event_counts[nonzero]
    )
    positive_fraction = np.zeros(total_pixels, dtype=np.float64)
    positive_fraction[nonzero] = positive_counts[nonzero] / event_counts[nonzero]
    polarity_dominance = np.abs(2.0 * positive_fraction - 1.0)
    polarity_dominance[~nonzero] = 0.0
    active_image = active_fraction.reshape(HEIGHT, WIDTH).astype(np.float32)
    neighbor_active = cv2.boxFilter(
        active_image,
        ddepth=-1,
        ksize=(3, 3),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    ).reshape(-1).astype(np.float64)

    minority_fraction = float(min(polarity.mean(), 1.0 - polarity.mean()))
    clipped_pairs = pair_event_counts > int(math.floor(math.expm1(LOG_COUNT_CLIP)))
    clipped_events = int(pair_event_counts[clipped_pairs].sum())
    active_values = active_bins[nonzero]
    longest_values = longest_runs[nonzero]
    summary = {
        "event_count": int(locations.shape[0]),
        "unique_pixel_count": int(nonzero.sum()),
        "unique_pixel_bin_count": int(unique_pairs.size),
        "pixel_bin_collision_fraction": float(1.0 - unique_pairs.size / locations.shape[0]),
        "polarity_minority_fraction": minority_fraction,
        "observable_domain": _observable_domain(minority_fraction),
        "log_count_clipped_pair_count": int(clipped_pairs.sum()),
        "log_count_clipped_event_fraction": float(clipped_events / locations.shape[0]),
        "active_bin_quantiles": {
            key: float(value)
            for key, value in zip(
                ("q50", "q90", "q99", "max"),
                np.quantile(active_values, (0.50, 0.90, 0.99, 1.0)),
            )
        },
        "longest_run_quantiles": {
            key: float(value)
            for key, value in zip(
                ("q50", "q90", "q99", "max"),
                np.quantile(longest_values, (0.50, 0.90, 0.99, 1.0)),
            )
        },
        "pixels_active_at_least_half_video": int(np.sum(active_bins >= TEMPORAL_BIN_COUNT / 2)),
    }
    return PixelPrior(
        event_pixel_ids=pixel_ids.astype(np.int64, copy=False),
        log_events=np.log1p(event_counts.astype(np.float64)),
        active_fraction=active_fraction,
        longest_run_fraction=longest_fraction,
        collision_fraction=collision_fraction,
        log_max_bin_events=np.log1p(max_bin_events.astype(np.float64)),
        polarity_dominance=polarity_dominance,
        neighbor_active_fraction=neighbor_active,
        summary=summary,
    )


def component_persistence_features(prior, event_indices):
    rows = []
    fields = (
        prior.log_events,
        prior.active_fraction,
        prior.longest_run_fraction,
        prior.collision_fraction,
        prior.log_max_bin_events,
        prior.polarity_dominance,
        prior.neighbor_active_fraction,
    )
    for indices in event_indices:
        pixels = prior.event_pixel_ids[np.asarray(indices, dtype=np.int64)]
        values = []
        for field in fields:
            selected = field[pixels]
            values.extend((float(selected.mean()), float(selected.max())))
        rows.append(values)
    result = np.asarray(rows, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(PERSISTENCE_FEATURE_NAMES):
        raise RuntimeError("Persistent-pixel feature width mismatch.")
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite persistent-pixel feature derived.")
    return result


@dataclass
class PriorVideo:
    prepared: component_crossfit.PreparedVideo
    persistence_features: np.ndarray
    input_summary: dict

    @property
    def source_name(self):
        return self.prepared.source_name

    @property
    def domain(self):
        return self.prepared.block


def _family_features(video, family):
    if family == "legacy":
        return video.prepared.features
    if family == "persistence":
        return video.persistence_features
    if family == "combined":
        return np.concatenate(
            (video.prepared.features, video.persistence_features), axis=1
        )
    raise ValueError("Unknown feature family: {}".format(family))


def _balanced_dataset(videos, family):
    features = []
    labels = []
    weights = []
    video_mass = 1.0 / len(videos)
    for video in videos:
        values = _family_features(video, family)
        count = values.shape[0]
        if count <= 0:
            raise ValueError("Every fit video must have component candidates.")
        features.append(values)
        labels.append(video.prepared.component_labels)
        weights.append(np.full(count, video_mass / count, dtype=np.float64))
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(weights),
    )


def _weighted_logistic_loss(design, labels, weights, parameters, l2):
    logits = design @ parameters
    losses = np.logaddexp(0.0, logits) - labels * logits
    return float(
        np.dot(weights, losses) / weights.sum()
        + 0.5 * l2 * np.dot(parameters[:-1], parameters[:-1])
    )


def _fit_balanced_logistic(
    features,
    labels,
    base_weights,
    positive_weight,
    l2=0.1,
    max_iterations=50,
):
    """Width-agnostic equivalent of the frozen reranker Newton fit."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    base_weights = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("Balanced fit features/labels have incompatible shapes.")
    if base_weights.size != labels.size or (base_weights <= 0).any():
        raise ValueError("Balanced fit base weights are invalid.")
    if not np.isfinite(features).all() or not np.isfinite(base_weights).all():
        raise ValueError("Balanced fit inputs must be finite.")
    if not np.isin(labels, (0.0, 1.0)).all() or np.unique(labels).size != 2:
        raise ValueError("Balanced fit labels must contain both binary classes.")
    if not math.isfinite(float(positive_weight)) or positive_weight <= 0:
        raise ValueError("positive_weight must be finite and positive.")

    normalized_base = base_weights / base_weights.sum()
    feature_mean = np.sum(features * normalized_base[:, None], axis=0)
    centered = features - feature_mean
    feature_scale = np.sqrt(
        np.sum(centered * centered * normalized_base[:, None], axis=0)
    )
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = centered / feature_scale
    design = np.column_stack(
        (standardized, np.ones(labels.size, dtype=np.float64))
    )
    sample_weights = base_weights * np.where(labels > 0.5, positive_weight, 1.0)
    weighted_positives = float(np.dot(sample_weights, labels))
    weighted_negatives = float(np.dot(sample_weights, 1.0 - labels))
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    parameters[-1] = math.log(
        max(weighted_positives, 1e-12) / max(weighted_negatives, 1e-12)
    )
    converged = False
    iterations = 0
    for iteration in range(int(max_iterations)):
        iterations = iteration + 1
        logits = design @ parameters
        probabilities = np.empty_like(logits)
        nonnegative = logits >= 0
        probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
        exp_negative = np.exp(logits[~nonnegative])
        probabilities[~nonnegative] = exp_negative / (1.0 + exp_negative)
        normalization = sample_weights.sum()
        gradient = (
            design.T @ (sample_weights * (probabilities - labels)) / normalization
        )
        gradient[:-1] += l2 * parameters[:-1]
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design / normalization
        hessian[:-1, :-1] += np.eye(features.shape[1]) * l2
        hessian[-1, -1] += 1e-12
        try:
            newton_step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            newton_step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if np.max(np.abs(newton_step)) < 1e-9:
            converged = True
            break
        current_loss = _weighted_logistic_loss(
            design, labels, sample_weights, parameters, l2
        )
        step_scale = 1.0
        accepted = False
        while step_scale >= 2.0**-20:
            candidate = parameters - step_scale * newton_step
            candidate_loss = _weighted_logistic_loss(
                design, labels, sample_weights, candidate, l2
            )
            if candidate_loss <= current_loss:
                parameters = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        if abs(current_loss - candidate_loss) < 1e-12:
            converged = True
            break
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "coefficients": parameters[:-1],
        "intercept": float(parameters[-1]),
        "iterations": iterations,
        "converged": converged,
        "weighted_loss": _weighted_logistic_loss(
            design, labels, sample_weights, parameters, l2
        ),
        "positive_weight": float(positive_weight),
    }


def _fit_models(fit_videos):
    specifications = sorted(
        {
            (candidate["family"], float(candidate["positive_weight"]))
            for candidate in CANDIDATES
        }
    )
    models = {}
    for family, positive_weight in specifications:
        features, labels, base_weights = _balanced_dataset(fit_videos, family)
        fitted = _fit_balanced_logistic(
            features,
            labels,
            base_weights,
            positive_weight,
        )
        models[(family, positive_weight)] = fitted
    return models


def _model_summary(fitted, family):
    if family == "legacy":
        names = component_crossfit.FEATURE_NAMES
    elif family == "persistence":
        names = PERSISTENCE_FEATURE_NAMES
    else:
        names = component_crossfit.FEATURE_NAMES + PERSISTENCE_FEATURE_NAMES
    standardized_coefficients = fitted["coefficients"]
    ranked = sorted(
        zip(names, standardized_coefficients), key=lambda item: -abs(float(item[1]))
    )
    return {
        "family": family,
        "feature_names": list(names),
        "feature_mean": fitted["feature_mean"].tolist(),
        "feature_scale": fitted["feature_scale"].tolist(),
        "coefficients": fitted["coefficients"].tolist(),
        "intercept": float(fitted["intercept"]),
        "positive_weight": float(fitted["positive_weight"]),
        "iterations": int(fitted["iterations"]),
        "converged": bool(fitted["converged"]),
        "weighted_loss": float(fitted["weighted_loss"]),
        "top_absolute_coefficients": [
            {"feature": name, "coefficient": float(value)}
            for name, value in ranked[:8]
        ],
    }


def _candidate_counts(video, fitted, candidate, cfg):
    prepared = video.prepared
    probabilities = component_crossfit._predict_probabilities(
        _family_features(video, candidate["family"]), fitted
    )
    keep = probabilities >= float(candidate["keep_probability"])
    scores = prepared.p0_scores.copy()
    removed_components = 0
    removed_events = 0
    for indices, keep_component in zip(prepared.event_indices, keep):
        if not keep_component:
            scores[indices] = 0.0
            removed_components += 1
            removed_events += int(len(indices))
    recovery = P18ScoreTrackRecovery.from_cfg(cfg, PREDICTION_THRESHOLD)
    score_tensor, _ = recovery.apply(
        torch.from_numpy(scores),
        torch.from_numpy(prepared.locations.astype(np.int64, copy=False)),
    )
    counts = component_crossfit.sufficient_counts_for_video(
        score_tensor.numpy(),
        prepared.event_labels,
        prepared.target_ids,
        prepared.locations,
    )
    return counts, {
        "candidate_component_count": int(keep.size),
        "removed_candidate_components": removed_components,
        "removed_candidate_events": removed_events,
    }


def _sum_counts(values):
    result = component_crossfit.SufficientCounts()
    for value in values:
        result = result + value
    return result


def _count_delta(candidate, baseline):
    return {
        field: int(getattr(candidate, field) - getattr(baseline, field))
        for field in candidate.__dataclass_fields__
    }


def _metric_delta(candidate, baseline):
    return {key: float(candidate[key] - baseline[key]) for key in candidate}


def _evaluate_fold(fold, videos, cfg):
    domain_videos = [video for video in videos if video.domain == fold["domain"]]
    held_names = set(fold["held_names"])
    held = [video for video in domain_videos if video.source_name in held_names]
    fit = [video for video in domain_videos if video.source_name not in held_names]
    if {video.source_name for video in held} != held_names:
        raise RuntimeError("Fold held-source population mismatch.")
    if not fit or not held:
        raise RuntimeError("Fold fit/held partition is empty.")
    if {video.source_name for video in fit} & held_names:
        raise RuntimeError("Source leakage in grouped OOF fold.")

    models = _fit_models(fit)
    baseline_counts = _sum_counts(video.prepared.baseline_counts for video in held)
    baseline_metrics = component_crossfit.metrics_from_counts(baseline_counts)
    candidate_results = []
    for candidate in CANDIDATES:
        fitted = models[(candidate["family"], candidate["positive_weight"])]
        per_video = []
        candidate_counts = component_crossfit.SufficientCounts()
        removed_components = 0
        removed_events = 0
        for video in held:
            counts, changes = _candidate_counts(video, fitted, candidate, cfg)
            candidate_counts = candidate_counts + counts
            removed_components += changes["removed_candidate_components"]
            removed_events += changes["removed_candidate_events"]
            per_video.append(
                {
                    "source_name": video.source_name,
                    "baseline_counts": video.prepared.baseline_counts.to_dict(),
                    "candidate_counts": counts.to_dict(),
                    "count_delta": _count_delta(
                        counts, video.prepared.baseline_counts
                    ),
                    **changes,
                }
            )
        metrics = component_crossfit.metrics_from_counts(candidate_counts)
        model = _model_summary(fitted, candidate["family"])
        candidate_results.append(
            {
                **candidate,
                "fit_model_sha256": sha256_json(model),
                "fit_model": model,
                "counts": candidate_counts.to_dict(),
                "metrics": metrics,
                "count_delta": _count_delta(candidate_counts, baseline_counts),
                "metric_delta": _metric_delta(metrics, baseline_metrics),
                "removed_candidate_components": removed_components,
                "removed_candidate_events": removed_events,
                "per_video": per_video,
            }
        )
    return {
        "fold_id": fold["fold_id"],
        "domain": fold["domain"],
        "fit_video_names": [video.source_name for video in fit],
        "held_video_names": [video.source_name for video in held],
        "baseline": {
            "counts": baseline_counts.to_dict(),
            "metrics": baseline_metrics,
        },
        "candidate_results": candidate_results,
    }


def _candidate_by_id(fold, candidate_id):
    return next(
        result
        for result in fold["candidate_results"]
        if result["candidate_id"] == candidate_id
    )


def _pooled_results(folds):
    baseline_counts = _sum_counts(
        component_crossfit.SufficientCounts(**fold["baseline"]["counts"])
        for fold in folds
    )
    baseline_metrics = component_crossfit.metrics_from_counts(baseline_counts)
    candidates = []
    for definition in CANDIDATES:
        candidate_id = definition["candidate_id"]
        counts = _sum_counts(
            component_crossfit.SufficientCounts(
                **_candidate_by_id(fold, candidate_id)["counts"]
            )
            for fold in folds
        )
        metrics = component_crossfit.metrics_from_counts(counts)
        fold_score_deltas = [
            float(_candidate_by_id(fold, candidate_id)["metric_delta"]["score"])
            for fold in folds
        ]
        candidates.append(
            {
                **definition,
                "counts": counts.to_dict(),
                "metrics": metrics,
                "count_delta": _count_delta(counts, baseline_counts),
                "metric_delta": _metric_delta(metrics, baseline_metrics),
                "false_component_reduction_fraction": float(
                    (baseline_counts.false_components - counts.false_components)
                    / baseline_counts.false_components
                ),
                "nonnegative_score_fold_count": int(
                    sum(delta >= 0.0 for delta in fold_score_deltas)
                ),
                "positive_score_fold_count": int(
                    sum(delta > 0.0 for delta in fold_score_deltas)
                ),
                "fold_score_deltas": fold_score_deltas,
            }
        )
    winner = sorted(
        candidates,
        key=lambda result: (
            -float(result["metrics"]["score"]), result["candidate_id"]
        ),
    )[0]
    conservative_gates = {
        "minimum_pooled_score_delta": 0.0002,
        "minimum_nonnegative_score_folds": 4,
        "minimum_false_component_reduction_fraction": 0.01,
        "require_zero_true_positive_event_loss": True,
        "require_zero_correct_object_loss": True,
        "require_nonnegative_iou_delta": True,
        "require_nonnegative_pd_delta": True,
    }
    conservative_eligible = []
    for result in candidates:
        checks = {
            "pooled_score_delta": result["metric_delta"]["score"]
            >= conservative_gates["minimum_pooled_score_delta"],
            "fold_consistency": result["nonnegative_score_fold_count"]
            >= conservative_gates["minimum_nonnegative_score_folds"],
            "false_component_reduction": result[
                "false_component_reduction_fraction"
            ]
            >= conservative_gates[
                "minimum_false_component_reduction_fraction"
            ],
            "true_positive_events_preserved": result["count_delta"][
                "true_positive_events"
            ]
            == 0,
            "correct_objects_preserved": result["count_delta"][
                "correct_objects"
            ]
            == 0,
            "iou_nonnegative": result["metric_delta"]["iou"] >= 0.0,
            "pd_nonnegative": result["metric_delta"]["pd"] >= 0.0,
        }
        result["conservative_gate_checks"] = checks
        result["conservative_gate_passed"] = all(checks.values())
        if result["conservative_gate_passed"]:
            conservative_eligible.append(result)
    conservative_winner = None
    if conservative_eligible:
        conservative_winner = sorted(
            conservative_eligible,
            key=lambda result: (
                -float(result["metrics"]["score"]), result["candidate_id"]
            ),
        )[0]["candidate_id"]
    return {
        "baseline": {
            "counts": baseline_counts.to_dict(),
            "metrics": baseline_metrics,
        },
        "candidates": candidates,
        "exploratory_winner_candidate_id": winner["candidate_id"],
        "conservative_gates": conservative_gates,
        "conservative_winner_candidate_id": conservative_winner,
        "selection_warning": (
            "The winner is selected on pooled train-only OOF and is not an "
            "independent generalization estimate. Freeze before any 24-val replay."
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--raw-train-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def run(args):
    raw_train_dir, cache_dir = _validate_train_paths(
        args.raw_train_dir, args.cache_dir
    )
    output_path = Path(args.output_json).resolve()
    if output_path.exists():
        raise FileExistsError("Output exists; refusing to overwrite: {}".format(output_path))
    reference_payload, overrides = _read_reference_protocol(args.reference_protocol)
    config_path = Path(args.config).resolve()
    expected_config_sha = reference_payload["definition"]["config"].get("sha256")
    current_config_sha = sha256_file(config_path)
    cfg = replay.load_flat_config(config_path, overrides)
    component_crossfit.validate_c00_config(cfg)

    cache_dir, manifest_path, manifest_sha, manifest = load_train_cache(cache_dir)
    records_by_name = {record["source_name"]: record for record in manifest["records"]}
    if not set(HIGH_NAMES).issubset(records_by_name):
        raise ValueError("Train cache lacks the frozen H1/H2 source population.")
    selected_manifest = dict(manifest)
    selected_manifest["records"] = [records_by_name[name] for name in HIGH_NAMES]
    prepared = component_crossfit._prepare_videos(
        cache_dir, selected_manifest, cfg, middle_names=set()
    )

    videos = []
    input_summaries = []
    for index, video in enumerate(prepared, start=1):
        metadata = records_by_name[video.source_name]
        raw_path = raw_train_dir / video.source_name
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if sha256_file(raw_path) != metadata["source_sha256"]:
            raise ValueError("Raw source SHA-256 mismatch: {}".format(raw_path))
        prior = derive_pixel_prior(raw_path, video.locations[:, 1:4])
        expected_domain = _domain_for_name(video.source_name)
        if prior.summary["observable_domain"] != expected_domain:
            raise RuntimeError(
                "Input-only H1/H2 route mismatch for {}.".format(video.source_name)
            )
        features = component_persistence_features(prior, video.event_indices)
        videos.append(PriorVideo(video, features, prior.summary))
        input_summaries.append(
            {
                "source_name": video.source_name,
                "expected_domain": expected_domain,
                **prior.summary,
            }
        )
        print(
            "prior {}/{}: {} [{}] collision={:.4f} minority={:.4f}".format(
                index,
                len(prepared),
                video.source_name,
                expected_domain,
                prior.summary["pixel_bin_collision_fraction"],
                prior.summary["polarity_minority_fraction"],
            ),
            flush=True,
        )

    folds = []
    for fold in FOLD_PLAN:
        result = _evaluate_fold(fold, videos, cfg)
        folds.append(result)
        print("evaluated fold:", fold["fold_id"], flush=True)
    pooled = _pooled_results(folds)

    code_paths = {
        "crossfit_persistent_pixel_prior.py": Path(__file__).resolve(),
        "crossfit_component_reranker.py": Path(component_crossfit.__file__).resolve(),
        "utils/component_reranker.py": Path(__file__).resolve().parent
        / "utils"
        / "component_reranker.py",
        "utils/postprocess.py": Path(__file__).resolve().parent
        / "utils"
        / "postprocess.py",
        "utils/eval.py": Path(__file__).resolve().parent / "utils" / "eval.py",
        "utils/challenge_eval.py": Path(__file__).resolve().parent
        / "utils"
        / "challenge_eval.py",
    }
    payload = {
        "schema": REPORT_SCHEMA,
        "created_utc": _utc_now(),
        "evidence_class": (
            "train_only_grouped_oof_candidate_selection_not_independent_final_estimate"
        ),
        "split_access": {
            "consumed": ["train raw x/y/t/p", "train M20 score cache", "train labels"],
            "forbidden": ["val", "test"],
            "validation_or_test_read": False,
        },
        "protocol": {
            "prediction_threshold": PREDICTION_THRESHOLD,
            "temporal_bin_size": TEMPORAL_BIN_SIZE,
            "temporal_bin_count": TEMPORAL_BIN_COUNT,
            "resolution": [WIDTH, HEIGHT],
            "domain_route": {
                "observable": "complete-video polarity minority fraction",
                "h1_operator": "<",
                "cutoff": DOMAIN_POLARITY_MINORITY_CUTOFF,
                "labels_used": False,
            },
            "fold_plan": [
                {
                    **fold,
                    "held_names": list(fold["held_names"]),
                }
                for fold in FOLD_PLAN
            ],
            "legacy_feature_names": list(component_crossfit.FEATURE_NAMES),
            "persistence_feature_names": list(PERSISTENCE_FEATURE_NAMES),
            "candidates": list(CANDIDATES),
        },
        "provenance": {
            "cache_manifest_path": str(manifest_path),
            "cache_manifest_sha256": manifest_sha,
            "reference_protocol_path": str(Path(args.reference_protocol).resolve()),
            "reference_protocol_sha256": sha256_file(args.reference_protocol),
            "config_path": str(config_path),
            "config_sha256": current_config_sha,
            "reference_config_sha256": expected_config_sha,
            "config_file_matches_reference": current_config_sha
            == expected_config_sha,
            "code_sha256": _sha256_mapping(code_paths),
            "selected_raw_source_sha256": {
                name: records_by_name[name]["source_sha256"] for name in HIGH_NAMES
            },
        },
        "input_domain_statistics": input_summaries,
        "fold_results": folds,
        "pooled_oof": pooled,
    }
    _atomic_json(output_path, payload)
    print("wrote:", output_path)
    print("sha256:", sha256_file(output_path))
    return payload


def main(argv=None):
    return 0 if run(parse_args(argv)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
