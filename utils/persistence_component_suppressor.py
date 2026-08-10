"""Standalone prediction-only persistence suppressor for full-stream H2 scores.

The runtime accepts one complete video's frozen P0/P0c score vector plus
observable x/y/t/p input.  The caller applies frozen P18 afterward.  It returns
the exact input object for every non-H2 route and does not invoke component
extraction there.  It has no T32 input or blend API.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np

from utils.component_reranker import (
    ComponentTopology,
    extract_component_examples,
    sha256_file,
)
from utils.temporal_memory_input_router import (
    HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE,
    POLARITY_MINORITY_CUTOFF,
)


ARTIFACT_SCHEMA = "ev-uav-persistence-component-suppressor-v1"
FEATURE_SEMANTICS_VERSION = "persistent-pixel-lifetime-component-v1"
PREDICTION_THRESHOLD = 0.719
WIDTH = 346
HEIGHT = 260
TEMPORAL_BIN_SIZE = 50
VIDEO_DURATION = 8000
TEMPORAL_BIN_COUNT = VIDEO_DURATION // TEMPORAL_BIN_SIZE
LOG_COUNT_CLIP = 4.0
FEATURE_NAMES = (
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
DEFAULT_TOPOLOGY = ComponentTopology(
    spatial_radius=1,
    temporal_bin_size=50,
    max_link_distance=6.0,
    max_gap_bins=1,
    max_component_events=3,
)


def _as_numpy(value):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _locations_xyt(locations, expected_count):
    values = _as_numpy(locations)
    if values.ndim != 2 or values.shape[0] != int(expected_count):
        raise ValueError("locations must align with the complete score vector.")
    if values.shape[1] == 3:
        xyt = values
    elif values.shape[1] >= 4:
        if np.unique(values[:, 0]).size != 1:
            raise ValueError("Standalone runtime accepts exactly one complete video.")
        xyt = values[:, 1:4]
    else:
        raise ValueError("locations must be [N,3] x/y/t or [N,4+] batch/x/y/t.")
    return np.ascontiguousarray(xyt, dtype=np.int64)


def _polarity_values(polarities, expected_count):
    values = _as_numpy(polarities).reshape(-1)
    if values.size != int(expected_count):
        raise ValueError("polarities must align with the complete score vector.")
    if values.size <= 0:
        raise ValueError("A complete video must contain at least one event.")
    return np.ascontiguousarray(values > 0, dtype=np.uint8)


def polarity_minority_fraction(polarities):
    values = np.asarray(polarities, dtype=np.uint8).reshape(-1)
    if values.size <= 0:
        raise ValueError("Cannot route an empty video.")
    positive_fraction = float(values.mean())
    return float(min(positive_fraction, 1.0 - positive_fraction))


def observable_route(event_count, polarities):
    event_count = int(event_count)
    values = _polarity_values(polarities, event_count)
    minority = polarity_minority_fraction(values)
    eligible = bool(
        event_count > HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE
        and minority >= POLARITY_MINORITY_CUTOFF
    )
    return {
        "route": "h2" if eligible else "non_h2",
        "eligible": eligible,
        "event_count": event_count,
        "event_count_cutoff_exclusive": HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE,
        "polarity_minority_fraction": minority,
        "polarity_minority_cutoff": POLARITY_MINORITY_CUTOFF,
    }


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


def derive_pixel_prior_from_arrays(locations, polarities):
    """Derive the frozen label-free lifetime fields from one x/y/t/p video."""
    polarities = np.asarray(polarities).reshape(-1)
    locations = _locations_xyt(locations, polarities.size)
    polarity = _polarity_values(polarities, locations.shape[0])
    if (
        locations[:, 0].min() < 0
        or locations[:, 0].max() >= WIDTH
        or locations[:, 1].min() < 0
        or locations[:, 1].max() >= HEIGHT
        or locations[:, 2].min() < 0
        or locations[:, 2].max() >= VIDEO_DURATION
    ):
        raise ValueError("Locations exceed the frozen resolution or duration.")

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

    minority = polarity_minority_fraction(polarity)
    clipped_pairs = pair_event_counts > int(math.floor(math.expm1(LOG_COUNT_CLIP)))
    clipped_events = int(pair_event_counts[clipped_pairs].sum())
    active_values = active_bins[nonzero]
    longest_values = longest_runs[nonzero]
    summary = {
        "event_count": int(locations.shape[0]),
        "unique_pixel_count": int(nonzero.sum()),
        "unique_pixel_bin_count": int(unique_pairs.size),
        "pixel_bin_collision_fraction": float(1.0 - unique_pairs.size / locations.shape[0]),
        "polarity_minority_fraction": minority,
        # Preserve the frozen train-audit summary semantics.  Deployment routing
        # additionally applies the event-count gate in ``observable_route``.
        "observable_domain": (
            "h1" if minority < POLARITY_MINORITY_CUTOFF else "h2"
        ),
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
        "pixels_active_at_least_half_video": int(
            np.sum(active_bins >= TEMPORAL_BIN_COUNT / 2)
        ),
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
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    result = np.asarray(rows, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("Persistence feature width mismatch.")
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite persistence feature derived.")
    return result


def extract_persistence_components(
    p0_p0c_full_scores,
    locations,
    polarities,
    topology=DEFAULT_TOPOLOGY,
    prediction_threshold=PREDICTION_THRESHOLD,
):
    scores = np.asarray(_as_numpy(p0_p0c_full_scores), dtype=np.float32).reshape(-1)
    xyt = _locations_xyt(locations, scores.size)
    polarity = _polarity_values(polarities, scores.size)
    locations4 = np.column_stack((np.zeros(scores.size, dtype=np.int64), xyt))
    examples = extract_component_examples(
        scores,
        locations4,
        float(prediction_threshold),
        topology,
        scores.size,
        labels=None,
    )
    event_indices = tuple(
        np.asarray(example.event_indices, dtype=np.int64) for example in examples
    )
    prior = derive_pixel_prior_from_arrays(xyt, polarity)
    features = component_persistence_features(prior, event_indices)
    return event_indices, features, prior.summary


@dataclass(frozen=True)
class PersistenceArtifact:
    artifact_sha256: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    keep_probability: float
    positive_weight: float
    topology: ComponentTopology

    @classmethod
    def from_payload(cls, payload, artifact_sha256="synthetic"):
        if not isinstance(payload, dict) or payload.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError("Persistence artifact schema differs.")
        if payload.get("candidate_id") != "persistence_pw08_kp050":
            raise ValueError("Persistence artifact candidate differs.")
        if payload.get("feature_semantics_version") != FEATURE_SEMANTICS_VERSION:
            raise ValueError("Persistence feature semantics differ.")
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Persistence feature names differ.")
        contract = payload.get("runtime_contract", {})
        if (
            contract.get("t32_allowed") is not False
            or contract.get("prediction_threshold") != PREDICTION_THRESHOLD
            or contract.get("event_count_cutoff_exclusive")
            != HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE
            or contract.get("polarity_minority_cutoff") != POLARITY_MINORITY_CUTOFF
        ):
            raise ValueError("Persistence runtime contract differs.")
        model = payload.get("model", {})
        feature_mean = np.asarray(model.get("feature_mean"), dtype=np.float64)
        feature_scale = np.asarray(model.get("feature_scale"), dtype=np.float64)
        coefficients = np.asarray(model.get("coefficients"), dtype=np.float64)
        width = len(FEATURE_NAMES)
        if any(value.shape != (width,) for value in (feature_mean, feature_scale, coefficients)):
            raise ValueError("Persistence model vector width differs.")
        if not all(
            np.isfinite(value).all()
            for value in (feature_mean, feature_scale, coefficients)
        ) or np.any(feature_scale <= 0):
            raise ValueError("Persistence model vectors are invalid.")
        intercept = float(model.get("intercept"))
        keep_probability = float(model.get("keep_probability"))
        positive_weight = float(model.get("positive_weight"))
        if (
            not math.isfinite(intercept)
            or keep_probability != 0.5
            or positive_weight != 8.0
        ):
            raise ValueError("Persistence model scalar contract differs.")
        topology = ComponentTopology.from_mapping(payload.get("component_topology"))
        if topology != DEFAULT_TOPOLOGY:
            raise ValueError("Persistence component topology differs.")
        return cls(
            artifact_sha256=str(artifact_sha256),
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            coefficients=coefficients,
            intercept=intercept,
            keep_probability=keep_probability,
            positive_weight=positive_weight,
            topology=topology,
        )

    @classmethod
    def load(cls, path, expected_sha256):
        path = Path(path).resolve()
        actual = sha256_file(path)
        if actual != str(expected_sha256).lower():
            raise ValueError("Persistence artifact SHA-256 differs.")
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if sha256_file(path) != actual:
            raise RuntimeError("Persistence artifact changed while being read.")
        return cls.from_payload(payload, artifact_sha256=actual)

    def predict_probabilities(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("Persistence feature matrix width differs.")
        standardized = (features - self.feature_mean) / self.feature_scale
        logits = standardized @ self.coefficients + self.intercept
        probabilities = np.empty_like(logits)
        nonnegative = logits >= 0
        probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
        negative_exp = np.exp(logits[~nonnegative])
        probabilities[~nonnegative] = negative_exp / (1.0 + negative_exp)
        return probabilities


@dataclass(frozen=True)
class PersistenceSuppressorStats:
    route: dict
    component_chain_called: bool
    candidate_component_count: int = 0
    kept_candidate_components: int = 0
    removed_candidate_components: int = 0
    removed_candidate_events: int = 0

    def to_dict(self):
        return {
            "route": dict(self.route),
            "component_chain_called": self.component_chain_called,
            "candidate_component_count": self.candidate_component_count,
            "kept_candidate_components": self.kept_candidate_components,
            "removed_candidate_components": self.removed_candidate_components,
            "removed_candidate_events": self.removed_candidate_events,
        }


class PersistenceComponentSuppressor:
    def __init__(self, artifact):
        if not isinstance(artifact, PersistenceArtifact):
            raise TypeError("artifact must be a PersistenceArtifact.")
        self.artifact = artifact

    def apply(self, p0_p0c_full_scores, locations, polarities):
        event_count = int(p0_p0c_full_scores.numel()) if hasattr(p0_p0c_full_scores, "numel") else int(np.asarray(p0_p0c_full_scores).size)
        xyt = _locations_xyt(locations, event_count)
        polarity = _polarity_values(polarities, event_count)
        route = observable_route(event_count, polarity)
        if not route["eligible"]:
            return p0_p0c_full_scores, PersistenceSuppressorStats(
                route=route, component_chain_called=False
            )

        scores_numpy = np.asarray(_as_numpy(p0_p0c_full_scores), dtype=np.float32).reshape(-1)
        if not np.isfinite(scores_numpy).all():
            raise ValueError("P0/P0c full-stream scores contain non-finite values.")
        event_indices, features, _ = extract_persistence_components(
            scores_numpy,
            xyt,
            polarity,
            topology=self.artifact.topology,
            prediction_threshold=PREDICTION_THRESHOLD,
        )
        probabilities = self.artifact.predict_probabilities(features)
        keep = probabilities >= self.artifact.keep_probability
        if hasattr(p0_p0c_full_scores, "clone"):
            output = p0_p0c_full_scores.clone()
            flattened = output.reshape(-1)
            for indices, keep_component in zip(event_indices, keep):
                if not keep_component:
                    flattened[indices.tolist()] = 0.0
        else:
            output = np.asarray(p0_p0c_full_scores).copy()
            flattened = output.reshape(-1)
            for indices, keep_component in zip(event_indices, keep):
                if not keep_component:
                    flattened[indices] = 0.0
        removed = [indices for indices, keep_component in zip(event_indices, keep) if not keep_component]
        output_numpy = np.asarray(_as_numpy(output), dtype=np.float32).reshape(-1)
        if np.any(output_numpy > scores_numpy):
            raise RuntimeError("Persistence suppression raised an input score.")
        changed = output_numpy != scores_numpy
        if np.any(output_numpy[changed] != 0.0):
            raise RuntimeError("Persistence suppression changed a score to a nonzero value.")
        return output, PersistenceSuppressorStats(
            route=route,
            component_chain_called=True,
            candidate_component_count=len(event_indices),
            kept_candidate_components=int(np.count_nonzero(keep)),
            removed_candidate_components=int(len(event_indices) - np.count_nonzero(keep)),
            removed_candidate_events=int(sum(indices.size for indices in removed)),
        )
