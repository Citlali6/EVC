"""Source-free relative features and positive-support guards for component retention.

The module deliberately does not know video names, source families, folds, labels,
or target identifiers at inference time.  Training code may pass a boolean mask to
``WeightedMarginalSupport.fit``; prediction consumes only observable component
features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils import contextual_deletion_head as contextual
from utils.allsize_deletion_head import AllSizeComponentBatch


CORE_EVIDENCE_NAMES = (
    "score_max",
    "score_mean",
    "log_component_events",
    "log_unique_cells",
    "track_bin_count",
    "log_track_events",
    "track_score_max",
    "graph_longest_path",
)

PERSISTENCE_EVIDENCE_NAMES = CORE_EVIDENCE_NAMES + (
    "track_score_mean",
    "cell_active_bins_mean",
    "cell_active_bins_max",
    "graph_prev_degree",
    "graph_next_degree",
    "gap_prev1_present",
    "gap_next1_present",
)

RELATIVE_BASE_NAMES = PERSISTENCE_EVIDENCE_NAMES
RELATIVE_FEATURE_NAMES = (
    ("frame_log_component_count",)
    + tuple(f"frame_percentile_{name}" for name in RELATIVE_BASE_NAMES)
    + tuple(f"video_percentile_{name}" for name in RELATIVE_BASE_NAMES)
    + tuple(f"frame_to_max_{name}" for name in RELATIVE_BASE_NAMES)
)
FEATURE_NAMES = contextual.FEATURE_NAMES + RELATIVE_FEATURE_NAMES

EVIDENCE_SETS = {
    "core_raw_relative": CORE_EVIDENCE_NAMES
    + tuple(f"frame_percentile_{name}" for name in CORE_EVIDENCE_NAMES)
    + tuple(f"video_percentile_{name}" for name in CORE_EVIDENCE_NAMES),
    "persistence_raw_relative": PERSISTENCE_EVIDENCE_NAMES
    + tuple(f"frame_percentile_{name}" for name in PERSISTENCE_EVIDENCE_NAMES)
    + tuple(f"video_percentile_{name}" for name in PERSISTENCE_EVIDENCE_NAMES),
    "relative_anchor": tuple(
        f"frame_percentile_{name}" for name in PERSISTENCE_EVIDENCE_NAMES
    )
    + tuple(f"video_percentile_{name}" for name in CORE_EVIDENCE_NAMES),
}


def _right_percentiles(values: np.ndarray) -> np.ndarray:
    """Return deterministic upper empirical ranks in (0, 1]."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    ordered = np.sort(values, kind="mergesort")
    return np.searchsorted(ordered, values, side="right") / float(values.size)


def _relative_features(features, event_indices, locations):
    features = np.asarray(features, dtype=np.float64)
    locations = np.asarray(locations)
    count = features.shape[0]
    if count != len(event_indices):
        raise ValueError("component features and index groups differ in length.")
    if count == 0:
        return np.empty((0, len(RELATIVE_FEATURE_NAMES)), dtype=np.float64)
    bins = np.asarray(
        [int(locations[np.asarray(indices, dtype=np.int64)[0], 3]) // 50 for indices in event_indices],
        dtype=np.int64,
    )
    contextual_index = {name: index for index, name in enumerate(contextual.FEATURE_NAMES)}
    raw = np.column_stack(
        [features[:, contextual_index[name]] for name in RELATIVE_BASE_NAMES]
    )
    frame_count = np.empty(count, dtype=np.float64)
    frame_percentiles = np.empty_like(raw)
    frame_to_max = np.empty_like(raw)
    for temporal_bin in np.unique(bins):
        rows = np.flatnonzero(bins == temporal_bin)
        values = raw[rows]
        frame_count[rows] = np.log1p(rows.size)
        for column in range(values.shape[1]):
            frame_percentiles[rows, column] = _right_percentiles(values[:, column])
            maximum = float(values[:, column].max())
            minimum = float(values[:, column].min())
            scale = maximum - minimum
            frame_to_max[rows, column] = (
                0.0 if scale < 1e-12 else (values[:, column] - maximum) / scale
            )
    video_percentiles = np.column_stack(
        [_right_percentiles(raw[:, column]) for column in range(raw.shape[1])]
    )
    result = np.column_stack(
        (frame_count, frame_percentiles, video_percentiles, frame_to_max)
    ).astype(np.float64, copy=False)
    if result.shape != (count, len(RELATIVE_FEATURE_NAMES)):
        raise RuntimeError("invalid relative feature shape.")
    if not np.isfinite(result).all():
        raise RuntimeError("relative component features must be finite.")
    return result


def extract_positive_support_components(
    prediction_scores,
    locations,
    prediction_threshold,
    topology,
    video_event_count,
    labels=None,
    context_scores=None,
):
    """Append within-frame/video ranks without consulting labels or source IDs."""

    base = contextual.extract_allsize_components(
        prediction_scores,
        locations,
        prediction_threshold,
        topology,
        video_event_count,
        labels=labels,
        context_scores=context_scores,
    )
    relative = _relative_features(base.features, base.event_indices, locations)
    features = np.column_stack((base.features, relative)).astype(np.float64, copy=False)
    if features.shape != (len(base.event_indices), len(FEATURE_NAMES)):
        raise RuntimeError("invalid positive-support feature matrix.")
    return AllSizeComponentBatch(base.event_indices, features, base.labels)


@dataclass(frozen=True)
class _WeightedCDF:
    values: np.ndarray
    cumulative_weights: np.ndarray

    @classmethod
    def fit(cls, values, weights):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if values.size == 0 or values.size != weights.size:
            raise ValueError("weighted CDF requires nonempty paired values and weights.")
        order = np.argsort(values, kind="mergesort")
        ordered_values = values[order]
        ordered_weights = weights[order]
        total = float(ordered_weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("weighted CDF weights must have positive finite mass.")
        return cls(ordered_values, np.cumsum(ordered_weights) / total)

    def __call__(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        positions = np.searchsorted(self.values, values, side="right") - 1
        result = np.zeros(values.size, dtype=np.float64)
        valid = positions >= 0
        result[valid] = self.cumulative_weights[positions[valid]]
        return result


class WeightedMarginalSupport:
    """Positive-only union-of-evidence support with equal source mass."""

    def __init__(self, evidence_names, aggregation="maximum"):
        if aggregation not in {"maximum", "second_maximum"}:
            raise KeyError(aggregation)
        unknown = set(evidence_names) - set(FEATURE_NAMES)
        if unknown:
            raise KeyError(f"unknown evidence features: {sorted(unknown)}")
        self.evidence_names = tuple(evidence_names)
        self.aggregation = aggregation
        self.feature_indices = tuple(FEATURE_NAMES.index(name) for name in evidence_names)
        self.distributions = None

    def fit(self, video_features, positive_masks):
        values = []
        weights = []
        for features, mask in zip(video_features, positive_masks):
            features = np.asarray(features, dtype=np.float64)
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            selected = features[mask][:, self.feature_indices]
            if selected.shape[0] == 0:
                continue
            values.append(selected)
            weights.append(np.full(selected.shape[0], 1.0 / selected.shape[0]))
        if not values:
            raise RuntimeError("positive-support fit received no positive components.")
        matrix = np.concatenate(values, axis=0)
        sample_weights = np.concatenate(weights)
        self.distributions = tuple(
            _WeightedCDF.fit(matrix[:, column], sample_weights)
            for column in range(matrix.shape[1])
        )
        return self

    def predict_support(self, features):
        if self.distributions is None:
            raise RuntimeError("positive-support guard is not fitted.")
        features = np.asarray(features, dtype=np.float64)
        marginal = np.column_stack(
            [
                distribution(features[:, feature_index])
                for distribution, feature_index in zip(
                    self.distributions, self.feature_indices
                )
            ]
        )
        if self.aggregation == "maximum":
            return marginal.max(axis=1)
        if marginal.shape[1] < 2:
            raise RuntimeError("second-maximum aggregation needs at least two features.")
        return np.partition(marginal, -2, axis=1)[:, -2]


__all__ = (
    "CORE_EVIDENCE_NAMES",
    "EVIDENCE_SETS",
    "FEATURE_NAMES",
    "PERSISTENCE_EVIDENCE_NAMES",
    "WeightedMarginalSupport",
    "extract_positive_support_components",
)
