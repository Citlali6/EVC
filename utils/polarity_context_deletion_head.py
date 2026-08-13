"""Polarity and fine-time context for source-free component decisions.

This module deliberately accepts only observable event arrays.  It extends the
existing contextual component representation with event-camera polarity and
sub-bin timing statistics; labels, target ids, source names, paths, and folds
are not accepted by the feature extractor.
"""

from __future__ import annotations

import numpy as np

from utils.contextual_deletion_head import (
    FEATURE_NAMES as CONTEXTUAL_FEATURE_NAMES,
    extract_allsize_components as extract_contextual_components,
)
from utils.allsize_deletion_head import AllSizeComponentBatch, suppress_components


COMPONENT_INPUT_NAMES = (
    "polarity_one_fraction",
    "polarity_minority_fraction",
    "polarity_entropy",
    "polarity_time_transition_fraction",
    "bipolar_cell_fraction",
    "cell_polarity_imbalance_mean",
    "cell_polarity_imbalance_max",
    "subbin_time_mean",
    "subbin_time_std",
    "subbin_time_span",
    "unique_fine_time_fraction",
    "max_same_millisecond_fraction",
    "polarity_zero_score_mean",
    "polarity_one_score_mean",
    "polarity_score_mean_gap",
    "fine_motion_speed",
    "fine_motion_residual",
)

LOCAL_INPUT_NAMES = tuple(
    f"input_r{radius}_dt{offset:+d}_{stat}"
    for radius in (1, 2)
    for offset in (-2, -1, 0, 1, 2)
    for stat in (
        "log_event_count",
        "polarity_one_fraction",
        "polarity_minority_fraction",
    )
)

RELATIVE_NAMES = (
    "same_bin_log_component_count",
    "same_bin_score_mean_rank",
    "same_bin_score_max_rank",
    "same_bin_size_rank",
    "same_bin_polarity_minority_rank",
)

FEATURE_NAMES = (
    CONTEXTUAL_FEATURE_NAMES
    + COMPONENT_INPUT_NAMES
    + LOCAL_INPUT_NAMES
    + RELATIVE_NAMES
)


def _fractional_ranks(values: np.ndarray) -> np.ndarray:
    """Deterministic average-tie ranks in [0, 1]."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return np.ones(values.size, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) / (values.size - 1)
        result[order[start:end]] = rank
        start = end
    return result


def _input_maps(locations: np.ndarray, polarities: np.ndarray):
    xyz = np.asarray(locations)[:, 1:4].astype(np.int64, copy=False)
    x = xyz[:, 0]
    y = xyz[:, 1]
    temporal_bin = np.floor_divide(xyz[:, 2], 50)
    if (
        np.any(x < 0)
        or np.any(x >= 346)
        or np.any(y < 0)
        or np.any(y >= 260)
        or np.any(temporal_bin < 0)
    ):
        raise ValueError("observable event coordinates are outside the frozen sensor.")
    shape = (int(temporal_bin.max()) + 1, 260, 346)
    total = np.zeros(shape, dtype=np.uint32)
    ones = np.zeros(shape, dtype=np.uint32)
    index = (temporal_bin, y, x)
    np.add.at(total, index, 1)
    np.add.at(ones, index, polarities.astype(np.uint32, copy=False))
    return total, ones


def _linear_motion(xy: np.ndarray, time: np.ndarray):
    if xy.shape[0] < 3 or float(np.ptp(time)) <= 1e-9:
        return 0.0, 0.0
    centered_time = time - float(time.mean())
    denominator = float(np.dot(centered_time, centered_time))
    velocity = centered_time @ xy / denominator
    fitted = xy.mean(axis=0) + centered_time[:, None] * velocity[None, :]
    residual = float(np.sqrt(np.mean(np.sum((xy - fitted) ** 2, axis=1))))
    return float(np.linalg.norm(velocity) * 50.0), residual / np.hypot(346.0, 260.0)


def extract_polarity_context_components(
    prediction_scores,
    locations,
    prediction_threshold,
    topology,
    video_event_count,
    polarities,
    fine_timestamps,
    context_scores=None,
):
    """Extract source-free component features from observable event inputs."""

    scores = np.asarray(prediction_scores, dtype=np.float32).reshape(-1)
    locations = np.asarray(locations)
    polarities = np.asarray(polarities).reshape(-1)
    fine_timestamps = np.asarray(fine_timestamps, dtype=np.float64).reshape(-1)
    if not (
        scores.size
        == locations.shape[0]
        == polarities.size
        == fine_timestamps.size
        == int(video_event_count)
    ):
        raise ValueError("observable arrays differ in length.")
    if not np.isin(polarities, (0, 1)).all() or not np.isfinite(fine_timestamps).all():
        raise ValueError("polarities/timestamps are invalid.")

    base = extract_contextual_components(
        scores,
        locations,
        prediction_threshold,
        topology,
        video_event_count,
        labels=None,
        context_scores=context_scores,
    )
    count = len(base.event_indices)
    if count == 0:
        return AllSizeComponentBatch(
            base.event_indices, np.empty((0, len(FEATURE_NAMES))), None
        )

    total_map, one_map = _input_maps(locations, polarities)
    component_input = np.zeros((count, len(COMPONENT_INPUT_NAMES)), dtype=np.float64)
    local_input = np.zeros((count, len(LOCAL_INPUT_NAMES)), dtype=np.float64)
    component_bins = np.empty(count, dtype=np.int64)
    centroids = np.empty((count, 2), dtype=np.float64)
    component_sizes = np.empty(count, dtype=np.float64)
    component_score_means = np.empty(count, dtype=np.float64)
    component_score_maxima = np.empty(count, dtype=np.float64)
    component_minority = np.empty(count, dtype=np.float64)

    for row, indices in enumerate(base.event_indices):
        indices = np.asarray(indices, dtype=np.int64)
        xy = locations[indices, 1:3].astype(np.float64, copy=False)
        integer_xy = locations[indices, 1:3].astype(np.int64, copy=False)
        p = polarities[indices].astype(np.uint8, copy=False)
        values = scores[indices].astype(np.float64, copy=False)
        time = fine_timestamps[indices]
        one_fraction = float(p.mean())
        minority = min(one_fraction, 1.0 - one_fraction)
        entropy = -sum(
            value * np.log(value + 1e-12)
            for value in (one_fraction, 1.0 - one_fraction)
        ) / np.log(2.0)
        time_order = np.argsort(time, kind="mergesort")
        transitions = (
            float(np.mean(p[time_order][1:] != p[time_order][:-1]))
            if p.size > 1
            else 0.0
        )
        unique_xy, inverse = np.unique(integer_xy, axis=0, return_inverse=True)
        cell_count = np.bincount(inverse)
        cell_ones = np.bincount(inverse, weights=p, minlength=unique_xy.shape[0])
        cell_zeros = cell_count - cell_ones
        bipolar = float(np.mean((cell_ones > 0) & (cell_zeros > 0)))
        imbalance = np.abs(cell_ones - cell_zeros) / np.maximum(cell_count, 1)
        phase = np.mod(time, 50.0) / 50.0
        milliseconds = np.floor(time).astype(np.int64)
        _, same_ms = np.unique(milliseconds, return_counts=True)
        zero_mean = float(values[p == 0].mean()) if np.any(p == 0) else 0.0
        one_mean = float(values[p == 1].mean()) if np.any(p == 1) else 0.0
        speed, residual = _linear_motion(xy, time)

        component_input[row] = (
            one_fraction,
            minority,
            float(entropy),
            transitions,
            bipolar,
            float(imbalance.mean()),
            float(imbalance.max()),
            float(phase.mean()),
            float(phase.std()),
            float(phase.max() - phase.min()),
            float(np.unique(time).size / time.size),
            float(same_ms.max() / time.size),
            zero_mean,
            one_mean,
            one_mean - zero_mean,
            speed,
            residual,
        )
        temporal_bin = int(np.floor_divide(int(locations[indices[0], 3]), 50))
        component_bins[row] = temporal_bin
        centroids[row] = xy.mean(axis=0)
        component_sizes[row] = float(indices.size)
        component_score_means[row] = float(values.mean())
        component_score_maxima[row] = float(values.max())
        component_minority[row] = minority

        x = int(np.clip(np.rint(centroids[row, 0]), 0, 345))
        y = int(np.clip(np.rint(centroids[row, 1]), 0, 259))
        column = 0
        for radius in (1, 2):
            x0, x1 = max(0, x - radius), min(346, x + radius + 1)
            y0, y1 = max(0, y - radius), min(260, y + radius + 1)
            for offset in (-2, -1, 0, 1, 2):
                local_bin = temporal_bin + offset
                total = ones = 0.0
                if 0 <= local_bin < total_map.shape[0]:
                    total = float(total_map[local_bin, y0:y1, x0:x1].sum())
                    ones = float(one_map[local_bin, y0:y1, x0:x1].sum())
                fraction = ones / total if total else 0.5
                local_input[row, column : column + 3] = (
                    np.log1p(total),
                    fraction,
                    min(fraction, 1.0 - fraction) if total else 0.0,
                )
                column += 3

    relative = np.zeros((count, len(RELATIVE_NAMES)), dtype=np.float64)
    for temporal_bin in np.unique(component_bins):
        members = np.flatnonzero(component_bins == temporal_bin)
        relative[members, 0] = np.log1p(members.size)
        relative[members, 1] = _fractional_ranks(component_score_means[members])
        relative[members, 2] = _fractional_ranks(component_score_maxima[members])
        relative[members, 3] = _fractional_ranks(component_sizes[members])
        relative[members, 4] = _fractional_ranks(component_minority[members])

    features = np.column_stack((base.features, component_input, local_input, relative))
    if features.shape != (count, len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise RuntimeError("invalid polarity-context feature matrix.")
    return AllSizeComponentBatch(base.event_indices, features, None)


__all__ = (
    "FEATURE_NAMES",
    "extract_polarity_context_components",
    "suppress_components",
)
