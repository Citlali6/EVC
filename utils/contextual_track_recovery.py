"""Label-free contextual features for P18-style track-edge recovery.

The proposal geometry is intentionally inherited from
``utils.track_edge_recovery``.  This module only enriches each proposal with
translation- and time-origin-invariant score/shape context.  It has no label,
target-id, source-name, path, or fold argument.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from utils.track_edge_recovery import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    FROZEN_TOPOLOGY,
    TrackEdgeTopology,
    extract_track_edge_candidates,
)


COMPONENT_FEATURE_NAMES = (
    "component_score_q10",
    "component_score_q25",
    "component_score_q50",
    "component_score_q75",
    "component_score_q90",
    "log_unique_cells",
    "bbox_width",
    "bbox_height",
    "bbox_fill_ratio",
    "events_per_cell",
    "cov_minor_major_ratio",
    "radial_compactness",
    "max_cell_multiplicity_fraction",
    "bin_score_percentile",
    "bin_weak_score_percentile",
)
CONTEXT_OFFSETS = tuple(range(-3, 4))
CONTEXT_RADII = (3, 8)
CONTEXT_FEATURE_NAMES = tuple(
    f"context_dt{offset:+d}_r{radius}_{stat}"
    for offset in CONTEXT_OFFSETS
    for radius in CONTEXT_RADII
    for stat in (
        "log_event_count",
        "raw_score_mean",
        "raw_score_max",
        "log_weak_event_count",
        "log_seed_event_count",
    )
)
SEED_NEIGHBOR_FEATURE_NAMES = tuple(
    f"seed_neighbor_dt{offset:+d}_{stat}"
    for offset in CONTEXT_OFFSETS
    for stat in ("present", "distance_over_link_radius", "raw_score")
)
FEATURE_NAMES = (
    BASE_FEATURE_NAMES
    + COMPONENT_FEATURE_NAMES
    + CONTEXT_FEATURE_NAMES
    + SEED_NEIGHBOR_FEATURE_NAMES
)
FEATURE_SEMANTICS_VERSION = "p18-track-edge-contextual-score-shape-v2"


@dataclass(frozen=True)
class ContextualTrackEdgeCandidate:
    """A P18 proposal carrying only inference-observable features."""

    event_index: int
    component_event_indices: np.ndarray
    endpoint_key: tuple[int, int]
    temporal_bin: int
    endpoint_side: int
    features: np.ndarray
    raw_score: float
    motion_residual: float


def _normalize_locations(locations, event_count):
    locations = np.asarray(locations)
    if locations.ndim != 2 or locations.shape[1] not in (3, 4):
        raise ValueError("locations must have shape [N,3] or [N,4].")
    if locations.shape[1] == 4:
        if np.unique(locations[:, 0]).size != 1:
            raise ValueError("Context extraction accepts one video at a time.")
        locations = locations[:, 1:4]
    locations = locations.astype(np.int64, copy=False)
    if locations.shape[0] != int(event_count):
        raise ValueError("locations and event_count differ.")
    return locations


def _indices_by_bin(event_bins):
    order = np.argsort(event_bins, kind="stable")
    ordered_bins = event_bins[order]
    values, starts, counts = np.unique(
        ordered_bins, return_index=True, return_counts=True
    )
    return {
        int(value): order[int(start) : int(start + count)]
        for value, start, count in zip(values, starts, counts)
    }


def _component_shape_features(candidate, raw_scores, locations, by_bin, topology):
    indices = np.asarray(candidate.component_event_indices, dtype=np.int64)
    scores = raw_scores[indices]
    coordinates = locations[indices, :2].astype(np.float64, copy=False)
    unique_cells, multiplicities = np.unique(
        coordinates.astype(np.int64, copy=False), axis=0, return_counts=True
    )
    width = float(unique_cells[:, 0].max() - unique_cells[:, 0].min() + 1)
    height = float(unique_cells[:, 1].max() - unique_cells[:, 1].min() + 1)
    fill = float(unique_cells.shape[0] / max(width * height, 1.0))
    events_per_cell = float(indices.size / max(unique_cells.shape[0], 1))
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    if indices.size >= 2:
        covariance = centered.T @ centered / float(indices.size)
        eigenvalues = np.linalg.eigvalsh(covariance)
        cov_ratio = float(max(eigenvalues[0], 0.0) / (max(eigenvalues[-1], 0.0) + 1e-8))
    else:
        cov_ratio = 0.0
    radial = np.linalg.norm(centered, axis=1)
    radial_compactness = float(radial.mean() / (math.hypot(width, height) + 1e-8))
    bin_indices = by_bin[int(candidate.temporal_bin)]
    bin_scores = raw_scores[bin_indices]
    candidate_max = float(scores.max())
    bin_percentile = float(np.mean(bin_scores <= candidate_max))
    weak = bin_scores[
        (bin_scores >= topology.weak_score_floor)
        & (bin_scores < topology.prediction_threshold)
    ]
    weak_percentile = float(np.mean(weak <= candidate_max)) if weak.size else 0.0
    return np.asarray(
        (
            *np.quantile(scores, (0.10, 0.25, 0.50, 0.75, 0.90)).tolist(),
            math.log1p(unique_cells.shape[0]),
            width,
            height,
            fill,
            events_per_cell,
            cov_ratio,
            radial_compactness,
            float(multiplicities.max() / indices.size),
            bin_percentile,
            weak_percentile,
        ),
        dtype=np.float64,
    )


def _context_features(
    candidate,
    raw_scores,
    baseline_scores,
    locations,
    by_bin,
    topology,
):
    component_coordinates = locations[
        np.asarray(candidate.component_event_indices, dtype=np.int64), :2
    ].astype(np.float64, copy=False)
    center = component_coordinates.mean(axis=0)
    context_values = []
    seed_values = []
    for offset in CONTEXT_OFFSETS:
        indices = by_bin.get(int(candidate.temporal_bin + offset))
        if indices is None or indices.size == 0:
            for _radius in CONTEXT_RADII:
                context_values.extend((0.0, 0.0, 0.0, 0.0, 0.0))
            seed_values.extend((0.0, 0.0, 0.0))
            continue
        coordinates = locations[indices, :2].astype(np.float64, copy=False)
        chebyshev = np.max(np.abs(coordinates - center[None, :]), axis=1)
        bin_raw_scores = raw_scores[indices]
        bin_baseline_scores = baseline_scores[indices]
        for radius in CONTEXT_RADII:
            local = chebyshev <= float(radius)
            if not local.any():
                context_values.extend((0.0, 0.0, 0.0, 0.0, 0.0))
                continue
            local_raw = bin_raw_scores[local]
            local_baseline = bin_baseline_scores[local]
            weak_count = int(
                np.sum(
                    (local_raw >= topology.weak_score_floor)
                    & (local_raw < topology.prediction_threshold)
                    & (local_baseline < topology.prediction_threshold)
                )
            )
            seed_count = int(np.sum(local_baseline >= topology.prediction_threshold))
            context_values.extend(
                (
                    math.log1p(local_raw.size),
                    float(local_raw.mean()),
                    float(local_raw.max()),
                    math.log1p(weak_count),
                    math.log1p(seed_count),
                )
            )
        seed_mask = bin_baseline_scores >= topology.prediction_threshold
        if seed_mask.any():
            seed_coordinates = coordinates[seed_mask]
            distances = np.linalg.norm(seed_coordinates - center[None, :], axis=1)
            nearest = int(np.argmin(distances))
            seed_raw = bin_raw_scores[seed_mask]
            seed_values.extend(
                (
                    1.0,
                    float(distances[nearest] / (topology.max_link_distance + 1e-8)),
                    float(seed_raw[nearest]),
                )
            )
        else:
            seed_values.extend((0.0, 0.0, 0.0))
    return np.asarray(context_values + seed_values, dtype=np.float64)


def extract_contextual_track_edge_candidates(
    raw_scores,
    baseline_scores,
    locations,
    event_count,
    topology=FROZEN_TOPOLOGY,
):
    """Extract enriched P18 proposals without accepting supervision."""

    if not isinstance(topology, TrackEdgeTopology):
        raise TypeError("topology must be TrackEdgeTopology.")
    raw_scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    baseline_scores = np.asarray(baseline_scores, dtype=np.float64).reshape(-1)
    locations3 = _normalize_locations(locations, event_count)
    if raw_scores.size != event_count or baseline_scores.size != event_count:
        raise ValueError("scores and event_count differ.")
    base_candidates = extract_track_edge_candidates(
        raw_scores,
        baseline_scores,
        locations3,
        event_count,
        topology,
    )
    if not base_candidates:
        return ()
    event_bins = np.floor_divide(locations3[:, 2], topology.temporal_bin_size)
    by_bin = _indices_by_bin(event_bins)
    result = []
    for candidate in base_candidates:
        component = _component_shape_features(
            candidate, raw_scores, locations3, by_bin, topology
        )
        context = _context_features(
            candidate,
            raw_scores,
            baseline_scores,
            locations3,
            by_bin,
            topology,
        )
        features = np.concatenate((candidate.features, component, context))
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise RuntimeError("Contextual track-recovery features are invalid.")
        result.append(
            ContextualTrackEdgeCandidate(
                event_index=int(candidate.event_index),
                component_event_indices=np.asarray(
                    candidate.component_event_indices, dtype=np.int64
                ).copy(),
                endpoint_key=tuple(candidate.endpoint_key),
                temporal_bin=int(candidate.temporal_bin),
                endpoint_side=int(candidate.endpoint_side),
                features=features,
                raw_score=float(candidate.raw_score),
                motion_residual=float(candidate.motion_residual),
            )
        )
    return tuple(result)

