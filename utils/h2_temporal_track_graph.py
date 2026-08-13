"""Input-only temporal component graph for the dense H2 regime.

The graph deliberately separates three concerns:

* nodes are complete post-C00, per-time-bin 8-connected components;
* candidate edges expose motion continuity to a learned message-passing model;
* edits are atomic: an inference result can remove a whole component or a
  whole deterministic track, but can never attenuate individual event scores.

Source identity, file paths, hashes, fold identifiers, target ids and labels
are not accepted by :func:`extract_temporal_track_graph`.  Train-only targets
are derived by separate functions after the input-only graph exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from utils.component_reranker import _spatial_components


DECODER_CHANNELS = 16
SPATIAL_WIDTH = 346
SPATIAL_HEIGHT = 260
TEMPORAL_BIN_SIZE = 50
GRAPH_NEIGHBORS_PER_GAP = 3
GRAPH_MAX_GAP_BINS = 2


BASE_NODE_FEATURE_NAMES = (
    "log_video_events",
    "log_component_events",
    "log_unique_cells",
    "score_max",
    "score_mean",
    "score_min",
    "score_std",
    "score_q25",
    "score_q75",
    "score_margin_max",
    "score_margin_mean",
    "centroid_x_norm",
    "centroid_y_norm",
    "time_position_norm",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_diagonal_norm",
    "compactness",
    "component_polarity_mean",
    "component_polarity_minority",
    "local_log_events",
    "local_predicted_fraction",
    "local_polarity_mean",
    "local_polarity_minority",
    "same_bin_log_component_count",
    "same_bin_size_percentile",
    "same_bin_score_percentile",
    "previous_bin_nearest_distance_norm",
    "next_bin_nearest_distance_norm",
    "previous_bin_score_difference",
    "next_bin_score_difference",
    "decoder_available",
)

NODE_FEATURE_NAMES = BASE_NODE_FEATURE_NAMES + tuple(
    "decoder_mean_{:02d}".format(index) for index in range(DECODER_CHANNELS)
) + tuple(
    "decoder_std_{:02d}".format(index) for index in range(DECODER_CHANNELS)
)

EDGE_FEATURE_NAMES = (
    "temporal_gap_fraction",
    "delta_x_norm",
    "delta_y_norm",
    "distance_norm",
    "distance_by_component_scale",
    "log_event_count_ratio",
    "score_max_delta",
    "score_mean_delta",
    "bbox_iou",
    "polarity_mean_delta",
    "log_local_activity_ratio",
    "origin_radius_norm",
    "destination_radius_norm",
    "forward_direction",
    "mutual_knn",
    "same_deterministic_track",
)

TRACK_FEATURE_NAMES = (
    "log_track_nodes",
    "log_track_events",
    "track_time_span_fraction",
    "track_displacement_norm",
    "track_step_distance_mean_norm",
    "track_step_distance_std_norm",
    "track_velocity_residual_mean_norm",
    "track_score_max",
    "track_score_mean",
    "track_polarity_mean",
    "track_local_activity_mean",
    "track_edge_density",
)

FORBIDDEN_INFERENCE_FEATURE_TOKENS = (
    "source",
    "file",
    "path",
    "hash",
    "fold",
    "target_id",
    "label",
    "sample_index",
)


@dataclass(frozen=True)
class TemporalTrackGraph:
    """One complete-video graph with no truth-bearing fields."""

    event_indices: tuple[np.ndarray, ...]
    temporal_bins: np.ndarray
    centroids: np.ndarray
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    component_to_track: np.ndarray
    track_component_rows: tuple[np.ndarray, ...]
    track_features: np.ndarray


@dataclass(frozen=True)
class AtomicGraphEditReceipt:
    mode: str
    component_count: int
    track_count: int
    deleted_component_count: int
    deleted_track_count: int
    deleted_event_count: int
    complete_components_only: bool
    complete_tracks_only: bool
    retained_scores_bitwise_equal: bool


def _signed_polarities(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("input polarities must be a non-empty finite vector")
    unique = np.unique(values)
    if np.all(np.isin(unique, np.asarray((0.0, 1.0), dtype=np.float32))):
        return values * np.float32(2.0) - np.float32(1.0)
    if np.all(np.isin(unique, np.asarray((-1.0, 1.0), dtype=np.float32))):
        return values.copy()
    raise ValueError("input polarities must use {0,1} or {-1,+1}")


def _minority_fraction(signed_values):
    values = np.asarray(signed_values).reshape(-1)
    if values.size == 0:
        return 0.0
    positive = float(np.mean(values > 0))
    return min(positive, 1.0 - positive)


def _percentile_rank(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    ordered = np.sort(values)
    return np.searchsorted(ordered, values, side="right") / float(values.size)


def _bbox_iou(first_minimum, first_maximum, second_minimum, second_maximum):
    overlap_minimum = np.maximum(first_minimum, second_minimum)
    overlap_maximum = np.minimum(first_maximum, second_maximum)
    overlap_extent = np.maximum(overlap_maximum - overlap_minimum + 1.0, 0.0)
    intersection = float(np.prod(overlap_extent))
    first_area = float(np.prod(first_maximum - first_minimum + 1.0))
    second_area = float(np.prod(second_maximum - second_minimum + 1.0))
    return intersection / max(first_area + second_area - intersection, 1.0)


def _nearest_other_bin_statistics(temporal_bins, centroids, score_means):
    count = temporal_bins.size
    diagonal = float(np.hypot(SPATIAL_WIDTH, SPATIAL_HEIGHT))
    distances = {
        -1: np.full(count, diagonal, dtype=np.float64),
        1: np.full(count, diagonal, dtype=np.float64),
    }
    score_differences = {
        -1: np.zeros(count, dtype=np.float64),
        1: np.zeros(count, dtype=np.float64),
    }
    by_bin = {
        int(bin_value): np.flatnonzero(temporal_bins == bin_value).astype(np.int64)
        for bin_value in np.unique(temporal_bins)
    }
    trees = {
        bin_value: cKDTree(centroids[rows]) for bin_value, rows in by_bin.items()
    }
    for bin_value, rows in by_bin.items():
        for direction in (-1, 1):
            other_rows = by_bin.get(bin_value + direction)
            if other_rows is None or other_rows.size == 0:
                continue
            queried_distance, queried_index = trees[bin_value + direction].query(
                centroids[rows], k=1
            )
            queried_distance = np.asarray(queried_distance, dtype=np.float64).reshape(-1)
            queried_index = np.asarray(queried_index, dtype=np.int64).reshape(-1)
            matched_rows = other_rows[queried_index]
            distances[direction][rows] = queried_distance
            score_differences[direction][rows] = (
                score_means[matched_rows] - score_means[rows]
            )
    return distances, score_differences


def _knn_links(rows_a, rows_b, centroids, neighbor_count):
    """Return directed local row pairs and their reciprocal-kNN flags."""

    if rows_a.size == 0 or rows_b.size == 0:
        return []
    k_ab = min(int(neighbor_count), int(rows_b.size))
    k_ba = min(int(neighbor_count), int(rows_a.size))
    tree_b = cKDTree(centroids[rows_b])
    _, indices_ab = tree_b.query(centroids[rows_a], k=k_ab)
    indices_ab = np.asarray(indices_ab, dtype=np.int64)
    if indices_ab.ndim == 1:
        indices_ab = indices_ab[:, None]
    tree_a = cKDTree(centroids[rows_a])
    _, indices_ba = tree_a.query(centroids[rows_b], k=k_ba)
    indices_ba = np.asarray(indices_ba, dtype=np.int64)
    if indices_ba.ndim == 1:
        indices_ba = indices_ba[:, None]
    reverse_sets = [set(rows_a[values].tolist()) for values in indices_ba]
    links = []
    for local_a, candidates in enumerate(indices_ab):
        row_a = int(rows_a[local_a])
        for local_b in candidates:
            row_b = int(rows_b[int(local_b)])
            reciprocal = row_a in reverse_sets[int(local_b)]
            links.append((row_a, row_b, reciprocal))
    return links


def _deterministic_tracks(temporal_bins, centroids):
    """Build conservative input-only tracks from adjacent-bin mutual nearest links.

    There is no fitted distance threshold.  A link is accepted only if both
    components select each other as their nearest adjacent-bin neighbor.
    """

    count = int(temporal_bins.size)
    parent = np.arange(count, dtype=np.int64)

    def find(row):
        row = int(row)
        while int(parent[row]) != row:
            parent[row] = parent[int(parent[row])]
            row = int(parent[row])
        return row

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    by_bin = {
        int(bin_value): np.flatnonzero(temporal_bins == bin_value).astype(np.int64)
        for bin_value in np.unique(temporal_bins)
    }
    for bin_value in sorted(by_bin):
        first_rows = by_bin[bin_value]
        second_rows = by_bin.get(bin_value + 1)
        if second_rows is None or second_rows.size == 0:
            continue
        tree_second = cKDTree(centroids[second_rows])
        _, first_to_second = tree_second.query(centroids[first_rows], k=1)
        tree_first = cKDTree(centroids[first_rows])
        _, second_to_first = tree_first.query(centroids[second_rows], k=1)
        first_to_second = np.asarray(first_to_second, dtype=np.int64).reshape(-1)
        second_to_first = np.asarray(second_to_first, dtype=np.int64).reshape(-1)
        for local_first, local_second in enumerate(first_to_second):
            if int(second_to_first[int(local_second)]) == local_first:
                union(first_rows[local_first], second_rows[int(local_second)])

    roots = np.asarray([find(row) for row in range(count)], dtype=np.int64)
    unique_roots = np.unique(roots)
    root_to_track = {int(root): index for index, root in enumerate(unique_roots)}
    component_to_track = np.asarray(
        [root_to_track[int(root)] for root in roots], dtype=np.int64
    )
    track_rows = tuple(
        np.flatnonzero(component_to_track == track).astype(np.int64)
        for track in range(unique_roots.size)
    )
    return component_to_track, track_rows


def _track_features(
    track_rows,
    temporal_bins,
    centroids,
    sizes,
    score_maxima,
    score_means,
    polarity_means,
    local_event_counts,
    edge_index,
):
    diagonal = float(np.hypot(SPATIAL_WIDTH, SPATIAL_HEIGHT))
    temporal_extent = max(int(temporal_bins.max() - temporal_bins.min()), 1)
    track_for_component = np.empty(temporal_bins.size, dtype=np.int64)
    for track_index, rows in enumerate(track_rows):
        track_for_component[rows] = int(track_index)
    within_track_edge_counts = np.zeros(len(track_rows), dtype=np.int64)
    if edge_index.shape[1]:
        same = track_for_component[edge_index[0]] == track_for_component[edge_index[1]]
        for track_index in track_for_component[edge_index[0, same]]:
            within_track_edge_counts[int(track_index)] += 1

    result = []
    for track_index, rows in enumerate(track_rows):
        ordered = rows[np.argsort(temporal_bins[rows], kind="stable")]
        step_distances = (
            np.linalg.norm(np.diff(centroids[ordered], axis=0), axis=1)
            if ordered.size > 1
            else np.empty(0, dtype=np.float64)
        )
        velocity_residuals = (
            np.linalg.norm(np.diff(centroids[ordered], n=2, axis=0), axis=1)
            if ordered.size > 2
            else np.empty(0, dtype=np.float64)
        )
        displacement = (
            float(np.linalg.norm(centroids[ordered[-1]] - centroids[ordered[0]]))
            if ordered.size > 1
            else 0.0
        )
        possible_directed_edges = max(int(ordered.size * max(ordered.size - 1, 1)), 1)
        result.append(
            (
                math.log1p(int(rows.size)),
                math.log1p(int(sizes[rows].sum())),
                float((temporal_bins[rows].max() - temporal_bins[rows].min()) / temporal_extent),
                displacement / diagonal,
                float(step_distances.mean() / diagonal) if step_distances.size else 0.0,
                float(step_distances.std() / diagonal) if step_distances.size else 0.0,
                float(velocity_residuals.mean() / diagonal)
                if velocity_residuals.size
                else 0.0,
                float(score_maxima[rows].max()),
                float(np.average(score_means[rows], weights=sizes[rows])),
                float(np.average(polarity_means[rows], weights=sizes[rows])),
                float(np.mean(np.log1p(local_event_counts[rows]))),
                float(within_track_edge_counts[track_index] / possible_directed_edges),
            )
        )
    return np.asarray(result, dtype=np.float64)


def extract_temporal_track_graph(
    prediction_scores,
    locations,
    input_polarities,
    prediction_threshold,
    video_event_count,
    *,
    decoder_event_features=None,
    temporal_bin_size=TEMPORAL_BIN_SIZE,
    spatial_width=SPATIAL_WIDTH,
    spatial_height=SPATIAL_HEIGHT,
    neighbors_per_gap=GRAPH_NEIGHBORS_PER_GAP,
    max_gap_bins=GRAPH_MAX_GAP_BINS,
):
    """Extract a complete-video graph from inputs and released-M20 outputs only."""

    scores = np.asarray(prediction_scores, dtype=np.float32).reshape(-1)
    locations = np.asarray(locations)
    if locations.ndim != 2 or locations.shape[1] < 4:
        raise ValueError("locations must be [N,4+] ordered [batch,x,y,t]")
    if scores.size != locations.shape[0] or scores.size != int(video_event_count):
        raise ValueError("score, location and complete-video event counts differ")
    if not np.isfinite(scores).all():
        raise ValueError("prediction scores must be finite")
    if int(temporal_bin_size) <= 0 or int(neighbors_per_gap) <= 0 or int(max_gap_bins) <= 0:
        raise ValueError("graph topology values must be positive")
    if int(spatial_width) != SPATIAL_WIDTH or int(spatial_height) != SPATIAL_HEIGHT:
        raise ValueError("track graph is frozen to the official 346x260 sensor")
    signed_polarities = _signed_polarities(input_polarities)
    if signed_polarities.size != scores.size:
        raise ValueError("input polarities and scores differ in length")
    decoder_available = decoder_event_features is not None
    if decoder_available:
        decoder = np.asarray(decoder_event_features, dtype=np.float32)
        if decoder.shape != (scores.size, DECODER_CHANNELS) or not np.isfinite(decoder).all():
            raise ValueError("decoder_event_features must be finite [N,16]")
    else:
        decoder = None

    effective_threshold = np.float32(prediction_threshold)
    positive_indices = np.flatnonzero(scores >= effective_threshold)
    if positive_indices.size == 0:
        return TemporalTrackGraph(
            tuple(),
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, len(NODE_FEATURE_NAMES)), dtype=np.float64),
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            tuple(),
            np.empty((0, len(TRACK_FEATURE_NAMES)), dtype=np.float64),
        )

    event_indices = []
    temporal_bins = []
    positive_bins = np.floor_divide(
        locations[positive_indices, 3].astype(np.int64), int(temporal_bin_size)
    )
    for temporal_bin in np.unique(positive_bins):
        rows = np.flatnonzero(positive_bins == temporal_bin)
        bin_positive_indices = positive_indices[rows]
        for component in _spatial_components(
            locations[bin_positive_indices, 1:4].astype(np.int64, copy=False),
            np.arange(bin_positive_indices.size, dtype=np.int64),
            1,
        ):
            event_indices.append(bin_positive_indices[np.asarray(component, dtype=np.int64)])
            temporal_bins.append(int(temporal_bin))
    event_indices = tuple(np.asarray(values, dtype=np.int64) for values in event_indices)
    temporal_bins = np.asarray(temporal_bins, dtype=np.int64)
    component_count = len(event_indices)
    if component_count == 0:
        raise RuntimeError("positive events produced no spatial component")

    centroids = np.empty((component_count, 2), dtype=np.float64)
    bbox_minima = np.empty((component_count, 2), dtype=np.float64)
    bbox_maxima = np.empty((component_count, 2), dtype=np.float64)
    sizes = np.empty(component_count, dtype=np.float64)
    unique_cell_counts = np.empty(component_count, dtype=np.float64)
    score_maxima = np.empty(component_count, dtype=np.float64)
    score_means = np.empty(component_count, dtype=np.float64)
    score_minima = np.empty(component_count, dtype=np.float64)
    score_stds = np.empty(component_count, dtype=np.float64)
    score_q25 = np.empty(component_count, dtype=np.float64)
    score_q75 = np.empty(component_count, dtype=np.float64)
    polarity_means = np.empty(component_count, dtype=np.float64)
    polarity_minorities = np.empty(component_count, dtype=np.float64)
    compactness = np.empty(component_count, dtype=np.float64)
    decoder_means = np.zeros((component_count, DECODER_CHANNELS), dtype=np.float64)
    decoder_stds = np.zeros((component_count, DECODER_CHANNELS), dtype=np.float64)
    seen = np.zeros(scores.size, dtype=bool)
    for row, indices in enumerate(event_indices):
        if indices.size == 0 or np.any(seen[indices]):
            raise RuntimeError("component indices are empty or overlapping")
        seen[indices] = True
        xy = locations[indices, 1:3].astype(np.float64, copy=False)
        component_scores = scores[indices].astype(np.float64, copy=False)
        component_polarities = signed_polarities[indices].astype(np.float64, copy=False)
        unique_cells = np.unique(xy.astype(np.int64), axis=0)
        minimum = xy.min(axis=0)
        maximum = xy.max(axis=0)
        extent = maximum - minimum + 1.0
        centroids[row] = xy.mean(axis=0)
        bbox_minima[row] = minimum
        bbox_maxima[row] = maximum
        sizes[row] = float(indices.size)
        unique_cell_counts[row] = float(unique_cells.shape[0])
        score_maxima[row] = float(component_scores.max())
        score_means[row] = float(component_scores.mean())
        score_minima[row] = float(component_scores.min())
        score_stds[row] = float(component_scores.std())
        score_q25[row], score_q75[row] = np.quantile(component_scores, (0.25, 0.75))
        polarity_means[row] = float(component_polarities.mean())
        polarity_minorities[row] = _minority_fraction(component_polarities)
        compactness[row] = float(unique_cells.shape[0] / max(float(np.prod(extent)), 1.0))
        if decoder is not None:
            decoder_means[row] = decoder[indices].mean(axis=0)
            decoder_stds[row] = decoder[indices].std(axis=0)

    by_bin = {
        int(bin_value): np.flatnonzero(temporal_bins == bin_value).astype(np.int64)
        for bin_value in np.unique(temporal_bins)
    }
    raw_bins = np.floor_divide(locations[:, 3].astype(np.int64), int(temporal_bin_size))
    raw_by_bin = {
        int(bin_value): np.flatnonzero(raw_bins == bin_value).astype(np.int64)
        for bin_value in np.unique(raw_bins)
    }
    raw_trees = {
        bin_value: cKDTree(locations[rows, 1:3].astype(np.float64, copy=False))
        for bin_value, rows in raw_by_bin.items()
    }
    local_event_counts = np.empty(component_count, dtype=np.float64)
    local_predicted_fractions = np.empty(component_count, dtype=np.float64)
    local_polarity_means = np.empty(component_count, dtype=np.float64)
    local_polarity_minorities = np.empty(component_count, dtype=np.float64)
    for row in range(component_count):
        bin_value = int(temporal_bins[row])
        raw_rows = raw_by_bin[bin_value]
        radius = max(
            1.0,
            float(np.linalg.norm(bbox_maxima[row] - bbox_minima[row] + 1.0)),
        )
        local_rows = np.asarray(
            raw_trees[bin_value].query_ball_point(centroids[row], r=radius),
            dtype=np.int64,
        )
        neighborhood_indices = raw_rows[local_rows]
        local_polarities = signed_polarities[neighborhood_indices]
        local_event_counts[row] = float(neighborhood_indices.size)
        local_predicted_fractions[row] = float(
            np.mean(scores[neighborhood_indices] >= effective_threshold)
        )
        local_polarity_means[row] = float(local_polarities.mean())
        local_polarity_minorities[row] = _minority_fraction(local_polarities)

    nearest_distances, nearest_score_differences = _nearest_other_bin_statistics(
        temporal_bins, centroids, score_means
    )
    diagonal = float(np.hypot(spatial_width, spatial_height))
    temporal_extent = max(int(temporal_bins.max() - temporal_bins.min()), 1)
    same_bin_counts = np.empty(component_count, dtype=np.float64)
    same_bin_size_ranks = np.empty(component_count, dtype=np.float64)
    same_bin_score_ranks = np.empty(component_count, dtype=np.float64)
    for rows in by_bin.values():
        same_bin_counts[rows] = float(rows.size)
        same_bin_size_ranks[rows] = _percentile_rank(sizes[rows])
        same_bin_score_ranks[rows] = _percentile_rank(score_maxima[rows])

    extent = bbox_maxima - bbox_minima + 1.0
    base_features = np.column_stack(
        (
            np.full(component_count, math.log1p(int(video_event_count))),
            np.log1p(sizes),
            np.log1p(unique_cell_counts),
            score_maxima,
            score_means,
            score_minima,
            score_stds,
            score_q25,
            score_q75,
            score_maxima - float(prediction_threshold),
            score_means - float(prediction_threshold),
            centroids[:, 0] / max(float(spatial_width - 1), 1.0),
            centroids[:, 1] / max(float(spatial_height - 1), 1.0),
            (temporal_bins - temporal_bins.min()) / float(temporal_extent),
            extent[:, 0] / float(spatial_width),
            extent[:, 1] / float(spatial_height),
            np.linalg.norm(extent, axis=1) / diagonal,
            compactness,
            polarity_means,
            polarity_minorities,
            np.log1p(local_event_counts),
            sizes / np.maximum(local_event_counts, 1.0),
            local_polarity_means,
            local_polarity_minorities,
            np.log1p(same_bin_counts),
            same_bin_size_ranks,
            same_bin_score_ranks,
            nearest_distances[-1] / diagonal,
            nearest_distances[1] / diagonal,
            nearest_score_differences[-1],
            nearest_score_differences[1],
            np.full(component_count, float(decoder_available)),
        )
    )
    node_features = np.column_stack((base_features, decoder_means, decoder_stds))
    if node_features.shape != (component_count, len(NODE_FEATURE_NAMES)):
        raise RuntimeError("node feature schema mismatch")
    if not np.isfinite(node_features).all():
        raise RuntimeError("node features must be finite")

    component_to_track, track_rows = _deterministic_tracks(temporal_bins, centroids)
    radii = np.sqrt(unique_cell_counts / np.pi)
    directed_edges = []
    directed_edge_features = []
    for bin_value in sorted(by_bin):
        rows_a = by_bin[bin_value]
        for gap in range(1, int(max_gap_bins) + 1):
            rows_b = by_bin.get(bin_value + gap)
            if rows_b is None:
                continue
            for row_a, row_b, reciprocal in _knn_links(
                rows_a, rows_b, centroids, int(neighbors_per_gap)
            ):
                for source, target, forward in (
                    (row_a, row_b, 1.0),
                    (row_b, row_a, 0.0),
                ):
                    delta = centroids[target] - centroids[source]
                    distance = float(np.linalg.norm(delta))
                    component_scale = max(float(radii[source] + radii[target]), 1.0)
                    edge_values = (
                        float(gap / max_gap_bins),
                        float(delta[0] / spatial_width),
                        float(delta[1] / spatial_height),
                        distance / diagonal,
                        distance / component_scale,
                        float(np.log((sizes[target] + 1.0) / (sizes[source] + 1.0))),
                        float(score_maxima[target] - score_maxima[source]),
                        float(score_means[target] - score_means[source]),
                        _bbox_iou(
                            bbox_minima[source],
                            bbox_maxima[source],
                            bbox_minima[target],
                            bbox_maxima[target],
                        ),
                        float(polarity_means[target] - polarity_means[source]),
                        float(
                            np.log(
                                (local_event_counts[target] + 1.0)
                                / (local_event_counts[source] + 1.0)
                            )
                        ),
                        float(radii[source] / diagonal),
                        float(radii[target] / diagonal),
                        forward,
                        float(reciprocal),
                        float(component_to_track[source] == component_to_track[target]),
                    )
                    directed_edges.append((source, target))
                    directed_edge_features.append(edge_values)
    edge_index = (
        np.asarray(directed_edges, dtype=np.int64).T
        if directed_edges
        else np.empty((2, 0), dtype=np.int64)
    )
    edge_features = (
        np.asarray(directed_edge_features, dtype=np.float64)
        if directed_edge_features
        else np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float64)
    )
    if edge_index.shape != (2, edge_features.shape[0]) or edge_features.shape[1] != len(
        EDGE_FEATURE_NAMES
    ):
        raise RuntimeError("edge feature schema mismatch")
    if not np.isfinite(edge_features).all():
        raise RuntimeError("edge features must be finite")

    track_features = _track_features(
        track_rows,
        temporal_bins,
        centroids,
        sizes,
        score_maxima,
        score_means,
        polarity_means,
        local_event_counts,
        edge_index,
    )
    if track_features.shape != (len(track_rows), len(TRACK_FEATURE_NAMES)):
        raise RuntimeError("track feature schema mismatch")
    if not np.isfinite(track_features).all():
        raise RuntimeError("track features must be finite")
    return TemporalTrackGraph(
        event_indices,
        temporal_bins,
        centroids,
        node_features.astype(np.float32, copy=False),
        edge_index,
        edge_features.astype(np.float32, copy=False),
        component_to_track,
        track_rows,
        track_features.astype(np.float32, copy=False),
    )


def pure_false_positive_component_targets(event_indices, labels):
    """Return 1 for a train-only component containing no target event."""

    labels = np.asarray(labels).reshape(-1)
    targets = []
    for indices in event_indices:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= labels.size):
            raise ValueError("component indices are empty or outside labels")
        targets.append(int(not np.any(labels[indices] > 0)))
    return np.asarray(targets, dtype=np.uint8)


def pure_false_positive_track_targets(graph, component_targets):
    """Return 1 only when every complete component in a track is pure FP."""

    component_targets = np.asarray(component_targets, dtype=np.uint8).reshape(-1)
    if component_targets.size != len(graph.event_indices):
        raise ValueError("component targets and graph components differ")
    return np.asarray(
        [int(np.all(component_targets[rows] == 1)) for rows in graph.track_component_rows],
        dtype=np.uint8,
    )


def aggregate_track_node_features(graph):
    """Fixed, threshold-free pooling used by the CPU separability diagnostic."""

    rows = []
    for component_rows in graph.track_component_rows:
        features = graph.node_features[np.asarray(component_rows, dtype=np.int64)]
        rows.append(
            np.concatenate(
                (
                    features.mean(axis=0),
                    features.max(axis=0),
                    features.min(axis=0),
                    features.std(axis=0),
                )
            )
        )
    pooled = (
        np.asarray(rows, dtype=np.float32)
        if rows
        else np.empty((0, 4 * len(NODE_FEATURE_NAMES)), dtype=np.float32)
    )
    return np.column_stack((pooled, graph.track_features)).astype(np.float32, copy=False)


def atomic_delete_from_graph(
    scores,
    graph,
    *,
    cutoff,
    component_pure_fp_probabilities=None,
    track_pure_fp_probabilities=None,
    mode="track",
    enabled=True,
):
    """Apply only complete-track or complete-component zeroing edits."""

    source_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    output = source_scores.copy()
    if not 0.0 <= float(cutoff) <= 1.0:
        raise ValueError("cutoff must be in [0,1]")
    if mode not in {"track", "component", "component_track_consensus"}:
        raise ValueError("unsupported atomic graph edit mode")
    component_probabilities = None
    if component_pure_fp_probabilities is not None:
        component_probabilities = np.asarray(
            component_pure_fp_probabilities, dtype=np.float64
        ).reshape(-1)
        if component_probabilities.size != len(graph.event_indices):
            raise ValueError("component probabilities and graph differ")
    track_probabilities = None
    if track_pure_fp_probabilities is not None:
        track_probabilities = np.asarray(
            track_pure_fp_probabilities, dtype=np.float64
        ).reshape(-1)
        if track_probabilities.size != len(graph.track_component_rows):
            raise ValueError("track probabilities and graph differ")
    if mode == "track" and track_probabilities is None:
        raise ValueError("track mode requires track probabilities")
    if mode in {"component", "component_track_consensus"} and component_probabilities is None:
        raise ValueError("component mode requires component probabilities")
    if mode == "component_track_consensus" and track_probabilities is None:
        raise ValueError("component-track consensus requires track probabilities")
    for values in (component_probabilities, track_probabilities):
        if values is not None and (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
            raise ValueError("pure-FP probabilities must be finite in [0,1]")

    delete_components = np.zeros(len(graph.event_indices), dtype=bool)
    delete_tracks = np.zeros(len(graph.track_component_rows), dtype=bool)
    if enabled:
        if mode == "track":
            delete_tracks = track_probabilities >= float(cutoff)
            for track_index in np.flatnonzero(delete_tracks):
                delete_components[graph.track_component_rows[int(track_index)]] = True
        elif mode == "component":
            delete_components = component_probabilities >= float(cutoff)
        else:
            component_track_probabilities = track_probabilities[graph.component_to_track]
            delete_components = (component_probabilities >= float(cutoff)) & (
                component_track_probabilities >= float(cutoff)
            )
            for track_index, rows in enumerate(graph.track_component_rows):
                delete_tracks[track_index] = bool(np.all(delete_components[rows]))

    deleted_mask = np.zeros(source_scores.size, dtype=bool)
    for row in np.flatnonzero(delete_components):
        indices = graph.event_indices[int(row)]
        if np.any(deleted_mask[indices]):
            raise RuntimeError("graph components overlap during atomic edit")
        deleted_mask[indices] = True
    output[deleted_mask] = np.float32(0.0)
    retained_equal = np.array_equal(output[~deleted_mask], source_scores[~deleted_mask])
    complete_tracks_only = bool(
        mode != "track"
        or all(
            np.all(delete_components[rows]) or not np.any(delete_components[rows])
            for rows in graph.track_component_rows
        )
    )
    receipt = AtomicGraphEditReceipt(
        mode=mode,
        component_count=len(graph.event_indices),
        track_count=len(graph.track_component_rows),
        deleted_component_count=int(delete_components.sum()),
        deleted_track_count=int(delete_tracks.sum()),
        deleted_event_count=int(deleted_mask.sum()),
        complete_components_only=True,
        complete_tracks_only=complete_tracks_only,
        retained_scores_bitwise_equal=bool(retained_equal),
    )
    return output, receipt


def derive_zero_observed_target_loss_cutoff(probabilities, pure_fp_targets):
    """Fit-only cutoff just above every observed target-bearing probability.

    This is an analytic risk rule, not a threshold grid.  It may deliberately
    return ``enabled=False`` when the inner OOF evidence supports no deletion.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    targets = np.asarray(pure_fp_targets, dtype=np.uint8).reshape(-1)
    if probabilities.size != targets.size or probabilities.size == 0:
        raise ValueError("cutoff probabilities and targets must align and be non-empty")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("cutoff probabilities must be finite in [0,1]")
    protected = probabilities[targets == 0]
    if protected.size == 0:
        raise ValueError("risk calibration needs target-bearing examples")
    cutoff = float(np.nextafter(protected.max(), np.inf))
    enabled = bool(cutoff <= 1.0 and np.any(probabilities[targets == 1] >= cutoff))
    return min(cutoff, 1.0), enabled


def validate_inference_feature_contract():
    names = NODE_FEATURE_NAMES + EDGE_FEATURE_NAMES + TRACK_FEATURE_NAMES
    violations = {
        name: token
        for name in names
        for token in FORBIDDEN_INFERENCE_FEATURE_TOKENS
        if token in name.lower()
    }
    if violations:
        raise RuntimeError("forbidden inference feature names: {!r}".format(violations))
    if len(set(names)) != len(names):
        raise RuntimeError("graph feature names must be globally unique")
    return True


validate_inference_feature_contract()


__all__ = (
    "AtomicGraphEditReceipt",
    "BASE_NODE_FEATURE_NAMES",
    "DECODER_CHANNELS",
    "EDGE_FEATURE_NAMES",
    "FORBIDDEN_INFERENCE_FEATURE_TOKENS",
    "GRAPH_MAX_GAP_BINS",
    "GRAPH_NEIGHBORS_PER_GAP",
    "NODE_FEATURE_NAMES",
    "TEMPORAL_BIN_SIZE",
    "TRACK_FEATURE_NAMES",
    "TemporalTrackGraph",
    "aggregate_track_node_features",
    "atomic_delete_from_graph",
    "derive_zero_observed_target_loss_cutoff",
    "extract_temporal_track_graph",
    "pure_false_positive_component_targets",
    "pure_false_positive_track_targets",
    "validate_inference_feature_contract",
)
