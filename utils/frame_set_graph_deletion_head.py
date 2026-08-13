"""Label-free frame-set graph features for conservative component deletion.

The unit of prediction is one 50-bin frame.  Every retained C00 component in
that frame receives the same keep probability, so the downstream decision is
set-valued rather than a second independent component classifier.  All graph
queries are bounded nearest-neighbour queries; no dense component-pair matrix
is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


FRAME_FEATURE_NAMES = (
    "log_video_events",
    "log_frame_component_count",
    "log_frame_component_events",
    "frame_score_top1",
    "frame_score_top2",
    "frame_score_top_gap",
    "frame_score_mean",
    "frame_score_std",
    "frame_score_hhi",
    "frame_largest_size_share",
    "frame_compactness_mean",
    "frame_bbox_span_diagonal_norm",
    "frame_nn_distance_mean_norm",
    "frame_nn_distance_min_norm",
    "graph_prev_match_r6_fraction",
    "graph_next_match_r6_fraction",
    "graph_bidirectional_r12_fraction",
    "graph_prev_distance_mean_norm",
    "graph_next_distance_mean_norm",
    "graph_score_residual_mean",
    "graph_velocity_residual_mean_norm",
    "graph_path_max_norm",
    "graph_path_ge3_fraction",
    "relative_component_count_percentile",
    "relative_top_score_percentile",
    "relative_total_events_percentile",
)


@dataclass(frozen=True)
class FrameSetGraphBatch:
    frame_bins: np.ndarray
    component_rows: tuple[np.ndarray, ...]
    features: np.ndarray


def _percentile_rank(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    ordered = np.sort(values)
    return np.searchsorted(ordered, values, side="right").astype(np.float64) / values.size


def _nearest(tree, points, target_count, upper_bound):
    if tree is None or target_count == 0:
        return (
            np.full(len(points), np.inf, dtype=np.float64),
            np.full(len(points), -1, dtype=np.int64),
        )
    distances, indices = tree.query(points, k=1, distance_upper_bound=upper_bound)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    valid = np.isfinite(distances) & (indices >= 0) & (indices < int(target_count))
    return np.where(valid, distances, np.inf), np.where(valid, indices, -1)


def extract_frame_set_graph_features(
    prediction_scores,
    locations,
    component_event_indices,
    video_event_count,
    temporal_bin_size=50,
    link_distance=6.0,
    context_distance=12.0,
):
    """Build one fixed-width, label-free feature row per non-empty frame."""

    scores = np.asarray(prediction_scores, dtype=np.float32).reshape(-1)
    locations = np.asarray(locations)
    if locations.ndim != 2 or locations.shape[1] < 4:
        raise ValueError("locations must be [N,4+] ordered [batch,x,y,t].")
    if scores.size != locations.shape[0] or scores.size != int(video_event_count):
        raise ValueError("complete-video score/location/event counts differ.")
    if not np.isfinite(scores).all():
        raise ValueError("prediction scores must be finite.")
    if temporal_bin_size <= 0 or link_distance <= 0 or context_distance < link_distance:
        raise ValueError("invalid frozen graph distances or temporal bin size.")

    component_count = len(component_event_indices)
    if component_count == 0:
        return FrameSetGraphBatch(
            np.empty(0, dtype=np.int64),
            tuple(),
            np.empty((0, len(FRAME_FEATURE_NAMES)), dtype=np.float64),
        )

    bins = np.empty(component_count, dtype=np.int64)
    centroids = np.empty((component_count, 2), dtype=np.float64)
    sizes = np.empty(component_count, dtype=np.float64)
    score_means = np.empty(component_count, dtype=np.float64)
    score_maxima = np.empty(component_count, dtype=np.float64)
    compactness = np.empty(component_count, dtype=np.float64)
    bbox_min = np.empty((component_count, 2), dtype=np.float64)
    bbox_max = np.empty((component_count, 2), dtype=np.float64)
    seen = np.zeros(scores.size, dtype=bool)

    normalized_indices = []
    for row, values in enumerate(component_event_indices):
        indices = np.asarray(values, dtype=np.int64).reshape(-1)
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= scores.size):
            raise ValueError("component event indices are empty or out of bounds.")
        if np.any(seen[indices]):
            raise ValueError("component event indices overlap.")
        seen[indices] = True
        component_bins = np.floor_divide(locations[indices, 3].astype(np.int64), temporal_bin_size)
        if np.unique(component_bins).size != 1:
            raise ValueError("a frame component crosses temporal bins.")
        xy = locations[indices, 1:3].astype(np.float64, copy=False)
        values_scores = scores[indices].astype(np.float64, copy=False)
        unique_cells = np.unique(xy.astype(np.int64), axis=0)
        minimum = xy.min(axis=0)
        maximum = xy.max(axis=0)
        extent = maximum - minimum + 1.0
        bins[row] = int(component_bins[0])
        centroids[row] = xy.mean(axis=0)
        sizes[row] = float(indices.size)
        score_means[row] = float(values_scores.mean())
        score_maxima[row] = float(values_scores.max())
        compactness[row] = float(unique_cells.shape[0] / max(float(np.prod(extent)), 1.0))
        bbox_min[row] = minimum
        bbox_max[row] = maximum
        normalized_indices.append(indices)

    by_bin = {
        int(frame_bin): np.flatnonzero(bins == frame_bin).astype(np.int64)
        for frame_bin in np.unique(bins)
    }
    trees = {
        frame_bin: cKDTree(centroids[rows])
        for frame_bin, rows in by_bin.items()
    }
    diagonal = float(np.hypot(346.0, 260.0))
    prev_distance = np.full(component_count, np.inf, dtype=np.float64)
    next_distance = np.full(component_count, np.inf, dtype=np.float64)
    prev_row = np.full(component_count, -1, dtype=np.int64)
    next_row = np.full(component_count, -1, dtype=np.int64)
    within_nn = np.full(component_count, diagonal, dtype=np.float64)

    for frame_bin, rows in by_bin.items():
        if rows.size > 1:
            distances, _ = trees[frame_bin].query(centroids[rows], k=2)
            within_nn[rows] = np.asarray(distances, dtype=np.float64)[:, 1]
        for offset, distance_output, row_output in (
            (-1, prev_distance, prev_row),
            (1, next_distance, next_row),
        ):
            neighbor_rows = by_bin.get(frame_bin + offset)
            tree = trees.get(frame_bin + offset)
            distances, local_indices = _nearest(
                tree,
                centroids[rows],
                0 if neighbor_rows is None else neighbor_rows.size,
                context_distance,
            )
            distance_output[rows] = distances
            valid = local_indices >= 0
            if neighbor_rows is not None and np.any(valid):
                row_output[rows[valid]] = neighbor_rows[local_indices[valid]]

    successor = np.where(next_distance <= link_distance, next_row, -1)
    forward = np.ones(component_count, dtype=np.float64)
    for row in np.argsort(bins, kind="stable"):
        neighbor = int(successor[row])
        if neighbor >= 0:
            forward[neighbor] = max(forward[neighbor], forward[row] + 1.0)
    backward = np.ones(component_count, dtype=np.float64)
    for row in np.argsort(bins, kind="stable")[::-1]:
        neighbor = int(successor[row])
        if neighbor >= 0:
            backward[row] = max(backward[row], backward[neighbor] + 1.0)
    path_length = forward + backward - 1.0

    velocity_residual = np.zeros(component_count, dtype=np.float64)
    score_residual = np.zeros(component_count, dtype=np.float64)
    for row in range(component_count):
        neighbors = [index for index in (int(prev_row[row]), int(next_row[row])) if index >= 0]
        if neighbors:
            score_residual[row] = float(
                np.mean(
                    np.abs(
                        np.log((score_means[np.asarray(neighbors)] + 1e-8) / (score_means[row] + 1e-8))
                    )
                )
            )
        if prev_row[row] >= 0 and next_row[row] >= 0:
            incoming = centroids[row] - centroids[prev_row[row]]
            outgoing = centroids[next_row[row]] - centroids[row]
            velocity_residual[row] = float(np.linalg.norm(outgoing - incoming) / diagonal)

    frame_bins = np.asarray(sorted(by_bin), dtype=np.int64)
    raw_rows = []
    rank_component_counts = []
    rank_top_scores = []
    rank_total_events = []
    component_rows = []
    for frame_bin in frame_bins:
        rows = by_bin[int(frame_bin)]
        component_rows.append(rows)
        frame_scores = score_maxima[rows]
        ordered_scores = np.sort(frame_scores)[::-1]
        top1 = float(ordered_scores[0])
        top2 = float(ordered_scores[1] if ordered_scores.size > 1 else ordered_scores[0])
        score_sum = float(frame_scores.sum())
        score_shares = frame_scores / max(score_sum, 1e-12)
        total_events = float(sizes[rows].sum())
        frame_minimum = bbox_min[rows].min(axis=0)
        frame_maximum = bbox_max[rows].max(axis=0)
        prev_valid = np.isfinite(prev_distance[rows])
        next_valid = np.isfinite(next_distance[rows])
        bidirectional = (prev_distance[rows] <= context_distance) & (
            next_distance[rows] <= context_distance
        )
        raw_rows.append(
            [
                np.log1p(float(video_event_count)),
                np.log1p(float(rows.size)),
                np.log1p(total_events),
                top1,
                top2,
                top1 - top2,
                float(frame_scores.mean()),
                float(frame_scores.std()),
                float(np.sum(score_shares * score_shares)),
                float(sizes[rows].max() / max(total_events, 1.0)),
                float(compactness[rows].mean()),
                float(np.linalg.norm(frame_maximum - frame_minimum) / diagonal),
                float(within_nn[rows].mean() / diagonal),
                float(within_nn[rows].min() / diagonal),
                float(np.mean(prev_distance[rows] <= link_distance)),
                float(np.mean(next_distance[rows] <= link_distance)),
                float(np.mean(bidirectional)),
                float(np.mean(np.where(prev_valid, prev_distance[rows], diagonal)) / diagonal),
                float(np.mean(np.where(next_valid, next_distance[rows], diagonal)) / diagonal),
                float(score_residual[rows].mean()),
                float(velocity_residual[rows].mean()),
                float(path_length[rows].max() / 160.0),
                float(np.mean(path_length[rows] >= 3.0)),
            ]
        )
        rank_component_counts.append(float(rows.size))
        rank_top_scores.append(top1)
        rank_total_events.append(total_events)

    features = np.column_stack(
        (
            np.asarray(raw_rows, dtype=np.float64),
            _percentile_rank(rank_component_counts),
            _percentile_rank(rank_top_scores),
            _percentile_rank(rank_total_events),
        )
    ).astype(np.float64, copy=False)
    if features.shape != (frame_bins.size, len(FRAME_FEATURE_NAMES)):
        raise RuntimeError("frame-set graph feature shape is inconsistent.")
    if not np.isfinite(features).all():
        raise RuntimeError("frame-set graph features must be finite.")
    return FrameSetGraphBatch(frame_bins, tuple(component_rows), features)


def broadcast_frame_probabilities(batch, frame_probabilities, component_count):
    """Broadcast one frame decision to all components in that frame."""

    probabilities = np.asarray(frame_probabilities, dtype=np.float64).reshape(-1)
    if probabilities.size != batch.frame_bins.size:
        raise ValueError("frame probabilities and frame rows differ in length.")
    output = np.empty(int(component_count), dtype=np.float64)
    assigned = np.zeros(int(component_count), dtype=bool)
    for probability, rows in zip(probabilities, batch.component_rows):
        rows = np.asarray(rows, dtype=np.int64)
        if np.any(rows < 0) or np.any(rows >= int(component_count)) or np.any(assigned[rows]):
            raise ValueError("frame component rows are invalid or overlapping.")
        output[rows] = float(probability)
        assigned[rows] = True
    if not assigned.all():
        raise ValueError("not every component received a frame probability.")
    return output


__all__ = (
    "FRAME_FEATURE_NAMES",
    "FrameSetGraphBatch",
    "broadcast_frame_probabilities",
    "extract_frame_set_graph_features",
)
