"""Continuous label-free context features for all-size component deletion."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from utils.allsize_deletion_head import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    AllSizeComponentBatch,
    extract_allsize_components as extract_base_components,
    suppress_components,
)


QUANTILE_NAMES = tuple(f"component_score_q{q}" for q in (10, 25, 50, 75, 90))
SHAPE_NAMES = (
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "cell_fill_ratio",
    "events_per_cell",
    "cov_minor_major_ratio",
    "radial_compactness",
    "normalized_border_distance",
    "relative_log_size_z",
    "relative_score_mean_z",
    "relative_score_max_z",
    "cell_active_bins_mean",
    "cell_active_bins_max",
)
GAP_NAMES = tuple(
    f"gap_{direction}{gap}_{value}"
    for direction in ("prev", "next")
    for gap in (1, 2)
    for value in ("present", "distance", "log_score_ratio", "log_size_ratio")
)
GRAPH_NAMES = (
    "graph_prev_degree",
    "graph_next_degree",
    "graph_longest_path",
    "graph_velocity_residual",
)
CONTEXT_NAMES = tuple(
    f"context_r{radius}_dt{offset:+d}_{stat}"
    for radius in (1, 2)
    for offset in (-2, -1, 0, 1, 2)
    for stat in ("event_count", "score_sum", "score_mean", "score_max")
)
FEATURE_NAMES = BASE_FEATURE_NAMES + QUANTILE_NAMES + SHAPE_NAMES + GAP_NAMES + GRAPH_NAMES + CONTEXT_NAMES


def _zscore(values):
    values = np.asarray(values, dtype=np.float64)
    scale = float(values.std())
    return (values - float(values.mean())) / (scale if scale >= 1e-8 else 1.0)


def _dense_context_maps(scores, locations):
    coordinates = np.asarray(locations)[:, 1:4].astype(np.int64, copy=False)
    x = np.clip(coordinates[:, 0], 0, 345)
    y = np.clip(coordinates[:, 1], 0, 259)
    temporal_bin = np.floor_divide(coordinates[:, 2], 50)
    bin_count = int(temporal_bin.max()) + 1
    shape = (bin_count, 260, 346)
    counts = np.zeros(shape, dtype=np.uint32)
    sums = np.zeros(shape, dtype=np.float32)
    maxima = np.zeros(shape, dtype=np.float32)
    index = (temporal_bin, y, x)
    np.add.at(counts, index, 1)
    np.add.at(sums, index, np.asarray(scores, dtype=np.float32))
    np.maximum.at(maxima, index, np.asarray(scores, dtype=np.float32))
    return counts, sums, maxima


def _graph_context(component_bins, centroids, score_means, sizes, max_distance):
    count = len(component_bins)
    by_bin = defaultdict(list)
    for index, temporal_bin in enumerate(component_bins):
        by_bin[int(temporal_bin)].append(index)
    prev_degree = np.zeros(count, dtype=np.float64)
    next_degree = np.zeros(count, dtype=np.float64)
    edges = [[] for _ in range(count)]
    gap_features = np.zeros((count, len(GAP_NAMES)), dtype=np.float64)
    for index in range(count):
        column = 0
        for direction, sign in (("prev", -1), ("next", 1)):
            for gap in (1, 2):
                candidates = by_bin.get(int(component_bins[index] + sign * gap), ())
                if candidates:
                    distances = np.linalg.norm(
                        centroids[np.asarray(candidates)] - centroids[index], axis=1
                    )
                    local = int(np.argmin(distances))
                    neighbor = int(candidates[local])
                    gap_features[index, column : column + 4] = (
                        1.0,
                        float(distances[local]) / np.hypot(346.0, 260.0),
                        float(np.log((score_means[neighbor] + 1e-8) / (score_means[index] + 1e-8))),
                        float(np.log((sizes[neighbor] + 1.0) / (sizes[index] + 1.0))),
                    )
                column += 4
        next_candidates = by_bin.get(int(component_bins[index] + 1), ())
        for neighbor in next_candidates:
            if float(np.linalg.norm(centroids[neighbor] - centroids[index])) <= max_distance:
                edges[index].append(int(neighbor))
                next_degree[index] += 1.0
                prev_degree[neighbor] += 1.0

    forward = np.ones(count, dtype=np.float64)
    for index in np.argsort(component_bins):
        for neighbor in edges[int(index)]:
            forward[neighbor] = max(forward[neighbor], forward[index] + 1.0)
    backward = np.ones(count, dtype=np.float64)
    for index in np.argsort(component_bins)[::-1]:
        for neighbor in edges[int(index)]:
            backward[index] = max(backward[index], backward[neighbor] + 1.0)
    longest = forward + backward - 1.0

    # Undirected graph components provide a deterministic local motion residual.
    undirected = [set(items) for items in edges]
    for left, items in enumerate(edges):
        for right in items:
            undirected[right].add(left)
    residual = np.zeros(count, dtype=np.float64)
    visited = np.zeros(count, dtype=bool)
    for start in range(count):
        if visited[start]:
            continue
        queue = deque([start]); visited[start] = True; members = []
        while queue:
            current = queue.popleft(); members.append(current)
            for neighbor in undirected[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True; queue.append(neighbor)
        member_array = np.asarray(members, dtype=np.int64)
        if member_array.size >= 3 and np.unique(component_bins[member_array]).size >= 2:
            time = component_bins[member_array].astype(np.float64)
            design = np.column_stack((time, np.ones_like(time)))
            fitted_x = design @ np.linalg.lstsq(design, centroids[member_array, 0], rcond=None)[0]
            fitted_y = design @ np.linalg.lstsq(design, centroids[member_array, 1], rcond=None)[0]
            value = float(np.sqrt(np.mean((centroids[member_array, 0] - fitted_x) ** 2 + (centroids[member_array, 1] - fitted_y) ** 2)))
            residual[member_array] = value / np.hypot(346.0, 260.0)
    return gap_features, np.column_stack((prev_degree, next_degree, longest, residual))


def extract_allsize_components(
    prediction_scores,
    locations,
    prediction_threshold,
    topology,
    video_event_count,
    labels=None,
    context_scores=None,
):
    """Return base features plus continuous spatial/temporal score context."""

    base = extract_base_components(
        prediction_scores,
        locations,
        prediction_threshold,
        topology,
        video_event_count,
        labels=labels,
    )
    if base.features.shape[0] == 0:
        return AllSizeComponentBatch(base.event_indices, np.empty((0, len(FEATURE_NAMES))), base.labels)
    scores = np.asarray(prediction_scores, dtype=np.float32).reshape(-1)
    context = scores if context_scores is None else np.asarray(context_scores, dtype=np.float32).reshape(-1)
    locations = np.asarray(locations)
    if context.size != scores.size:
        raise ValueError("context scores and prediction scores differ in length.")
    counts_map, sums_map, maxima_map = _dense_context_maps(context, locations)

    component_count = len(base.event_indices)
    bins = np.empty(component_count, dtype=np.int64)
    centroids = np.empty((component_count, 2), dtype=np.float64)
    sizes = np.empty(component_count, dtype=np.float64)
    score_means = np.empty(component_count, dtype=np.float64)
    score_maxima = np.empty(component_count, dtype=np.float64)
    quantiles = np.empty((component_count, len(QUANTILE_NAMES)), dtype=np.float64)
    shapes = np.empty((component_count, len(SHAPE_NAMES)), dtype=np.float64)

    positive_cells = locations[scores >= np.float32(prediction_threshold), 1:4].astype(np.int64, copy=False)
    if positive_cells.size:
        flat = np.unique(np.floor_divide(positive_cells[:, 2], 50) * (260 * 346) + positive_cells[:, 1] * 346 + positive_cells[:, 0])
        active_bins = np.bincount(flat % (260 * 346), minlength=260 * 346).reshape(260, 346)
    else:
        active_bins = np.zeros((260, 346), dtype=np.int64)

    raw_shape_parts = []
    for row, indices in enumerate(base.event_indices):
        coordinates = locations[indices, 1:4].astype(np.float64, copy=False)
        values = scores[indices].astype(np.float64, copy=False)
        xy = coordinates[:, :2]
        unique = np.unique(xy.astype(np.int64), axis=0)
        extent = xy.max(axis=0) - xy.min(axis=0) + 1.0
        width, height = float(extent[0]), float(extent[1])
        area = width * height
        centered = xy - xy.mean(axis=0)
        covariance = centered.T @ centered / max(1, xy.shape[0])
        eigenvalues = np.linalg.eigvalsh(covariance)
        radial = np.sqrt(np.sum(centered * centered, axis=1))
        border = min(float(xy[:, 0].min()), float(345 - xy[:, 0].max()), float(xy[:, 1].min()), float(259 - xy[:, 1].max())) / 259.0
        activity = active_bins[unique[:, 1], unique[:, 0]].astype(np.float64)
        bins[row] = int(np.floor_divide(int(coordinates[0, 2]), 50))
        centroids[row] = xy.mean(axis=0)
        sizes[row] = float(indices.size)
        score_means[row] = float(values.mean())
        score_maxima[row] = float(values.max())
        quantiles[row] = np.quantile(values, (0.10, 0.25, 0.50, 0.75, 0.90))
        raw_shape_parts.append((width, height, area, len(unique) / area, indices.size / len(unique), float(eigenvalues[0] / (eigenvalues[-1] + 1e-8)), float(indices.size / (np.pi * (float(radial.max()) + 1.0) ** 2)), border, float(activity.mean()), float(activity.max())))

    size_z = _zscore(np.log1p(sizes)); mean_z = _zscore(score_means); max_z = _zscore(score_maxima)
    for row, values in enumerate(raw_shape_parts):
        shapes[row] = values[:8] + (float(size_z[row]), float(mean_z[row]), float(max_z[row])) + values[8:]
    gap, graph = _graph_context(bins, centroids, score_means, sizes, float(topology.max_link_distance))

    context_features = np.zeros((component_count, len(CONTEXT_NAMES)), dtype=np.float64)
    for row in range(component_count):
        x = int(np.clip(np.rint(centroids[row, 0]), 0, 345)); y = int(np.clip(np.rint(centroids[row, 1]), 0, 259)); column = 0
        for radius in (1, 2):
            x0, x1 = max(0, x - radius), min(346, x + radius + 1); y0, y1 = max(0, y - radius), min(260, y + radius + 1)
            for offset in (-2, -1, 0, 1, 2):
                temporal_bin = int(bins[row] + offset)
                if 0 <= temporal_bin < counts_map.shape[0]:
                    count = float(counts_map[temporal_bin, y0:y1, x0:x1].sum()); score_sum = float(sums_map[temporal_bin, y0:y1, x0:x1].sum()); score_max = float(maxima_map[temporal_bin, y0:y1, x0:x1].max())
                    score_mean = score_sum / count if count else 0.0
                    context_features[row, column:column + 4] = (np.log1p(count), score_sum, score_mean, score_max)
                column += 4
    features = np.column_stack((base.features, quantiles, shapes, gap, graph, context_features)).astype(np.float64, copy=False)
    if features.shape != (component_count, len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise RuntimeError("invalid contextual feature matrix.")
    return AllSizeComponentBatch(base.event_indices, features, base.labels)


__all__ = ("FEATURE_NAMES", "extract_allsize_components", "suppress_components")
