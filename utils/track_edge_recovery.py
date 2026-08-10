"""Label-free candidates for conservative track-end weak-event recovery.

This module deliberately stops before runtime integration.  Candidate
extraction receives only observable scores, locations, and the complete-video
event count.  Training labels and target identifiers enter through the
separate :func:`attach_training_targets` function, which makes accidental
label use in feature construction straightforward to test and audit.

The frozen MVP extends only the two ends of stable seed tracks.  It never
fills arbitrary gaps, changes an existing prediction, or recovers more than
one event at either end of a track.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


FEATURE_SEMANTICS_VERSION = "p18-seed-track-end-one-bin-v1"
FEATURE_NAMES = (
    "log_video_events",
    "log_component_events",
    "candidate_score_max",
    "candidate_score_mean",
    "candidate_score_std",
    "score_margin_to_threshold",
    "log_seed_track_bins",
    "seed_endpoint_score_max",
    "last_speed",
    "motion_residual",
    "turn_residual",
    "direction_cosine",
    "speed_ratio",
    "log_local_event_count",
    "endpoint_side",
)


@dataclass(frozen=True)
class TrackEdgeTopology:
    """Frozen observable rules used by the train-only MVP."""

    deployment_event_count_cutoff: int = 100000
    weak_score_floor: float = 0.53
    prediction_threshold: float = 0.719
    spatial_radius: int = 5
    temporal_bin_size: int = 50
    max_link_distance: float = 8.0
    max_gap_bins: int = 1
    min_seed_track_bins: int = 4
    local_density_radius: int = 5

    def __post_init__(self):
        if self.deployment_event_count_cutoff <= 0:
            raise ValueError("deployment_event_count_cutoff must be positive.")
        if not 0.0 <= self.weak_score_floor < self.prediction_threshold <= 1.0:
            raise ValueError(
                "Scores must satisfy 0 <= weak_score_floor < "
                "prediction_threshold <= 1."
            )
        if self.spatial_radius < 0:
            raise ValueError("spatial_radius must be non-negative.")
        if self.temporal_bin_size <= 0:
            raise ValueError("temporal_bin_size must be positive.")
        if self.max_link_distance < 0.0:
            raise ValueError("max_link_distance must be non-negative.")
        if self.max_gap_bins != 1:
            raise ValueError("The frozen MVP links only adjacent temporal bins.")
        if self.min_seed_track_bins < 4:
            raise ValueError("min_seed_track_bins must be at least four.")
        if self.local_density_radius < 0:
            raise ValueError("local_density_radius must be non-negative.")

    def to_dict(self):
        return {
            "deployment_event_count_cutoff": self.deployment_event_count_cutoff,
            "weak_score_floor": self.weak_score_floor,
            "prediction_threshold": self.prediction_threshold,
            "spatial_radius": self.spatial_radius,
            "temporal_bin_size": self.temporal_bin_size,
            "max_link_distance": self.max_link_distance,
            "max_gap_bins": self.max_gap_bins,
            "min_seed_track_bins": self.min_seed_track_bins,
            "local_density_radius": self.local_density_radius,
        }


FROZEN_TOPOLOGY = TrackEdgeTopology()


@dataclass(frozen=True)
class SpatialComponent:
    component_id: Tuple[int, int]
    temporal_bin: int
    event_indices: np.ndarray
    centroid: np.ndarray
    score_max: float
    score_mean: float
    score_std: float


@dataclass(frozen=True)
class SeedTrack:
    track_id: int
    components: Tuple[SpatialComponent, ...]

    @property
    def first_bin(self):
        return self.components[0].temporal_bin

    @property
    def last_bin(self):
        return self.components[-1].temporal_bin


@dataclass(frozen=True)
class TrackEdgeCandidate:
    """One label-free action candidate.

    ``endpoint_key`` groups mutually exclusive actions.  Later selection may
    recover at most one candidate event for that key.
    """

    event_index: int
    component_event_indices: np.ndarray
    endpoint_key: Tuple[int, int]
    temporal_bin: int
    endpoint_side: int
    features: np.ndarray
    raw_score: float
    motion_residual: float


@dataclass(frozen=True)
class TrackEdgeTrainingTarget:
    """Train-only supervision attached after label-free extraction."""

    label: int
    recovers_target_group: bool
    false_component_delta: int
    official_frame_index: Optional[int]


def _normalize_observable_inputs(raw_scores, baseline_scores, locations, event_count):
    raw_scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    baseline_scores = np.asarray(baseline_scores, dtype=np.float64).reshape(-1)
    locations = np.asarray(locations)
    if locations.ndim != 2 or locations.shape[1] not in (3, 4):
        raise ValueError("locations must have shape [N,3] or [N,4].")
    if locations.shape[1] == 4:
        if np.unique(locations[:, 0]).size != 1:
            raise ValueError("Candidate extraction accepts one video at a time.")
        locations = locations[:, 1:4]
    locations = locations.astype(np.int64, copy=False)
    if not (
        raw_scores.size
        == baseline_scores.size
        == locations.shape[0]
        == int(event_count)
    ):
        raise ValueError("Scores, locations, and event_count must align.")
    if not np.isfinite(raw_scores).all() or not np.isfinite(baseline_scores).all():
        raise ValueError("Candidate scores must be finite.")
    if (
        (raw_scores < 0.0).any()
        or (raw_scores > 1.0).any()
        or (baseline_scores < 0.0).any()
        or (baseline_scores > 1.0).any()
    ):
        raise ValueError("Candidate scores must be probabilities in [0,1].")
    return raw_scores, baseline_scores, locations


def _neighbor_offsets(radius):
    return tuple(
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if (dx, dy) != (0, 0)
    )


def _spatial_components_for_bin(
    event_indices,
    locations,
    scores,
    temporal_bin,
    spatial_radius,
):
    event_indices = np.asarray(event_indices, dtype=np.int64).reshape(-1)
    if event_indices.size == 0:
        return ()
    coordinates = locations[event_indices, :2]
    unique_cells, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    cell_events = [event_indices[inverse == index] for index in range(len(unique_cells))]
    lookup = {
        (int(cell[0]), int(cell[1])): index
        for index, cell in enumerate(unique_cells)
    }
    offsets = _neighbor_offsets(spatial_radius)
    visited = np.zeros(len(unique_cells), dtype=bool)
    components = []
    for start in range(len(unique_cells)):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        component_cells = []
        while stack:
            cell_index = stack.pop()
            component_cells.append(cell_index)
            x, y = unique_cells[cell_index]
            for dx, dy in offsets:
                neighbor = lookup.get((int(x + dx), int(y + dy)))
                if neighbor is not None and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        component_indices = np.concatenate(
            [cell_events[index] for index in component_cells]
        ).astype(np.int64, copy=False)
        component_indices.sort()
        component_scores = scores[component_indices]
        component_coordinates = locations[component_indices, :2].astype(
            np.float64, copy=False
        )
        components.append(
            SpatialComponent(
                component_id=(int(temporal_bin), len(components)),
                temporal_bin=int(temporal_bin),
                event_indices=component_indices,
                centroid=component_coordinates.mean(axis=0),
                score_max=float(component_scores.max()),
                score_mean=float(component_scores.mean()),
                score_std=float(component_scores.std()),
            )
        )
    return tuple(components)


def _components_by_temporal_bin(mask, locations, scores, topology):
    event_bins = np.floor_divide(locations[:, 2], topology.temporal_bin_size)
    by_bin: Dict[int, Tuple[SpatialComponent, ...]] = {}
    selected_bins = np.unique(event_bins[mask])
    for temporal_bin in selected_bins.tolist():
        indices = np.flatnonzero(mask & (event_bins == temporal_bin))
        by_bin[int(temporal_bin)] = _spatial_components_for_bin(
            indices,
            locations,
            scores,
            temporal_bin,
            topology.spatial_radius,
        )
    return by_bin, event_bins


def _link_seed_tracks(components_by_bin, topology):
    mutable_tracks = []
    for temporal_bin in sorted(components_by_bin):
        components = components_by_bin[temporal_bin]
        links = []
        for track_index, track in enumerate(mutable_tracks):
            gap = temporal_bin - track[-1].temporal_bin
            if gap != topology.max_gap_bins:
                continue
            for component_index, component in enumerate(components):
                distance = float(
                    np.linalg.norm(component.centroid - track[-1].centroid)
                )
                if distance <= topology.max_link_distance:
                    links.append((distance, track_index, component_index))
        assigned_tracks = set()
        assigned_components = set()
        for _, track_index, component_index in sorted(links):
            if track_index in assigned_tracks or component_index in assigned_components:
                continue
            mutable_tracks[track_index].append(components[component_index])
            assigned_tracks.add(track_index)
            assigned_components.add(component_index)
        for component_index, component in enumerate(components):
            if component_index not in assigned_components:
                mutable_tracks.append([component])
    stable = []
    for components in mutable_tracks:
        if len(components) >= topology.min_seed_track_bins:
            stable.append(SeedTrack(track_id=len(stable), components=tuple(components)))
    return tuple(stable)


def _endpoint_motion(track, endpoint_side):
    if endpoint_side not in (-1, 1):
        raise ValueError("endpoint_side must be -1 or +1.")
    if len(track.components) < 3:
        raise ValueError("Endpoint motion requires at least three seed components.")
    if endpoint_side < 0:
        endpoint, inward, inward2 = track.components[:3]
    else:
        endpoint, inward, inward2 = track.components[-1:-4:-1]
    outward_velocity = endpoint.centroid - inward.centroid
    previous_outward_velocity = inward.centroid - inward2.centroid
    predicted_centroid = endpoint.centroid + outward_velocity
    last_speed = float(np.linalg.norm(outward_velocity))
    turn_residual = float(
        np.linalg.norm(outward_velocity - previous_outward_velocity)
    )
    return endpoint, predicted_centroid, outward_velocity, last_speed, turn_residual


def _direction_features(outward_velocity, endpoint_centroid, candidate_centroid):
    candidate_displacement = candidate_centroid - endpoint_centroid
    last_speed = float(np.linalg.norm(outward_velocity))
    candidate_speed = float(np.linalg.norm(candidate_displacement))
    denominator = last_speed * candidate_speed
    direction_cosine = (
        float(np.dot(outward_velocity, candidate_displacement) / denominator)
        if denominator > 1e-12
        else 0.0
    )
    speed_ratio = candidate_speed / (last_speed + 1e-6)
    return direction_cosine, speed_ratio


def _local_event_count(locations, event_bins, temporal_bin, x, y, radius):
    indices = np.flatnonzero(event_bins == int(temporal_bin))
    if indices.size == 0:
        return 0
    coordinates = locations[indices, :2]
    return int(
        np.sum(
            (np.abs(coordinates[:, 0] - int(x)) <= radius)
            & (np.abs(coordinates[:, 1] - int(y)) <= radius)
        )
    )


def extract_track_edge_candidates(
    raw_scores,
    baseline_scores,
    locations,
    event_count,
    topology=FROZEN_TOPOLOGY,
):
    """Return deterministic label-free recovery candidates.

    No label or target-id argument is accepted.  Stable tracks are built from
    final C00 positives, while weak components use the immutable raw M20
    scores.  The deployment cutoff is intentionally *not* applied here so
    middle-density train videos can supply a separately weighted auxiliary
    fit domain.  Deployment eligibility is enforced by the cross-source
    protocol and, later, by any runtime integration.
    """
    if not isinstance(topology, TrackEdgeTopology):
        raise TypeError("topology must be TrackEdgeTopology.")
    raw_scores, baseline_scores, locations = _normalize_observable_inputs(
        raw_scores, baseline_scores, locations, event_count
    )
    seed_mask = baseline_scores >= topology.prediction_threshold
    weak_mask = (
        (raw_scores >= topology.weak_score_floor)
        & (raw_scores < topology.prediction_threshold)
        & ~seed_mask
    )
    if not seed_mask.any() or not weak_mask.any():
        return ()
    seed_components, event_bins = _components_by_temporal_bin(
        seed_mask, locations, baseline_scores, topology
    )
    tracks = _link_seed_tracks(seed_components, topology)
    if not tracks:
        return ()
    weak_components, _ = _components_by_temporal_bin(
        weak_mask, locations, raw_scores, topology
    )

    # A weak component may be close to two seed endpoints.  Assign it once,
    # using motion residual before endpoint distance and deterministic IDs.
    assignments = {}
    endpoint_metadata = {}
    for track in tracks:
        for endpoint_side in (-1, 1):
            endpoint, predicted, velocity, last_speed, turn = _endpoint_motion(
                track, endpoint_side
            )
            candidate_bin = endpoint.temporal_bin + endpoint_side
            endpoint_key = (track.track_id, endpoint_side)
            endpoint_metadata[endpoint_key] = (
                track,
                endpoint,
                predicted,
                velocity,
                last_speed,
                turn,
            )
            for component in weak_components.get(candidate_bin, ()):
                endpoint_distance = float(
                    np.linalg.norm(component.centroid - endpoint.centroid)
                )
                if endpoint_distance > topology.max_link_distance:
                    continue
                motion_residual = float(
                    np.linalg.norm(component.centroid - predicted)
                )
                ordering = (
                    motion_residual,
                    endpoint_distance,
                    track.track_id,
                    endpoint_side,
                )
                previous = assignments.get(component.component_id)
                if previous is None or ordering < previous[0]:
                    assignments[component.component_id] = (
                        ordering,
                        endpoint_key,
                        component,
                    )

    candidates = []
    for _, endpoint_key, component in sorted(
        assignments.values(),
        key=lambda item: (item[1][0], item[1][1], item[2].component_id),
    ):
        track, endpoint, predicted, velocity, last_speed, turn = endpoint_metadata[
            endpoint_key
        ]
        component_scores = raw_scores[component.event_indices]
        event_index = int(
            sorted(
                component.event_indices.tolist(),
                key=lambda index: (-raw_scores[index], index),
            )[0]
        )
        event_x, event_y = locations[event_index, :2]
        motion_residual = float(np.linalg.norm(component.centroid - predicted))
        direction_cosine, speed_ratio = _direction_features(
            velocity, endpoint.centroid, component.centroid
        )
        local_count = _local_event_count(
            locations,
            event_bins,
            component.temporal_bin,
            event_x,
            event_y,
            topology.local_density_radius,
        )
        endpoint_scores = baseline_scores[endpoint.event_indices]
        features = np.asarray(
            (
                math.log1p(int(event_count)),
                math.log1p(component.event_indices.size),
                float(component_scores.max()),
                float(component_scores.mean()),
                float(component_scores.std()),
                topology.prediction_threshold - float(component_scores.max()),
                math.log1p(len(track.components)),
                float(endpoint_scores.max()),
                last_speed,
                motion_residual,
                turn,
                direction_cosine,
                speed_ratio,
                math.log1p(local_count),
                float(endpoint_key[1]),
            ),
            dtype=np.float64,
        )
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise RuntimeError("Track-edge candidate features are invalid.")
        candidates.append(
            TrackEdgeCandidate(
                event_index=event_index,
                component_event_indices=component.event_indices.copy(),
                endpoint_key=endpoint_key,
                temporal_bin=component.temporal_bin,
                endpoint_side=endpoint_key[1],
                features=features,
                raw_score=float(raw_scores[event_index]),
                motion_residual=motion_residual,
            )
        )
    return tuple(candidates)


def _official_frame_index(timestamp, temporal_bin_size):
    timestamp = int(timestamp)
    if timestamp <= 0 or timestamp % int(temporal_bin_size) == 0:
        return None
    return timestamp // int(temporal_bin_size)


def _connected_component_labels(cells):
    cells = set(cells)
    labels = {}
    component_id = 0
    offsets = _neighbor_offsets(1)
    for start in sorted(cells):
        if start in labels:
            continue
        labels[start] = component_id
        stack = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in offsets:
                neighbor = (x + dx, y + dy)
                if neighbor in cells and neighbor not in labels:
                    labels[neighbor] = component_id
                    stack.append(neighbor)
        component_id += 1
    return labels


def attach_training_targets(
    candidates,
    labels,
    target_ids,
    baseline_scores,
    locations,
    topology=FROZEN_TOPOLOGY,
    correct_threshold=0.0001,
):
    """Attach train-only labels without changing candidate features.

    ``false_component_delta`` is the exact local 8-connected change caused by
    adding a previously absent false cell.  The fitting script conservatively
    clips negative merge deltas to zero when constructing metric utility, so a
    false event can never receive a positive training reward.
    """
    candidates = tuple(candidates)
    labels = np.asarray(labels).reshape(-1)
    target_ids = np.asarray(target_ids).reshape(-1)
    baseline_scores = np.asarray(baseline_scores, dtype=np.float64).reshape(-1)
    locations = np.asarray(locations)
    if locations.ndim != 2 or locations.shape[1] not in (3, 4):
        raise ValueError("locations must have shape [N,3] or [N,4].")
    if locations.shape[1] == 4:
        if np.unique(locations[:, 0]).size != 1:
            raise ValueError("Training targets accept one video at a time.")
        locations = locations[:, 1:4]
    locations = locations.astype(np.int64, copy=False)
    correct_threshold = float(correct_threshold)
    if not math.isfinite(correct_threshold) or not 0.0 < correct_threshold <= 1.0:
        raise ValueError("correct_threshold must be finite and in (0,1].")
    if not (
        labels.size
        == target_ids.size
        == baseline_scores.size
        == locations.shape[0]
    ):
        raise ValueError("Training target arrays must align.")
    if not np.isin(labels, (0, 1, 0.0, 1.0)).all():
        raise ValueError("Training labels must be binary.")
    if not np.issubdtype(target_ids.dtype, np.integer):
        if not (
            np.isfinite(target_ids).all()
            and np.equal(target_ids, np.floor(target_ids)).all()
        ):
            raise ValueError("Training target IDs must be finite integers.")
    target_ids = target_ids.astype(np.int64, copy=False)
    binary_labels = labels > 0.5
    if not np.array_equal(binary_labels, target_ids > 0):
        raise ValueError(
            "Positive labels and positive target IDs must identify the same events."
        )
    baseline_positive = baseline_scores >= topology.prediction_threshold

    target_group_total: Dict[Tuple[int, int], int] = {}
    target_group_correct: Dict[Tuple[int, int], int] = {}
    # The unchanged evaluator accumulates false-event multiplicity in a uint8
    # image before connected components.  Preserve counts (rather than only
    # occupied cells) so the rare 255 -> 0 wrap on one added event is exact.
    false_cell_counts_by_frame: Dict[int, Dict[Tuple[int, int], int]] = {}
    for index in range(labels.size):
        frame = _official_frame_index(
            locations[index, 2], topology.temporal_bin_size
        )
        if frame is None:
            continue
        if binary_labels[index] and int(target_ids[index]) > 0:
            group = (frame, int(target_ids[index]))
            target_group_total[group] = target_group_total.get(group, 0) + 1
            if baseline_positive[index]:
                target_group_correct[group] = target_group_correct.get(group, 0) + 1
        elif not binary_labels[index] and baseline_positive[index]:
            cell = (int(locations[index, 0]), int(locations[index, 1]))
            cell_counts = false_cell_counts_by_frame.setdefault(frame, {})
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
    false_cells_by_frame = {
        frame: {
            cell for cell, count in cell_counts.items() if count % 256 != 0
        }
        for frame, cell_counts in false_cell_counts_by_frame.items()
    }
    false_labels_by_frame = {
        frame: _connected_component_labels(cells)
        for frame, cells in false_cells_by_frame.items()
    }

    targets = []
    for candidate in candidates:
        index = int(candidate.event_index)
        if index < 0 or index >= labels.size:
            raise ValueError("Candidate event index is outside the training record.")
        if baseline_positive[index]:
            raise ValueError("A recovery candidate must be baseline-negative.")
        frame = _official_frame_index(
            locations[index, 2], topology.temporal_bin_size
        )
        label = int(binary_labels[index])
        recovers_target_group = False
        false_component_delta = 0
        if label:
            target_id = int(target_ids[index])
            if target_id <= 0:
                raise ValueError("Positive training events must have positive target IDs.")
            if frame is not None:
                group = (frame, target_id)
                group_total = target_group_total.get(group, 0)
                group_correct = target_group_correct.get(group, 0)
                if group_total <= 0:
                    raise RuntimeError("Positive candidate has no official target group.")
                detected_before = (
                    group_correct / group_total >= correct_threshold
                )
                detected_after = (
                    (group_correct + 1) / group_total >= correct_threshold
                )
                recovers_target_group = not detected_before and detected_after
        elif frame is not None:
            cell = (int(locations[index, 0]), int(locations[index, 1]))
            component_labels = false_labels_by_frame.get(frame, {})
            baseline_count = false_cell_counts_by_frame.get(frame, {}).get(cell, 0)
            present_before = baseline_count % 256 != 0
            present_after = (baseline_count + 1) % 256 != 0
            if not present_before and present_after:
                neighbor_components = {
                    component_labels[neighbor]
                    for dx, dy in _neighbor_offsets(1)
                    for neighbor in ((cell[0] + dx, cell[1] + dy),)
                    if neighbor in component_labels
                }
                false_component_delta = 1 - len(neighbor_components)
            elif present_before and not present_after:
                before_cells = set(component_labels)
                after_cells = before_cells - {cell}
                before_components = len(set(component_labels.values()))
                after_components = len(
                    set(_connected_component_labels(after_cells).values())
                )
                false_component_delta = after_components - before_components
        targets.append(
            TrackEdgeTrainingTarget(
                label=label,
                recovers_target_group=bool(recovers_target_group),
                false_component_delta=int(false_component_delta),
                official_frame_index=frame,
            )
        )
    return tuple(targets)


def select_endpoint_recoveries(candidates, logits, decision_logit=0.0):
    """Select at most one event per endpoint from fixed model logits."""
    candidates = tuple(candidates)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if logits.size != len(candidates):
        raise ValueError("Candidate and logit counts differ.")
    if not np.isfinite(logits).all() or not math.isfinite(float(decision_logit)):
        raise ValueError("Recovery logits and decision_logit must be finite.")
    by_endpoint: Dict[Tuple[int, int], list] = {}
    for candidate, logit in zip(candidates, logits.tolist()):
        by_endpoint.setdefault(candidate.endpoint_key, []).append((logit, candidate))
    selected = []
    for endpoint_key in sorted(by_endpoint):
        best_logit, best_candidate = sorted(
            by_endpoint[endpoint_key],
            key=lambda item: (-item[0], item[1].event_index),
        )[0]
        if best_logit >= float(decision_logit):
            selected.append(best_candidate.event_index)
    return np.asarray(sorted(set(selected)), dtype=np.int64)
