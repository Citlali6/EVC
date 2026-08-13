"""Label-free all-size component features for a post-C00 deletion head.

This module deliberately knows nothing about dataset source names or folds.  A
caller supplies one complete video's scores/locations and, during train-only
fitting, optional event labels.  Components are the frozen per-50-bin,
8-connected topology already used by the component-reranker work, but no
component-size cutoff is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from utils.component_reranker import ComponentTopology, _build_components_and_tracks


FEATURE_NAMES = (
    "log_video_events",
    "log_component_events",
    "score_max",
    "score_mean",
    "score_min",
    "score_std",
    "score_margin_max",
    "log_unique_cells",
    "bbox_diagonal",
    "track_bin_count",
    "log_track_events",
    "track_score_max",
    "track_score_mean",
    "track_displacement_per_bin",
)


@dataclass(frozen=True)
class AllSizeComponentBatch:
    event_indices: tuple[np.ndarray, ...]
    features: np.ndarray
    labels: Optional[np.ndarray]


def extract_allsize_components(
    prediction_scores,
    locations,
    prediction_threshold: float,
    topology: ComponentTopology,
    video_event_count: int,
    labels=None,
    context_scores=None,
) -> AllSizeComponentBatch:
    """Extract every retained component and frozen cross-bin track features."""

    scores = np.asarray(prediction_scores, dtype=np.float32).reshape(-1)
    locations = np.asarray(locations)
    if locations.ndim != 2 or locations.shape[1] < 4:
        raise ValueError("locations must be [N,4+] ordered [batch,x,y,t].")
    if scores.size != locations.shape[0] or scores.size != int(video_event_count):
        raise ValueError("complete-video score/location/event counts differ.")
    if not np.isfinite(scores).all():
        raise ValueError("prediction scores must be finite.")
    if not isinstance(topology, ComponentTopology):
        raise TypeError("topology must be ComponentTopology.")
    label_values = None if labels is None else np.asarray(labels).reshape(-1)
    if label_values is not None and label_values.size != scores.size:
        raise ValueError("labels and scores differ in length.")

    effective_threshold = np.float32(prediction_threshold)
    all_indices = []
    all_features = []
    all_labels = []
    for batch_id in np.unique(locations[:, 0]):
        positive_indices = np.flatnonzero(
            (locations[:, 0] == batch_id) & (scores >= effective_threshold)
        )
        if positive_indices.size == 0:
            continue
        positive_scores = scores[positive_indices].astype(np.float64, copy=False)
        positive_coordinates = locations[positive_indices, 1:4].astype(
            np.int64, copy=False
        )
        components, tracks, component_to_track = _build_components_and_tracks(
            positive_scores, positive_coordinates, topology
        )
        for component_index, component in enumerate(components):
            local_indices = component["event_indices"]
            event_indices = positive_indices[local_indices].astype(np.int64, copy=False)
            component_scores = positive_scores[local_indices]
            component_coordinates = positive_coordinates[local_indices]
            track = tracks[int(component_to_track[component_index])]
            track_component_indices = track["component_indices"]
            track_event_indices = np.concatenate(
                [components[index]["event_indices"] for index in track_component_indices]
            )
            track_scores = positive_scores[track_event_indices]
            bin_span = max(int(track["last_bin"] - track["first_bin"]), 1)
            displacement = float(
                np.linalg.norm(track["last_centroid"] - track["first_centroid"])
                / bin_span
            )
            unique_cells = np.unique(component_coordinates[:, :2], axis=0).shape[0]
            extent = (
                component_coordinates[:, :2].max(axis=0)
                - component_coordinates[:, :2].min(axis=0)
            )
            values = np.asarray(
                (
                    math.log1p(int(video_event_count)),
                    math.log1p(int(event_indices.size)),
                    float(component_scores.max()),
                    float(component_scores.mean()),
                    float(component_scores.min()),
                    float(component_scores.std()),
                    float(component_scores.max() - prediction_threshold),
                    math.log1p(int(unique_cells)),
                    float(np.linalg.norm(extent)),
                    float(len(track_component_indices)),
                    math.log1p(int(track_event_indices.size)),
                    float(track_scores.max()),
                    float(track_scores.mean()),
                    displacement,
                ),
                dtype=np.float64,
            )
            if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
                raise RuntimeError("invalid all-size component feature row.")
            all_indices.append(event_indices)
            all_features.append(values)
            if label_values is not None:
                all_labels.append(int(np.any(label_values[event_indices] > 0.5)))

    features = (
        np.stack(all_features).astype(np.float64, copy=False)
        if all_features
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    )
    component_labels = (
        None
        if label_values is None
        else np.asarray(all_labels, dtype=np.uint8)
    )
    return AllSizeComponentBatch(tuple(all_indices), features, component_labels)


def suppress_components(scores, event_indices, keep_probabilities, keep_threshold):
    """Return a score copy with low-probability whole components suppressed."""

    output = np.asarray(scores, dtype=np.float32).reshape(-1).copy()
    probabilities = np.asarray(keep_probabilities, dtype=np.float64).reshape(-1)
    if len(event_indices) != probabilities.size:
        raise ValueError("component indices/probabilities differ in length.")
    for indices, probability in zip(event_indices, probabilities):
        if float(probability) < float(keep_threshold):
            output[np.asarray(indices, dtype=np.int64)] = np.float32(0.0)
    return output
