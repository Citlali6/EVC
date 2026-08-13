"""Source-free helpers for atomic post-C00 component deletion.

The only mutable unit is one complete connected component of the frozen M20
``0.719 + C00`` output.  Components use the same x/y/time-bin neighbourhood as
C00.  Labels are accepted only by :func:`pure_false_positive_targets`; they are
never accepted by component construction, patch queries, calibration scoring,
or the atomic edit itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


H2_EVENT_COUNT_CUTOFF = 200000
H2_POLARITY_MINORITY_CUTOFF = 0.20


@dataclass(frozen=True)
class AtomicComponentBatch:
    """A deterministic partition of all post-C00 threshold-positive events."""

    event_indices: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ComponentPatchQuery:
    """One recentered spatial query for one occupied bin of a component."""

    temporal_bin: int
    center_x: int
    center_y: int
    component_mask: np.ndarray


@dataclass(frozen=True)
class AtomicEditReceipt:
    enabled: bool
    fallback_reason: Optional[str]
    component_count: int
    deleted_component_count: int
    kept_component_count: int
    deleted_event_count: int
    retained_scores_bitwise_equal: bool
    complete_components_only: bool


def complete_input_polarity_minority_fraction(polarities) -> float:
    """Return the minority fraction of a complete normalized polarity vector."""

    values = np.asarray(polarities)
    if values.ndim != 1 or values.size <= 0:
        raise ValueError("complete-input polarities must be a non-empty vector")
    if values.dtype.kind not in "biuf":
        raise TypeError("polarities must be numeric")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("normalized polarities must be finite and lie in [0,1]")
    positive = int(np.count_nonzero(values > 0.5))
    return float(min(positive, int(values.size) - positive) / int(values.size))


def use_h2_atomic_deletion(event_count, polarities) -> bool:
    """Apply V3 only to the pre-existing input-only H2 domain."""

    if isinstance(event_count, (bool, np.bool_)):
        raise ValueError("event_count must be a non-negative integer")
    event_count = int(event_count)
    if event_count < 0 or event_count != len(polarities):
        raise ValueError("event_count must match the complete polarity vector")
    return bool(
        event_count > H2_EVENT_COUNT_CUTOFF
        and complete_input_polarity_minority_fraction(polarities)
        >= H2_POLARITY_MINORITY_CUTOFF
    )


def _validate_score_location_inputs(scores, locations):
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    coordinates = np.asarray(locations)
    if coordinates.ndim != 2 or coordinates.shape[1] < 4:
        raise ValueError("locations must have shape [N,4+] ordered [batch,x,y,t]")
    if coordinates.shape[0] != values.size:
        raise ValueError("scores and locations must align")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return values, coordinates


def extract_atomic_components(
    scores,
    locations,
    prediction_threshold: float,
    *,
    spatial_radius: int = 2,
    temporal_bin_size: int = 50,
    temporal_radius_bins: int = 1,
) -> AtomicComponentBatch:
    """Partition post-C00 positives with C00's exact cell connectivity.

    Event multiplicity is retained, while graph connectivity is defined over
    unique ``(x, y, temporal_bin)`` cells just as in ``P0ClusterFilter``.
    """

    values, coordinates = _validate_score_location_inputs(scores, locations)
    spatial_radius = int(spatial_radius)
    temporal_bin_size = int(temporal_bin_size)
    temporal_radius_bins = int(temporal_radius_bins)
    if spatial_radius < 0 or temporal_bin_size <= 0 or temporal_radius_bins < 0:
        raise ValueError("invalid component topology")
    threshold = np.float32(prediction_threshold)
    offsets = tuple(
        (dx, dy, dt)
        for dx in range(-spatial_radius, spatial_radius + 1)
        for dy in range(-spatial_radius, spatial_radius + 1)
        for dt in range(-temporal_radius_bins, temporal_radius_bins + 1)
        if (dx, dy, dt) != (0, 0, 0)
    )
    output = []
    positive_global = values >= threshold
    for batch_id in np.unique(coordinates[:, 0]):
        positive_indices = np.flatnonzero(
            (coordinates[:, 0] == batch_id) & positive_global
        )
        if positive_indices.size == 0:
            continue
        positive_locations = coordinates[positive_indices, 1:4].astype(
            np.int64, copy=False
        )
        cells = np.column_stack(
            (
                positive_locations[:, 0],
                positive_locations[:, 1],
                np.floor_divide(positive_locations[:, 2], temporal_bin_size),
            )
        )
        unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
        cell_events = [[] for _ in range(unique_cells.shape[0])]
        for local_event, cell_index in enumerate(inverse):
            cell_events[int(cell_index)].append(int(local_event))
        lookup = {
            (int(cell[0]), int(cell[1]), int(cell[2])): index
            for index, cell in enumerate(unique_cells)
        }
        visited = np.zeros(unique_cells.shape[0], dtype=bool)
        for start in range(unique_cells.shape[0]):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            local_events = []
            while stack:
                cell_index = stack.pop()
                local_events.extend(cell_events[cell_index])
                x, y, temporal_bin = unique_cells[cell_index]
                for dx, dy, dt in offsets:
                    neighbor = lookup.get(
                        (int(x + dx), int(y + dy), int(temporal_bin + dt))
                    )
                    if neighbor is not None and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            indices = np.sort(
                positive_indices[np.asarray(local_events, dtype=np.int64)]
            ).astype(np.int64, copy=False)
            output.append(indices)

    output.sort(key=lambda item: int(item[0]))
    flattened = (
        np.concatenate(output)
        if output
        else np.empty(0, dtype=np.int64)
    )
    expected = np.flatnonzero(positive_global)
    if (
        flattened.size != expected.size
        or np.unique(flattened).size != flattened.size
        or not np.array_equal(np.sort(flattened), expected)
    ):
        raise RuntimeError("atomic components do not exactly partition positives")
    return AtomicComponentBatch(tuple(output))


def pure_false_positive_targets(event_indices, labels) -> np.ndarray:
    """Return one train-only class per component (1 means pure false positive)."""

    label_values = np.asarray(labels).reshape(-1)
    targets = []
    for indices in event_indices:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= label_values.size):
            raise ValueError("component indices are empty or out of label bounds")
        targets.append(int(not np.any(label_values[indices] > 0.5)))
    return np.asarray(targets, dtype=np.uint8)


def build_component_patch_queries(
    event_indices: Sequence[np.ndarray],
    locations,
    *,
    patch_radius: int,
    temporal_bin_size: int = 50,
) -> tuple[tuple[ComponentPatchQuery, ...], ...]:
    """Build label-free, motion-recentered component query sequences."""

    coordinates = np.asarray(locations)
    if coordinates.ndim != 2 or coordinates.shape[1] < 4:
        raise ValueError("locations must have shape [N,4+] ordered [batch,x,y,t]")
    patch_radius = int(patch_radius)
    temporal_bin_size = int(temporal_bin_size)
    if patch_radius < 1 or temporal_bin_size <= 0:
        raise ValueError("patch_radius and temporal_bin_size must be positive")
    patch_size = 2 * patch_radius + 1
    all_queries = []
    seen = np.zeros(coordinates.shape[0], dtype=bool)
    for component in event_indices:
        indices = np.asarray(component, dtype=np.int64).reshape(-1)
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= len(seen)):
            raise ValueError("component indices are empty or out of bounds")
        if np.any(seen[indices]):
            raise ValueError("component indices overlap")
        seen[indices] = True
        component_locations = coordinates[indices, 1:4].astype(np.int64, copy=False)
        bins = np.floor_divide(component_locations[:, 2], temporal_bin_size)
        unique_bins = np.unique(bins)
        if unique_bins.size > 1 and not np.all(np.diff(unique_bins) == 1):
            raise ValueError("an atomic component has a temporal gap")
        queries = []
        for temporal_bin in unique_bins:
            local = component_locations[bins == temporal_bin, :2]
            # floor(mean + .5) is stable and avoids round-to-even ambiguity.
            center_x = int(np.floor(float(local[:, 0].mean()) + 0.5))
            center_y = int(np.floor(float(local[:, 1].mean()) + 0.5))
            mask = np.zeros((patch_size, patch_size), dtype=np.float32)
            patch_x = local[:, 0] - center_x + patch_radius
            patch_y = local[:, 1] - center_y + patch_radius
            valid = (
                (patch_x >= 0)
                & (patch_x < patch_size)
                & (patch_y >= 0)
                & (patch_y < patch_size)
            )
            mask[patch_y[valid], patch_x[valid]] = np.float32(1.0)
            if not np.any(mask):
                raise RuntimeError("component query mask is unexpectedly empty")
            queries.append(
                ComponentPatchQuery(
                    temporal_bin=int(temporal_bin),
                    center_x=center_x,
                    center_y=center_y,
                    component_mask=mask,
                )
            )
        all_queries.append(tuple(queries))
    return tuple(all_queries)


def derive_strict_safe_cutoff(
    pure_fp_probabilities,
    pure_fp_targets,
) -> tuple[float, bool, dict]:
    """Derive the sole V3 cutoff from grouped inner-OOF predictions.

    The deletion score is a pure-FP probability.  The cutoff is the next
    representable float64 above the maximum score of any target-bearing
    component.  A candidate is safe only when at least one OOF *pure FP* lies
    at or above that strict upper bound; otherwise deployment is identity.
    """

    probabilities = np.asarray(pure_fp_probabilities, dtype=np.float64).reshape(-1)
    targets = np.asarray(pure_fp_targets, dtype=np.uint8).reshape(-1)
    if probabilities.size == 0 or probabilities.size != targets.size:
        raise ValueError("calibration probabilities and targets must align and be non-empty")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("calibration probabilities must be finite and in [0,1]")
    target_bearing = targets == 0
    if not np.any(target_bearing):
        raise RuntimeError("inner OOF calibration contains no target-bearing component")
    maximum_target_score = float(np.max(probabilities[target_bearing]))
    cutoff = float(np.nextafter(np.float64(maximum_target_score), np.float64(np.inf)))
    safe_pure_fp = (targets == 1) & (probabilities >= cutoff)
    enabled = bool(np.any(safe_pure_fp) and cutoff <= 1.0)
    diagnostics = {
        "target_bearing_component_count": int(np.count_nonzero(target_bearing)),
        "pure_fp_component_count": int(np.count_nonzero(targets == 1)),
        "maximum_target_bearing_score": maximum_target_score,
        "strict_safe_cutoff": cutoff,
        "safe_pure_fp_component_count": int(np.count_nonzero(safe_pure_fp)),
        "identity_due_to_no_safe_component": not enabled,
    }
    return cutoff, enabled, diagnostics


def verify_atomic_candidate(
    base_scores,
    candidate_scores,
    event_indices: Sequence[np.ndarray],
    deleted_components,
) -> tuple[bool, bool]:
    """Verify whole-component edits and bitwise preservation of every keep."""

    base = np.asarray(base_scores, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate_scores, dtype=np.float32).reshape(-1)
    deleted = np.asarray(deleted_components, dtype=bool).reshape(-1)
    if base.shape != candidate.shape or deleted.size != len(event_indices):
        raise ValueError("candidate audit inputs do not align")
    assigned = np.zeros(base.size, dtype=bool)
    retained_equal = True
    complete_only = True
    for component_index, indices in enumerate(event_indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= base.size):
            raise ValueError("component index is empty or out of bounds")
        if np.any(assigned[indices]):
            raise ValueError("component indices overlap")
        assigned[indices] = True
        if deleted[component_index]:
            complete_only &= bool(np.all(candidate[indices] == np.float32(0.0)))
        else:
            equal = np.array_equal(
                candidate[indices].view(np.uint32), base[indices].view(np.uint32)
            )
            retained_equal &= equal
            complete_only &= equal
    untouched = ~assigned
    retained_equal &= np.array_equal(
        candidate[untouched].view(np.uint32), base[untouched].view(np.uint32)
    )
    complete_only &= retained_equal
    return bool(retained_equal), bool(complete_only)


def atomic_delete_or_identity(
    base_scores,
    event_indices: Sequence[np.ndarray],
    pure_fp_probabilities,
    cutoff: float,
    *,
    enabled: bool,
) -> tuple[np.ndarray, AtomicEditReceipt]:
    """Delete complete components, or return exact identity on any violation."""

    base = np.asarray(base_scores, dtype=np.float32).reshape(-1)
    probabilities = np.asarray(pure_fp_probabilities, dtype=np.float64).reshape(-1)
    component_count = len(event_indices)
    if probabilities.size != component_count:
        return base.copy(), AtomicEditReceipt(
            False,
            "component_probability_count_mismatch",
            component_count,
            0,
            component_count,
            0,
            True,
            False,
        )
    if not enabled:
        return base.copy(), AtomicEditReceipt(
            False, "identity_policy", component_count, 0, component_count, 0, True, True
        )
    if not np.isfinite(probabilities).all() or not np.isfinite(float(cutoff)):
        return base.copy(), AtomicEditReceipt(
            False, "non_finite_probability_or_cutoff", component_count, 0,
            component_count, 0, True, False
        )
    deleted = probabilities >= float(cutoff)
    candidate = base.copy()
    try:
        assigned = np.zeros(base.size, dtype=bool)
        deleted_events = 0
        for component_index, indices in enumerate(event_indices):
            indices = np.asarray(indices, dtype=np.int64).reshape(-1)
            if indices.size == 0 or np.any(indices < 0) or np.any(indices >= base.size):
                raise ValueError("invalid component index")
            if np.any(assigned[indices]):
                raise ValueError("overlapping component index")
            assigned[indices] = True
            if deleted[component_index]:
                candidate[indices] = np.float32(0.0)
                deleted_events += int(indices.size)
        retained_equal, complete_only = verify_atomic_candidate(
            base, candidate, event_indices, deleted
        )
        if not retained_equal or not complete_only:
            raise RuntimeError("atomic edit invariant failed")
    except (ValueError, RuntimeError):
        return base.copy(), AtomicEditReceipt(
            False, "atomic_integrity_failure", component_count, 0,
            component_count, 0, True, False
        )
    deleted_count = int(np.count_nonzero(deleted))
    return candidate, AtomicEditReceipt(
        True,
        None,
        component_count,
        deleted_count,
        component_count - deleted_count,
        deleted_events,
        retained_equal,
        complete_only,
    )


__all__ = (
    "AtomicComponentBatch",
    "AtomicEditReceipt",
    "ComponentPatchQuery",
    "H2_EVENT_COUNT_CUTOFF",
    "H2_POLARITY_MINORITY_CUTOFF",
    "atomic_delete_or_identity",
    "build_component_patch_queries",
    "complete_input_polarity_minority_fraction",
    "derive_strict_safe_cutoff",
    "extract_atomic_components",
    "pure_false_positive_targets",
    "use_h2_atomic_deletion",
    "verify_atomic_candidate",
)
