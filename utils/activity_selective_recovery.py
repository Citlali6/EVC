"""Source-free mechanics for activity suppression plus atomic M20 recovery.

Stage 1 may change the dense event scores.  Stage 2 is deliberately narrower:
it can only restore an entire connected component from the frozen post-C00 M20
output.  Labels and target IDs are accepted only by a caller-supplied fit-time
metric closure when marginal recovery classes are constructed; they are never
accepted by disagreement construction, feature construction, calibration, or
the inference-time atomic action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from utils.atomic_component_deletion import (
    ComponentPatchQuery,
    build_component_patch_queries,
    extract_atomic_components,
)


@dataclass(frozen=True)
class DisagreementComponentBatch:
    """Complete M20 components for which activity suppresses at least one event."""

    event_indices: tuple[np.ndarray, ...]
    m20_component_ids: np.ndarray
    missing_event_counts: np.ndarray
    activity_supported_event_counts: np.ndarray
    m20_component_count: int
    activity_component_count: int


@dataclass(frozen=True)
class AtomicRecoveryReceipt:
    enabled: bool
    fallback_reason: str | None
    disagreement_component_count: int
    recovered_component_count: int
    unrecovered_component_count: int
    recovered_event_count: int
    activity_positive_event_count: int
    final_positive_event_count: int
    activity_outside_recovery_bitwise_equal: bool
    recovered_m20_scores_bitwise_equal: bool
    complete_components_only: bool


def _scores(values, name):
    output = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(output).all():
        raise ValueError("{} scores must be finite".format(name))
    return output


def extract_disagreement_components(
    m20_post_scores,
    activity_post_scores,
    locations,
    prediction_threshold: float,
    *,
    spatial_radius: int = 2,
    temporal_bin_size: int = 50,
    temporal_radius_bins: int = 1,
) -> DisagreementComponentBatch:
    """Return complete post-C00 M20 components partly absent from activity.

    Component membership is defined only by frozen scores, locations, and the
    C00 topology.  No label-derived quantity is accepted by this function.
    """

    m20 = _scores(m20_post_scores, "M20")
    activity = _scores(activity_post_scores, "activity")
    if m20.size != activity.size:
        raise ValueError("M20 and activity score lengths differ")
    coordinates = np.asarray(locations)
    if coordinates.ndim != 2 or coordinates.shape != (m20.size, 4):
        raise ValueError("locations must have shape [N,4]")
    kwargs = {
        "spatial_radius": int(spatial_radius),
        "temporal_bin_size": int(temporal_bin_size),
        "temporal_radius_bins": int(temporal_radius_bins),
    }
    m20_components = extract_atomic_components(
        m20, coordinates, float(prediction_threshold), **kwargs
    ).event_indices
    activity_components = extract_atomic_components(
        activity, coordinates, float(prediction_threshold), **kwargs
    ).event_indices
    activity_positive = activity >= np.float32(prediction_threshold)
    selected = []
    component_ids = []
    missing_counts = []
    supported_counts = []
    for component_id, indices in enumerate(m20_components):
        indices = np.asarray(indices, dtype=np.int64)
        supported = int(np.count_nonzero(activity_positive[indices]))
        missing = int(indices.size - supported)
        if missing <= 0:
            continue
        selected.append(indices.copy())
        component_ids.append(component_id)
        missing_counts.append(missing)
        supported_counts.append(supported)
    return DisagreementComponentBatch(
        event_indices=tuple(selected),
        m20_component_ids=np.asarray(component_ids, dtype=np.int32),
        missing_event_counts=np.asarray(missing_counts, dtype=np.int32),
        activity_supported_event_counts=np.asarray(supported_counts, dtype=np.int32),
        m20_component_count=len(m20_components),
        activity_component_count=len(activity_components),
    )


def _identity_receipt(activity, threshold, component_count, reason, complete=False):
    positive = int(np.count_nonzero(activity >= np.float32(threshold)))
    return AtomicRecoveryReceipt(
        enabled=False,
        fallback_reason=reason,
        disagreement_component_count=int(component_count),
        recovered_component_count=0,
        unrecovered_component_count=int(component_count),
        recovered_event_count=0,
        activity_positive_event_count=positive,
        final_positive_event_count=positive,
        activity_outside_recovery_bitwise_equal=True,
        recovered_m20_scores_bitwise_equal=True,
        complete_components_only=bool(complete),
    )


def atomic_recover_or_identity(
    m20_post_scores,
    activity_post_scores,
    disagreement_event_indices: Sequence[np.ndarray],
    recovery_confidences,
    cutoff: float,
    prediction_threshold: float,
    *,
    enabled: bool,
) -> tuple[np.ndarray, AtomicRecoveryReceipt]:
    """Restore selected complete M20 components or fail closed to activity."""

    try:
        m20 = _scores(m20_post_scores, "M20")
        activity = _scores(activity_post_scores, "activity")
    except ValueError:
        # There is no trustworthy same-shape identity when activity itself is
        # non-finite, so propagate instead of fabricating a score vector.
        raise
    if m20.size != activity.size:
        raise ValueError("M20 and activity score lengths differ")
    component_count = len(disagreement_event_indices)
    if not enabled:
        return activity.copy(), _identity_receipt(
            activity, prediction_threshold, component_count, "identity_policy", True
        )
    confidences = np.asarray(recovery_confidences, dtype=np.float64).reshape(-1)
    if confidences.size != component_count:
        return activity.copy(), _identity_receipt(
            activity,
            prediction_threshold,
            component_count,
            "component_confidence_count_mismatch",
        )
    if not np.isfinite(confidences).all() or not np.isfinite(float(cutoff)):
        return activity.copy(), _identity_receipt(
            activity,
            prediction_threshold,
            component_count,
            "non_finite_confidence_or_cutoff",
        )

    selected = confidences >= float(cutoff)
    candidate = activity.copy()
    assigned = np.zeros(activity.size, dtype=bool)
    recovered_union = np.zeros(activity.size, dtype=bool)
    recovered_events = 0
    try:
        threshold32 = np.float32(prediction_threshold)
        for component_id, raw_indices in enumerate(disagreement_event_indices):
            indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
            if (
                indices.size == 0
                or np.any(indices < 0)
                or np.any(indices >= activity.size)
                or np.any(assigned[indices])
                or np.any(m20[indices] < threshold32)
                or np.all(activity[indices] >= threshold32)
            ):
                raise ValueError("invalid or non-disagreement component")
            assigned[indices] = True
            if selected[component_id]:
                recovered_union[indices] = True
                candidate[indices] = m20[indices]
                recovered_events += int(indices.size)
        outside_equal = np.array_equal(
            candidate[~recovered_union], activity[~recovered_union]
        )
        recovered_equal = np.array_equal(
            candidate[recovered_union], m20[recovered_union]
        )
        if not outside_equal or not recovered_equal:
            raise RuntimeError("atomic recovery invariant failed")
    except (ValueError, RuntimeError):
        return activity.copy(), _identity_receipt(
            activity,
            prediction_threshold,
            component_count,
            "atomic_integrity_failure",
        )

    recovered_count = int(np.count_nonzero(selected))
    activity_positive = int(
        np.count_nonzero(activity >= np.float32(prediction_threshold))
    )
    final_positive = int(
        np.count_nonzero(candidate >= np.float32(prediction_threshold))
    )
    return candidate, AtomicRecoveryReceipt(
        enabled=True,
        fallback_reason=None,
        disagreement_component_count=component_count,
        recovered_component_count=recovered_count,
        unrecovered_component_count=component_count - recovered_count,
        recovered_event_count=recovered_events,
        activity_positive_event_count=activity_positive,
        final_positive_event_count=final_positive,
        activity_outside_recovery_bitwise_equal=outside_equal,
        recovered_m20_scores_bitwise_equal=recovered_equal,
        complete_components_only=True,
    )


def marginal_recovery_targets(
    m20_post_scores,
    activity_post_scores,
    disagreement_event_indices: Sequence[np.ndarray],
    prediction_threshold: float,
    official_score_fn: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit-only class: restoring this component alone raises official Score.

    ``official_score_fn`` is the only place a caller may close over labels and
    target IDs.  Neither the returned targets nor score deltas are inference
    features.
    """

    if not callable(official_score_fn):
        raise TypeError("official_score_fn must be callable")
    activity = _scores(activity_post_scores, "activity")
    component_count = len(disagreement_event_indices)
    base_score = float(official_score_fn(activity.copy()))
    if not np.isfinite(base_score):
        raise ValueError("base official Score must be finite")
    deltas = np.empty(component_count, dtype=np.float64)
    for component_id in range(component_count):
        confidences = np.zeros(component_count, dtype=np.float64)
        confidences[component_id] = 1.0
        candidate, receipt = atomic_recover_or_identity(
            m20_post_scores,
            activity,
            disagreement_event_indices,
            confidences,
            cutoff=1.0,
            prediction_threshold=prediction_threshold,
            enabled=True,
        )
        if not receipt.complete_components_only or receipt.recovered_component_count != 1:
            raise RuntimeError("fit-only marginal recovery action failed")
        value = float(official_score_fn(candidate))
        if not np.isfinite(value):
            raise ValueError("candidate official Score must be finite")
        deltas[component_id] = value - base_score
    return (deltas > 0.0).astype(np.uint8), deltas


def negative_reference_conformal_confidence(
    raw_recovery_probabilities,
    fit_negative_reference_probabilities,
) -> np.ndarray:
    """Map raw recovery scores to a model-local, conservative common scale.

    The reference contains only fit-time marginal-negative disagreement
    components for that trained model.  A score must strictly exceed a
    negative reference value to gain confidence; ties never gain credit.
    """

    scores = np.asarray(raw_recovery_probabilities, dtype=np.float64).reshape(-1)
    reference = np.asarray(
        fit_negative_reference_probabilities, dtype=np.float64
    ).reshape(-1)
    if reference.size <= 0:
        raise ValueError("at least one fit-negative reference is required")
    if not np.isfinite(scores).all() or not np.isfinite(reference).all():
        raise ValueError("recovery probabilities must be finite")
    sorted_reference = np.sort(reference)
    strictly_lower = np.searchsorted(sorted_reference, scores, side="left")
    return (1.0 + strictly_lower.astype(np.float64)) / float(reference.size + 1)


def unique_recovery_cutoffs(conformal_confidences) -> np.ndarray:
    """Identity boundary followed by every exact observed action breakpoint."""

    values = np.asarray(conformal_confidences, dtype=np.float64).reshape(-1)
    if values.size <= 0 or not np.isfinite(values).all():
        raise ValueError("finite non-empty conformal confidences are required")
    unique = np.unique(values)[::-1]
    identity = np.nextafter(unique[0], np.inf)
    return np.concatenate((np.asarray([identity]), unique))


def disagreement_trajectory_context(
    event_indices,
    locations,
    m20_post_scores,
    activity_post_scores,
    prediction_threshold: float,
    *,
    patch_radius: int,
    temporal_bin_size: int = 50,
    stream_bin_count: int = 160,
    width: int = 346,
    height: int = 260,
) -> tuple[tuple[ComponentPatchQuery, ...], np.ndarray]:
    """Build long-track input-only scalars aligned with component patch queries."""

    indices = np.asarray(event_indices, dtype=np.int64).reshape(-1)
    coordinates = np.asarray(locations)
    m20 = _scores(m20_post_scores, "M20")
    activity = _scores(activity_post_scores, "activity")
    if (
        indices.size <= 0
        or coordinates.ndim != 2
        or coordinates.shape != (m20.size, 4)
        or activity.size != m20.size
        or np.any(indices < 0)
        or np.any(indices >= m20.size)
    ):
        raise ValueError("invalid disagreement trajectory inputs")
    queries = build_component_patch_queries(
        (indices,),
        coordinates,
        patch_radius=int(patch_radius),
        temporal_bin_size=int(temporal_bin_size),
    )[0]
    event_bins = np.floor_divide(
        coordinates[indices, 3].astype(np.int64), int(temporal_bin_size)
    )
    duration = len(queries)
    previous_x = None
    previous_y = None
    rows = []
    for order, query in enumerate(queries):
        local = indices[event_bins == int(query.temporal_bin)]
        center_x = float(query.center_x)
        center_y = float(query.center_y)
        dx = 0.0 if previous_x is None else (center_x - previous_x) / max(width - 1, 1)
        dy = 0.0 if previous_y is None else (center_y - previous_y) / max(height - 1, 1)
        previous_x, previous_y = center_x, center_y
        activity_support = float(
            np.mean(activity[local] >= np.float32(prediction_threshold))
        )
        mean_delta = float(np.mean(m20[local].astype(np.float64) - activity[local]))
        relative_time = 0.0 if duration == 1 else 2.0 * order / (duration - 1) - 1.0
        rows.append(
            (
                relative_time,
                duration / float(max(int(stream_bin_count), 1)),
                2.0 * center_x / max(width - 1, 1) - 1.0,
                2.0 * center_y / max(height - 1, 1) - 1.0,
                dx,
                dy,
                activity_support,
                float(np.clip(mean_delta, -1.0, 1.0)),
            )
        )
    features = np.asarray(rows, dtype=np.float32)
    if features.shape != (duration, 8) or not np.isfinite(features).all():
        raise RuntimeError("trajectory context is invalid")
    return queries, features


__all__ = (
    "AtomicRecoveryReceipt",
    "DisagreementComponentBatch",
    "atomic_recover_or_identity",
    "disagreement_trajectory_context",
    "extract_disagreement_components",
    "marginal_recovery_targets",
    "negative_reference_conformal_confidence",
    "unique_recovery_cutoffs",
)
