"""Atomic recovery actions for the H2 pyramid selective-recovery stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BreakpointRecord:
    """One exact fit-only component-probability breakpoint evaluation."""

    cutoff: float
    score_gain_vs_m20: float
    stage2_true_positive_events: int
    stage1_true_positive_events: int
    stage2_correct_objects: int
    stage1_correct_objects: int


def validate_disjoint_components(components, event_count):
    event_count = int(event_count)
    if event_count < 0:
        raise ValueError("event_count must be non-negative")
    normalized = []
    owner = np.full(event_count, -1, dtype=np.int64)
    for component_index, values in enumerate(components):
        indices = np.asarray(values, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("components must be non-empty")
        if int(indices.min()) < 0 or int(indices.max()) >= event_count:
            raise ValueError("component index is outside the event vector")
        if np.unique(indices).size != indices.size:
            raise ValueError("a component contains duplicate event indices")
        if np.any(owner[indices] >= 0):
            raise ValueError("M20 components must be disjoint")
        owner[indices] = component_index
        normalized.append(indices)
    return tuple(normalized)


def restore_whole_components_bitwise(
    pyramid_scores,
    m20_scores,
    components,
    recover_components,
):
    """Restore complete selected M20 components; preserve all other bits.

    There is deliberately no event-level weight or attenuation argument.  The
    only allowed action is a Boolean choice for each complete component.
    """

    pyramid = np.asarray(pyramid_scores)
    m20 = np.asarray(m20_scores)
    if pyramid.ndim != 1 or m20.ndim != 1 or pyramid.shape != m20.shape:
        raise ValueError("paired score vectors must be aligned one-dimensional arrays")
    if pyramid.dtype != np.float32 or m20.dtype != np.float32:
        raise ValueError("bitwise recovery requires float32 score vectors")
    normalized = validate_disjoint_components(components, pyramid.size)
    decisions = np.asarray(recover_components, dtype=np.bool_).reshape(-1)
    if decisions.size != len(normalized):
        raise ValueError("one Boolean recovery decision is required per component")
    output = pyramid.copy()
    restored_mask = np.zeros(pyramid.size, dtype=np.bool_)
    for decision, indices in zip(decisions, normalized):
        if decision:
            output[indices] = m20[indices]
            restored_mask[indices] = True
    if not np.array_equal(output[restored_mask], m20[restored_mask]):
        raise RuntimeError("selected components were not restored bitwise to M20")
    if not np.array_equal(output[~restored_mask], pyramid[~restored_mask]):
        raise RuntimeError("events outside selected components changed")
    return output


def exact_risk_controlled_breakpoint(records):
    """Select one exact fit-only breakpoint without a numerical threshold grid.

    Feasible breakpoints must recover Stage1 TP or CO, may not reduce either
    quantity relative to Stage1, and are ranked by official Score gain.  Ties
    prefer greater CO recovery, then TP recovery, then the more conservative
    (higher) cutoff.
    """

    values = tuple(records)
    if not values:
        raise ValueError("at least one exact breakpoint record is required")
    feasible = []
    for value in values:
        if not isinstance(value, BreakpointRecord):
            raise TypeError("records must contain BreakpointRecord values")
        numeric = np.asarray((value.cutoff, value.score_gain_vs_m20), dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("breakpoint metrics must be finite")
        tp_gain = value.stage2_true_positive_events - value.stage1_true_positive_events
        co_gain = value.stage2_correct_objects - value.stage1_correct_objects
        if tp_gain >= 0 and co_gain >= 0 and (tp_gain > 0 or co_gain > 0):
            feasible.append(value)
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda value: (
            value.score_gain_vs_m20,
            value.stage2_correct_objects - value.stage1_correct_objects,
            value.stage2_true_positive_events - value.stage1_true_positive_events,
            value.cutoff,
        ),
    )


__all__ = (
    "BreakpointRecord",
    "exact_risk_controlled_breakpoint",
    "restore_whole_components_bitwise",
    "validate_disjoint_components",
)
