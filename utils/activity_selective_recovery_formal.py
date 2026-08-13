"""Pure formal-training and exact inner-gate helpers for activity recovery."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


INNER_GROUPS = ("g1_088_091", "g3_095_098")


def source_class_balanced_weights(source_ids, targets):
    """Give every observed source/class cell equal total loss mass."""

    sources = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    labels = np.asarray(targets, dtype=np.uint8).reshape(-1)
    if sources.shape != labels.shape or sources.size == 0:
        raise ValueError("source_ids and targets must be nonempty aligned vectors")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("targets must be binary")
    weights = np.empty(labels.size, dtype=np.float64)
    for source in np.unique(sources):
        source_mask = sources == source
        for target in (0, 1):
            mask = source_mask & (labels == target)
            count = int(np.count_nonzero(mask))
            if count:
                weights[mask] = 1.0 / count
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("source/class weights are invalid")
    weights /= weights.mean()
    return weights


def deterministic_epoch_batches(item_count, batch_size, epochs, seed):
    """Freeze the final-epoch training schedule without checkpoint selection."""

    item_count = int(item_count)
    batch_size = int(batch_size)
    epochs = int(epochs)
    if item_count <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("item_count, batch_size, and epochs must be positive")
    generator = np.random.default_rng(int(seed))
    output = []
    for epoch in range(epochs):
        permutation = generator.permutation(item_count)
        for start in range(0, item_count, batch_size):
            output.append(
                {
                    "epoch": epoch,
                    "indices": permutation[start : start + batch_size].astype(
                        np.int64, copy=False
                    ),
                }
            )
    expected = epochs * int(math.ceil(item_count / batch_size))
    if len(output) != expected:
        raise RuntimeError("deterministic batch schedule length changed")
    return tuple(output)


def exact_confidence_cutoffs(confidence_arrays: Iterable[np.ndarray]):
    """Return identity plus every unique observed confidence, never a grid."""

    arrays = [np.asarray(values, dtype=np.float64).reshape(-1) for values in confidence_arrays]
    if not arrays or not any(values.size for values in arrays):
        raise ValueError("at least one confidence is required")
    flattened = np.concatenate([values for values in arrays if values.size])
    if not np.isfinite(flattened).all():
        raise ValueError("confidences must be finite")
    unique = np.unique(flattened)
    identity = np.nextafter(float(unique[-1]), math.inf)
    return np.concatenate(
        (np.asarray([identity], dtype=np.float64), unique[::-1])
    )


def _counts(payload):
    return payload["counts"]


def _metrics(payload):
    return payload["metrics"]


def assess_inner_replay(
    replay: Mapping,
    *,
    groups: Sequence[str] = INNER_GROUPS,
    pooled_score_gain_minimum: float = 0.01,
    absolute_pd_delta_maximum: float = 0.005,
):
    """Apply the frozen parent-specified inner promotion gates exactly."""

    checks = {}
    score_gains = []
    absolute_pd_deltas = []
    for group in groups:
        payload = replay["groups"][group]
        m20 = payload["m20"]
        activity = payload["activity"]
        candidate = payload["candidate"]
        m20_counts = _counts(m20)
        activity_counts = _counts(activity)
        candidate_counts = _counts(candidate)
        m20_metrics = _metrics(m20)
        candidate_metrics = _metrics(candidate)
        score_gain = float(candidate_metrics["score"] - m20_metrics["score"])
        pd_delta = float(candidate_metrics["pd"] - m20_metrics["pd"])
        score_gains.append(score_gain)
        absolute_pd_deltas.append(abs(pd_delta))
        checks[group] = {
            "score_strictly_positive": score_gain > 0.0,
            "false_positive_events_strictly_lower": int(
                candidate_counts["false_positive_events"]
            )
            < int(m20_counts["false_positive_events"]),
            "false_components_strictly_lower": int(
                candidate_counts["false_components"]
            )
            < int(m20_counts["false_components"]),
            "absolute_pd_delta_at_most": abs(pd_delta)
            <= float(absolute_pd_delta_maximum),
            "recovers_tp_or_co_relative_to_stage1": int(
                candidate_counts["true_positive_events"]
            )
            > int(activity_counts["true_positive_events"])
            or int(candidate_counts["correct_objects"])
            > int(activity_counts["correct_objects"]),
            "atomic_integrity": bool(payload["atomic_integrity"]),
        }
    pooled_m20 = replay["pooled"]["m20"]
    pooled_candidate = replay["pooled"]["candidate"]
    pooled_gain = float(
        _metrics(pooled_candidate)["score"] - _metrics(pooled_m20)["score"]
    )
    pooled_checks = {
        "score_gain_at_least": pooled_gain >= float(pooled_score_gain_minimum),
        "false_positive_events_strictly_lower": int(
            _counts(pooled_candidate)["false_positive_events"]
        )
        < int(_counts(pooled_m20)["false_positive_events"]),
        "false_components_strictly_lower": int(
            _counts(pooled_candidate)["false_components"]
        )
        < int(_counts(pooled_m20)["false_components"]),
        "nonidentity_recovery": int(replay["recovered_component_count"]) > 0,
    }
    passed = all(
        value
        for group_checks in checks.values()
        for value in group_checks.values()
    ) and all(pooled_checks.values())
    objective = (
        min(score_gains),
        pooled_gain,
        -max(absolute_pd_deltas),
        float(replay["cutoff"]),
    )
    return {
        "passed": bool(passed),
        "group_checks": checks,
        "pooled_checks": pooled_checks,
        "group_score_gains": {
            group: score_gains[index] for index, group in enumerate(groups)
        },
        "pooled_score_gain": pooled_gain,
        "maximum_absolute_group_pd_delta": max(absolute_pd_deltas),
        "objective": objective,
    }


def select_qualifying_inner_replay(replays, **gate_kwargs):
    """Select by maximin, pooled gain, Pd preservation, then conservative cutoff."""

    audited = []
    for replay in replays:
        gate = assess_inner_replay(replay, **gate_kwargs)
        audited.append({"replay": replay, "gate": gate})
    qualifying = [item for item in audited if item["gate"]["passed"]]
    if not qualifying:
        return None, tuple(audited)
    selected = max(qualifying, key=lambda item: item["gate"]["objective"])
    return selected, tuple(audited)

