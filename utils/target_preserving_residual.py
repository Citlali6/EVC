"""Train-only primal/dual loss helpers for target-preserving residual V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F


H2_EVENT_COUNT_CUTOFF = 200000
H2_POLARITY_MINORITY_CUTOFF = 0.20


def complete_input_polarity_minority_fraction(polarities):
    """Return the source-free minority fraction of a complete polarity vector."""
    values = np.asarray(polarities)
    if values.ndim != 1 or values.size <= 0:
        raise ValueError('Complete-input polarities must be a non-empty vector.')
    if values.dtype.kind not in 'biuf':
        raise TypeError('Polarities must be numeric.')
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError('Polarities must be finite.')
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError('Normalized polarities must lie in [0,1].')
    positives = int(np.count_nonzero(values > 0.5))
    return float(min(positives, int(values.size) - positives) / int(values.size))


def use_h2_residual_refiner(event_count, polarities):
    """Select H2 using only complete-input event count and polarity balance."""
    if isinstance(event_count, (bool, np.bool_)):
        raise ValueError('event_count must be a non-negative integer.')
    event_count = int(event_count)
    if event_count < 0 or event_count != len(polarities):
        raise ValueError('event_count must match complete-input polarities.')
    minority = complete_input_polarity_minority_fraction(polarities)
    return bool(
        event_count > H2_EVENT_COUNT_CUTOFF
        and minority >= H2_POLARITY_MINORITY_CUTOFF
    )


def input_only_routed_scores(base_scores, candidate_scores, polarities):
    """Return candidate only for H2; otherwise preserve base bitwise."""
    if base_scores.shape != candidate_scores.shape:
        raise ValueError('Base and candidate scores must align.')
    if base_scores.shape[0] != len(polarities):
        raise ValueError('Scores must align with complete-input polarities.')
    if use_h2_residual_refiner(base_scores.shape[0], polarities):
        return candidate_scores
    return base_scores


@dataclass
class TargetRetentionDualState:
    """Non-negative multipliers updated by unit-step projected dual ascent."""

    positive_event: float = 1.0
    target_group: float = 1.0

    def __post_init__(self):
        if self.positive_event < 0.0 or self.target_group < 0.0:
            raise ValueError('Dual multipliers must be non-negative.')

    def update(self, positive_event_constraint, target_group_constraint):
        event_value = float(positive_event_constraint.detach())
        group_value = float(target_group_constraint.detach())
        if not torch.isfinite(torch.tensor((event_value, group_value))).all():
            raise RuntimeError('Cannot update dual state from non-finite constraints.')
        # Unit step is the canonical unscaled projected subgradient update; it
        # is fixed structurally and is not a searched loss weight.
        self.positive_event = max(0.0, self.positive_event + event_value)
        self.target_group = max(0.0, self.target_group + group_value)

    def to_dict(self):
        return asdict(self)


def target_retention_constraints(refined, base, labels, target_ids, times):
    """Return positive-event and target-frame deficits relative to M20."""
    for name, tensor in (
        ('refined', refined),
        ('base', base),
        ('labels', labels),
        ('target_ids', target_ids),
        ('times', times),
    ):
        if tensor.ndim != 1:
            raise ValueError('{} must be one-dimensional.'.format(name))
    length = refined.numel()
    if any(tensor.numel() != length for tensor in (base, labels, target_ids, times)):
        raise ValueError('Event tensors must align.')

    positive = labels > 0.5
    zero = refined.sum() * 0.0
    if bool(torch.any(positive)):
        positive_event_constraint = F.relu(
            base[positive].detach() - refined[positive]
        ).mean()
    else:
        positive_event_constraint = zero

    positive_target = positive & (target_ids > 0)
    group_deficits = []
    if bool(torch.any(positive_target)):
        selected_ids = target_ids[positive_target]
        selected_times = times[positive_target]
        multiplier = torch.max(selected_ids) + 1
        keys = selected_times * multiplier + selected_ids
        candidate_positive = refined[positive_target]
        base_positive = base[positive_target].detach()
        for key in torch.unique(keys):
            mask = keys == key
            group_deficits.append(F.relu(
                torch.max(base_positive[mask]) - torch.max(candidate_positive[mask])
            ))
    target_group_constraint = (
        torch.stack(group_deficits).mean() if group_deficits else zero
    )
    return positive_event_constraint, target_group_constraint, len(group_deficits)


def target_preserving_event_loss(
    refined,
    base,
    labels,
    target_ids,
    times,
    dual_state,
):
    """Balanced classification objective under dynamic retention constraints."""
    if not isinstance(dual_state, TargetRetentionDualState):
        raise TypeError('dual_state must be TargetRetentionDualState.')
    positive = labels > 0.5
    negative = ~positive
    zero = refined.sum() * 0.0
    positive_term = F.softplus(-refined[positive]).mean() if bool(torch.any(positive)) else zero
    if bool(torch.any(negative)):
        negative_weight = torch.sigmoid(base[negative]).detach()
        negative_term = (
            negative_weight * F.softplus(refined[negative])
        ).sum() / negative_weight.sum().clamp_min(torch.finfo(refined.dtype).eps)
    else:
        negative_term = zero
    classification = 0.5 * (positive_term + negative_term)
    event_constraint, group_constraint, group_count = target_retention_constraints(
        refined, base, labels, target_ids, times,
    )
    # Standard augmented Lagrangian: dynamic multipliers plus unit quadratic
    # curvature. No loss coefficient is selected from a held fold.
    retention = (
        dual_state.positive_event * event_constraint
        + dual_state.target_group * group_constraint
        + 0.5 * (event_constraint.square() + group_constraint.square())
    )
    loss = classification + retention
    diagnostics = {
        'loss': float(loss.detach()),
        'classification': float(classification.detach()),
        'positive_event_constraint': float(event_constraint.detach()),
        'target_group_constraint': float(group_constraint.detach()),
        'retention_term': float(retention.detach()),
        'positive_count': int(torch.count_nonzero(positive)),
        'target_group_count': int(group_count),
        'dual_positive_event_before': float(dual_state.positive_event),
        'dual_target_group_before': float(dual_state.target_group),
    }
    return loss, event_constraint, group_constraint, diagnostics


def validate_all_step_diagnostics(records, expected_steps):
    """Fail closed unless every optimizer step has a complete finite record."""
    records = list(records)
    expected_steps = int(expected_steps)
    if expected_steps <= 0 or len(records) != expected_steps:
        raise RuntimeError('All-step diagnostic count mismatch.')
    required = {
        'step',
        'loss',
        'gradient_norm',
        'positive_event_constraint',
        'target_group_constraint',
        'protection_gate_after',
        'suppression_gate_after',
        'dual_positive_event_after',
        'dual_target_group_after',
    }
    for index, record in enumerate(records, start=1):
        if int(record.get('step', -1)) != index or not required.issubset(record):
            raise RuntimeError('Missing or misordered all-step diagnostics.')
        for key in required - {'step'}:
            if not np.isfinite(float(record[key])):
                raise RuntimeError('Non-finite all-step diagnostic: {}.'.format(key))
    return True
