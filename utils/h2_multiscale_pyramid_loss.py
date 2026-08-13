"""Scale-normalized dynamic constraints for the H2 temporal pyramid expert."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass
class PyramidDualState:
    """Projected dual multipliers; no held-fold loss-weight grid is needed."""

    target_time_recall: float = 1.0
    hard_negative_suppression: float = 1.0

    def __post_init__(self):
        if self.target_time_recall < 0 or self.hard_negative_suppression < 0:
            raise ValueError("dual multipliers must be non-negative")

    def update(self, recall_violation, suppression_violation):
        recall = float(recall_violation.detach())
        suppression = float(suppression_violation.detach())
        values = torch.tensor((recall, suppression), dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise RuntimeError("dual update received a non-finite violation")
        self.target_time_recall = max(0.0, self.target_time_recall + recall)
        self.hard_negative_suppression = max(
            0.0, self.hard_negative_suppression + suppression
        )

    def to_dict(self):
        return asdict(self)


def _validate_event_vectors(refined, base, labels, target_ids, times):
    for name, tensor in (
        ("refined", refined),
        ("base", base),
        ("labels", labels),
        ("target_ids", target_ids),
        ("times", times),
    ):
        if tensor.ndim != 1:
            raise ValueError("{} must be one-dimensional".format(name))
    if any(value.numel() != refined.numel() for value in (base, labels, target_ids, times)):
        raise ValueError("event vectors must align")
    if not torch.isfinite(refined).all() or not torch.isfinite(base).all():
        raise ValueError("event logits must be finite")


def _self_normalizing_scale(values):
    """Detached RMS deviation with a data-magnitude fallback."""

    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("cannot derive scale from an empty tensor")
    eps = torch.finfo(values.dtype).eps
    centered = torch.sqrt(torch.mean((values - values.mean()).square()))
    magnitude = torch.sqrt(torch.mean(values.square()))
    return torch.where(centered > eps, centered, magnitude).clamp_min(eps)


def _balanced_event_objective(logits, base, labels):
    positive = labels > 0.5
    negative = ~positive
    if not bool(torch.any(positive)) or not bool(torch.any(negative)):
        raise ValueError("event objective needs positive and negative events")
    positive_term = F.softplus(-logits[positive]).mean()
    hard_negative_weight = torch.sigmoid(base[negative]).detach()
    negative_term = (
        hard_negative_weight * F.softplus(logits[negative])
    ).sum() / hard_negative_weight.sum().clamp_min(torch.finfo(logits.dtype).eps)
    return 0.5 * (positive_term + negative_term)


def target_time_group_recall_constraint(refined, base, labels, target_ids, times):
    """Normalized deficit of each target/time-bin maximum relative to M20."""

    _validate_event_vectors(refined, base, labels, target_ids, times)
    positive_target = (labels > 0.5) & (target_ids > 0)
    zero = refined.sum() * 0.0
    if not bool(torch.any(positive_target)):
        return zero, 0
    ids = target_ids[positive_target].long()
    temporal = times[positive_target].long()
    key_multiplier = ids.max() + 1
    keys = temporal * key_multiplier + ids
    refined_values = refined[positive_target]
    base_values = base[positive_target].detach()
    base_group_maxima = []
    refined_group_maxima = []
    for key in torch.unique(keys):
        mask = keys == key
        base_group_maxima.append(base_values[mask].max())
        refined_group_maxima.append(refined_values[mask].max())
    base_group_maxima = torch.stack(base_group_maxima)
    refined_group_maxima = torch.stack(refined_group_maxima)
    scale = _self_normalizing_scale(base_group_maxima)
    violation = F.relu(
        (base_group_maxima - refined_group_maxima) / scale
    ).mean()
    return violation, int(base_group_maxima.numel())


def hard_negative_component_suppression_constraint(
    refined,
    base,
    labels,
    hard_negative_components,
):
    """Require a one-fit-scale decrease of every train-only pure-FP component."""

    components = tuple(hard_negative_components)
    if not components:
        raise ValueError("hard-negative component constraint needs components")
    base_maxima = []
    refined_maxima = []
    for indices in components:
        indices = torch.as_tensor(indices, device=refined.device, dtype=torch.long).reshape(-1)
        if indices.numel() == 0 or int(indices.min()) < 0 or int(indices.max()) >= refined.numel():
            raise ValueError("hard-negative component indices are invalid")
        if bool(torch.any(labels[indices] > 0.5)):
            raise ValueError("hard-negative components must be pure FP in train labels")
        base_maxima.append(base[indices].detach().max())
        refined_maxima.append(refined[indices].max())
    base_maxima = torch.stack(base_maxima)
    refined_maxima = torch.stack(refined_maxima)
    scale = _self_normalizing_scale(base_maxima)
    # Identity has normalized violation one.  Classification gradients and the
    # dynamic multiplier must earn a full fit-derived logit-scale reduction.
    violation = F.relu(
        (refined_maxima - base_maxima + scale) / scale
    ).mean()
    return violation, int(base_maxima.numel()), scale


def multiscale_pyramid_constrained_loss(
    refined,
    base,
    labels,
    target_ids,
    times,
    hard_negative_components,
    dual_state,
):
    """Balanced event classification plus two dynamic normalized constraints."""

    if not isinstance(dual_state, PyramidDualState):
        raise TypeError("dual_state must be PyramidDualState")
    _validate_event_vectors(refined, base, labels, target_ids, times)
    candidate_objective = _balanced_event_objective(refined, base, labels)
    base_objective = _balanced_event_objective(base.detach(), base.detach(), labels).detach()
    classification = candidate_objective / base_objective.clamp_min(
        torch.finfo(candidate_objective.dtype).eps
    )
    recall, target_group_count = target_time_group_recall_constraint(
        refined, base, labels, target_ids, times
    )
    suppression, component_count, component_scale = (
        hard_negative_component_suppression_constraint(
            refined, base, labels, hard_negative_components
        )
    )
    constraints = (
        dual_state.target_time_recall * recall
        + dual_state.hard_negative_suppression * suppression
        + 0.5 * (recall.square() + suppression.square())
    )
    loss = classification + constraints
    diagnostics = {
        "loss": float(loss.detach()),
        "classification_normalized": float(classification.detach()),
        "target_time_recall_violation": float(recall.detach()),
        "hard_negative_suppression_violation": float(suppression.detach()),
        "hard_negative_component_scale": float(component_scale.detach()),
        "target_time_group_count": int(target_group_count),
        "hard_negative_component_count": int(component_count),
        "dual_target_time_recall_before": float(dual_state.target_time_recall),
        "dual_hard_negative_suppression_before": float(
            dual_state.hard_negative_suppression
        ),
    }
    return loss, recall, suppression, diagnostics


def validate_pyramid_step_diagnostics(records, expected_steps):
    records = list(records)
    expected_steps = int(expected_steps)
    if expected_steps <= 0 or len(records) != expected_steps:
        raise RuntimeError("pyramid all-step diagnostic count mismatch")
    required = {
        "step",
        "loss",
        "gradient_norm",
        "classification_normalized",
        "target_time_recall_violation",
        "hard_negative_suppression_violation",
        "dual_target_time_recall_after",
        "dual_hard_negative_suppression_after",
        "mixture_entropy",
        "correction_abs_mean",
    }
    for expected, record in enumerate(records, start=1):
        if int(record.get("step", -1)) != expected or not required.issubset(record):
            raise RuntimeError("pyramid step diagnostic is incomplete or misordered")
        values = torch.tensor(
            [float(record[key]) for key in required if key != "step"],
            dtype=torch.float64,
        )
        if not torch.isfinite(values).all():
            raise RuntimeError("pyramid step diagnostic contains non-finite values")
    return True


__all__ = (
    "PyramidDualState",
    "hard_negative_component_suppression_constraint",
    "multiscale_pyramid_constrained_loss",
    "target_time_group_recall_constraint",
    "validate_pyramid_step_diagnostics",
)
