"""Event-point losses for full-frame temporal segmentation."""

import math

import torch
import torch.nn.functional as functional


def build_target_center_heatmaps(
    event_x,
    event_y,
    labels,
    target_ids,
    event_batch_indices,
    batch_size,
    height,
    width,
    sigma=2.5,
    radius=6,
):
    """Build one soft target-centre map for every labelled time-frame view.

    A target ID identifies a physical target in the centre temporal bin. Its
    positive event centroid is rendered as a clipped Gaussian so sparse
    targets still provide a spatially dense supervisory signal. This helper
    is used only during training; inference receives no target labels.
    """
    tensors = (event_x, event_y, labels, target_ids, event_batch_indices)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError('Target-centre inputs must be flat tensors.')
    if not (
        event_x.shape
        == event_y.shape
        == labels.shape
        == target_ids.shape
        == event_batch_indices.shape
    ):
        raise ValueError('Target-centre inputs must have matching shapes.')
    batch_size = int(batch_size)
    height = int(height)
    width = int(width)
    sigma = float(sigma)
    radius = int(radius)
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError('Target-centre map dimensions must be positive.')
    if sigma <= 0.0:
        raise ValueError('target-centre sigma must be positive.')
    if radius <= 0:
        raise ValueError('target-centre radius must be positive.')

    heatmaps = labels.new_zeros((batch_size, 1, height, width))
    if labels.numel() == 0:
        return heatmaps

    valid = (
        (labels > 0.5)
        & (target_ids.long() > 0)
        & (event_batch_indices >= 0)
        & (event_batch_indices < batch_size)
        & (event_x >= 0)
        & (event_x < width)
        & (event_y >= 0)
        & (event_y < height)
    )
    if not bool(valid.any()):
        return heatmaps

    for batch_index in event_batch_indices[valid].unique(sorted=True):
        sample_mask = valid & (event_batch_indices == batch_index)
        for target_id in target_ids[sample_mask].unique(sorted=True):
            target_mask = sample_mask & (target_ids == target_id)
            center_x = event_x[target_mask].float().mean()
            center_y = event_y[target_mask].float().mean()
            center_x_floor = int(torch.floor(center_x).item())
            center_y_floor = int(torch.floor(center_y).item())
            x_start = max(0, center_x_floor - radius)
            x_end = min(width - 1, center_x_floor + radius)
            y_start = max(0, center_y_floor - radius)
            y_end = min(height - 1, center_y_floor + radius)
            x_coordinates = torch.arange(
                x_start,
                x_end + 1,
                device=labels.device,
                dtype=labels.dtype,
            )
            y_coordinates = torch.arange(
                y_start,
                y_end + 1,
                device=labels.device,
                dtype=labels.dtype,
            )
            squared_distance = (
                (y_coordinates[:, None] - center_y).square()
                + (x_coordinates[None, :] - center_x).square()
            )
            gaussian = torch.exp(
                -squared_distance / (2.0 * sigma * sigma)
            )
            region = heatmaps[
                int(batch_index.item()),
                0,
                y_start:y_end + 1,
                x_start:x_end + 1,
            ]
            heatmaps[
                int(batch_index.item()),
                0,
                y_start:y_end + 1,
                x_start:x_end + 1,
            ] = torch.maximum(region, gaussian)
    return heatmaps


def target_center_heatmap_loss(
    logits,
    target_heatmaps,
    target_positive_loss_mass=0.20,
    max_positive_weight=512.0,
    empty_loss_weight=0.10,
):
    """Balanced BCE for soft target-centre heatmaps.

    The Gaussian target mass is much smaller than a full event frame. Each
    non-empty frame is balanced independently, while target-free views retain
    a small negative-only term so the centre head does not activate globally.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Target-centre logits must have shape [B, 1, H, W].')
    if target_heatmaps.shape != logits.shape:
        raise ValueError('Target-centre logits and heatmaps must match.')
    target_positive_loss_mass = float(target_positive_loss_mass)
    max_positive_weight = float(max_positive_weight)
    empty_loss_weight = float(empty_loss_weight)
    if not 0.0 < target_positive_loss_mass < 1.0:
        raise ValueError('target_positive_loss_mass must be in (0, 1).')
    if max_positive_weight < 1.0:
        raise ValueError('max_positive_weight must be at least one.')
    if not 0.0 <= empty_loss_weight <= 1.0:
        raise ValueError('empty_loss_weight must be in [0, 1].')

    target_heatmaps = target_heatmaps.to(dtype=logits.dtype)
    pixel_loss = functional.binary_cross_entropy_with_logits(
        logits,
        target_heatmaps,
        reduction='none',
    )
    per_view_losses = []
    positive_weights = []
    nonempty_views = 0
    pixel_count = logits.shape[2] * logits.shape[3]
    for batch_index in range(logits.shape[0]):
        sample_target = target_heatmaps[batch_index]
        sample_loss = pixel_loss[batch_index]
        target_mass = float(sample_target.detach().sum().item())
        if target_mass > 0.0:
            nonempty_views += 1
            positive_weight = min(
                max_positive_weight,
                max(
                    1.0,
                    (
                        (pixel_count - target_mass)
                        / target_mass
                        * target_positive_loss_mass
                        / (1.0 - target_positive_loss_mass)
                    ),
                ),
            )
            weights = 1.0 + sample_target * (positive_weight - 1.0)
            per_view_losses.append(
                (sample_loss * weights).sum() / weights.sum()
            )
            positive_weights.append(positive_weight)
        else:
            per_view_losses.append(sample_loss.mean() * empty_loss_weight)
            positive_weights.append(1.0)

    return torch.stack(per_view_losses).mean(), {
        'nonempty_view_fraction': float(nonempty_views / logits.shape[0]),
        'mean_positive_weight': float(
            sum(positive_weights) / len(positive_weights)
        ),
        'mean_target_mass': float(
            target_heatmaps.detach().sum().item() / logits.shape[0]
        ),
    }


def frame_balanced_event_bce(
    logits,
    labels,
    event_batch_indices,
    target_positive_loss_mass=0.20,
    max_positive_weight=16.0,
):
    """Compute BCE at event coordinates with bounded per-frame balancing.

    Each video-time view contributes one equally weighted loss. The positive
    factor is chosen so positives make up at most the configured fraction of
    that view's total loss mass. This increases recall without allowing an
    exceptionally sparse frame to dominate the whole minibatch.
    """
    if logits.ndim != 1 or labels.ndim != 1 or event_batch_indices.ndim != 1:
        raise ValueError('logits, labels, and event_batch_indices must be flat.')
    if not (
        logits.shape == labels.shape == event_batch_indices.shape
    ):
        raise ValueError('logits, labels, and event_batch_indices must match.')
    target_positive_loss_mass = float(target_positive_loss_mass)
    max_positive_weight = float(max_positive_weight)
    if not 0.0 < target_positive_loss_mass < 1.0:
        raise ValueError('target_positive_loss_mass must be in (0, 1).')
    if max_positive_weight < 1.0:
        raise ValueError('max_positive_weight must be at least one.')
    if logits.numel() == 0:
        return logits.sum() * 0.0, {
            'positive_fraction': 0.0,
            'mean_positive_weight': 1.0,
        }

    labels = labels.float()
    point_loss = functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction='none',
    )
    per_view_losses = []
    positive_weights = []
    for batch_index in event_batch_indices.unique(sorted=True):
        sample_mask = event_batch_indices == batch_index
        sample_labels = labels[sample_mask]
        sample_loss = point_loss[sample_mask]
        positive_mask = sample_labels > 0.5
        positive_count = int(positive_mask.sum().item())
        negative_count = int((~positive_mask).sum().item())
        if positive_count == 0 or negative_count == 0:
            positive_weight = 1.0
        else:
            positive_weight = min(
                max_positive_weight,
                max(
                    1.0,
                    (
                        negative_count
                        / float(positive_count)
                        * target_positive_loss_mass
                        / (1.0 - target_positive_loss_mass)
                    ),
                ),
            )
        weights = torch.ones_like(sample_loss)
        if positive_weight != 1.0:
            weights[positive_mask] = positive_weight
        per_view_losses.append((sample_loss * weights).sum() / weights.sum())
        positive_weights.append(positive_weight)

    return torch.stack(per_view_losses).mean(), {
        'positive_fraction': float(labels.mean().detach().item()),
        'mean_positive_weight': float(sum(positive_weights) / len(positive_weights)),
    }


def target_group_coverage_loss(
    logits,
    labels,
    target_ids,
    event_batch_indices,
    score_floor=0.70,
    correct_fraction=0.0001,
):
    """Ensure each labelled target-time group has enough confident events.

    Challenge Pd marks a target group detected once a very small fraction of
    its positive events is classified correctly. This loss uses the kth
    highest positive logit, where k is that fraction rounded up, then applies
    a hinge at a fixed score margin. Already covered targets produce no loss,
    so ordinary point-wise BCE remains responsible for IoU and false alarms.
    """
    if not (
        logits.ndim
        == labels.ndim
        == target_ids.ndim
        == event_batch_indices.ndim
        == 1
    ):
        raise ValueError('Target coverage inputs must be flat tensors.')
    if not (
        logits.shape
        == labels.shape
        == target_ids.shape
        == event_batch_indices.shape
    ):
        raise ValueError('Target coverage inputs must have matching shapes.')
    score_floor = float(score_floor)
    correct_fraction = float(correct_fraction)
    if not 0.0 < score_floor < 1.0:
        raise ValueError('score_floor must be in (0, 1).')
    if not 0.0 < correct_fraction <= 1.0:
        raise ValueError('correct_fraction must be in (0, 1].')
    if logits.numel() == 0:
        return logits.sum() * 0.0, {
            'target_group_count': 0,
            'uncovered_group_count': 0,
        }

    target_ids = target_ids.long()
    labels = labels.float()
    floor_logit = math.log(score_floor / (1.0 - score_floor))
    group_losses = []
    uncovered_group_count = 0
    for batch_index in event_batch_indices.unique(sorted=True):
        batch_mask = event_batch_indices == batch_index
        target_mask = batch_mask & (labels > 0.5) & (target_ids != 0)
        for target_id in target_ids[target_mask].unique(sorted=True):
            group_mask = target_mask & (target_ids == target_id)
            group_logits = logits[group_mask]
            required_count = max(
                1,
                int(math.ceil(group_logits.numel() * correct_fraction)),
            )
            kth_logit = torch.topk(
                group_logits,
                k=required_count,
                largest=True,
                sorted=False,
            ).values.min()
            group_loss = functional.relu(kth_logit.new_tensor(floor_logit) - kth_logit)
            group_losses.append(group_loss)

    if not group_losses:
        return logits.sum() * 0.0, {
            'target_group_count': 0,
            'uncovered_group_count': 0,
        }
    group_loss_tensor = torch.stack(group_losses)
    uncovered_group_count = int(
        (group_loss_tensor.detach() > 0.0).sum().item()
    )
    return group_loss_tensor.mean(), {
        'target_group_count': len(group_losses),
        'uncovered_group_count': uncovered_group_count,
    }
