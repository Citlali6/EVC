"""Training-aligned sliding-window inference for temporal-memory models.

The released inference path treats every temporal bin in a video as one
bidirectional sequence.  Training instead resets the recurrent/attention
state for fixed-length sequences.  This module offers an opt-in diagnostic
path that preserves the training-time reset semantics without changing the
released full-stream default.

Overlapping windows are stitched by assigning every temporal bin to the
window whose centre is closest to that bin.  The assignment is deterministic,
covers every bin exactly once, and retains the complete first/last video edge.
"""

from dataclasses import dataclass

import numpy as np
import torch

from dataset.temporal_frame import build_temporal_context_frame


@dataclass(frozen=True)
class TemporalWindowStitch:
    """One inference window and the global output interval retained from it."""

    window_start: int
    window_stop: int
    keep_start: int
    keep_stop: int

    def __post_init__(self):
        values = (
            self.window_start,
            self.window_stop,
            self.keep_start,
            self.keep_stop,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError('Temporal window boundaries must be integers.')
        if not (
            0 <= self.window_start
            <= self.keep_start
            < self.keep_stop
            <= self.window_stop
        ):
            raise ValueError('The retained interval must be non-empty and inside its window.')

    @property
    def window_length(self):
        return self.window_stop - self.window_start

    @property
    def keep_length(self):
        return self.keep_stop - self.keep_start


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError('{} must be a positive integer.'.format(name))
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('{} must be a positive integer.'.format(name)) from error
    if normalized != value or normalized <= 0:
        raise ValueError('{} must be a positive integer.'.format(name))
    return normalized


def temporal_center_stitch_plan(temporal_bin_count, window_length, stride=None):
    """Build a deterministic nearest-centre overlap/stitch plan.

    ``stride`` defaults to half the window length.  A shorter final shift is
    inserted when the video length is not stride-aligned, so the final window
    always ends exactly at the video boundary.  Every bin is then assigned to
    the containing window with the nearest centre; ties go to the earlier
    window.  Returned ``keep`` intervals are therefore disjoint and cover
    ``[0, temporal_bin_count)`` exactly once.
    """

    temporal_bin_count = _positive_integer(
        temporal_bin_count,
        'temporal_bin_count',
    )
    window_length = _positive_integer(window_length, 'window_length')
    if window_length > temporal_bin_count:
        raise ValueError('window_length must not exceed temporal_bin_count.')
    if stride is None:
        stride = max(1, window_length // 2)
    stride = _positive_integer(stride, 'stride')
    if stride > window_length:
        raise ValueError('stride must not exceed window_length.')

    last_start = temporal_bin_count - window_length
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    starts = tuple(dict.fromkeys(starts))

    owners = np.empty(temporal_bin_count, dtype=np.int64)
    doubled_bin_centres = np.arange(
        1,
        temporal_bin_count * 2,
        2,
        dtype=np.int64,
    )
    doubled_window_centres = np.asarray(
        [start * 2 + window_length for start in starts],
        dtype=np.int64,
    )
    for temporal_bin, doubled_bin_centre in enumerate(doubled_bin_centres):
        candidates = [
            window_index
            for window_index, start in enumerate(starts)
            if start <= temporal_bin < start + window_length
        ]
        if not candidates:
            raise RuntimeError('Internal error: temporal stitch plan has a coverage gap.')
        owners[temporal_bin] = min(
            candidates,
            key=lambda window_index: (
                abs(
                    int(doubled_bin_centre)
                    - int(doubled_window_centres[window_index])
                ),
                window_index,
            ),
        )

    plan = []
    for window_index, start in enumerate(starts):
        retained = np.flatnonzero(owners == window_index)
        if retained.size == 0:
            continue
        keep_start = int(retained[0])
        keep_stop = int(retained[-1]) + 1
        if not np.array_equal(
            retained,
            np.arange(keep_start, keep_stop, dtype=np.int64),
        ):
            raise RuntimeError('Internal error: a retained window interval is not contiguous.')
        plan.append(
            TemporalWindowStitch(
                window_start=int(start),
                window_stop=int(start + window_length),
                keep_start=keep_start,
                keep_stop=keep_stop,
            )
        )

    cursor = 0
    for item in plan:
        if item.keep_start != cursor:
            raise RuntimeError('Internal error: retained intervals do not tile the video.')
        cursor = item.keep_stop
    if cursor != temporal_bin_count:
        raise RuntimeError('Internal error: retained intervals do not cover the video.')
    return tuple(plan)


def stitch_temporal_window_tensors(window_tensors, plan, temporal_bin_count):
    """Stitch window-local tensors according to a validated centre plan.

    This small public helper keeps coverage and boundary behavior directly
    testable.  Window tensors use time as their first dimension.
    """

    temporal_bin_count = _positive_integer(
        temporal_bin_count,
        'temporal_bin_count',
    )
    plan = tuple(plan)
    window_tensors = tuple(window_tensors)
    if len(window_tensors) != len(plan):
        raise ValueError('window_tensors and plan must have matching lengths.')
    if not plan:
        raise ValueError('plan must not be empty.')

    output = None
    covered = torch.zeros(temporal_bin_count, dtype=torch.int64)
    for tensor, item in zip(window_tensors, plan):
        if not isinstance(item, TemporalWindowStitch):
            raise TypeError('plan items must be TemporalWindowStitch instances.')
        if not torch.is_tensor(tensor):
            raise TypeError('window_tensors must contain torch tensors.')
        if tensor.ndim < 1 or tensor.shape[0] != item.window_length:
            raise ValueError('A window tensor has an unexpected time dimension.')
        if output is None:
            output = tensor.new_empty((temporal_bin_count,) + tuple(tensor.shape[1:]))
        elif tensor.shape[1:] != output.shape[1:]:
            raise ValueError('All window tensors must have matching non-time dimensions.')
        local_start = item.keep_start - item.window_start
        local_stop = item.keep_stop - item.window_start
        output[item.keep_start:item.keep_stop] = tensor[local_start:local_stop]
        covered[item.keep_start:item.keep_stop] += 1

    if output is None or not torch.equal(covered, torch.ones_like(covered)):
        raise RuntimeError('The temporal stitch plan must cover every bin exactly once.')
    return output


def _frame_tensor(
    video,
    temporal_bins,
    context_bins,
    width,
    height,
    log_count_clip,
    device,
):
    frames = np.stack(
        [
            build_temporal_context_frame(
                video,
                temporal_bin,
                context_bins,
                width,
                height,
                log_count_clip,
            )
            for temporal_bin in temporal_bins
        ],
        axis=0,
    )
    return torch.from_numpy(frames).float().to(device)


def predict_temporal_memory_scores_windowed(
    model,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    window_length,
    stride=None,
    log_count_clip=4.0,
):
    """Return one event probability using fixed-length reset/stitch inference.

    Encoder bottlenecks are computed once per temporal bin.  Only the temporal
    memory is rerun for overlapping windows, and decoder skips are recomputed
    once, matching the memory discipline of the released full-stream helper.
    """

    context_bins = _positive_integer(context_bins, 'context_bins')
    width = _positive_integer(width, 'width')
    height = _positive_integer(height, 'height')
    inference_batch_size = _positive_integer(
        inference_batch_size,
        'inference_batch_size',
    )
    if context_bins % 2 == 0:
        raise ValueError('context_bins must be odd.')
    if not np.isfinite(float(log_count_clip)) or float(log_count_clip) <= 0:
        raise ValueError('log_count_clip must be positive and finite.')
    temporal_bin_count = len(video.event_indices_by_bin)
    if temporal_bin_count <= 0:
        raise ValueError('video must contain temporal bins.')
    plan = temporal_center_stitch_plan(
        temporal_bin_count,
        window_length,
        stride=stride,
    )

    bottlenecks = []
    with torch.no_grad():
        for start in range(0, temporal_bin_count, inference_batch_size):
            temporal_bins = list(
                range(start, min(start + inference_batch_size, temporal_bin_count))
            )
            frames = _frame_tensor(
                video,
                temporal_bins,
                context_bins,
                width,
                height,
                log_count_clip,
                device,
            )
            bottlenecks.append(model.encode_bottleneck(frames))
        bottlenecks = torch.cat(bottlenecks, dim=0)
        residual_windows = [
            model.temporal_residual(
                bottlenecks[item.window_start:item.window_stop]
            )
            for item in plan
        ]
        residuals = stitch_temporal_window_tensors(
            residual_windows,
            plan,
            temporal_bin_count,
        )

    scores = np.empty(video.locations.shape[0], dtype=np.float32)
    confidence_enabled = bool(getattr(model, 'confidence_head_enabled', False))
    with torch.no_grad():
        for start in range(0, temporal_bin_count, inference_batch_size):
            temporal_bins = list(
                range(start, min(start + inference_batch_size, temporal_bin_count))
            )
            frames = _frame_tensor(
                video,
                temporal_bins,
                context_bins,
                width,
                height,
                log_count_clip,
                device,
            )
            decoded = model.decode_with_residual(
                frames,
                residuals[start:start + len(temporal_bins)],
                return_confidence_logits=confidence_enabled,
            )
            if confidence_enabled:
                logit_maps, confidence_maps = decoded
                probabilities = torch.sigmoid(logit_maps).squeeze(1).cpu().numpy()
                confidence_probabilities = (
                    torch.sigmoid(confidence_maps).squeeze(1).cpu().numpy()
                )
            else:
                probabilities = torch.sigmoid(decoded).squeeze(1).cpu().numpy()
                confidence_probabilities = None
            for local_index, temporal_bin in enumerate(temporal_bins):
                event_indices = video.event_indices_by_bin[temporal_bin]
                if event_indices.size == 0:
                    continue
                locations = video.locations[event_indices]
                event_probabilities = probabilities[
                    local_index,
                    locations[:, 1],
                    locations[:, 0],
                ]
                if confidence_probabilities is not None:
                    event_probabilities = event_probabilities * confidence_probabilities[
                        local_index,
                        locations[:, 1],
                        locations[:, 0],
                    ]
                scores[event_indices] = event_probabilities

    if not np.isfinite(scores).all():
        raise RuntimeError('Windowed temporal-memory inference produced non-finite scores.')
    return torch.from_numpy(scores)
