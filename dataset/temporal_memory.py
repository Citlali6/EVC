"""Sequence views for the bidirectional full-stream temporal memory model."""

from collections import OrderedDict
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.temporal_frame import (
    build_temporal_context_frame,
    load_temporal_frame_video,
)


def temporal_sequence_start(center_bin, bin_count, sequence_length):
    """Choose a fixed-length sequence centred on an observed time bin."""
    center_bin = int(center_bin)
    bin_count = int(bin_count)
    sequence_length = int(sequence_length)
    if bin_count <= 0 or sequence_length <= 0:
        raise ValueError('bin_count and sequence_length must be positive.')
    if sequence_length > bin_count:
        raise ValueError('sequence_length must not exceed bin_count.')
    if center_bin < 0 or center_bin >= bin_count:
        raise ValueError('center_bin is outside the available range.')
    return min(
        max(center_bin - sequence_length // 2, 0),
        bin_count - sequence_length,
    )


def _integer_tuple(values, name):
    """Normalize a configuration sequence without silently rounding values."""
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise ValueError('{} must be a sequence of integers.'.format(name))
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(
            '{} must be a sequence of integers.'.format(name)
        ) from error

    normalized = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('{} must contain only integers.'.format(name))
        try:
            normalized_value = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                '{} must contain only integers.'.format(name)
            ) from error
        if normalized_value != value:
            raise ValueError('{} must contain only integers.'.format(name))
        normalized.append(normalized_value)
    return tuple(normalized)


def normalize_density_bucket_config(boundaries=None, views=None):
    """Validate explicit density buckets and return immutable integer tuples.

    Boundaries are inclusive upper bounds. For example, boundaries
    ``[30000, 200000]`` produce the buckets ``<=30000``, ``30001..200000``
    and ``>200000``. Views are absolute per-video sequence counts, not
    multipliers. Empty boundaries and views disable this sampling mode.
    """
    boundaries = _integer_tuple(boundaries, 'density_bucket_boundaries')
    views = _integer_tuple(views, 'density_bucket_views')
    enabled = bool(boundaries or views)
    if not enabled:
        return (), ()
    if len(views) != len(boundaries) + 1:
        raise ValueError(
            'density_bucket_views must contain exactly one more value than '
            'density_bucket_boundaries.'
        )
    if any(boundary <= 0 for boundary in boundaries):
        raise ValueError('density bucket boundaries must be positive.')
    if any(
        left >= right
        for left, right in zip(boundaries, boundaries[1:])
    ):
        raise ValueError(
            'density bucket boundaries must be strictly ascending.'
        )
    if any(view_count <= 0 for view_count in views):
        raise ValueError('density bucket views must be positive.')
    return boundaries, views


def normalize_min_event_count_exclusive(value):
    """Normalize an optional strict lower event-count filter."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('min_event_count_exclusive must be an integer or null.')
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            'min_event_count_exclusive must be an integer or null.'
        ) from error
    if normalized != value:
        raise ValueError('min_event_count_exclusive must be an integer or null.')
    if normalized < 0:
        raise ValueError('min_event_count_exclusive must not be negative.')
    return normalized


def sparse_target_support_bins(video, max_events=3):
    """Return time bins containing a genuinely sparse labelled target.

    A target-time group is eligible when it contains between one and
    ``max_events`` positive events and has a strictly positive target id.
    ``video.event_bins`` already uses the dataset's metric-time binning, so
    the training configuration's 50-unit bins are preserved exactly.
    """
    if isinstance(max_events, (bool, np.bool_)):
        raise ValueError('max_events must be a positive integer.')
    try:
        max_events = int(max_events)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('max_events must be a positive integer.') from error
    if max_events <= 0:
        raise ValueError('max_events must be a positive integer.')

    event_bins = np.asarray(video.event_bins, dtype=np.int64).reshape(-1)
    labels = np.asarray(video.labels).reshape(-1)
    target_ids = np.asarray(video.target_ids, dtype=np.int64).reshape(-1)
    if not (
        event_bins.shape[0] == labels.shape[0] == target_ids.shape[0]
    ):
        raise ValueError('video event bins, labels, and target ids must align.')

    positive_target = (labels > 0.5) & (target_ids > 0)
    if not np.any(positive_target):
        return np.empty(0, dtype=np.int64)
    target_time_pairs = np.stack(
        (event_bins[positive_target], target_ids[positive_target]),
        axis=1,
    )
    unique_pairs, support = np.unique(
        target_time_pairs,
        axis=0,
        return_counts=True,
    )
    eligible = (support >= 1) & (support <= max_events)
    return np.unique(unique_pairs[eligible, 0]).astype(
        np.int64,
        copy=False,
    )


def temporal_memory_views_by_video(
    event_counts,
    views_per_video,
    dense_sampling_enabled=False,
    dense_event_count_cutoff=200000,
    dense_view_multiplier=2,
    density_bucket_boundaries=None,
    density_bucket_views=None,
):
    """Return per-video view counts and explicit bucket assignments.

    Explicit buckets take precedence when configured. With empty bucket
    configuration, the legacy dense multiplier (including its strict ``>``
    cutoff) is preserved.
    """
    event_counts = np.asarray(event_counts, dtype=np.int64).reshape(-1)
    views_per_video = int(views_per_video)
    if views_per_video <= 0:
        raise ValueError('views_per_video must be positive.')
    if np.any(event_counts < 0):
        raise ValueError('event_counts must not contain negative values.')

    boundaries, bucket_views = normalize_density_bucket_config(
        density_bucket_boundaries,
        density_bucket_views,
    )
    if bucket_views:
        bucket_indices = np.searchsorted(
            np.asarray(boundaries, dtype=np.int64),
            event_counts,
            side='left',
        ).astype(np.int64, copy=False)
        views_by_video = np.asarray(
            bucket_views,
            dtype=np.int64,
        )[bucket_indices]
        return views_by_video, bucket_indices

    views_by_video = np.full(
        event_counts.shape,
        views_per_video,
        dtype=np.int64,
    )
    bucket_indices = np.full(event_counts.shape, -1, dtype=np.int64)
    if dense_sampling_enabled:
        dense_event_count_cutoff = int(dense_event_count_cutoff)
        dense_view_multiplier = int(dense_view_multiplier)
        if dense_event_count_cutoff <= 0:
            raise ValueError('dense_event_count_cutoff must be positive.')
        if dense_view_multiplier < 2:
            raise ValueError('dense_view_multiplier must be at least two.')
        views_by_video[event_counts > dense_event_count_cutoff] *= (
            dense_view_multiplier
        )
    return views_by_video, bucket_indices


def npz_event_count(path, array_name='ev_loc'):
    """Read an event-array row count from its NPY header inside an NPZ file."""
    path = Path(path)
    member_name = '{}.npy'.format(array_name)
    try:
        with zipfile.ZipFile(path, mode='r') as archive:
            with archive.open(member_name, mode='r') as stream:
                version = np.lib.format.read_magic(stream)
                shape, _, _ = np.lib.format._read_array_header(
                    stream,
                    version,
                )
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(
            '{}: unable to read the {} array header.'.format(path, array_name)
        ) from error
    if not shape:
        raise ValueError(
            '{}: {} must have at least one dimension.'.format(path, array_name)
        )
    return int(shape[0])


class TemporalMemoryTrainDataset(Dataset):
    """Sample contiguous full-stream frame sequences without validation data."""

    def __init__(
        self,
        root,
        whole_t,
        temporal_bin_size,
        context_bins,
        sequence_length,
        width,
        height,
        views_per_video,
        positive_frame_probability,
        random_seed,
        log_count_clip=4.0,
        cache_all_videos=True,
        cache_video_count=16,
        dense_sampling_enabled=False,
        dense_event_count_cutoff=200000,
        dense_view_multiplier=2,
        density_bucket_boundaries=None,
        density_bucket_views=None,
        min_event_count_exclusive=None,
        source_name_include=None,
        sparse_target_support_sampling_enabled=False,
        sparse_target_support_max_events=3,
        sparse_target_support_probability=0.75,
    ):
        self.root = Path(root)
        self.whole_t = int(whole_t)
        self.temporal_bin_size = int(temporal_bin_size)
        self.context_bins = int(context_bins)
        self.sequence_length = int(sequence_length)
        self.width = int(width)
        self.height = int(height)
        self.views_per_video = int(views_per_video)
        self.positive_frame_probability = float(positive_frame_probability)
        self.random_seed = int(random_seed)
        self.log_count_clip = float(log_count_clip)
        self.cache_all_videos = bool(cache_all_videos)
        self.cache_video_count = int(cache_video_count)
        self.dense_sampling_enabled = bool(dense_sampling_enabled)
        self.dense_event_count_cutoff = int(dense_event_count_cutoff)
        self.dense_view_multiplier = int(dense_view_multiplier)
        self.min_event_count_exclusive = normalize_min_event_count_exclusive(
            min_event_count_exclusive
        )
        if source_name_include is not None:
            if isinstance(source_name_include, str):
                source_name_include = [
                    str(name).strip()
                    for name in source_name_include.split(",")
                    if str(name).strip()
                ]
            else:
                source_name_include = [
                    str(name) for name in source_name_include
                ]
        self.source_name_include = source_name_include
        self.sparse_target_support_sampling_enabled = bool(
            sparse_target_support_sampling_enabled
        )
        if isinstance(sparse_target_support_max_events, (bool, np.bool_)):
            raise ValueError(
                'sparse_target_support_max_events must be a positive integer.'
            )
        try:
            self.sparse_target_support_max_events = int(
                sparse_target_support_max_events
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                'sparse_target_support_max_events must be a positive integer.'
            ) from error
        if (
            self.sparse_target_support_max_events <= 0
            or self.sparse_target_support_max_events
            != sparse_target_support_max_events
        ):
            raise ValueError(
                'sparse_target_support_max_events must be a positive integer.'
            )
        if isinstance(sparse_target_support_probability, (bool, np.bool_)):
            raise ValueError(
                'sparse_target_support_probability must be in [0, 1].'
            )
        try:
            self.sparse_target_support_probability = float(
                sparse_target_support_probability
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                'sparse_target_support_probability must be in [0, 1].'
            ) from error
        if not (
            np.isfinite(self.sparse_target_support_probability)
            and 0.0 <= self.sparse_target_support_probability <= 1.0
        ):
            raise ValueError(
                'sparse_target_support_probability must be in [0, 1].'
            )
        if (
            self.sparse_target_support_sampling_enabled
            and self.temporal_bin_size != 50
        ):
            raise ValueError(
                'Sparse-target-support sampling requires 50-unit time bins.'
            )
        (
            self.density_bucket_boundaries,
            self.density_bucket_views,
        ) = normalize_density_bucket_config(
            density_bucket_boundaries,
            density_bucket_views,
        )
        self.density_bucket_sampling_enabled = bool(
            self.density_bucket_views
        )
        self.current_epoch = 0

        all_file_paths = sorted(self.root.glob('*.npz'))
        if not all_file_paths:
            raise RuntimeError('No npz files found in {}'.format(self.root))
        self.source_video_count = len(all_file_paths)
        self.source_video_indices = np.arange(
            self.source_video_count,
            dtype=np.int64,
        )
        filtered_event_counts = None
        if self.min_event_count_exclusive is None:
            self.file_paths = all_file_paths
        else:
            all_event_counts = np.asarray(
                [npz_event_count(path) for path in all_file_paths],
                dtype=np.int64,
            )
            retained = all_event_counts > self.min_event_count_exclusive
            self.file_paths = [
                path
                for path, keep in zip(all_file_paths, retained)
                if bool(keep)
            ]
            self.source_video_indices = self.source_video_indices[retained]
            filtered_event_counts = all_event_counts[retained]
            if not self.file_paths:
                raise RuntimeError(
                    'Event-count filter >{} retained no npz files in {}.'.format(
                        self.min_event_count_exclusive,
                        self.root,
                    )
                )
        if self.source_name_include is not None:
            included = set(self.source_name_include)
            retained_by_name = [
                path for path in self.file_paths
                if path.name in included
            ]
            if not retained_by_name:
                raise RuntimeError(
                    'Source-name filter retained no npz files in {}.'.format(
                        self.root,
                    )
                )
            self.file_paths = retained_by_name
        if self.context_bins < 1 or self.context_bins % 2 == 0:
            raise ValueError('context_bins must be a positive odd integer.')
        if self.sequence_length <= 0:
            raise ValueError('sequence_length must be positive.')
        if self.views_per_video <= 0:
            raise ValueError('views_per_video must be positive.')
        if not 0.0 <= self.positive_frame_probability <= 1.0:
            raise ValueError('positive_frame_probability must be in [0, 1].')
        if self.cache_video_count <= 0:
            raise ValueError('cache_video_count must be positive.')
        if self.dense_sampling_enabled:
            if self.dense_event_count_cutoff <= 0:
                raise ValueError('dense_event_count_cutoff must be positive.')
            if self.dense_view_multiplier < 2:
                raise ValueError('dense_view_multiplier must be at least two.')

        self._videos = {}
        self._lru = OrderedDict()
        if self.cache_all_videos:
            for video_index in range(len(self.file_paths)):
                video = self._load_video(video_index)
                if self.sequence_length > len(video.event_indices_by_bin):
                    raise ValueError(
                        'sequence_length exceeds the available temporal bins.'
                )
                self._videos[video_index] = video

        self.sparse_target_support_bins_by_video = ()
        self.sparse_target_support_video_count = 0
        self.sparse_target_support_bin_count = 0
        if self.sparse_target_support_sampling_enabled:
            sparse_bins = []
            for video_index in range(len(self.file_paths)):
                video = self._videos.get(video_index)
                if video is None:
                    video = self._load_video(video_index)
                sparse_bins.append(
                    sparse_target_support_bins(
                        video,
                        max_events=self.sparse_target_support_max_events,
                    )
                )
            self.sparse_target_support_bins_by_video = tuple(sparse_bins)
            self.sparse_target_support_video_count = sum(
                bins.size > 0 for bins in sparse_bins
            )
            self.sparse_target_support_bin_count = sum(
                bins.size for bins in sparse_bins
            )

        needs_event_counts = (
            self.dense_sampling_enabled
            or self.density_bucket_sampling_enabled
        )
        if filtered_event_counts is not None:
            self.event_counts_by_video = filtered_event_counts
        elif needs_event_counts:
            if self.cache_all_videos:
                self.event_counts_by_video = np.asarray(
                    [
                        self._videos[video_index].locations.shape[0]
                        for video_index in range(len(self.file_paths))
                    ],
                    dtype=np.int64,
                )
            else:
                self.event_counts_by_video = np.asarray(
                    [npz_event_count(path) for path in self.file_paths],
                    dtype=np.int64,
                )
        else:
            self.event_counts_by_video = np.zeros(
                len(self.file_paths),
                dtype=np.int64,
            )

        self.views_by_video, bucket_indices = temporal_memory_views_by_video(
            self.event_counts_by_video,
            self.views_per_video,
            self.dense_sampling_enabled,
            self.dense_event_count_cutoff,
            self.dense_view_multiplier,
            self.density_bucket_boundaries,
            self.density_bucket_views,
        )
        dense_mask = np.zeros(len(self.file_paths), dtype=bool)
        if (
            self.dense_sampling_enabled
            and not self.density_bucket_sampling_enabled
        ):
            dense_mask = (
                self.event_counts_by_video > self.dense_event_count_cutoff
            )
        self.dense_video_count = int(dense_mask.sum())
        view_delta = int(
            self.views_by_video.sum()
            - len(self.file_paths) * self.views_per_video
        )
        self.extra_dense_views = (
            view_delta
            if self.dense_sampling_enabled
            and not self.density_bucket_sampling_enabled
            else 0
        )
        self.density_bucket_view_delta = (
            view_delta if self.density_bucket_sampling_enabled else 0
        )
        if self.density_bucket_sampling_enabled:
            bucket_count = len(self.density_bucket_views)
            self.density_bucket_video_counts = tuple(
                int(np.count_nonzero(bucket_indices == bucket_index))
                for bucket_index in range(bucket_count)
            )
            self.density_bucket_sequence_counts = tuple(
                video_count * view_count
                for video_count, view_count in zip(
                    self.density_bucket_video_counts,
                    self.density_bucket_views,
                )
            )
        else:
            self.density_bucket_video_counts = ()
            self.density_bucket_sequence_counts = ()
        self.view_offsets = np.concatenate((
            np.zeros(1, dtype=np.int64),
            np.cumsum(self.views_by_video, dtype=np.int64),
        ))

    def _load_video(self, video_index):
        return load_temporal_frame_video(
            self.file_paths[video_index],
            self.temporal_bin_size,
            self.whole_t,
        )

    def _video(self, video_index):
        cached = self._videos.get(video_index)
        if cached is not None:
            return cached
        cached = self._lru.pop(video_index, None)
        if cached is not None:
            self._lru[video_index] = cached
            return cached
        cached = self._load_video(video_index)
        if self.sequence_length > len(cached.event_indices_by_bin):
            raise ValueError(
                'sequence_length exceeds the available temporal bins.'
            )
        self._lru[video_index] = cached
        while len(self._lru) > self.cache_video_count:
            self._lru.popitem(last=False)
        return cached

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def __len__(self):
        return int(self.view_offsets[-1])

    def sampling_summary(self):
        """Return JSON-friendly sampling diagnostics for experiment logs."""
        if self.density_bucket_sampling_enabled:
            mode = 'density_buckets'
        elif self.dense_sampling_enabled:
            mode = 'dense_multiplier'
        else:
            mode = 'uniform'
        summary = {
            'mode': mode,
            'source_video_count': self.source_video_count,
            'video_count': len(self.file_paths),
            'excluded_video_count': (
                self.source_video_count - len(self.file_paths)
            ),
            'sequence_count': len(self),
            'views_per_video': self.views_per_video,
            'min_event_count_exclusive': self.min_event_count_exclusive,
            'dense_event_count_cutoff': self.dense_event_count_cutoff,
            'dense_view_multiplier': self.dense_view_multiplier,
            'dense_video_count': self.dense_video_count,
            'extra_dense_views': self.extra_dense_views,
            'density_bucket_boundaries': list(
                self.density_bucket_boundaries
            ),
            'density_bucket_views': list(self.density_bucket_views),
            'density_bucket_video_counts': list(
                self.density_bucket_video_counts
            ),
            'density_bucket_sequence_counts': list(
                self.density_bucket_sequence_counts
            ),
            'density_bucket_view_delta': self.density_bucket_view_delta,
        }
        if self.sparse_target_support_sampling_enabled:
            summary.update(
                {
                    'sparse_target_support_sampling_enabled': True,
                    'sparse_target_support_max_events': (
                        self.sparse_target_support_max_events
                    ),
                    'sparse_target_support_probability': (
                        self.sparse_target_support_probability
                    ),
                    'sparse_target_support_video_count': (
                        self.sparse_target_support_video_count
                    ),
                    'sparse_target_support_bin_count': (
                        self.sparse_target_support_bin_count
                    ),
                    'sparse_target_support_bin_counts_by_video': [
                        int(bins.size)
                        for bins in self.sparse_target_support_bins_by_video
                    ],
                }
            )
        return summary

    def _sample_center_bin(self, video_index, view_index, video):
        source_video_index = int(self.source_video_indices[int(video_index)])
        seed = (
            self.random_seed
            + 1000003 * self.current_epoch
            + 1009 * source_video_index
            + view_index
        )
        rng = np.random.default_rng(seed)
        if self.sparse_target_support_sampling_enabled:
            sparse_bins = self.sparse_target_support_bins_by_video[
                int(video_index)
            ]
            use_sparse_target = (
                sparse_bins.size > 0
                and rng.random() < self.sparse_target_support_probability
            )
            if use_sparse_target:
                return int(sparse_bins[rng.integers(sparse_bins.size)])
        use_positive = (
            video.positive_bins.size > 0
            and rng.random() < self.positive_frame_probability
        )
        candidates = video.positive_bins if use_positive else video.occupied_bins
        if candidates.size == 0:
            raise RuntimeError('{} contains no valid event-time bins.'.format(video.name))
        return int(candidates[rng.integers(candidates.size)])

    def __getitem__(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError('Temporal-memory sample index is out of range.')
        video_index = int(np.searchsorted(self.view_offsets, index, side='right') - 1)
        view_index = int(index - self.view_offsets[video_index])
        video = self._video(video_index)
        center_bin = self._sample_center_bin(video_index, view_index, video)
        start_bin = temporal_sequence_start(
            center_bin,
            len(video.event_indices_by_bin),
            self.sequence_length,
        )

        frames = []
        event_time_indices = []
        event_timestamps = []
        event_x = []
        event_y = []
        labels = []
        target_ids = []
        for sequence_index, temporal_bin in enumerate(
            range(start_bin, start_bin + self.sequence_length)
        ):
            frames.append(
                build_temporal_context_frame(
                    video,
                    temporal_bin,
                    self.context_bins,
                    self.width,
                    self.height,
                    self.log_count_clip,
                )
            )
            event_indices = video.event_indices_by_bin[temporal_bin]
            if event_indices.size == 0:
                continue
            locations = video.locations[event_indices]
            event_time_indices.append(
                np.full(event_indices.shape, sequence_index, dtype=np.int64)
            )
            event_timestamps.append(
                locations[:, 2].astype(np.int64, copy=False)
            )
            event_x.append(locations[:, 0].astype(np.int64, copy=False))
            event_y.append(locations[:, 1].astype(np.int64, copy=False))
            labels.append(video.labels[event_indices].astype(np.float32, copy=False))
            target_ids.append(
                video.target_ids[event_indices].astype(np.int64, copy=False)
            )

        if not event_time_indices:
            raise RuntimeError('Sampled sequence contains no events.')
        return {
            'frames': np.stack(frames, axis=0),
            'event_time_indices': np.concatenate(event_time_indices),
            'event_timestamps': np.concatenate(event_timestamps),
            'event_x': np.concatenate(event_x),
            'event_y': np.concatenate(event_y),
            'labels': np.concatenate(labels),
            'target_ids': np.concatenate(target_ids),
        }


def temporal_memory_collate(samples):
    """Keep one variable-event sequence per GPU step for predictable memory."""
    if len(samples) != 1:
        raise ValueError('Temporal-memory training requires batch_size=1.')
    sample = samples[0]
    return {
        'frames': torch.from_numpy(sample['frames']).float(),
        'event_time_indices': torch.from_numpy(
            sample['event_time_indices']
        ).long(),
        'event_timestamps': torch.from_numpy(
            sample['event_timestamps']
        ).long(),
        'event_x': torch.from_numpy(sample['event_x']).long(),
        'event_y': torch.from_numpy(sample['event_y']).long(),
        'labels': torch.from_numpy(sample['labels']).float(),
        'target_ids': torch.from_numpy(sample['target_ids']).long(),
    }
