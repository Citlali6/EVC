"""CPU-only tests for train-label sparse-target-support sampling."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dataset.temporal_frame import temporal_frame_video_from_events
from dataset.temporal_memory import (
    TemporalMemoryTrainDataset,
    sparse_target_support_bins,
)


def write_video(path):
    groups = (
        (0, 1, 1, 1.0),
        (1, 1, 3, 1.0),
        (2, 1, 4, 1.0),
        (3, 0, 2, 1.0),
        (4, 2, 2, 1.0),
        (5, 3, 1, 0.0),
    )
    locations = []
    rows = []
    event_index = 0
    for temporal_bin, target_id, count, label in groups:
        for _ in range(count):
            locations.append(
                (event_index % 16, event_index % 12, temporal_bin * 50 + 1)
            )
            row = np.zeros(6, dtype=np.float32)
            row[3] = 1.0
            row[4] = label
            row[5] = target_id
            rows.append(row)
            event_index += 1
    np.savez(
        path,
        ev_loc=np.asarray(locations, dtype=np.int64),
        evs_norm=np.stack(rows),
    )


def make_dataset(root, **overrides):
    options = {
        'root': root,
        'whole_t': 400,
        'temporal_bin_size': 50,
        'context_bins': 5,
        'sequence_length': 2,
        'width': 16,
        'height': 12,
        'views_per_video': 4,
        'positive_frame_probability': 0.75,
        'random_seed': 49,
        'cache_all_videos': False,
        'cache_video_count': 1,
        'dense_sampling_enabled': False,
        'density_bucket_boundaries': [],
        'density_bucket_views': [],
    }
    options.update(overrides)
    return TemporalMemoryTrainDataset(**options)


def legacy_center(random_seed, epoch, source_index, view_index, video):
    rng = np.random.default_rng(
        random_seed + 1000003 * epoch + 1009 * source_index + view_index
    )
    use_positive = (
        video.positive_bins.size > 0 and rng.random() < 0.75
    )
    candidates = video.positive_bins if use_positive else video.occupied_bins
    return int(candidates[rng.integers(candidates.size)])


class SparseTargetSupportPoolTest(unittest.TestCase):
    def test_pool_uses_positive_target_groups_with_support_one_to_three(self):
        locations = np.asarray(
            [(0, 0, temporal_bin * 50 + 1) for temporal_bin in range(6)],
            dtype=np.int64,
        )
        video = temporal_frame_video_from_events(
            name='unit',
            locations=locations,
            polarities=np.ones(6, dtype=np.float32),
            temporal_bin_size=50,
            whole_t=400,
            labels=np.asarray([1, 1, 1, 1, 1, 0], dtype=np.float32),
            target_ids=np.asarray([1, 1, 2, 0, 3, 4], dtype=np.int64),
        )
        # Give target 1 support three in bin 1, target 2 support four in bin 2.
        video.event_bins = np.asarray([0, 1, 1, 3, 4, 5], dtype=np.int64)
        video.labels = np.asarray([1, 1, 1, 1, 1, 0], dtype=np.float32)
        video.target_ids = np.asarray([1, 1, 1, 0, 3, 4], dtype=np.int64)

        np.testing.assert_array_equal(
            sparse_target_support_bins(video, max_events=3),
            np.asarray([0, 1, 4]),
        )

    def test_pool_rejects_invalid_support_limit(self):
        video = SimpleNamespace(
            event_bins=np.asarray([0]),
            labels=np.asarray([1.0]),
            target_ids=np.asarray([1]),
        )
        for value in (0, -1, True, 'invalid'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    sparse_target_support_bins(video, max_events=value)


class SparseTargetSupportDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write_video(self.root / 'train_000.npz')

    def test_enabled_sampler_builds_pool_and_summary(self):
        dataset = make_dataset(
            self.root,
            sparse_target_support_sampling_enabled=True,
        )

        np.testing.assert_array_equal(
            dataset.sparse_target_support_bins_by_video[0],
            np.asarray([0, 1, 4]),
        )
        summary = dataset.sampling_summary()
        self.assertTrue(summary['sparse_target_support_sampling_enabled'])
        self.assertEqual(summary['sparse_target_support_max_events'], 3)
        self.assertEqual(summary['sparse_target_support_probability'], 0.75)
        self.assertEqual(summary['sparse_target_support_video_count'], 1)
        self.assertEqual(summary['sparse_target_support_bin_count'], 3)
        self.assertEqual(
            summary['sparse_target_support_bin_counts_by_video'],
            [3],
        )

    def test_default_off_does_not_load_or_change_summary_schema(self):
        with mock.patch.object(
            TemporalMemoryTrainDataset,
            '_load_video',
            side_effect=AssertionError('disabled sampler must not read labels'),
        ):
            dataset = make_dataset(self.root)

        self.assertNotIn(
            'sparse_target_support_sampling_enabled',
            dataset.sampling_summary(),
        )

    def test_default_off_preserves_legacy_random_draws_exactly(self):
        dataset = make_dataset(self.root)
        video = SimpleNamespace(
            positive_bins=np.asarray([1, 3, 5, 7], dtype=np.int64),
            occupied_bins=np.arange(8, dtype=np.int64),
        )
        for epoch in (0, 1, 5):
            dataset.set_epoch(epoch)
            for view_index in range(12):
                self.assertEqual(
                    dataset._sample_center_bin(0, view_index, video),
                    legacy_center(49, epoch, 0, view_index, video),
                )

    def test_enabled_probability_is_deterministic_by_epoch_and_view(self):
        first = make_dataset(
            self.root,
            sparse_target_support_sampling_enabled=True,
        )
        second = make_dataset(
            self.root,
            sparse_target_support_sampling_enabled=True,
        )
        video = first._load_video(0)
        for epoch in (0, 2, 7):
            first.set_epoch(epoch)
            second.set_epoch(epoch)
            first_centers = [
                first._sample_center_bin(0, view_index, video)
                for view_index in range(20)
            ]
            second_centers = [
                second._sample_center_bin(0, view_index, video)
                for view_index in range(20)
            ]
            self.assertEqual(first_centers, second_centers)

    def test_probability_one_always_selects_sparse_pool(self):
        dataset = make_dataset(
            self.root,
            sparse_target_support_sampling_enabled=True,
            sparse_target_support_probability=1.0,
        )
        video = dataset._load_video(0)
        sparse_bins = set(
            dataset.sparse_target_support_bins_by_video[0].tolist()
        )
        for epoch in (0, 3):
            dataset.set_epoch(epoch)
            for view_index in range(20):
                self.assertIn(
                    dataset._sample_center_bin(0, view_index, video),
                    sparse_bins,
                )

    def test_empty_sparse_pool_falls_back_to_legacy_sampling(self):
        dataset = make_dataset(
            self.root,
            sparse_target_support_sampling_enabled=True,
        )
        dataset.sparse_target_support_bins_by_video = (
            np.empty(0, dtype=np.int64),
        )
        video = SimpleNamespace(
            positive_bins=np.asarray([1, 3, 5, 7], dtype=np.int64),
            occupied_bins=np.arange(8, dtype=np.int64),
        )
        for epoch in (0, 4):
            dataset.set_epoch(epoch)
            for view_index in range(8):
                self.assertEqual(
                    dataset._sample_center_bin(0, view_index, video),
                    legacy_center(49, epoch, 0, view_index, video),
                )

    def test_invalid_configuration_is_rejected(self):
        invalid = (
            {'sparse_target_support_max_events': 0},
            {'sparse_target_support_max_events': True},
            {'sparse_target_support_probability': -0.1},
            {'sparse_target_support_probability': 1.1},
            {'sparse_target_support_probability': True},
            {
                'sparse_target_support_sampling_enabled': True,
                'temporal_bin_size': 25,
            },
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    make_dataset(self.root, **overrides)


if __name__ == '__main__':
    unittest.main()
