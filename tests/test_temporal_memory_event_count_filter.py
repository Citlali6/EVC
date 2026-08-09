"""CPU-only tests for strict temporal-memory training video filtering."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dataset.temporal_memory import (
    TemporalMemoryTrainDataset,
    normalize_min_event_count_exclusive,
)


def make_dataset(root, **overrides):
    options = {
        'root': root,
        'whole_t': 1000,
        'temporal_bin_size': 50,
        'context_bins': 5,
        'sequence_length': 2,
        'width': 32,
        'height': 24,
        'views_per_video': 1,
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


class TemporalMemoryEventCountFilterTests(unittest.TestCase):
    def file_root(self, names):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        for name in names:
            (root / name).touch()
        self.addCleanup(temporary.cleanup)
        return root

    def test_strict_cutoff_retains_only_counts_above_boundary(self):
        root = self.file_root(('a.npz', 'b.npz', 'c.npz'))
        counts = {'a.npz': 29999, 'b.npz': 30000, 'c.npz': 30001}
        with mock.patch(
            'dataset.temporal_memory.npz_event_count',
            side_effect=lambda path: counts[Path(path).name],
        ):
            dataset = make_dataset(
                root,
                min_event_count_exclusive=30000,
            )

        self.assertEqual([path.name for path in dataset.file_paths], ['c.npz'])
        self.assertEqual(dataset.source_video_indices.tolist(), [2])
        self.assertEqual(dataset.event_counts_by_video.tolist(), [30001])
        self.assertEqual(dataset.views_by_video.tolist(), [1])
        self.assertEqual(dataset.view_offsets.tolist(), [0, 1])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            dataset.sampling_summary(),
            {
                'mode': 'uniform',
                'source_video_count': 3,
                'video_count': 1,
                'excluded_video_count': 2,
                'sequence_count': 1,
                'views_per_video': 1,
                'min_event_count_exclusive': 30000,
                'dense_event_count_cutoff': 200000,
                'dense_view_multiplier': 2,
                'dense_video_count': 0,
                'extra_dense_views': 0,
                'density_bucket_boundaries': [],
                'density_bucket_views': [],
                'density_bucket_video_counts': [],
                'density_bucket_sequence_counts': [],
                'density_bucket_view_delta': 0,
            },
        )

    def test_disabled_filter_preserves_files_without_header_reads(self):
        root = self.file_root(('a.npz', 'b.npz', 'c.npz'))
        with mock.patch(
            'dataset.temporal_memory.npz_event_count'
        ) as event_count:
            dataset = make_dataset(root, min_event_count_exclusive=None)

        event_count.assert_not_called()
        self.assertEqual(len(dataset.file_paths), 3)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.source_video_indices.tolist(), [0, 1, 2])
        self.assertIsNone(
            dataset.sampling_summary()['min_event_count_exclusive']
        )

    def test_filter_that_excludes_every_video_fails(self):
        root = self.file_root(('a.npz', 'b.npz'))
        with mock.patch(
            'dataset.temporal_memory.npz_event_count',
            return_value=30000,
        ):
            with self.assertRaisesRegex(RuntimeError, 'retained no npz files'):
                make_dataset(root, min_event_count_exclusive=30000)

    def test_cutoff_validation_rejects_ambiguous_values(self):
        invalid_values = (True, False, -1, 2.5, 'not-an-integer')
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_min_event_count_exclusive(value)
        self.assertIsNone(normalize_min_event_count_exclusive(None))
        self.assertEqual(normalize_min_event_count_exclusive(30000), 30000)

    def test_filtered_sampling_preserves_original_video_seed(self):
        root = self.file_root(('a.npz', 'b.npz', 'c.npz'))
        counts = {'a.npz': 100, 'b.npz': 200, 'c.npz': 30001}
        with mock.patch(
            'dataset.temporal_memory.npz_event_count',
            side_effect=lambda path: counts[Path(path).name],
        ):
            filtered = make_dataset(
                root,
                min_event_count_exclusive=30000,
            )
        unfiltered = make_dataset(root)
        video = SimpleNamespace(
            positive_bins=np.asarray([1, 3, 5, 7], dtype=np.int64),
            occupied_bins=np.asarray([0, 1, 2, 3, 4, 5, 6, 7]),
        )
        for epoch in (0, 1, 5):
            filtered.set_epoch(epoch)
            unfiltered.set_epoch(epoch)
            self.assertEqual(
                filtered._sample_center_bin(0, 0, video),
                unfiltered._sample_center_bin(2, 0, video),
            )


if __name__ == '__main__':
    unittest.main()
