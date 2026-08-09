import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dataset.temporal_memory import (
    TemporalMemoryTrainDataset,
    normalize_density_bucket_config,
    npz_event_count,
    temporal_memory_views_by_video,
)


class DensityBucketScheduleTest(unittest.TestCase):
    def test_boundaries_are_inclusive_upper_bounds(self):
        views, buckets = temporal_memory_views_by_video(
            [0, 29999, 30000, 30001, 200000, 200001],
            views_per_video=2,
            density_bucket_boundaries=[30000, 200000],
            density_bucket_views=[1, 6, 7],
        )

        np.testing.assert_array_equal(views, [1, 1, 1, 6, 6, 7])
        np.testing.assert_array_equal(buckets, [0, 0, 0, 1, 1, 2])

    def test_empty_bucket_config_preserves_legacy_dense_cutoff(self):
        views, buckets = temporal_memory_views_by_video(
            [199999, 200000, 200001],
            views_per_video=2,
            dense_sampling_enabled=True,
            dense_event_count_cutoff=200000,
            dense_view_multiplier=8,
            density_bucket_boundaries=[],
            density_bucket_views=[],
        )

        np.testing.assert_array_equal(views, [2, 2, 16])
        np.testing.assert_array_equal(buckets, [-1, -1, -1])

    def test_explicit_buckets_take_precedence_over_dense_multiplier(self):
        views, _ = temporal_memory_views_by_video(
            [10000, 250000],
            views_per_video=2,
            dense_sampling_enabled=True,
            dense_event_count_cutoff=200000,
            dense_view_multiplier=8,
            density_bucket_boundaries=[30000],
            density_bucket_views=[1, 5],
        )

        np.testing.assert_array_equal(views, [1, 5])

    def test_invalid_bucket_configs_are_rejected(self):
        invalid_configs = (
            ([30000], []),
            ([30000], [1]),
            ([30000, 30000], [1, 2, 3]),
            ([200000, 30000], [1, 2, 3]),
            ([0], [1, 2]),
            ([30000], [0, 2]),
            ([30000.5], [1, 2]),
        )
        for boundaries, views in invalid_configs:
            with self.subTest(boundaries=boundaries, views=views):
                with self.assertRaises(ValueError):
                    normalize_density_bucket_config(boundaries, views)


class DensityBucketDatasetTest(unittest.TestCase):
    @staticmethod
    def _write_npz(path, event_count):
        np.savez(
            path,
            ev_loc=np.zeros((event_count, 3), dtype=np.int64),
        )

    def test_npz_event_count_reads_array_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'video.npz'
            self._write_npz(path, 17)

            self.assertEqual(npz_event_count(path), 17)

    def test_dataset_builds_bucket_summary_without_loading_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, event_count in enumerate((2, 4, 7)):
                self._write_npz(root / 'train_{:03d}.npz'.format(index), event_count)

            with mock.patch.object(
                TemporalMemoryTrainDataset,
                '_load_video',
                side_effect=AssertionError('full video load is not allowed'),
            ):
                dataset = TemporalMemoryTrainDataset(
                    root=root,
                    whole_t=8000,
                    temporal_bin_size=50,
                    context_bins=5,
                    sequence_length=16,
                    width=346,
                    height=260,
                    views_per_video=2,
                    positive_frame_probability=0.75,
                    random_seed=49,
                    cache_all_videos=False,
                    cache_video_count=2,
                    density_bucket_boundaries=[2, 5],
                    density_bucket_views=[1, 3, 4],
                )

            np.testing.assert_array_equal(dataset.event_counts_by_video, [2, 4, 7])
            np.testing.assert_array_equal(dataset.views_by_video, [1, 3, 4])
            self.assertEqual(len(dataset), 8)
            self.assertEqual(dataset.density_bucket_video_counts, (1, 1, 1))
            self.assertEqual(dataset.density_bucket_sequence_counts, (1, 3, 4))
            self.assertEqual(
                dataset.sampling_summary(),
                {
                    'mode': 'density_buckets',
                    'video_count': 3,
                    'sequence_count': 8,
                    'views_per_video': 2,
                    'dense_event_count_cutoff': 200000,
                    'dense_view_multiplier': 2,
                    'dense_video_count': 0,
                    'extra_dense_views': 0,
                    'density_bucket_boundaries': [2, 5],
                    'density_bucket_views': [1, 3, 4],
                    'density_bucket_video_counts': [1, 1, 1],
                    'density_bucket_sequence_counts': [1, 3, 4],
                    'density_bucket_view_delta': 2,
                },
            )


if __name__ == '__main__':
    unittest.main()
