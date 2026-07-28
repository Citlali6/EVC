import unittest

import numpy as np

from utils.postprocess import (
    P0ClusterFilterConfig,
    P0bTrackFilterConfig,
    P18ScoreTrackRecoveryConfig,
    filter_positive_events,
    filter_positive_events_by_tracks,
    recover_seed_supported_track_events,
)


class P0ClusterFilterTests(unittest.TestCase):
    def test_disabled_filter_returns_original_mask(self):
        locations = np.array(
            [
                [0, 10, 10, 100],
                [0, 80, 80, 100],
            ]
        )
        positive_mask = np.array([True, False])
        config = P0ClusterFilterConfig(enabled=False)

        kept_mask, stats = filter_positive_events(positive_mask, locations, config)

        np.testing.assert_array_equal(kept_mask, positive_mask)
        self.assertFalse(stats.enabled)

    def test_small_clusters_are_removed_per_video(self):
        locations = np.array(
            [
                [0, 10, 10, 100],
                [0, 11, 10, 105],
                [0, 10, 11, 110],
                [0, 80, 80, 100],
                [1, 10, 10, 100],
            ]
        )
        positive_mask = np.ones(len(locations), dtype=bool)
        config = P0ClusterFilterConfig(
            enabled=True,
            spatial_radius=1,
            temporal_bin_size=50,
            temporal_radius_bins=1,
            min_cluster_events=2,
            min_duration_bins=1,
        )

        kept_mask, stats = filter_positive_events(positive_mask, locations, config)

        np.testing.assert_array_equal(
            kept_mask,
            np.array([True, True, True, False, False]),
        )
        self.assertEqual(stats.component_count, 3)
        self.assertEqual(stats.removed_components, 2)
        self.assertEqual(stats.removed_positive_events, 2)

    def test_minimum_duration_is_applied_after_cluster_size(self):
        locations = np.array(
            [
                [0, 10, 10, 100],
                [0, 10, 10, 105],
                [0, 20, 20, 100],
                [0, 20, 20, 151],
            ]
        )
        positive_mask = np.ones(len(locations), dtype=bool)
        config = P0ClusterFilterConfig(
            enabled=True,
            spatial_radius=1,
            temporal_bin_size=50,
            temporal_radius_bins=1,
            min_cluster_events=2,
            min_duration_bins=2,
        )

        kept_mask, stats = filter_positive_events(positive_mask, locations, config)

        np.testing.assert_array_equal(
            kept_mask,
            np.array([False, False, True, True]),
        )
        self.assertEqual(stats.kept_components, 1)
        self.assertEqual(stats.removed_components, 1)

    def test_p0c_recovers_only_high_confidence_small_clusters(self):
        locations = np.array(
            [
                [0, 10, 10, 100],
                [0, 10, 11, 101],
                [0, 80, 80, 100],
                [0, 80, 81, 101],
            ]
        )
        positive_mask = np.ones(len(locations), dtype=bool)
        prediction_scores = np.array([0.995, 0.93, 0.97, 0.94])
        config = P0ClusterFilterConfig(
            enabled=True,
            spatial_radius=1,
            temporal_bin_size=50,
            temporal_radius_bins=1,
            min_cluster_events=3,
            min_duration_bins=1,
            high_confidence_recovery_enabled=True,
            retain_min_score=0.98,
        )

        kept_mask, stats = filter_positive_events(
            positive_mask,
            locations,
            config,
            prediction_scores=prediction_scores,
        )

        np.testing.assert_array_equal(
            kept_mask,
            np.array([True, True, False, False]),
        )
        self.assertEqual(stats.recovered_components, 1)
        self.assertEqual(stats.recovered_positive_events, 2)

    def test_p0c_requires_scores_when_enabled(self):
        locations = np.array([[0, 10, 10, 100]])
        config = P0ClusterFilterConfig(
            enabled=True,
            high_confidence_recovery_enabled=True,
        )

        with self.assertRaisesRegex(ValueError, 'requires prediction scores'):
            filter_positive_events(
                np.array([True]),
                locations,
                config,
            )

    def test_p0b_retains_a_moving_two_frame_track_that_p0_drops(self):
        locations = np.array(
            [
                [0, 10, 10, 0],
                [0, 10, 11, 1],
                [0, 13, 10, 50],
                [0, 13, 11, 51],
            ]
        )
        positive_mask = np.ones(len(locations), dtype=bool)
        p0_config = P0ClusterFilterConfig(
            enabled=True,
            spatial_radius=1,
            temporal_bin_size=50,
            temporal_radius_bins=1,
            min_cluster_events=3,
            min_duration_bins=1,
        )
        p0b_config = P0bTrackFilterConfig(
            enabled=True,
            spatial_radius=1,
            temporal_bin_size=50,
            max_link_distance=5.0,
            max_gap_bins=1,
            min_track_events=3,
            min_track_frames=1,
        )

        p0_kept_mask, _ = filter_positive_events(
            positive_mask,
            locations,
            p0_config,
        )
        p0b_kept_mask, p0b_stats = filter_positive_events_by_tracks(
            positive_mask,
            locations,
            p0b_config,
        )

        np.testing.assert_array_equal(p0_kept_mask, np.zeros(4, dtype=bool))
        np.testing.assert_array_equal(p0b_kept_mask, np.ones(4, dtype=bool))
        self.assertEqual(p0b_stats.track_count, 1)
        self.assertEqual(p0b_stats.kept_tracks, 1)

    def test_p0b_removes_isolated_noise_component(self):
        locations = np.array(
            [
                [0, 50, 50, 0],
                [0, 50, 51, 1],
            ]
        )
        positive_mask = np.ones(len(locations), dtype=bool)
        config = P0bTrackFilterConfig(
            enabled=True,
            spatial_radius=1,
            temporal_bin_size=50,
            max_link_distance=5.0,
            max_gap_bins=1,
            min_track_events=3,
            min_track_frames=1,
        )

        kept_mask, stats = filter_positive_events_by_tracks(
            positive_mask,
            locations,
            config,
        )

        np.testing.assert_array_equal(kept_mask, np.zeros(2, dtype=bool))
        self.assertEqual(stats.track_count, 1)
        self.assertEqual(stats.removed_tracks, 1)

    def test_disabled_p0b_returns_original_mask(self):
        locations = np.array(
            [
                [0, 10, 10, 100],
                [0, 80, 80, 100],
            ]
        )
        positive_mask = np.array([True, False])
        config = P0bTrackFilterConfig(enabled=False)

        kept_mask, stats = filter_positive_events_by_tracks(
            positive_mask,
            locations,
            config,
        )

        np.testing.assert_array_equal(kept_mask, positive_mask)
        self.assertFalse(stats.enabled)

    def test_p18_restores_one_event_from_a_seed_supported_dense_track(self):
        locations = np.array(
            [
                [0, 10, 10, 0],
                [0, 13, 10, 50],
                [0, 13, 11, 51],
                [0, 80, 80, 50],
                [0, 80, 81, 51],
            ]
        )
        scores = np.array([0.95, 0.83, 0.87, 0.86, 0.82])
        config = P18ScoreTrackRecoveryConfig(
            enabled=True,
            event_count_cutoff=4,
            candidate_floor=0.80,
            spatial_radius=1,
            temporal_bin_size=50,
            max_link_distance=5.0,
            max_gap_bins=1,
            min_track_bins=2,
        )

        recovery_mask, stats = recover_seed_supported_track_events(
            scores,
            locations,
            config,
            prediction_threshold=0.90,
        )

        np.testing.assert_array_equal(
            recovery_mask,
            np.array([False, False, True, False, False]),
        )
        self.assertEqual(stats.eligible_videos, 1)
        self.assertEqual(stats.supported_tracks, 1)
        self.assertEqual(stats.restored_components, 1)
        self.assertEqual(stats.restored_events, 1)

    def test_p18_component_mode_restores_all_weak_component_events(self):
        locations = np.array(
            [
                [0, 10, 10, 0],
                [0, 13, 10, 50],
                [0, 13, 11, 51],
                [0, 14, 10, 52],
            ]
        )
        scores = np.array([0.95, 0.83, 0.87, 0.82])
        config = P18ScoreTrackRecoveryConfig(
            enabled=True,
            event_count_cutoff=1,
            candidate_floor=0.80,
            spatial_radius=1,
            temporal_bin_size=50,
            max_link_distance=5.0,
            max_gap_bins=1,
            min_track_bins=2,
            restore_mode='component',
        )

        recovery_mask, stats = recover_seed_supported_track_events(
            scores,
            locations,
            config,
            prediction_threshold=0.90,
        )

        np.testing.assert_array_equal(
            recovery_mask,
            np.array([False, True, True, True]),
        )
        self.assertEqual(stats.restored_components, 1)
        self.assertEqual(stats.restored_events, 3)

    def test_p18_topk_mode_restores_only_best_weak_events(self):
        locations = np.array(
            [
                [0, 10, 10, 0],
                [0, 13, 10, 50],
                [0, 13, 11, 51],
                [0, 14, 10, 52],
            ]
        )
        scores = np.array([0.95, 0.83, 0.87, 0.82])
        config = P18ScoreTrackRecoveryConfig(
            enabled=True,
            event_count_cutoff=1,
            candidate_floor=0.80,
            spatial_radius=1,
            temporal_bin_size=50,
            max_link_distance=5.0,
            max_gap_bins=1,
            min_track_bins=2,
            restore_mode='topk',
            max_restore_events_per_component=2,
        )

        recovery_mask, stats = recover_seed_supported_track_events(
            scores,
            locations,
            config,
            prediction_threshold=0.90,
        )

        np.testing.assert_array_equal(
            recovery_mask,
            np.array([False, True, True, False]),
        )
        self.assertEqual(stats.restored_events, 2)

    def test_p18_does_not_recover_unseeded_or_small_videos(self):
        locations = np.array(
            [
                [0, 10, 10, 0],
                [0, 13, 10, 50],
                [1, 20, 20, 0],
                [1, 40, 20, 50],
            ]
        )
        scores = np.array([0.85, 0.87, 0.95, 0.86])
        config = P18ScoreTrackRecoveryConfig(
            enabled=True,
            event_count_cutoff=1,
            candidate_floor=0.80,
            spatial_radius=1,
            temporal_bin_size=50,
            max_link_distance=5.0,
            max_gap_bins=1,
            min_track_bins=2,
        )

        recovery_mask, stats = recover_seed_supported_track_events(
            scores,
            locations,
            config,
            prediction_threshold=0.90,
        )

        np.testing.assert_array_equal(recovery_mask, np.zeros(4, dtype=bool))
        self.assertEqual(stats.eligible_videos, 2)
        self.assertEqual(stats.restored_events, 0)

    def test_p18_respects_the_event_count_cutoff(self):
        locations = np.array(
            [
                [0, 10, 10, 0],
                [0, 13, 10, 50],
            ]
        )
        scores = np.array([0.95, 0.86])
        config = P18ScoreTrackRecoveryConfig(
            enabled=True,
            event_count_cutoff=2,
            candidate_floor=0.80,
            spatial_radius=1,
            temporal_bin_size=50,
            max_link_distance=5.0,
            max_gap_bins=1,
            min_track_bins=2,
        )

        recovery_mask, stats = recover_seed_supported_track_events(
            scores,
            locations,
            config,
            prediction_threshold=0.90,
        )

        np.testing.assert_array_equal(recovery_mask, np.zeros(2, dtype=bool))
        self.assertEqual(stats.eligible_videos, 0)


if __name__ == '__main__':
    unittest.main()
