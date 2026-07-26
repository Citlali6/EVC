import unittest

import numpy as np

from dataset.sampling import (
    dense_target_oversample_repeats,
    density_dual_view_modes,
    select_training_event_indices,
)


class TargetPreservingSamplingTests(unittest.TestCase):
    def test_all_positive_events_are_kept_when_they_fit_the_budget(self):
        labels = np.array([0, 1, 0, 1, 0, 0, 1, 0])
        selected = select_training_event_indices(
            labels,
            max_events_num=5,
            target_preserving_enabled=True,
            rng=np.random.default_rng(37),
        )

        self.assertEqual(len(selected), 5)
        self.assertEqual(len(np.unique(selected)), 5)
        self.assertTrue({1, 3, 6}.issubset(set(selected.tolist())))

    def test_disabled_mode_uses_a_uniform_subset_without_duplicates(self):
        labels = np.array([0, 1, 0, 1, 0, 0, 1, 0])
        selected = select_training_event_indices(
            labels,
            max_events_num=5,
            target_preserving_enabled=False,
            rng=np.random.default_rng(37),
        )

        self.assertEqual(len(selected), 5)
        self.assertEqual(len(np.unique(selected)), 5)
        self.assertTrue(np.all((selected >= 0) & (selected < len(labels))))

    def test_positive_only_fallback_respects_the_budget(self):
        labels = np.array([1, 1, 1, 1, 1, 0])
        selected = select_training_event_indices(
            labels,
            max_events_num=3,
            target_preserving_enabled=True,
            rng=np.random.default_rng(37),
        )

        self.assertEqual(len(selected), 3)
        self.assertTrue(np.all(labels[selected] == 1))

    def test_exact_budget_keeps_the_original_event_order(self):
        labels = np.array([0, 1, 0, 1])
        selected = select_training_event_indices(
            labels,
            max_events_num=4,
            target_preserving_enabled=False,
            rng=np.random.default_rng(37),
        )

        np.testing.assert_array_equal(selected, np.arange(4))

    def test_density_dual_view_adds_a_uniform_complement_only_above_cutoff(self):
        self.assertEqual(
            density_dual_view_modes(100000, 100000, density_dual_view_enabled=True),
            ('standard',),
        )
        self.assertEqual(
            density_dual_view_modes(100001, 100000, density_dual_view_enabled=True),
            ('target_preserving', 'uniform'),
        )

    def test_density_dual_view_disabled_keeps_the_standard_view(self):
        self.assertEqual(
            density_dual_view_modes(250000, 100000, density_dual_view_enabled=False),
            ('standard',),
        )

    def test_dense_target_oversampling_repeats_only_videos_above_cutoff(self):
        self.assertEqual(
            dense_target_oversample_repeats(
                100000,
                100000,
                dense_target_oversampling_enabled=True,
                factor=5,
            ),
            1,
        )
        self.assertEqual(
            dense_target_oversample_repeats(
                100001,
                100000,
                dense_target_oversampling_enabled=True,
                factor=5,
            ),
            5,
        )

    def test_dense_target_oversampling_disabled_keeps_one_view(self):
        self.assertEqual(
            dense_target_oversample_repeats(
                250000,
                100000,
                dense_target_oversampling_enabled=False,
                factor=5,
            ),
            1,
        )


if __name__ == '__main__':
    unittest.main()
