import unittest

from utils.density_threshold import (
    ChallengeCountTotals,
    DensityAdaptiveThresholdConfig,
    aggregate_challenge_counts,
    select_density_threshold,
)


class DensityThresholdTests(unittest.TestCase):
    def test_selects_threshold_from_event_count(self):
        self.assertEqual(select_density_threshold(80000, 80000, 0.7, 0.9), 0.7)
        self.assertEqual(select_density_threshold(80001, 80000, 0.7, 0.9), 0.9)

    def test_aggregates_event_and_detection_counts(self):
        metrics = aggregate_challenge_counts([
            ChallengeCountTotals(8, 2, 10, 4, 5, 3, 2),
            ChallengeCountTotals(6, 4, 10, 3, 5, 1, 2),
        ])

        self.assertAlmostEqual(metrics.iou, 14 / 26)
        self.assertAlmostEqual(metrics.acc, 14 / 20)
        self.assertAlmostEqual(metrics.pd, 7 / 10)
        self.assertAlmostEqual(metrics.fa, 4 / (4 * 346 * 260))

    def test_invalid_thresholds_fail_early(self):
        with self.assertRaises(ValueError):
            select_density_threshold(1, 0, 0.0, 0.9)
        with self.assertRaises(ValueError):
            select_density_threshold(1, 0, 0.7, 1.0)

    def test_disabled_policy_preserves_the_static_threshold(self):
        policy = DensityAdaptiveThresholdConfig(enabled=False)
        self.assertEqual(policy.threshold_for_event_count(500000, 0.9), 0.9)

    def test_enabled_policy_selects_density_threshold(self):
        policy = DensityAdaptiveThresholdConfig(
            enabled=True,
            event_count_cutoff=100000,
            low_density_threshold=0.7,
            high_density_threshold=0.92,
        )
        self.assertEqual(policy.threshold_for_event_count(100000, 0.9), 0.7)
        self.assertEqual(policy.threshold_for_event_count(100001, 0.9), 0.92)


if __name__ == '__main__':
    unittest.main()
