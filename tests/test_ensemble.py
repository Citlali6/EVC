import unittest

import numpy as np

from utils.ensemble import DenseExpertConfig, EnsembleConfig, weighted_average


class EnsembleTests(unittest.TestCase):
    def test_weighted_average_preserves_event_order(self):
        primary = np.array([0.95, 0.50, 0.10])
        secondary = np.array([0.75, 0.70, 0.90])

        result = weighted_average(primary, secondary, primary_weight=0.80)

        np.testing.assert_allclose(
            result,
            np.array([0.91, 0.54, 0.26]),
            rtol=0.0,
            atol=1e-6,
        )

    def test_enabled_ensemble_requires_secondary_path(self):
        with self.assertRaises(ValueError):
            EnsembleConfig(enabled=True, secondary_model_path="")

    def test_invalid_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            EnsembleConfig(primary_weight=1.01)
        with self.assertRaises(ValueError):
            DenseExpertConfig(base_weight=-0.01)

    def test_dense_expert_uses_only_videos_above_the_cutoff(self):
        config = DenseExpertConfig(
            enabled=True,
            model_path='dense.pt',
            event_count_cutoff=100000,
            base_weight=0.8,
        )

        self.assertFalse(config.should_use(100000))
        self.assertTrue(config.should_use(100001))
        with self.assertRaises(ValueError):
            DenseExpertConfig(enabled=True, model_path='')


if __name__ == "__main__":
    unittest.main()
