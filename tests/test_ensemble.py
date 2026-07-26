import unittest

import numpy as np

from utils.ensemble import EnsembleConfig, weighted_average


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


if __name__ == "__main__":
    unittest.main()
