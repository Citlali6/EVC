import unittest

import numpy as np

from dataset.event_features import (
    build_local_activity_feature,
    build_local_spatiotemporal_density_feature,
)


class LocalActivityFeatureTests(unittest.TestCase):
    def test_same_pixel_temporal_neighborhood_controls_activity(self):
        locations = np.array([
            [0, 0, 10],
            [0, 0, 20],
            [0, 0, 30],
            [1, 0, 20],
        ])

        feature = build_local_activity_feature(
            locations,
            width=2,
            height=1,
            temporal_size=64,
            temporal_radius=10,
        )

        self.assertEqual(feature.shape, (4,))
        self.assertAlmostEqual(float(feature.mean()), 0.0, places=6)
        self.assertGreater(feature[1], feature[0])
        self.assertAlmostEqual(float(feature[0]), float(feature[2]), places=6)
        self.assertGreater(feature[0], feature[3])

    def test_invalid_locations_leave_zero_features(self):
        locations = np.array([[3, 0, 1], [0, -1, 2], [0, 0, 8]])

        feature = build_local_activity_feature(
            locations,
            width=3,
            height=2,
            temporal_size=8,
            temporal_radius=2,
        )

        np.testing.assert_array_equal(feature, np.zeros(3, dtype=np.float32))

    def test_negative_radius_is_rejected(self):
        with self.assertRaises(ValueError):
            build_local_activity_feature(
                np.array([[0, 0, 0]]),
                width=1,
                height=1,
                temporal_size=1,
                temporal_radius=-1,
            )

    def test_local_density_counts_neighboring_cells(self):
        locations = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [7, 7, 7],
        ])

        feature = build_local_spatiotemporal_density_feature(
            locations,
            width=8,
            height=8,
            temporal_size=8,
            spatial_cell_size=1,
            temporal_cell_size=1,
            neighborhood_radius=1,
        )

        self.assertEqual(feature.shape, (4,))
        self.assertAlmostEqual(float(feature.mean()), 0.0, places=6)
        self.assertGreater(feature[1], feature[0])
        self.assertGreater(feature[1], feature[2])
        self.assertLess(feature[3], feature[0])

    def test_local_density_uses_log_standardization_and_keeps_invalid_zero(self):
        locations = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [9, 0, 0],
        ])

        feature = build_local_spatiotemporal_density_feature(
            locations,
            width=2,
            height=1,
            temporal_size=4,
            spatial_cell_size=1,
            temporal_cell_size=2,
            neighborhood_radius=0,
        )

        self.assertEqual(feature.shape, (4,))
        self.assertAlmostEqual(float(feature[:3].mean()), 0.0, places=6)
        self.assertGreater(feature[0], feature[2])
        self.assertAlmostEqual(float(feature[0]), float(feature[1]), places=6)
        self.assertEqual(float(feature[3]), 0.0)

    def test_local_density_rejects_invalid_cell_configuration(self):
        with self.assertRaises(ValueError):
            build_local_spatiotemporal_density_feature(
                np.array([[0, 0, 0]]),
                width=1,
                height=1,
                temporal_size=1,
                spatial_cell_size=0,
            )


if __name__ == '__main__':
    unittest.main()
