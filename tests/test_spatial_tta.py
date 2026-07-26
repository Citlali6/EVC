import unittest
from types import SimpleNamespace

import numpy as np

from utils.spatial_tta import (
    HorizontalFlipTTAConfig,
    horizontal_flip_event_inputs,
    horizontal_flip_sample,
    padded_feature_width,
)


class SpatialTtaTests(unittest.TestCase):
    def test_padded_feature_width(self):
        self.assertEqual(padded_feature_width(346), 352)
        self.assertEqual(padded_feature_width(352), 352)
        self.assertEqual(padded_feature_width(353), 384)

    def test_horizontal_flip_tta_config_reads_explicit_options(self):
        config = HorizontalFlipTTAConfig.from_cfg(SimpleNamespace(
            p14_horizontal_flip_enabled=True,
            p14_horizontal_flip_original_weight=0.5,
        ))

        self.assertTrue(config.enabled)
        self.assertEqual(config.original_weight, 0.5)
        self.assertIn('flipped_weight=0.500', config.describe())

    def test_horizontal_flip_tta_config_rejects_invalid_weight(self):
        with self.assertRaises(ValueError):
            HorizontalFlipTTAConfig(enabled=True, original_weight=1.1)

    def test_horizontal_flip_preserves_order_and_updates_normalized_x(self):
        sample = {
            'ev_loc': np.array([[0, 2, 3], [2, 4, 5], [5, 6, 7]], dtype=np.int64),
            'evs_norm': np.array([[0.0, 0.2, 0.3, 1.0], [0.25, 0.4, 0.5, 0.0], [0.625, 0.6, 0.7, 1.0]], dtype=np.float32),
            'seg_label': np.array([0.0, 1.0, 0.0]),
            'idx': np.array([1, 2, 3]),
        }

        flipped = horizontal_flip_sample(sample, image_width=6, feature_width=8)
        restored = horizontal_flip_sample(flipped, image_width=6, feature_width=8)

        np.testing.assert_array_equal(flipped['ev_loc'][:, 0], np.array([5, 3, 0]))
        np.testing.assert_allclose(flipped['evs_norm'][:, 0], np.array([5 / 8, 3 / 8, 0.0]))
        np.testing.assert_array_equal(flipped['idx'], sample['idx'])
        np.testing.assert_array_equal(restored['ev_loc'], sample['ev_loc'])
        np.testing.assert_allclose(restored['evs_norm'], sample['evs_norm'])

    def test_event_input_flip_preserves_non_coordinate_features(self):
        locations = np.array([[0, 2, 3], [5, 4, 5]], dtype=np.int64)
        features = np.array([
            [0.0, 0.2, 0.3, 1.0, -0.4],
            [0.625, 0.4, 0.5, 0.0, 0.8],
        ], dtype=np.float32)

        flipped_locations, flipped_features = horizontal_flip_event_inputs(
            locations,
            features,
            image_width=6,
            feature_width=8,
        )

        np.testing.assert_array_equal(flipped_locations[:, 0], np.array([5, 0]))
        np.testing.assert_allclose(flipped_features[:, 0], np.array([5 / 8, 0.0]))
        np.testing.assert_allclose(flipped_features[:, 1:], features[:, 1:])

    def test_horizontal_flip_rejects_invalid_coordinates(self):
        sample = {
            'ev_loc': np.array([[6, 0, 0]], dtype=np.int64),
            'evs_norm': np.zeros((1, 4), dtype=np.float32),
            'seg_label': np.zeros(1),
            'idx': np.zeros(1),
        }
        with self.assertRaises(ValueError):
            horizontal_flip_sample(sample, image_width=6)


if __name__ == '__main__':
    unittest.main()
