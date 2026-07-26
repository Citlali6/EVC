import unittest

import numpy as np

from dataset.sampling import target_context_mask


class TargetContextSamplingTests(unittest.TestCase):
    def test_marks_neighboring_coarse_cells(self):
        locations = np.asarray([
            [1, 1, 5],
            [3, 3, 15],
            [7, 7, 35],
        ])
        labels = np.asarray([1, 0, 0])

        mask = target_context_mask(
            labels,
            locations,
            width=8,
            height=8,
            temporal_size=40,
            spatial_cell_size=2,
            temporal_cell_size=10,
            spatial_radius_cells=1,
            temporal_radius_cells=1,
        )

        np.testing.assert_array_equal(mask, np.asarray([True, True, False]))

    def test_returns_empty_context_when_no_positive_exists(self):
        locations = np.asarray([[1, 1, 5], [3, 3, 15]])
        labels = np.asarray([0, 0])

        mask = target_context_mask(
            labels,
            locations,
            width=8,
            height=8,
            temporal_size=40,
        )

        np.testing.assert_array_equal(mask, np.asarray([False, False]))

    def test_rejects_mismatched_labels_and_locations(self):
        with self.assertRaisesRegex(ValueError, 'same length'):
            target_context_mask(
                np.asarray([1]),
                np.asarray([[1, 1, 5], [2, 2, 5]]),
                width=8,
                height=8,
                temporal_size=40,
            )


if __name__ == '__main__':
    unittest.main()
