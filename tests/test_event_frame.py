import unittest

import numpy as np

from dataset.event_frame import build_event_frame


class EventFrameTests(unittest.TestCase):
    def test_polarity_and_time_bins_are_separated(self):
        locations = np.array(
            [
                [1, 2, 0],
                [1, 2, 0],
                [3, 0, 7],
            ]
        )
        polarities = np.array([0, 1, 1])

        frame = build_event_frame(
            locations,
            polarities,
            width=4,
            height=3,
            temporal_bins=2,
            temporal_size=8,
        )

        self.assertEqual(frame.shape, (4, 3, 4))
        self.assertAlmostEqual(frame[0, 2, 1], np.log(2.0))
        self.assertAlmostEqual(frame[1, 2, 1], np.log(2.0))
        self.assertAlmostEqual(frame[3, 0, 3], np.log(2.0))
        self.assertEqual(frame[2].sum(), 0.0)

    def test_invalid_coordinates_do_not_write_outside_the_frame(self):
        locations = np.array([[4, 0, 1], [0, -1, 1], [0, 0, -1]])
        frame = build_event_frame(
            locations,
            np.array([1, 1, 1]),
            width=4,
            height=3,
            temporal_bins=2,
            temporal_size=8,
        )

        self.assertEqual(frame.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
