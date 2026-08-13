import inspect
import unittest

import numpy as np

from utils.contextual_track_recovery import (
    FEATURE_NAMES,
    extract_contextual_track_edge_candidates,
)
from utils.track_edge_recovery import attach_training_targets


def _video():
    events = [
        (10, 10, 101, 0.90, 0.90),
        (11, 10, 151, 0.91, 0.91),
        (12, 10, 201, 0.92, 0.92),
        (13, 10, 251, 0.93, 0.93),
        (9, 10, 51, 0.61, 0.61),
        (14, 10, 301, 0.64, 0.64),
        (14, 11, 302, 0.57, 0.57),
        (15, 10, 302, 0.20, 0.20),
    ]
    locations = np.asarray([item[:3] for item in events], dtype=np.int64)
    raw = np.asarray([item[3] for item in events], dtype=np.float32)
    baseline = np.asarray([item[4] for item in events], dtype=np.float32)
    return raw, baseline, locations


class ContextualTrackRecoveryTests(unittest.TestCase):
    def test_feature_schema_is_unique_and_finite(self):
        raw, baseline, locations = _video()
        candidates = extract_contextual_track_edge_candidates(
            raw, baseline, locations, len(raw)
        )
        self.assertEqual(len(FEATURE_NAMES), 121)
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))
        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertEqual(candidate.features.shape, (len(FEATURE_NAMES),))
            self.assertTrue(np.isfinite(candidate.features).all())

    def test_extractor_has_no_supervision_or_identity_argument(self):
        parameters = inspect.signature(
            extract_contextual_track_edge_candidates
        ).parameters
        forbidden = {"labels", "target_ids", "source_name", "path", "fold"}
        self.assertFalse(forbidden & set(parameters))

    def test_features_are_deterministic_and_supervision_cannot_mutate_them(self):
        raw, baseline, locations = _video()
        first = extract_contextual_track_edge_candidates(
            raw, baseline, locations, len(raw)
        )
        second = extract_contextual_track_edge_candidates(
            raw, baseline, locations, len(raw)
        )
        before = [candidate.features.copy() for candidate in first]
        labels = np.asarray([1, 1, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
        target_ids = np.asarray([7, 7, 7, 7, 7, 0, 0, 0], dtype=np.int64)
        attach_training_targets(first, labels, target_ids, baseline, locations)
        for expected, left, right in zip(before, first, second):
            np.testing.assert_array_equal(expected, left.features)
            np.testing.assert_array_equal(left.features, right.features)

    def test_relative_features_ignore_spatial_and_temporal_origin(self):
        raw, baseline, locations = _video()
        shifted = locations + np.asarray([40, 25, 500], dtype=np.int64)
        first = extract_contextual_track_edge_candidates(
            raw, baseline, locations, len(raw)
        )
        second = extract_contextual_track_edge_candidates(
            raw, baseline, shifted, len(raw)
        )
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            np.testing.assert_allclose(left.features, right.features, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

