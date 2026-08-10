import unittest

import numpy as np

from utils.track_edge_recovery import (
    FEATURE_NAMES,
    FROZEN_TOPOLOGY,
    TrackEdgeCandidate,
    TrackEdgeTopology,
    attach_training_targets,
    extract_track_edge_candidates,
    select_endpoint_recoveries,
)


def _video(extra_events=()):
    # Four stable seed bins moving one pixel to the right per bin, plus a weak
    # extension at both ends.  Timestamps avoid official open-interval edges.
    events = [
        (10, 10, 101, 0.90, 0.90),
        (11, 10, 151, 0.91, 0.91),
        (12, 10, 201, 0.92, 0.92),
        (13, 10, 251, 0.93, 0.93),
        (9, 10, 51, 0.61, 0.61),
        (14, 10, 301, 0.64, 0.64),
    ]
    events.extend(extra_events)
    locations = np.asarray([event[:3] for event in events], dtype=np.int64)
    raw = np.asarray([event[3] for event in events], dtype=np.float32)
    baseline = np.asarray([event[4] for event in events], dtype=np.float32)
    return raw, baseline, locations


class TrackEdgeRecoveryTests(unittest.TestCase):
    def test_frozen_feature_width_is_fifteen(self):
        self.assertEqual(len(FEATURE_NAMES), 15)

    def test_extracts_only_adjacent_track_ends(self):
        raw, baseline, locations = _video(
            extra_events=(
                (40, 40, 301, 0.70, 0.70),  # too far from the end
                (12, 10, 351, 0.70, 0.70),  # two bins after the end
            )
        )
        candidates = extract_track_edge_candidates(
            raw, baseline, locations, len(raw)
        )
        self.assertEqual([candidate.event_index for candidate in candidates], [4, 5])
        self.assertEqual(
            [candidate.endpoint_side for candidate in candidates], [-1, 1]
        )
        for candidate in candidates:
            self.assertEqual(candidate.features.shape, (15,))
            self.assertTrue(np.isfinite(candidate.features).all())

    def test_four_seed_bins_are_required(self):
        raw, baseline, locations = _video()
        baseline[3] = 0.70
        candidates = extract_track_edge_candidates(
            raw, baseline, locations, len(raw)
        )
        self.assertEqual(candidates, ())

    def test_candidate_features_do_not_accept_or_depend_on_labels(self):
        raw, baseline, locations = _video()
        first = extract_track_edge_candidates(raw, baseline, locations, len(raw))
        second = extract_track_edge_candidates(raw, baseline, locations, len(raw))
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.features, right.features)
            self.assertEqual(left.event_index, right.event_index)

        labels_a = np.ones(len(raw), dtype=np.uint8)
        labels_b = np.zeros(len(raw), dtype=np.uint8)
        target_ids = np.arange(1, len(raw) + 1, dtype=np.int64)
        targets_a = attach_training_targets(
            first, labels_a, target_ids, baseline, locations
        )
        targets_b = attach_training_targets(
            first, labels_b, np.zeros(len(raw), dtype=np.int64), baseline, locations
        )
        self.assertNotEqual(
            [target.label for target in targets_a],
            [target.label for target in targets_b],
        )
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.features, right.features)

    def test_train_targets_record_pd_miss_and_false_component_delta(self):
        raw, baseline, locations = _video()
        candidates = extract_track_edge_candidates(raw, baseline, locations, len(raw))
        labels = np.asarray([1, 1, 1, 1, 1, 0], dtype=np.uint8)
        target_ids = np.asarray([7, 7, 7, 7, 7, 0], dtype=np.int64)
        targets = attach_training_targets(
            candidates, labels, target_ids, baseline, locations
        )
        self.assertEqual(targets[0].label, 1)
        self.assertTrue(targets[0].recovers_target_group)
        self.assertEqual(targets[0].false_component_delta, 0)
        self.assertEqual(targets[1].label, 0)
        self.assertFalse(targets[1].recovers_target_group)
        self.assertEqual(targets[1].false_component_delta, 1)

    def test_training_identity_mismatch_is_rejected_after_feature_extraction(self):
        raw, baseline, locations = _video()
        candidates = extract_track_edge_candidates(raw, baseline, locations, len(raw))
        labels = np.asarray([1, 1, 1, 1, 1, 0], dtype=np.uint8)
        mismatched_target_ids = np.asarray([7, 7, 7, 7, 0, 0], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "same events"):
            attach_training_targets(
                candidates,
                labels,
                mismatched_target_ids,
                baseline,
                locations,
            )

    @staticmethod
    def _manual_candidate(event_index=0):
        return TrackEdgeCandidate(
            event_index=event_index,
            component_event_indices=np.asarray([event_index], dtype=np.int64),
            endpoint_key=(0, 1),
            temporal_bin=1,
            endpoint_side=1,
            features=np.zeros(15, dtype=np.float64),
            raw_score=0.6,
            motion_residual=0.0,
        )

    def test_one_event_does_not_fake_pd_when_fraction_remains_too_small(self):
        locations = np.asarray(
            [(10, 10, 51), (11, 10, 52), (12, 10, 53)], dtype=np.int64
        )
        targets = attach_training_targets(
            (self._manual_candidate(),),
            np.ones(3, dtype=np.uint8),
            np.full(3, 7, dtype=np.int64),
            np.full(3, 0.2, dtype=np.float32),
            locations,
            correct_threshold=0.5,
        )
        self.assertFalse(targets[0].recovers_target_group)

    def test_existing_correct_events_plus_candidate_can_cross_pd_fraction(self):
        locations = np.asarray(
            [(10, 10, 51), (11, 10, 52), (12, 10, 53), (13, 10, 54)],
            dtype=np.int64,
        )
        baseline = np.asarray([0.2, 0.9, 0.2, 0.2], dtype=np.float32)
        targets = attach_training_targets(
            (self._manual_candidate(),),
            np.ones(4, dtype=np.uint8),
            np.full(4, 7, dtype=np.int64),
            baseline,
            locations,
            correct_threshold=0.5,
        )
        self.assertTrue(targets[0].recovers_target_group)

    def test_official_boundary_event_has_no_pd_or_fa_group(self):
        raw, baseline, locations = _video()
        locations[4, 2] = 50
        candidates = extract_track_edge_candidates(raw, baseline, locations, len(raw))
        labels = np.asarray([1, 1, 1, 1, 1, 0], dtype=np.uint8)
        target_ids = np.asarray([7, 7, 7, 7, 7, 0], dtype=np.int64)
        targets = attach_training_targets(
            candidates, labels, target_ids, baseline, locations
        )
        start_target = next(
            target
            for candidate, target in zip(candidates, targets)
            if candidate.endpoint_side == -1
        )
        self.assertIsNone(start_target.official_frame_index)
        self.assertFalse(start_target.recovers_target_group)

    def test_false_cell_merge_delta_is_exact(self):
        raw, baseline, locations = _video(
            extra_events=(
                (13, 9, 301, 0.90, 0.90),
                (15, 11, 301, 0.90, 0.90),
            )
        )
        candidates = (
            TrackEdgeCandidate(
                event_index=5,
                component_event_indices=np.asarray([5], dtype=np.int64),
                endpoint_key=(0, 1),
                temporal_bin=6,
                endpoint_side=1,
                features=np.zeros(15, dtype=np.float64),
                raw_score=float(raw[5]),
                motion_residual=0.0,
            ),
        )
        labels = np.asarray([1, 1, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
        target_ids = np.asarray([7, 7, 7, 7, 7, 0, 0, 0], dtype=np.int64)
        targets = attach_training_targets(
            candidates, labels, target_ids, baseline, locations
        )
        end_target = targets[0]
        # The recovered cell at (14,10) bridges the two diagonal false cells.
        self.assertEqual(end_target.false_component_delta, -1)

    def test_false_cell_delta_matches_official_uint8_wrap(self):
        event_count = 256
        locations = np.tile(np.asarray([[14, 10, 301]], dtype=np.int64), (event_count, 1))
        baseline = np.full(event_count, 0.90, dtype=np.float32)
        baseline[-1] = 0.60
        targets = attach_training_targets(
            (self._manual_candidate(event_count - 1),),
            np.zeros(event_count, dtype=np.uint8),
            np.zeros(event_count, dtype=np.int64),
            baseline,
            locations,
        )
        # The 256th false event wraps the official uint8 cell from 255 to 0,
        # removing its one connected component.
        self.assertEqual(targets[0].false_component_delta, -1)

    def test_training_target_rejects_already_positive_candidate(self):
        locations = np.asarray([(10, 10, 51)], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "baseline-negative"):
            attach_training_targets(
                (self._manual_candidate(),),
                np.ones(1, dtype=np.uint8),
                np.ones(1, dtype=np.int64),
                np.asarray([0.90], dtype=np.float32),
                locations,
            )

    def test_selects_at_most_one_event_per_endpoint(self):
        raw, baseline, locations = _video(
            extra_events=(
                (6, 10, 301, 0.63, 0.63),
                (20, 10, 301, 0.62, 0.62),
            )
        )
        candidates = extract_track_edge_candidates(raw, baseline, locations, len(raw))
        end_positions = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.endpoint_side == 1
        ]
        self.assertGreaterEqual(len(end_positions), 2)
        logits = np.full(len(candidates), -1.0, dtype=np.float64)
        logits[end_positions[0]] = 0.2
        logits[end_positions[1]] = 0.9
        selected = select_endpoint_recoveries(candidates, logits)
        self.assertEqual(selected.size, 1)
        self.assertEqual(selected[0], candidates[end_positions[1]].event_index)

    def test_topology_rejects_non_adjacent_gap(self):
        with self.assertRaisesRegex(ValueError, "adjacent"):
            TrackEdgeTopology(max_gap_bins=2)


if __name__ == "__main__":
    unittest.main()
