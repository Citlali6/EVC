import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.temporal_memory_input_router import (
    EXPECTED_TEMPORAL_BIN_COUNT,
    HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE,
    LOW_DENSITY_MAX_EVENT_COUNT,
    assert_full_window_identity,
    polarity_minority_fraction,
    predict_temporal_memory_scores_input_routed,
    require_persistence_second_stage_disabled,
    route_policy_definition,
    route_policy_sha256,
    select_temporal_memory_input_route,
)


def _polarities(event_count, positive_count):
    values = np.zeros(event_count, dtype=np.float32)
    values[-positive_count:] = 1.0
    return values


class _RouteOnlyVideo:
    def __init__(self, polarities):
        self.polarities = polarities
        self.event_indices_by_bin = tuple(
            np.empty(0, dtype=np.int64) for _ in range(EXPECTED_TEMPORAL_BIN_COUNT)
        )

    def __getattribute__(self, name):
        if name in {"name", "labels", "target_ids"}:
            raise AssertionError("The runtime route accessed forbidden {}.".format(name))
        return object.__getattribute__(self, name)


class TemporalInputRouteSelectionTests(unittest.TestCase):
    def test_released_low_density_m10_gate_precedes_polarity(self):
        decision = select_temporal_memory_input_route(
            _polarities(LOW_DENSITY_MAX_EVENT_COUNT, 15_000),
            EXPECTED_TEMPORAL_BIN_COUNT,
        )

        self.assertEqual(decision.domain, "low")
        self.assertEqual(decision.checkpoint_role, "m10")
        self.assertEqual(decision.mode, "full_stream")
        self.assertEqual(decision.prediction_threshold, 0.718)

    def test_unassessed_middle_density_never_routes_to_t32(self):
        decision = select_temporal_memory_input_route(
            _polarities(HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE, 100_000),
            EXPECTED_TEMPORAL_BIN_COUNT,
        )

        self.assertEqual(decision.domain, "middle")
        self.assertEqual(decision.checkpoint_role, "m20")
        self.assertEqual(decision.mode, "full_stream")
        self.assertEqual(decision.prediction_threshold, 0.719)

    def test_high_density_fraction_below_cutoff_is_h1_full(self):
        decision = select_temporal_memory_input_route(
            _polarities(200_001, 40_000),
            EXPECTED_TEMPORAL_BIN_COUNT,
        )

        self.assertLess(decision.polarity_minority_fraction, 0.20)
        self.assertEqual((decision.domain, decision.mode), ("h1", "full_stream"))
        self.assertIsNone(decision.window_length)

    def test_high_density_fraction_at_or_above_cutoff_is_h2_t32(self):
        decision = select_temporal_memory_input_route(
            _polarities(200_001, 40_001),
            EXPECTED_TEMPORAL_BIN_COUNT,
        )

        self.assertGreaterEqual(decision.polarity_minority_fraction, 0.20)
        self.assertEqual((decision.domain, decision.mode), ("h2", "window_t32"))
        self.assertEqual((decision.window_length, decision.stride), (32, 16))

    def test_complete_vector_includes_the_last_event(self):
        h1 = polarity_minority_fraction(np.zeros(5, dtype=np.float32))
        h2_boundary = polarity_minority_fraction(
            np.asarray([0, 0, 0, 0, 1], dtype=np.float32)
        )

        self.assertEqual(h1, 0.0)
        self.assertEqual(h2_boundary, 0.2)

    def test_invalid_input_and_temporal_length_fail_closed(self):
        invalid_polarities = (
            np.empty((2, 2), dtype=np.float32),
            np.asarray([], dtype=np.float32),
            np.asarray([0.0, np.nan], dtype=np.float32),
            np.asarray([-1.0, 1.0], dtype=np.float32),
        )
        for values in invalid_polarities:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                polarity_minority_fraction(values)
        with self.assertRaisesRegex(ValueError, "exactly 160"):
            select_temporal_memory_input_route(np.zeros(10), 159)

    def test_policy_digest_binds_label_name_and_persistence_guardrails(self):
        policy = route_policy_definition()

        self.assertEqual(len(route_policy_sha256()), 64)
        self.assertFalse(policy["labels_used_for_route"])
        self.assertFalse(policy["source_name_used_for_route"])
        self.assertFalse(policy["persistent_pixel_second_stage"]["enabled"])
        self.assertEqual(policy["high_density_min_event_count_exclusive"], 200_000)


class TemporalInputRoutePredictionTests(unittest.TestCase):
    def _call(self, polarities):
        video = _RouteOnlyVideo(polarities)
        return predict_temporal_memory_scores_input_routed(
            m10_model="m10",
            m20_model="m20",
            video=video,
            device=torch.device("cpu"),
            context_bins=5,
            width=346,
            height=260,
            inference_batch_size=8,
            log_count_clip=4.0,
        )

    @patch("utils.temporal_memory_input_router.predict_temporal_memory_scores_windowed")
    @patch("utils.temporal_memory_input_router.predict_temporal_memory_scores")
    def test_low_route_uses_only_m10_and_never_reads_name_or_labels(
        self, full_predictor, window_predictor
    ):
        full_predictor.return_value = torch.zeros(30_000)

        scores, decision = self._call(_polarities(30_000, 15_000))

        self.assertEqual(scores.numel(), 30_000)
        self.assertEqual(decision.checkpoint_role, "m10")
        self.assertEqual(full_predictor.call_args.kwargs["model"], "m10")
        window_predictor.assert_not_called()

    @patch("utils.temporal_memory_input_router.predict_temporal_memory_scores_windowed")
    @patch("utils.temporal_memory_input_router.predict_temporal_memory_scores")
    def test_middle_route_uses_only_m20_full(self, full_predictor, window_predictor):
        full_predictor.return_value = torch.zeros(30_001)

        _, decision = self._call(_polarities(30_001, 15_000))

        self.assertEqual(decision.domain, "middle")
        self.assertEqual(full_predictor.call_args.kwargs["model"], "m20")
        window_predictor.assert_not_called()

    @patch("utils.temporal_memory_input_router.predict_temporal_memory_scores_windowed")
    @patch("utils.temporal_memory_input_router.predict_temporal_memory_scores")
    def test_h2_route_uses_m20_t32_stride16(self, full_predictor, window_predictor):
        polarities = _polarities(200_001, 40_001)
        window_predictor.return_value = torch.zeros(len(polarities))

        _, decision = self._call(polarities)

        self.assertEqual(decision.domain, "h2")
        self.assertEqual(window_predictor.call_args.kwargs["model"], "m20")
        self.assertEqual(window_predictor.call_args.kwargs["window_length"], 32)
        self.assertEqual(window_predictor.call_args.kwargs["stride"], 16)
        full_predictor.assert_not_called()

    @patch("utils.temporal_memory_input_router.predict_temporal_memory_scores_windowed")
    @patch("utils.temporal_memory_input_router.predict_temporal_memory_scores")
    def test_l_full_identity_is_bitwise_and_fails_closed(
        self, full_predictor, window_predictor
    ):
        video = _RouteOnlyVideo(np.zeros(7, dtype=np.float32))
        full_predictor.return_value = torch.tensor([0.1] * 7, dtype=torch.float32)
        window_predictor.return_value = full_predictor.return_value.clone()

        result = assert_full_window_identity(
            model="m20",
            video=video,
            device=torch.device("cpu"),
            context_bins=5,
            width=346,
            height=260,
            inference_batch_size=8,
        )

        self.assertTrue(result["bitwise_equal"])
        self.assertEqual(window_predictor.call_args.kwargs["window_length"], 160)
        window_predictor.return_value = torch.tensor(
            [0.2] + [0.1] * 6, dtype=torch.float32
        )
        with self.assertRaisesRegex(RuntimeError, "identity failed"):
            assert_full_window_identity(
                model="m20",
                video=video,
                device=torch.device("cpu"),
                context_bins=5,
                width=346,
                height=260,
                inference_batch_size=8,
            )

    def test_persistence_stage_is_explicitly_off(self):
        require_persistence_second_stage_disabled(False)
        with self.assertRaisesRegex(RuntimeError, "interaction"):
            require_persistence_second_stage_disabled(True)


if __name__ == "__main__":
    unittest.main()
