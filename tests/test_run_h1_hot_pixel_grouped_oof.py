import json
from pathlib import Path
import unittest

import numpy as np

import run_h1_hot_pixel_grouped_oof as runner
from crossfit_persistent_pixel_prior import PixelPrior


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "h1_hot_pixel_grouped_oof_science.json"
)


def _prior():
    zeros = np.zeros(runner.PIXEL_COUNT, dtype=np.float64)
    return PixelPrior(
        event_pixel_ids=np.asarray([0], dtype=np.int64),
        log_events=zeros.copy(),
        active_fraction=zeros.copy(),
        longest_run_fraction=zeros.copy(),
        collision_fraction=zeros.copy(),
        log_max_bin_events=zeros.copy(),
        polarity_dominance=zeros.copy(),
        neighbor_active_fraction=zeros.copy(),
        summary={},
    )


class H1HotPixelProtocolTests(unittest.TestCase):
    def test_frozen_protocol_validates(self):
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

        runner.validate_protocol(payload)

        self.assertEqual(len(payload["candidates"]), 15)

    def test_unknown_candidate_rule_fails_closed(self):
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        payload["candidates"][0]["hot_rule"] = "unknown"

        with self.assertRaisesRegex(ValueError, "unknown hot rule"):
            runner.validate_protocol(payload)

    def test_hot_rules_have_frozen_boundary_semantics(self):
        prior = _prior()
        prior.active_fraction[1] = 1.0
        prior.longest_run_fraction[1] = 1.0
        prior.active_fraction[2] = 0.90
        prior.longest_run_fraction[2] = 0.50
        prior.polarity_dominance[2] = 0.90
        prior.log_max_bin_events[3] = np.log1p(54.0)
        prior.collision_fraction[3] = 0.50
        prior.polarity_dominance[3] = 0.90

        self.assertTrue(runner.hot_mask(prior, "full_life")[1])
        self.assertTrue(runner.hot_mask(prior, "persistent_polar")[2])
        self.assertTrue(runner.hot_mask(prior, "saturated")[3])

    def test_component_gate_is_inclusive_and_single_coordinate_only(self):
        candidate = {
            "max_component_events": 3,
            "max_track_bins": 2,
            "max_score": 0.85,
        }
        descriptor = {
            "pixels": np.asarray([7], dtype=np.int64),
            "event_count": 3,
            "track_bins": 2,
            "score_max": 0.85,
        }

        self.assertTrue(runner.component_passes_gates(descriptor, candidate))
        descriptor["pixels"] = np.asarray([7, 8], dtype=np.int64)
        self.assertFalse(runner.component_passes_gates(descriptor, candidate))

    def test_exact_identity_cannot_pass_clear_pooled_gain_gate(self):
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        candidate_id = protocol["candidates"][0]["candidate_id"]
        metric_delta = {
            key: 0.0 for key in ("iou", "acc", "pd", "fa", "score_fa", "score")
        }
        count_delta = {
            key: 0
            for key in (
                "true_positive_events",
                "false_positive_events",
                "false_negative_events",
                "correct_objects",
                "object_count",
                "false_components",
                "frame_count",
                "event_count",
            )
        }
        result = {
            "candidate_id": candidate_id,
            "metric_delta": metric_delta,
            "count_delta": count_delta,
        }
        folds = [
            {
                "fold_id": fold_id,
                "candidate_results": [result],
            }
            for fold_id in ("holdout_044_045", "holdout_046_047")
        ]

        checks = runner.promotion_checks(
            result, folds, protocol["promotion_gates"]
        )

        self.assertTrue(all(value["passed"] for value in checks["folds"]))
        self.assertFalse(checks["pooled"]["score_clearly_positive"])
        self.assertFalse(checks["passed"])


if __name__ == "__main__":
    unittest.main()
