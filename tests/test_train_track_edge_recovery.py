import unittest
from pathlib import Path
import tempfile

import numpy as np

from crossfit_component_reranker import SufficientCounts, metrics_from_counts
from train_track_edge_recovery import (
    FROZEN_TOPOLOGY,
    PD_DETECTION_INTERVAL,
    TRAIN_STEPS,
    _safe_new_output,
    evaluate_gates,
    fit_metric_weighted_mlp,
    marginal_score_utility,
    predict_mlp_logits,
)
from utils.track_edge_recovery import TrackEdgeTrainingTarget


class TrackEdgeTrainingTests(unittest.TestCase):
    def test_candidate_bins_match_official_pd_interval(self):
        self.assertEqual(
            FROZEN_TOPOLOGY.temporal_bin_size,
            PD_DETECTION_INTERVAL,
        )

    def test_metric_utility_has_correct_sign_and_pd_weight(self):
        baseline = SufficientCounts(
            true_positive_events=100,
            false_positive_events=10,
            false_negative_events=20,
            correct_objects=8,
            object_count=10,
            false_components=5,
            frame_count=100,
            event_count=130,
        )
        hit_positive = TrackEdgeTrainingTarget(1, False, 0, 1)
        missed_positive = TrackEdgeTrainingTarget(1, True, 0, 1)
        false_event = TrackEdgeTrainingTarget(0, False, 1, 1)
        hit_utility = marginal_score_utility(baseline, hit_positive)
        missed_utility = marginal_score_utility(baseline, missed_positive)
        false_utility = marginal_score_utility(baseline, false_event)
        self.assertGreater(hit_utility, 0.0)
        self.assertGreater(missed_utility, hit_utility)
        self.assertLess(false_utility, 0.0)

    def test_false_component_merge_never_becomes_reward(self):
        baseline = SufficientCounts(
            true_positive_events=100,
            false_positive_events=10,
            false_negative_events=20,
            correct_objects=8,
            object_count=10,
            false_components=5,
            frame_count=100,
            event_count=130,
        )
        merge = TrackEdgeTrainingTarget(0, False, -2, 1)
        self.assertLess(marginal_score_utility(baseline, merge), 0.0)

    def test_real_adamw_training_is_deterministic_and_auditable(self):
        rng = np.random.default_rng(53)
        features = rng.normal(size=(40, 15))
        labels = (features[:, 0] + 0.5 * features[:, 1] > 0.0).astype(np.float64)
        base_weights = np.full(40, 1.0 / 40.0, dtype=np.float64)
        utilities = np.where(labels > 0.5, 0.00008, -0.00001)
        first = fit_metric_weighted_mlp(
            features, labels, base_weights, utilities
        )
        second = fit_metric_weighted_mlp(
            features, labels, base_weights, utilities
        )
        evidence = first["training_evidence"]
        self.assertEqual(evidence["optimizer_steps"], TRAIN_STEPS)
        self.assertEqual(evidence["parameter_count"], 137)
        self.assertEqual(evidence["optimizer_moment_tensor_count"], 8)
        self.assertGreater(evidence["optimizer_moment_l2"], 0.0)
        self.assertGreater(evidence["parameter_delta_l2"], 0.0)
        self.assertLess(evidence["final_loss"], evidence["initial_loss"])
        self.assertEqual(first["model_state"], second["model_state"])
        logits = predict_mlp_logits(features, first)
        self.assertEqual(logits.shape, (40,))
        self.assertTrue(np.isfinite(logits).all())

    def test_fit_rejects_single_class_pseudo_training(self):
        features = np.zeros((4, 15), dtype=np.float64)
        labels = np.ones(4, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            fit_metric_weighted_mlp(
                features,
                labels,
                np.full(4, 0.25),
                np.full(4, 0.00008),
            )

    def test_audit_outputs_cannot_mutate_cache_or_alias_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                _safe_new_output(cache / "report.json", cache, "report")
            protocol = root / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "alias"):
                _safe_new_output(
                    protocol,
                    cache,
                    "report",
                    forbidden=(protocol,),
                )
            allowed = _safe_new_output(root / "report.json", cache, "report")
            self.assertEqual(allowed, (root / "report.json").resolve())

    @staticmethod
    def _fold(block):
        baseline_counts = SufficientCounts(
            true_positive_events=100,
            false_positive_events=10,
            false_negative_events=20,
            correct_objects=8,
            object_count=10,
            false_components=5,
            frame_count=100,
            event_count=130,
        )
        recovered_counts = SufficientCounts(
            true_positive_events=102,
            false_positive_events=10,
            false_negative_events=18,
            correct_objects=9,
            object_count=10,
            false_components=6,
            frame_count=100,
            event_count=130,
        )
        baseline_metrics = metrics_from_counts(baseline_counts)
        recovered_metrics = metrics_from_counts(recovered_counts)
        return {
            "held_block": block,
            "baseline": {
                "counts": baseline_counts.to_dict(),
                "metrics": baseline_metrics,
            },
            "recovered": {
                "counts": recovered_counts.to_dict(),
                "metrics": recovered_metrics,
                "delta": {
                    name: recovered_metrics[name] - baseline_metrics[name]
                    for name in recovered_metrics
                },
            },
            "new_pd_groups": 1,
            "false_component_delta": 1,
            "false_components_per_new_pd_group": 1.0,
            "positive_candidate_videos": 2,
            "model": {
                "training_evidence": {
                    "optimizer_steps": 200,
                    "parameter_delta_l2": 1.0,
                    "initial_loss": 1.0,
                    "final_loss": 0.5,
                    "optimizer_moment_l2": 1.0,
                }
            },
        }

    def test_cross_source_gates_require_both_blocks(self):
        middle = SufficientCounts(
            true_positive_events=1000,
            false_positive_events=20,
            false_negative_events=30,
            correct_objects=80,
            object_count=90,
            false_components=10,
            frame_count=500,
            event_count=1050,
        )
        result = evaluate_gates([self._fold("h1"), self._fold("h2")], middle)
        self.assertTrue(result["passed"])
        broken = self._fold("h2")
        broken["new_pd_groups"] = 0
        broken["false_components_per_new_pd_group"] = None
        result = evaluate_gates([self._fold("h1"), broken], middle)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
