import unittest

import numpy as np

import run_confidence_score_blend_grouped_oof as blend


class ConfidenceBlendGroupedOOFTests(unittest.TestCase):
    def test_alpha_zero_is_bitwise_identity_and_positive_alpha_attenuates(self):
        baseline = np.asarray([0.1, 0.719, 0.9, 1.0], dtype=np.float32)
        confidence = np.asarray([0.05, 0.7, 0.45, 0.99], dtype=np.float32)
        identity = blend.blend_scores(baseline, confidence, 0.0)
        candidate = blend.blend_scores(baseline, confidence, 0.08)
        self.assertTrue(np.array_equal(identity, baseline))
        self.assertTrue(np.all(candidate <= baseline))
        self.assertTrue(np.any(candidate < baseline))

    def test_fold_plan_is_disjoint_and_covers_each_source_once(self):
        held = []
        for fold in blend.FOLD_PLAN:
            fit = set(fold["fit_names"])
            hold = set(fold["held_names"])
            self.assertFalse(fit & hold)
            domain_names = (
                set(blend.H1_NAMES) if fold["domain"] == "h1" else set(blend.H2_NAMES)
            )
            self.assertEqual(fit | hold, domain_names)
            held.extend(fold["held_names"])
        self.assertEqual(tuple(held), blend.SOURCE_NAMES)
        self.assertEqual(len(set(held)), len(blend.SOURCE_NAMES))

    def test_fit_gate_requires_real_false_alarm_evidence_and_no_tp_loss(self):
        neutral = {
            "metrics": {"score": 1e-6, "pd": 0.0, "iou": 0.0, "fa": 0.0},
            "counts": {key: 0 for key in blend.COUNT_KEYS},
        }
        passed, checks = blend.fit_gate(neutral)
        self.assertFalse(passed)
        self.assertFalse(checks["false_alarm_evidence"])
        useful = {
            "metrics": dict(neutral["metrics"]),
            "counts": dict(neutral["counts"]),
        }
        useful["counts"]["false_positive_events"] = -1
        passed, _ = blend.fit_gate(useful)
        self.assertTrue(passed)
        useful["counts"]["true_positive_events"] = -1
        passed, _ = blend.fit_gate(useful)
        self.assertFalse(passed)

    def test_select_alpha_uses_highest_score_then_smallest_alpha(self):
        results = {}
        for alpha in blend.ALPHAS:
            delta = {
                "metrics": {"score": 0.0, "pd": 0.0, "iou": 0.0, "fa": 0.0},
                "counts": {key: 0 for key in blend.COUNT_KEYS},
            }
            results[alpha] = {"delta": delta}
        for alpha in (0.001, 0.0025):
            results[alpha]["delta"]["metrics"]["score"] = 0.01
            results[alpha]["delta"]["counts"]["false_components"] = -1
        self.assertEqual(blend.select_alpha(results), 0.001)

    def test_forbidden_split_component_is_rejected(self):
        with self.assertRaises(ValueError):
            blend._reject_forbidden_path("C:/science/validation/cache", "cache")


if __name__ == "__main__":
    unittest.main()
