import copy
from pathlib import Path
import sys
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_metric_aux_h2_grouped_oof_v4 as public_v4

_PUBLIC_V4_STATE = {
    "protocol": public_v4.PROTOCOL_PATH,
    "output": public_v4.OUTPUT_ROOT,
    "file": Path(public_v4.__file__).resolve(),
    "core_file": Path(public_v4.core.__file__).resolve(),
}

import run_metric_aux_task_arithmetic_h2_grouped_oof as runner


class MetricAuxTaskArithmeticGroupedOofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = runner.load_protocol()
        cls.overlay, cls.overlay_sha = runner.core.load_json_snapshot(
            runner.PROTOCOL_PATH
        )

    def test_protocol_is_frozen_adaptive_alpha_one_without_search(self):
        self.assertEqual(self.protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(self.overlay_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(
            self.protocol["status"],
            "frozen_after_v4_train_only_results_before_any_task_arithmetic_checkpoint_synthesis_or_inference",
        )
        disclosure = self.protocol[
            "adaptive_selection_disclosure_task_arithmetic"
        ]
        self.assertTrue(disclosure["v4_held_train_results_observed_before_candidate_definition"])
        self.assertTrue(disclosure["new_results_must_not_be_called_independent_held_or_unbiased_oof"])
        candidate = self.protocol["task_arithmetic_candidate"]
        self.assertEqual(candidate["candidate_count"], 1)
        self.assertEqual(candidate["alpha"], 1.0)
        self.assertFalse(candidate["parameter_grid_allowed"])
        self.assertFalse(candidate["threshold_search_allowed"])
        self.assertFalse(candidate["weight_search_allowed"])
        self.assertEqual(candidate["new_training_optimizer_steps"], 0)

    def test_correct_c00_and_bound_failed_v4_report(self):
        self.assertEqual(
            self.overlay["evaluation_contract"]["effective_c00_canonical_sha256"],
            runner.EFFECTIVE_C00_SHA256,
        )
        report = self.protocol["_task_arithmetic_bound_v4_evidence"]["report"]
        self.assertFalse(report["passed"])
        self.assertEqual(report["protocol_sha256"], runner.V4_PROTOCOL_SHA256)
        self.assertEqual(report["runner_sha256"], runner.V4_RUNNER_SHA256)
        self.assertAlmostEqual(
            report["pooled"]["metric_aux"]["delta_vs_baseline"]["score"],
            0.0005062585837782851,
        )

    def test_private_v4_import_does_not_mutate_public_v4(self):
        self.assertIsNot(runner.v4, public_v4)
        self.assertIsNot(runner.core, public_v4.core)
        self.assertEqual(public_v4.PROTOCOL_PATH, _PUBLIC_V4_STATE["protocol"])
        self.assertEqual(public_v4.OUTPUT_ROOT, _PUBLIC_V4_STATE["output"])
        self.assertEqual(Path(public_v4.__file__).resolve(), _PUBLIC_V4_STATE["file"])
        self.assertEqual(
            Path(public_v4.core.__file__).resolve(), _PUBLIC_V4_STATE["core_file"]
        )

    def test_synthesis_specs_are_exactly_three_fold_specific_outputs(self):
        specs = runner.synthesis_specs(self.protocol)
        self.assertEqual([spec["fold_id"] for spec in specs], ["hold_g1", "hold_g2", "hold_g3"])
        self.assertEqual(len({spec["output"] for spec in specs}), 3)
        for spec in specs:
            self.assertIn(
                "20260810_metric_aux_task_arithmetic_h2_grouped_oof_v1",
                spec["output"],
            )
            self.assertTrue(spec["output"].endswith("isolated_metric_aux_alpha1.pt"))

    def test_alpha_zero_identity_and_alpha_one_float64_formula(self):
        import torch

        parent = {"weight": torch.tensor([1.0, -2.0], dtype=torch.float32)}
        baseline = {"weight": torch.tensor([1.25, -1.5], dtype=torch.float32)}
        aux = {"weight": torch.tensor([1.5, -1.75], dtype=torch.float32)}
        alpha_zero = runner.synthesize_state_dict(parent, baseline, aux, alpha=0.0)
        isolated = runner.synthesize_state_dict(parent, baseline, aux, alpha=1.0)
        expected = (
            parent["weight"].to(torch.float64)
            + aux["weight"].to(torch.float64)
            - baseline["weight"].to(torch.float64)
        ).to(torch.float32)
        self.assertTrue(torch.equal(alpha_zero["weight"], parent["weight"]))
        self.assertTrue(torch.equal(isolated["weight"], expected))
        self.assertEqual(isolated["weight"].dtype, torch.float32)

    def test_state_arithmetic_fails_closed_on_key_shape_or_dtype_drift(self):
        import torch

        base = {"x": torch.ones(2, dtype=torch.float32)}
        with self.assertRaises(RuntimeError):
            runner.synthesize_state_dict(base, {"y": torch.ones(2)}, base)
        with self.assertRaises(RuntimeError):
            runner.synthesize_state_dict(base, {"x": torch.ones(3)}, base)
        with self.assertRaises(RuntimeError):
            runner.synthesize_state_dict(
                base, {"x": torch.ones(2, dtype=torch.float64)}, base
            )

    def test_real_cpu_preflight_checks_alpha_identity_formula_and_strict_load(self):
        result = runner.task_arithmetic_preflight(self.protocol)
        self.assertTrue(result["passed"])
        self.assertTrue(result["cuda_not_initialized"])
        self.assertEqual(len(result["records"]), 3)
        for record in result["records"]:
            self.assertTrue(record["alpha_zero_parent_identity"])
            self.assertTrue(record["alpha_one_formula_bitwise"])
            self.assertTrue(record["strict_model_load_cpu"]["passed"])
            self.assertEqual(record["stats"]["changed_tensor_count"], 85)
            self.assertGreater(record["stats"]["task_over_baseline_drift"], 0.0)
            self.assertLess(record["stats"]["task_over_baseline_drift"], 0.1)

    def test_three_evaluation_specs_use_only_synthesized_candidates(self):
        records = []
        for spec in runner.synthesis_specs(self.protocol):
            records.append(
                {
                    "fold_id": spec["fold_id"],
                    "output_path": spec["output"],
                    "output_sha256": spec["metric_aux_sha256"],
                }
            )
        specs = runner.evaluation_specs(self.protocol, {"records": records})
        self.assertEqual(len(specs), 3)
        self.assertEqual(
            [spec["eval_id"] for spec in specs],
            [
                "hold_g1_isolated_metric_aux",
                "hold_g2_isolated_metric_aux",
                "hold_g3_isolated_metric_aux",
            ],
        )
        self.assertTrue(all(spec["variant"] == "isolated_metric_aux" for spec in specs))

    def _counts(self, tp=100, fp=10, fn=10, correct=9, false_components=5):
        return {
            "true_positive_events": tp,
            "false_positive_events": fp,
            "false_negative_events": fn,
            "correct_objects": correct,
            "object_count": 10,
            "false_components": false_components,
            "frame_count": 20,
            "event_count": 200,
        }

    def test_revised_official_gate_allows_raw_tp_loss_but_not_official_metric_loss(self):
        anchor_counts = self._counts()
        candidate_counts = self._counts(tp=99, fp=9, fn=11)
        anchor_metrics = {"score": 0.7, "pd": 0.9, "iou": 0.6, "fa": 0.01}
        candidate_metrics = {"score": 0.7003, "pd": 0.9, "iou": 0.601, "fa": 0.009}
        gate = runner._official_gate(
            candidate_counts,
            candidate_metrics,
            anchor_counts,
            anchor_metrics,
            pooled=True,
        )
        self.assertTrue(gate["passed"])
        self.assertNotIn("true_positive_events_not_lower", gate["checks"])
        bad = dict(candidate_metrics, iou=0.599)
        self.assertFalse(
            runner._official_gate(
                candidate_counts, bad, anchor_counts, anchor_metrics, pooled=True
            )["passed"]
        )

    def test_pooled_gate_requires_delta_and_one_strict_false_positive_reduction(self):
        counts = self._counts()
        anchor_metrics = {"score": 0.7, "pd": 0.9, "iou": 0.6, "fa": 0.01}
        candidate_metrics = {"score": 0.7001, "pd": 0.9, "iou": 0.6, "fa": 0.01}
        gate = runner._official_gate(
            counts, candidate_metrics, counts, anchor_metrics, pooled=True
        )
        self.assertFalse(gate["checks"]["score_delta_at_least_0p0002"])
        self.assertFalse(
            gate["checks"]["false_positive_events_or_false_components_strictly_lower"]
        )

    def test_cli_has_cpu_synthesis_but_no_train_probe_or_search(self):
        parser = runner.build_parser()
        action = next(
            value
            for value in parser._actions
            if value.__class__.__name__ == "_SubParsersAction"
        )
        self.assertEqual(
            set(action.choices),
            {"audit", "synthesize", "evaluate", "report", "all-evaluate-report"},
        )
        for forbidden in ("train", "probe", "search", "grid"):
            self.assertNotIn(forbidden, action.choices)

    def test_bound_m20_anchors_are_v4_artifacts_and_not_new_specs(self):
        for fold_id in ("hold_g1", "hold_g2", "hold_g3"):
            payload, digest = runner.load_m20_anchor(self.protocol, fold_id)
            record = self.protocol["v4_evaluation_artifacts_task_arithmetic"][
                "{}_released_m20".format(fold_id)
            ]
            self.assertEqual(digest, record["sha256"])
            self.assertEqual(payload["checkpoint_sha256"], runner.M20_SHA256)
            self.assertEqual(payload["protocol_sha256"], runner.V4_PROTOCOL_SHA256)

    def test_temporary_inference_helper_is_restored_after_failure(self):
        spec = {
            "eval_id": "hold_g1_isolated_metric_aux",
            "fold_id": "hold_g1",
            "variant": "isolated_metric_aux",
            "checkpoint_sha256": "1" * 64,
            "result_path": str(runner.EVALUATION_ROOT / "synthetic" / "evaluation.json"),
        }
        with mock.patch.object(
            runner, "_BASE_EVALUATE_SPEC", side_effect=RuntimeError("synthetic failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                runner.evaluate_spec(self.protocol, spec, "2" * 64)
        import utils.temporal_memory_inference as memory_inference

        self.assertFalse(hasattr(memory_inference, "temporal_frame_video_from_sample"))
        self.assertIs(runner.core.write_new_json, runner._BASE_WRITE_NEW_JSON)


if __name__ == "__main__":
    unittest.main()
