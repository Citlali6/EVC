from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_metric_aux_task_arithmetic_alpha130_h2_grouped_oof as runner


class Alpha130GroupedOofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = runner.load_protocol()
        cls.overlay, cls.overlay_sha = runner.core.load_json_snapshot(runner.PROTOCOL_PATH)

    def test_protocol_is_one_fixed_train_only_alpha130_candidate(self):
        self.assertEqual(self.protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(self.overlay_sha, runner.EXPECTED_PROTOCOL_SHA256)
        candidate = self.overlay["fixed_candidate"]
        self.assertEqual(candidate["alpha"], 1.30)
        self.assertEqual(candidate["candidate_count"], 1)
        self.assertEqual(candidate["new_training_optimizer_steps"], 0)
        self.assertFalse(candidate["alpha_grid_allowed"])
        self.assertFalse(candidate["module_mask_or_projection_allowed"])
        self.assertFalse(candidate["threshold_search_allowed"])
        self.assertFalse(candidate["weight_or_c00_search_allowed"])
        self.assertFalse(self.overlay["validation_or_test_read_allowed"])
        self.assertFalse(self.overlay["current_failed_validation_report_read_allowed"])

    def test_synthesis_specs_are_exactly_three_new_fold_outputs(self):
        specs = runner.synthesis_specs(self.protocol)
        self.assertEqual([item["fold_id"] for item in specs], ["hold_g1", "hold_g2", "hold_g3"])
        self.assertEqual(len({item["output"] for item in specs}), 3)
        self.assertTrue(all(item["output"].endswith("isolated_metric_aux_alpha130.pt") for item in specs))

    def test_alpha_zero_identity_and_alpha130_float64_single_cast(self):
        import torch

        parent = {"weight": torch.tensor([1.0, -2.0], dtype=torch.float32)}
        baseline = {"weight": torch.tensor([1.25, -1.5], dtype=torch.float32)}
        aux = {"weight": torch.tensor([1.5, -1.75], dtype=torch.float32)}
        alpha_zero = runner.synthesize_state_dict(parent, baseline, aux, 0.0)
        candidate = runner.synthesize_state_dict(parent, baseline, aux, 1.30)
        expected = (
            parent["weight"].to(torch.float64)
            + 1.30 * (aux["weight"].to(torch.float64) - baseline["weight"].to(torch.float64))
        ).to(torch.float32)
        self.assertTrue(torch.equal(alpha_zero["weight"], parent["weight"]))
        self.assertTrue(torch.equal(candidate["weight"], expected))
        self.assertTrue(runner._independent_formula_equal(parent, baseline, aux, candidate, 1.30))

    def test_arithmetic_fails_closed_on_key_shape_and_dtype_drift(self):
        import torch

        base = {"x": torch.ones(2, dtype=torch.float32)}
        with self.assertRaises(RuntimeError):
            runner.synthesize_state_dict(base, {"y": torch.ones(2)}, base)
        with self.assertRaises(RuntimeError):
            runner.synthesize_state_dict(base, {"x": torch.ones(3)}, base)
        with self.assertRaises(RuntimeError):
            runner.synthesize_state_dict(base, {"x": torch.ones(2, dtype=torch.float64)}, base)

    def test_real_cpu_preflight_passes_fold_formula_and_all11_geometry(self):
        result = runner.task_arithmetic_preflight(self.protocol)
        self.assertTrue(result["passed"])
        self.assertTrue(result["cuda_not_initialized"])
        self.assertEqual(len(result["records"]), 3)
        for record in result["records"]:
            self.assertTrue(record["alpha_zero_parent_identity"])
            self.assertTrue(record["alpha130_independent_formula_bitwise"])
            self.assertTrue(record["strict_model_load_cpu"]["passed"])
            self.assertEqual(record["raw_task"]["changed_tensor_count"], 85)
        geometry = result["all11_geometry"]
        self.assertTrue(geometry["passed"])
        self.assertTrue(all(geometry["checks"].values()))
        self.assertAlmostEqual(
            geometry["applied_task"]["l2"],
            self.overlay["all11_geometry_evidence"]["alpha130_applied_l2"],
            places=15,
        )

    def test_evaluation_specs_only_add_three_formal_alpha130_inferences(self):
        records = [
            {"fold_id": spec["fold_id"], "output_path": spec["output"], "output_sha256": spec["metric_aux_sha256"]}
            for spec in runner.synthesis_specs(self.protocol)
        ]
        specs = runner.evaluation_specs(self.protocol, {"records": records})
        self.assertEqual([item["eval_id"] for item in specs], ["hold_g1_alpha130", "hold_g2_alpha130", "hold_g3_alpha130"])
        self.assertTrue(all(item["variant"] == "isolated_metric_aux_alpha130" for item in specs))
        self.assertTrue(all(item["training_result_path"] is None for item in specs))

    def _counts(self, tp=100, fp=10, fn=10, correct=9, fc=5):
        return {
            "true_positive_events": tp,
            "false_positive_events": fp,
            "false_negative_events": fn,
            "correct_objects": correct,
            "object_count": 10,
            "false_components": fc,
            "frame_count": 20,
            "event_count": 200,
        }

    def test_dual_anchor_gate_enforces_alpha1_tp_overshoot(self):
        m20 = self._counts()
        alpha1 = self._counts(tp=99, fp=8, fn=11, fc=4)
        candidate = self._counts(tp=99, fp=7, fn=11, fc=3)
        m20_metrics = {"score": 0.7000, "pd": 0.9, "iou": 0.60, "fa": 0.010}
        alpha1_metrics = {"score": 0.7002, "pd": 0.9, "iou": 0.61, "fa": 0.009}
        candidate_metrics = {"score": 0.7004, "pd": 0.9, "iou": 0.62, "fa": 0.008}
        gate = runner.dual_anchor_gate(candidate, candidate_metrics, m20, m20_metrics, alpha1, alpha1_metrics, pooled=True)
        self.assertTrue(gate["passed"])
        overshoot = self._counts(tp=98, fp=7, fn=12, fc=3)
        bad = runner.dual_anchor_gate(overshoot, candidate_metrics, m20, m20_metrics, alpha1, alpha1_metrics, pooled=True)
        self.assertFalse(bad["against_alpha1"]["checks"]["raw_true_positive_events_not_lower"])
        self.assertFalse(bad["passed"])

    def test_pooled_gate_requires_both_score_margins_and_strict_reduction(self):
        counts = self._counts()
        metrics = {"score": 0.7000, "pd": 0.9, "iou": 0.60, "fa": 0.010}
        candidate = dict(metrics, score=0.7003)
        gate = runner.dual_anchor_gate(counts, candidate, counts, metrics, counts, dict(metrics, score=0.70026), pooled=True)
        self.assertFalse(gate["against_released_m20"]["checks"]["score_delta_at_least_0p00032"])
        self.assertFalse(gate["against_alpha1"]["checks"]["score_delta_at_least_0p00005"])
        self.assertFalse(gate["against_alpha1"]["checks"]["false_positive_events_or_false_components_strictly_lower"])

    def test_count_validator_rejects_bool_negative_and_extra_fields(self):
        runner._validate_count_dict(self._counts(), "valid")
        with self.assertRaises(RuntimeError):
            runner._validate_count_dict(dict(self._counts(), true_positive_events=True), "bool")
        with self.assertRaises(RuntimeError):
            runner._validate_count_dict(dict(self._counts(), false_components=-1), "negative")
        with self.assertRaises(RuntimeError):
            runner._validate_count_dict(dict(self._counts(), extra=1), "extra")

    def test_exclusive_json_writer_refuses_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            runner._write_json_exclusive(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                runner._write_json_exclusive(path, {"value": 2})

    def test_cli_has_no_train_probe_grid_threshold_or_module_command(self):
        parser = runner.build_parser()
        action = next(value for value in parser._actions if value.__class__.__name__ == "_SubParsersAction")
        self.assertEqual(set(action.choices), {"audit", "synthesize", "evaluate", "report", "all-evaluate-report"})
        for forbidden in ("train", "probe", "grid", "search", "threshold", "module"):
            self.assertNotIn(forbidden, action.choices)

    def test_helper_and_writer_are_restored_after_base_evaluation_failure(self):
        import utils.temporal_memory_inference as memory_inference

        spec = {
            "eval_id": "hold_g1_alpha130",
            "fold_id": "hold_g1",
            "variant": "isolated_metric_aux_alpha130",
            "checkpoint_sha256": "1" * 64,
            "result_path": str(runner.EVALUATION_ROOT / "synthetic" / "evaluation.json"),
        }
        original_writer = runner.core.write_new_json
        fake_assets = {"checkpoint": {"path": "x", "sha256": "1" * 64}}
        with mock.patch.object(runner, "_immutable_inference_asset_hashes", return_value=fake_assets):
            with mock.patch.object(runner, "_BASE_EVALUATE_SPEC", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    runner.evaluate_spec(self.protocol, spec, "2" * 64)
        self.assertFalse(hasattr(memory_inference, "temporal_frame_video_from_sample"))
        self.assertIs(runner.core.write_new_json, original_writer)


if __name__ == "__main__":
    unittest.main()
