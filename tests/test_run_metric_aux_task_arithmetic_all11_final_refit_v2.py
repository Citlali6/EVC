import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest


EVC_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = EVC_ROOT / "run_metric_aux_task_arithmetic_all11_final_refit_v2.py"
SPEC = importlib.util.spec_from_file_location("_test_all11_final_v2", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class All11FinalV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = runner.load_protocol()
        cls.geometry = runner.geometry_preflight(cls.protocol)

    def test_protocol_and_v1_failure_are_bound(self):
        self.assertEqual(self.protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(self.protocol["all11_v2"]["schema"], runner.EXPECTED_SCHEMA)
        self.assertFalse(self.protocol["_v1_failure"]["passed"])
        self.assertTrue(self.protocol["_v1_failure"]["v1_attempt_must_remain_failed"])
        self.assertEqual(
            self.protocol["_v1_failure"]["failed_gate"],
            "task_over_drift_open_interval",
        )

    def test_no_v1_pair_or_synthesis_artifact(self):
        evidence = self.protocol["all11_v2"]["v1_evidence"]
        for key in (
            "pair_audit_must_remain_absent",
            "final_checkpoint_must_remain_absent",
            "synthesis_manifest_must_remain_absent",
        ):
            self.assertFalse(runner.core.workspace_path(evidence[key]).exists())

    def test_all_inherited_v1_and_replacement_gates_pass(self):
        self.assertTrue(self.geometry["passed"])
        self.assertTrue(all(self.geometry["old_v1_checks_except_replaced_ratio"].values()))
        self.assertTrue(all(self.geometry["replacement_gates"].values()))
        self.assertGreater(self.geometry["task_over_drift_reported_not_gated"], 0.1)

    def test_step_normalized_task_is_inside_frozen_oof_envelope(self):
        lower, upper = self.protocol["all11_v2"]["v2_replacement_safety_gates"][
            "step_normalized_task_l2_inclusive_oof_envelope"
        ]
        value = self.geometry["step_normalized_task_l2"]
        self.assertLessEqual(lower, value)
        self.assertLessEqual(value, upper)
        self.assertAlmostEqual(value, 0.0001059710701593259, places=16)

    def test_direction_parent_and_module_gates(self):
        floor = self.protocol["all11_v2"]["v2_replacement_safety_gates"][
            "cosine_with_every_oof_task_at_least_oof_pairwise_minimum"
        ]
        self.assertTrue(all(value >= floor for value in self.geometry["cosines_with_oof_tasks"].values()))
        self.assertEqual(
            set(self.geometry["task"]["module_energy_share"]),
            {"base", "forward_memory", "backward_memory", "memory_projection", "temporal_attn"},
        )
        self.assertTrue(all(self.geometry["module_gates"].values()))
        self.assertLessEqual(
            self.geometry["task_over_m20"],
            self.protocol["all11_v2"]["v2_replacement_safety_gates"][
                "task_over_m20_maximum_scaled_by_sqrt_264_over_168"
            ],
        )

    def test_no_train_gpu_evaluate_or_search_cli(self):
        parser = runner.build_parser()
        for forbidden in ("train", "probe", "evaluate", "report", "validation", "search"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([forbidden])
        for allowed in ("audit", "audit-training", "synthesize"):
            self.assertEqual(parser.parse_args([allowed]).command, allowed)

    def test_alpha_zero_and_alpha_one_formula(self):
        import torch

        parent = {"a": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        baseline = {"a": torch.tensor([0.5, 2.5], dtype=torch.float32)}
        metric = {"a": torch.tensor([0.25, 3.0], dtype=torch.float32)}
        zero = runner.v1._V5_SYNTHESIZE_STATE_DICT(parent, baseline, metric, 0.0)
        one = runner.v1._V5_SYNTHESIZE_STATE_DICT(parent, baseline, metric, 1.0)
        self.assertTrue(runner.v1._V5_STATE_EQUAL(zero, parent))
        expected = (
            parent["a"].to(torch.float64)
            + metric["a"].to(torch.float64)
            - baseline["a"].to(torch.float64)
        ).to(torch.float32)
        self.assertTrue(torch.equal(one["a"], expected))

    def test_access_contract_is_train_only_cpu_geometry(self):
        overlay = self.protocol["all11_v2"]
        self.assertFalse(overlay["validation_or_test_read_allowed"])
        self.assertFalse(overlay["t32_allowed"])
        self.assertEqual(overlay["amendment_disclosure"]["new_training_optimizer_steps"], 0)
        self.assertFalse(
            overlay["synthesis_contract"]["evaluation_or_score_computation_allowed"]
        )


if __name__ == "__main__":
    unittest.main()
