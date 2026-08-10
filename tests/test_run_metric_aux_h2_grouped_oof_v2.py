import copy
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_metric_aux_h2_grouped_oof as public_v1

_PUBLIC_V1_PATHS_BEFORE_V2 = {
    "protocol": public_v1.PROTOCOL_PATH,
    "output": public_v1.OUTPUT_ROOT,
    "runner_file": Path(public_v1.__file__).resolve(),
}

import run_metric_aux_h2_grouped_oof_v2 as runner


class MetricAuxGroupedOofRecoveryV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = runner.load_protocol()
        cls.overlay, cls.overlay_sha = runner.core.load_json_snapshot(
            runner.PROTOCOL_PATH
        )
        failure_record = cls.overlay["inheritance"]["attempt1_failure_receipt"]
        cls.failure_path = runner.core.workspace_path(
            failure_record["workspace_relative_path"]
        )
        cls.failure, cls.failure_sha = runner.core.load_json_snapshot(
            cls.failure_path
        )

    def test_v2_overlay_hash_status_and_v1_evidence_bindings_are_frozen(self):
        self.assertEqual(self.protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(self.overlay_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(
            self.overlay["status"],
            "frozen_before_any_v2_probe_formal_training_or_held_evaluation",
        )
        expected = {
            "base_science_protocol": runner.V1_PROTOCOL_SHA256,
            "base_runner": runner.V1_RUNNER_SHA256,
            "base_command_audit": runner.V1_COMMAND_AUDIT_SHA256,
            "attempt1_failure_receipt": runner.V1_FAILURE_SHA256,
        }
        for key, digest in expected.items():
            record = self.overlay["inheritance"][key]
            path = runner.core.workspace_path(record["workspace_relative_path"])
            self.assertEqual(record["sha256"], digest)
            self.assertEqual(runner.core.sha256_file(path), digest)

    def test_private_v1_core_does_not_mutate_public_v1_module(self):
        self.assertIsNot(runner.core, public_v1)
        self.assertEqual(
            public_v1.PROTOCOL_PATH, _PUBLIC_V1_PATHS_BEFORE_V2["protocol"]
        )
        self.assertEqual(public_v1.OUTPUT_ROOT, _PUBLIC_V1_PATHS_BEFORE_V2["output"])
        self.assertEqual(
            Path(public_v1.__file__).resolve(),
            _PUBLIC_V1_PATHS_BEFORE_V2["runner_file"],
        )
        self.assertEqual(runner.core.PROTOCOL_PATH, runner.PROTOCOL_PATH)
        self.assertEqual(runner.core.OUTPUT_ROOT, runner.OUTPUT_ROOT)
        self.assertEqual(
            Path(runner.core.__file__).resolve(), Path(runner.__file__).resolve()
        )

    def test_v1_attempt1_remains_failed_and_cannot_be_a_v2_receipt(self):
        self.assertEqual(self.failure_sha, runner.V1_FAILURE_SHA256)
        self.assertEqual(self.failure["status"], "failed")
        self.assertFalse(self.failure["passed"])
        self.assertFalse(self.failure["formal_training_started"])
        self.assertFalse(self.failure["held_train_evaluation_started"])
        self.assertTrue(
            self.failure["recovery_policy"][
                "retroactive_reclassification_forbidden"
            ]
        )
        self.assertNotEqual(self.failure_path.resolve(), runner.PROBE_RESULT_PATH.resolve())
        self.assertFalse(runner.PROBE_RESULT_PATH.exists())
        self.assertFalse(runner.PROBE_FAILURE_PATH.exists())
        with self.assertRaises(FileNotFoundError):
            runner.require_probe_passed()

    def test_numeric_recovery_thresholds_are_round_and_apply_to_formal_pairs(self):
        contract = self.protocol["numeric_near_identity_contract"]
        e1 = contract["e1_zero_based_epoch0"]
        self.assertEqual(
            (
                e1["model_max_abs_maximum"],
                e1["model_relative_l2_maximum"],
                e1["optimizer_max_abs_maximum"],
                e1["optimizer_global_l2_maximum"],
                e1["epoch_loss_abs_delta_maximum"],
            ),
            (1e-6, 1e-7, 1e-5, 1e-4, 1e-7),
        )
        self.assertEqual(
            contract["e2_zero_based_epoch1"][
                "candidate_model_global_l2_over_e1_numerical_floor_minimum"
            ],
            10.0,
        )
        self.assertTrue(
            contract["formal_pair_reuse"][
                "same_e1_numeric_near_identity_limits_required_for_all_three_fold_pairs"
            ]
        )

    def _attempt1_results(self):
        evidence = self.failure["frozen_evidence"]
        baseline, baseline_sha = runner.core.load_json_snapshot(
            runner.core.workspace_path(
                evidence["baseline_training_result"]["path"]
            )
        )
        candidate, candidate_sha = runner.core.load_json_snapshot(
            runner.core.workspace_path(
                evidence["candidate_training_result"]["path"]
            )
        )
        self.assertEqual(
            baseline_sha, evidence["baseline_training_result"]["sha256"]
        )
        self.assertEqual(
            candidate_sha, evidence["candidate_training_result"]["sha256"]
        )
        return baseline, candidate

    def test_attempt1_diagnostics_fit_v2_envelope_without_reclassification(self):
        baseline, candidate = self._attempt1_results()
        audit = runner.compare_pair_checkpoints(baseline, candidate)
        self.assertTrue(audit["passed"])
        self.assertAlmostEqual(audit["e1_model"]["max_abs"], 9.685754776000977e-08)
        self.assertAlmostEqual(audit["e1_model"]["global_l2"], 6.309269390648273e-07)
        self.assertAlmostEqual(
            audit["e1_model"]["relative_l2"], 1.2138687507618539e-08
        )
        self.assertAlmostEqual(
            audit["e1_optimizer"]["global_l2"], 4.501978219619991e-06
        )
        self.assertAlmostEqual(
            audit["e2_model_global_l2_over_e1_numerical_floor"],
            137.45372022069103,
        )
        self.assertFalse(self.failure["passed"])

    def test_attempt1_optimizer_steps_are_checked_for_all_89_states(self):
        baseline, candidate = self._attempt1_results()
        audit = runner.compare_pair_checkpoints(baseline, candidate)
        steps = audit["optimizer_step_audits"]
        for key in ("baseline_e1", "candidate_e1"):
            self.assertTrue(steps[key]["passed"])
            self.assertEqual(steps[key]["observed_unique_steps"], [8.0])
            self.assertEqual(steps[key]["optimizer_state_entry_count"], 89)
        for key in ("baseline_e2", "candidate_e2"):
            self.assertTrue(steps[key]["passed"])
            self.assertEqual(steps[key]["observed_unique_steps"], [16.0])

    def test_optimizer_step_audit_fails_one_corrupted_state(self):
        import torch

        state = {
            "param_groups": [{"params": list(range(89))}],
            "state": {index: {"step": torch.tensor(8.0)} for index in range(89)},
        }
        contract = self.protocol["numeric_near_identity_contract"][
            "optimizer_step_contract"
        ]
        self.assertTrue(runner._optimizer_step_audit(state, 8, contract)["passed"])
        corrupted = copy.deepcopy(state)
        corrupted["state"][17]["step"] = torch.tensor(7.0)
        result = runner._optimizer_step_audit(corrupted, 8, contract)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["every_state_step_matches_expected"])

    def test_model_numeric_gate_rejects_over_limit_difference(self):
        import torch

        baseline = {"weight": torch.zeros(4, dtype=torch.float32)}
        candidate = {"weight": torch.tensor([2e-6, 0.0, 0.0, 0.0])}
        metrics = runner._tensor_model_difference(baseline, candidate)
        limit = self.protocol["numeric_near_identity_contract"][
            "e1_zero_based_epoch0"
        ]["model_max_abs_maximum"]
        self.assertTrue(metrics["structure_exact"])
        self.assertGreater(metrics["max_abs"], limit)

    def test_m23_train_only_sampling_record_is_bound_and_not_claimed_as_exact_reuse(self):
        m23 = self.protocol["historical_m23_sampling_audit_v2"]
        self.assertEqual(m23["source_video_count"], 99)
        self.assertEqual(m23["dense_video_count_over_200k"], 15)
        self.assertEqual(m23["h2_dense_video_count"], 11)
        self.assertEqual(m23["non_h2_dense_video_count"], 4)
        self.assertEqual(m23["non_dense_video_count"], 84)
        self.assertEqual(
            m23["h2_dense_sequences_per_epoch"]
            + m23["non_h2_dense_sequences_per_epoch"]
            + m23["non_dense_sequences_per_epoch"],
            408,
        )
        self.assertEqual(408 * m23["epochs"], 1632)
        self.assertIn("mixed-population", m23["disclosure"])
        self.assertIn("reuses only M23 loss hyperparameters", m23["disclosure"])

    def test_inherited_single_candidate_folds_and_double_anchor_promotion(self):
        candidate = self.protocol["training"]["candidate"]
        self.assertEqual(
            (
                candidate["metric_target_weight"],
                candidate["metric_component_weight"],
                candidate["metric_warmup_epochs"],
                candidate["metric_activation_threshold"],
            ),
            (0.005, 0.001, 1, 0.719),
        )
        self.assertTrue(self.protocol["training"]["no_parameter_grid"])
        self.assertEqual(len(self.protocol["dataset"]["folds"]), 3)
        gates = self.protocol["promotion_gates"]
        self.assertEqual(gates["comparators"], ["paired_baseline_e3", "released_m20"])
        self.assertTrue(gates["against_each_comparator_each_fold_score_not_lower"])

    def test_optimizer_step_formula_covers_probe_and_all_formal_fold_sizes(self):
        contract = self.protocol["numeric_near_identity_contract"][
            "optimizer_step_contract"
        ]
        self.assertEqual((contract["probe_e1_step"], contract["probe_e2_step"]), (8, 16))
        observed = []
        for fold in self.protocol["dataset"]["folds"]:
            per_epoch = int(fold["fit_video_count"]) * 8
            observed.append((per_epoch, per_epoch * 2, per_epoch * 3))
        self.assertEqual(observed, [(56, 112, 168), (64, 128, 192), (56, 112, 168)])

    def test_gpu_entry_is_authorization_fail_closed(self):
        with self.assertRaises(PermissionError):
            runner.run_probe(False)


if __name__ == "__main__":
    unittest.main()
