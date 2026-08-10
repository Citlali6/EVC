import copy
from pathlib import Path
import sys
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_metric_aux_h2_grouped_oof as public_v1
import run_metric_aux_h2_grouped_oof_v2 as public_v2

_PUBLIC_STATE_BEFORE_V3 = {
    "v1_protocol": public_v1.PROTOCOL_PATH,
    "v1_output": public_v1.OUTPUT_ROOT,
    "v1_file": Path(public_v1.__file__).resolve(),
    "v2_protocol": public_v2.PROTOCOL_PATH,
    "v2_output": public_v2.OUTPUT_ROOT,
    "v2_file": Path(public_v2.__file__).resolve(),
    "v2_core_file": Path(public_v2.core.__file__).resolve(),
}

import run_metric_aux_h2_grouped_oof_v3 as runner


class MetricAuxGroupedOofAuditRecoveryV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = runner.load_protocol()
        cls.overlay, cls.overlay_sha = runner.core.load_json_snapshot(
            runner.PROTOCOL_PATH
        )
        failure_record = cls.overlay["inheritance"]["v2_failure_receipt"]
        cls.failure, cls.failure_sha = runner.core.load_json_snapshot(
            runner.core.workspace_path(failure_record["workspace_relative_path"])
        )
        cls.v2_probe_root = (
            runner.WORKSPACE_ROOT
            / "experiments"
            / "20260810_metric_aux_h2_grouped_oof_v2"
            / "data_views"
            / "probe"
        )

    def test_v3_protocol_hash_status_and_recovery_scope_are_frozen(self):
        self.assertEqual(self.protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(self.overlay_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(
            self.protocol["status"],
            "frozen_before_any_v3_gpu_gradient_audit_formal_training_or_held_evaluation",
        )
        recovery = self.protocol["recovery_amendment_v3"]
        self.assertTrue(recovery["v2_attempt_remains_failed"])
        self.assertTrue(recovery["retroactive_v2_pass_forbidden"])
        self.assertTrue(recovery["v2_fresh_pair_training_reuse_allowed"])
        self.assertTrue(recovery["repeat_32_step_pair_training_forbidden"])
        self.assertFalse(
            recovery["scientific_candidate_training_evaluation_or_promotion_change"]
        )

    def test_v2_failure_receipt_is_bound_and_v2_remains_failed(self):
        self.assertEqual(self.failure_sha, runner.V2_FAILURE_SHA256)
        self.assertEqual(self.failure["status"], "failed")
        self.assertFalse(self.failure["passed"])
        self.assertFalse(self.failure["candidate_or_training_failure"])
        self.assertFalse(self.failure["formal_training_started"])
        self.assertFalse(self.failure["held_train_evaluation_started"])
        self.assertTrue(self.failure["completed_numeric_pair_audit"]["passed"])
        self.assertTrue(
            self.failure["recovery_policy"][
                "repeat_paired_32_step_training_forbidden"
            ]
        )

    def test_private_v2_and_v1_cores_do_not_mutate_public_modules(self):
        self.assertIsNot(runner.v2, public_v2)
        self.assertIsNot(runner.core, public_v2.core)
        self.assertEqual(public_v1.PROTOCOL_PATH, _PUBLIC_STATE_BEFORE_V3["v1_protocol"])
        self.assertEqual(public_v1.OUTPUT_ROOT, _PUBLIC_STATE_BEFORE_V3["v1_output"])
        self.assertEqual(Path(public_v1.__file__).resolve(), _PUBLIC_STATE_BEFORE_V3["v1_file"])
        self.assertEqual(public_v2.PROTOCOL_PATH, _PUBLIC_STATE_BEFORE_V3["v2_protocol"])
        self.assertEqual(public_v2.OUTPUT_ROOT, _PUBLIC_STATE_BEFORE_V3["v2_output"])
        self.assertEqual(Path(public_v2.__file__).resolve(), _PUBLIC_STATE_BEFORE_V3["v2_file"])
        self.assertEqual(
            Path(public_v2.core.__file__).resolve(),
            _PUBLIC_STATE_BEFORE_V3["v2_core_file"],
        )

    def test_core_paths_and_identity_are_patched_to_v3_for_future_artifacts(self):
        self.assertEqual(runner.core.PROTOCOL_PATH, runner.PROTOCOL_PATH)
        self.assertEqual(runner.core.OUTPUT_ROOT, runner.OUTPUT_ROOT)
        self.assertEqual(runner.core.COMMAND_AUDIT_PATH, runner.COMMAND_AUDIT_PATH)
        self.assertEqual(runner.core.PROBE_RESULT_PATH, runner.PROBE_RESULT_PATH)
        self.assertEqual(runner.core.FORMAL_ROOT, runner.FORMAL_ROOT)
        self.assertEqual(runner.core.EVALUATION_ROOT, runner.EVALUATION_ROOT)
        self.assertEqual(runner.core.REPORT_PATH, runner.REPORT_PATH)
        self.assertEqual(runner.core.EXPECTED_PROTOCOL_SHA256, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(Path(runner.core.__file__).resolve(), Path(runner.__file__).resolve())

    def test_v2_pair_results_and_checkpoints_are_exactly_bound(self):
        baseline, candidate = runner._load_v2_pair_results(self.protocol)
        self.assertEqual(baseline["variant"], "baseline")
        self.assertEqual(candidate["variant"], "metric_aux")
        self.assertEqual(baseline["expected_source_names"], ["train_096.npz"])
        self.assertEqual(candidate["expected_source_names"], ["train_096.npz"])
        self.assertEqual(baseline["expected_optimizer_steps"], 16)
        self.assertEqual(candidate["expected_optimizer_steps"], 16)
        self.assertNotIn("e3", baseline["checkpoints"])
        self.assertNotIn("e3", candidate["checkpoints"])
        evidence = self.protocol["v2_pair_evidence_v3"]
        self.assertEqual(
            baseline["checkpoints"]["e1"]["sha256"],
            evidence["baseline_e1_checkpoint"]["sha256"],
        )
        self.assertEqual(
            candidate["checkpoints"]["e1"]["sha256"],
            evidence["candidate_e1_checkpoint"]["sha256"],
        )

    def test_v2_numeric_pair_recomputes_pass_without_new_training(self):
        baseline, candidate = runner._load_v2_pair_results(self.protocol)
        audit = runner._V2_COMPARE_PAIR(baseline, candidate)
        self.assertTrue(audit["passed"])
        self.assertAlmostEqual(
            audit["e1_model"]["global_l2"], 1.9947395589793698e-07
        )
        self.assertAlmostEqual(
            audit["e2_model_global_l2_over_e1_numerical_floor"],
            434.7515348425672,
        )
        for key in ("baseline_e1", "candidate_e1"):
            self.assertEqual(
                audit["optimizer_step_audits"][key]["observed_unique_steps"],
                [8.0],
            )
        for key in ("baseline_e2", "candidate_e2"):
            self.assertEqual(
                audit["optimizer_step_audits"][key]["observed_unique_steps"],
                [16.0],
            )
        self.assertEqual(
            self.protocol["v3_resource_probe"]["new_training_optimizer_steps"], 0
        )

    def test_cpu_resolution_audit_matches_all_eight_fixed_views(self):
        result = runner.cpu_resolution_audit(self.protocol, self.v2_probe_root)
        self.assertTrue(result["passed"])
        self.assertEqual(result["spatial_resolution"], [346, 260])
        self.assertEqual(result["total_event_count"], 496290)
        self.assertEqual(result["total_outside_rejected_128x128_count"], 404304)
        self.assertEqual(len(result["records"]), 8)
        for record in result["records"]:
            self.assertEqual(record["frame_shape"], [16, 10, 260, 346])
            self.assertEqual(record["event_time_index_range"], [0, 15])
            self.assertEqual(record["event_x_range"], [0, 345])
            self.assertEqual(record["event_y_range"], [0, 259])
            self.assertEqual(record["outside_formal_resolution_count"], 0)

    def test_old_128_spatial_shape_is_deterministically_out_of_bounds(self):
        from dataset.temporal_memory import (
            TemporalMemoryTrainDataset,
            temporal_memory_collate,
        )

        dataset = TemporalMemoryTrainDataset(
            root=self.v2_probe_root / "train",
            whole_t=8000,
            temporal_bin_size=50,
            context_bins=5,
            sequence_length=16,
            width=128,
            height=128,
            views_per_video=8,
            positive_frame_probability=0.75,
            random_seed=49,
            cache_all_videos=False,
            cache_video_count=2,
            dense_sampling_enabled=False,
            density_bucket_boundaries=[],
            density_bucket_views=[],
            min_event_count_exclusive=200000,
            sparse_target_support_sampling_enabled=False,
        )
        dataset.set_epoch(1)
        batch = temporal_memory_collate([dataset[0]])
        out_of_bounds = (
            (batch["event_x"] >= batch["frames"].shape[3])
            | (batch["event_y"] >= batch["frames"].shape[2])
        )
        self.assertEqual(list(batch["frames"].shape), [16, 10, 128, 128])
        self.assertEqual(int(batch["event_x"].max()), 345)
        self.assertEqual(int(batch["event_y"].max()), 259)
        self.assertEqual(int(out_of_bounds.sum()), 50579)

    def test_resolution_and_model_feature_width_are_distinct_and_frozen(self):
        resolution = self.protocol["audit_resolution_contract_v3"]
        self.assertEqual(
            (
                resolution["spatial_width"],
                resolution["spatial_height"],
                resolution["model_feature_width"],
            ),
            (346, 260, 16),
        )
        self.assertIn("No event may be clamped", resolution["event_handling"])
        self.assertIn("filtered", resolution["event_handling"])
        self.assertIn("rescaled", resolution["event_handling"])

    def test_command_audit_overlay_discards_pair_probe_training_commands(self):
        fake_payload = {
            "schema": "old",
            "probe_commands": {"baseline": {}, "metric_aux": {}},
            "data_use_statement": "old",
        }
        fake_pair = {"passed": True}
        fake_synthetic = {"passed": True}
        fake_bounds = {"passed": True}
        with mock.patch.object(
            runner, "_V2_COMMAND_AUDIT_PAYLOAD", return_value=fake_payload
        ), mock.patch.object(
            runner, "_load_v2_pair_results", return_value=({}, {})
        ), mock.patch.object(
            runner, "_V2_COMPARE_PAIR", return_value=fake_pair
        ), mock.patch.object(
            runner.core, "synthetic_metric_gradient_probe", return_value=fake_synthetic
        ), mock.patch.object(
            runner, "cpu_resolution_audit", return_value=fake_bounds
        ):
            payload = runner.command_audit_payload(
                self.protocol, self.protocol_sha, {}, {"probe": {"root": "unused"}}
            )
        self.assertNotIn("probe_commands", payload)
        self.assertEqual(payload["probe_command"]["mode"], "audit_only")
        self.assertEqual(payload["probe_command"]["new_training_optimizer_steps"], 0)
        self.assertEqual(
            payload["v3_audit_only_recovery"][
                "discarded_inherited_probe_training_command_count"
            ],
            2,
        )

    def test_v3_probe_cannot_reuse_v2_failure_as_a_success_receipt(self):
        self.assertNotEqual(
            runner.core.workspace_path(
                self.overlay["inheritance"]["v2_failure_receipt"][
                    "workspace_relative_path"
                ]
            ).resolve(),
            runner.PROBE_RESULT_PATH.resolve(),
        )
        self.assertFalse(runner.PROBE_RESULT_PATH.exists())
        self.assertFalse(runner.PROBE_FAILURE_PATH.exists())
        with self.assertRaises(FileNotFoundError):
            runner.require_probe_passed()

    def test_gpu_entry_is_authorization_fail_closed_before_any_probe_action(self):
        with self.assertRaises(PermissionError):
            runner.run_probe(False)

    def test_cli_retains_formal_evaluate_report_delegation_without_v4(self):
        parser = runner.build_parser()
        for argv, expected in (
            (["train", "--run-id", "hold_g1_baseline"], "train"),
            (["audit-training"], "audit-training"),
            (["evaluate", "--eval-id", "hold_g1_released_m20"], "evaluate"),
            (["report"], "report"),
            (["all-after-probe"], "all-after-probe"),
        ):
            self.assertEqual(parser.parse_args(argv).command, expected)
        self.assertEqual(
            self.protocol["outputs"]["workspace_relative_directory"],
            "experiments/20260810_metric_aux_h2_grouped_oof_v3",
        )

    def test_inherited_candidate_folds_and_promotion_gates_are_unchanged(self):
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
        self.assertEqual(len(self.protocol["dataset"]["folds"]), 3)
        self.assertTrue(self.protocol["training"]["no_parameter_grid"])
        gates = self.protocol["promotion_gates"]
        self.assertEqual(gates["comparators"], ["paired_baseline_e3", "released_m20"])
        self.assertTrue(gates["against_each_comparator_each_fold_score_not_lower"])


if __name__ == "__main__":
    unittest.main()
