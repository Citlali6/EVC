from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_metric_aux_task_arithmetic_alpha130_h2_grouped_oof_v2 as runner


class Alpha130CpuPublishRecoveryV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = runner.load_protocol()
        cls.overlay, cls.overlay_sha = runner.core.load_json_snapshot(runner.PROTOCOL_PATH)

    def test_recovery_protocol_binds_failed_v1_without_published_candidate(self):
        self.assertEqual(self.protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(self.overlay_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(runner.core.sha256_file(runner._V1_PATH), runner.V1_RUNNER_SHA256)
        failure = self.overlay["v1_failure_receipt"]
        self.assertEqual(failure["candidate_checkpoint_publish_count"], 0)
        self.assertFalse(failure["synthesis_manifest_published"])
        self.assertEqual(failure["formal_alpha130_inference_count"], 0)
        self.assertFalse(failure["gpu_or_cuda_used"])

    def test_recovery_changes_no_science_or_search_dimension(self):
        recovery = self.overlay["recovery_amendment"]
        self.assertEqual(recovery["only_code_change_allowed"], "open_the_fully_written_temporary_torch_file_as_rb_plus_instead_of_rb_before_fsync")
        self.assertFalse(recovery["science_candidate_alpha_threshold_c00_fold_source_geometry_and_gates_changed"])
        self.assertEqual(recovery["alpha"], 1.3)
        self.assertEqual(recovery["candidate_count"], 1)
        self.assertEqual(recovery["new_training_optimizer_steps"], 0)
        self.assertFalse(recovery["alpha_grid_module_projection_threshold_or_c00_search_allowed"])
        self.assertEqual(self.protocol["alpha130_contract"]["fixed_candidate"]["alpha"], 1.3)
        self.assertEqual(self.protocol["evaluation"]["prediction_threshold"], 0.719)

    def test_v2_output_plan_is_disjoint_from_failed_v1(self):
        specs = runner.synthesis_specs(self.protocol)
        self.assertEqual(len(specs), 3)
        for spec in specs:
            self.assertIn("alpha130_h2_grouped_oof_v2", spec["output"])
            self.assertNotIn("alpha130_h2_grouped_oof_v1", spec["output"])

    def test_real_cpu_geometry_preflight_still_passes(self):
        result = runner.task_arithmetic_preflight(self.protocol)
        self.assertTrue(result["passed"])
        self.assertTrue(result["cuda_not_initialized"])
        self.assertTrue(result["all11_geometry"]["passed"])
        self.assertTrue(all(result["all11_geometry"]["checks"].values()))

    def test_repaired_torch_publisher_is_exclusive_and_reloadable(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            payload = {"value": torch.tensor([1.0], dtype=torch.float32)}
            runner._atomic_torch_save_exclusive(payload, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(loaded["value"], payload["value"]))
            with self.assertRaises(FileExistsError):
                runner._atomic_torch_save_exclusive(payload, path)

    def test_audit_payload_discloses_recovery_and_binds_v2_identities(self):
        payload = runner.command_audit_payload(
            self.protocol,
            self.protocol_sha,
            {},
            {"passed": True, "cuda_not_initialized": True},
        )
        self.assertEqual(payload["protocol_sha256"], runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(payload["runner_sha256"], runner.core.sha256_file(runner.RUNNER_PATH))
        self.assertEqual(payload["tests_sha256"], runner.core.sha256_file(runner.TEST_PATH))
        self.assertTrue(payload["cpu_publish_recovery_v2"]["science_definition_unchanged"])
        self.assertEqual(payload["cpu_publish_recovery_v2"]["only_execution_change"], "temporary_torch_fsync_handle_rb_to_rb_plus")

    def test_cli_remains_exactly_five_nontraining_commands(self):
        parser = runner.build_parser()
        action = next(value for value in parser._actions if value.__class__.__name__ == "_SubParsersAction")
        self.assertEqual(set(action.choices), {"audit", "synthesize", "evaluate", "report", "all-evaluate-report"})


if __name__ == "__main__":
    unittest.main()
