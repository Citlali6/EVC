import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


EVC_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = EVC_ROOT / "run_metric_aux_task_arithmetic_all11_final_refit.py"
SPEC = importlib.util.spec_from_file_location("_test_all11_final_refit", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class All11FinalRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = runner.load_protocol()

    def test_protocol_hash_status_and_access_contract(self):
        self.assertEqual(self.protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        overlay = self.protocol["all11_final_refit_contract"]
        self.assertEqual(overlay["schema"], runner.EXPECTED_SCHEMA)
        self.assertFalse(overlay["validation_or_test_read_allowed"])
        self.assertFalse(overlay["t32_allowed"])
        self.assertEqual(
            overlay["cli_contract"]["allowed_commands"],
            ["audit", "train", "audit-training", "synthesize"],
        )

    def test_v5_pass_is_bound_before_all11(self):
        evidence = self.protocol["_all11_bound_evidence"]["v5"]["report"]
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(
            evidence["decision"],
            "eligible_for_preregistered_all11_paired_fit_before_any_validation",
        )
        self.assertFalse(evidence["released_m20_anchor_reinference"])
        self.assertTrue(all(evidence["promotion_gates"].values()))

    def test_v3_resource_probe_is_reuse_only_and_passed(self):
        probe = self.protocol["_all11_bound_evidence"]["v3_probe"]
        self.assertTrue(probe["passed"])
        self.assertTrue(all(probe["checks"].values()))
        self.assertEqual(probe["new_pair_training_optimizer_steps"], 0)
        overlay = self.protocol["all11_final_refit_contract"]
        self.assertTrue(overlay["resource_feasibility_evidence"]["reuse_only_no_new_probe"])

    def test_all11_source_order_and_pair_budget(self):
        names = [item["name"] for item in runner.all11_items(self.protocol)]
        self.assertEqual(names, ["train_{:03d}.npz".format(i) for i in range(88, 99)])
        view = {"root": str(Path("dummy_all11_view").resolve()), "records": []}
        specs = runner.training_specs(self.protocol, view)
        self.assertEqual([item["variant"] for item in specs], ["baseline", "metric_aux"])
        self.assertEqual(sum(item["expected_optimizer_steps"] for item in specs), 528)
        for spec in specs:
            self.assertEqual(spec["expected_videos"], 11)
            self.assertEqual(spec["expected_sequences_per_epoch"], 88)
            self.assertEqual(spec["expected_optimizer_steps"], 264)
            self.assertEqual(spec["epochs"], 3)
            self.assertIsNone(spec["held_group"])
            self.assertEqual(spec["expected_source_names"], names)

    def test_pair_command_diff_is_exactly_two_paths(self):
        view = {"root": str(Path("dummy_all11_view").resolve()), "records": []}
        specs = runner.training_specs(self.protocol, view)
        commands = {}
        for spec in specs:
            _, overrides = runner.core.training_argv(
                self.protocol,
                spec["data_root"],
                spec["model_root"],
                spec["variant"],
                spec["epochs"],
            )
            commands[spec["run_id"]] = {"overrides": overrides}
        differences = runner._pair_command_diff(self.protocol, specs, commands)
        self.assertEqual(
            set(differences),
            {"TRAIN.model_save_root", "TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled"},
        )
        self.assertEqual(
            differences["TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled"],
            {"baseline": "false", "metric_aux": "true"},
        )

    def test_frozen_training_hyperparameters_and_scope(self):
        frozen = self.protocol["all11_final_refit_contract"]["paired_refit_contract"]
        self.assertEqual(
            (
                frozen["metric_target_weight"],
                frozen["metric_component_weight"],
                frozen["metric_warmup_epochs"],
                frozen["metric_spatial_cell_size"],
                frozen["metric_min_cell_events"],
                frozen["metric_component_ratio"],
                frozen["metric_activation_threshold"],
                frozen["metric_activation_temperature"],
            ),
            (0.005, 0.001, 1, 3, 2, 0.01, 0.719, 0.1),
        )
        self.assertEqual(frozen["seed"], 49)
        self.assertEqual(frozen["epochs"], 3)
        self.assertEqual(frozen["views_per_video"], 8)
        self.assertEqual(frozen["trainable_state_tensor_count"], 89)
        self.assertEqual(frozen["trainable_parameter_count"], 1924716)
        self.assertEqual(frozen["frozen_parameter_count"], 0)

    def test_alpha_zero_and_alpha_one_formula(self):
        import torch

        parent = {"a": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        baseline = {"a": torch.tensor([0.75, 2.5], dtype=torch.float32)}
        metric_aux = {"a": torch.tensor([0.5, 3.0], dtype=torch.float32)}
        alpha_zero = runner._V5_SYNTHESIZE_STATE_DICT(parent, baseline, metric_aux, 0.0)
        alpha_one = runner._V5_SYNTHESIZE_STATE_DICT(parent, baseline, metric_aux, 1.0)
        self.assertTrue(runner._V5_STATE_EQUAL(alpha_zero, parent))
        expected = (
            parent["a"].to(torch.float64)
            + (metric_aux["a"].to(torch.float64) - baseline["a"].to(torch.float64))
        ).to(torch.float32)
        self.assertTrue(torch.equal(alpha_one["a"], expected))

    def test_model_difference_is_finite_and_nonzero(self):
        import torch

        left = {"model_state_dict": {"a": torch.tensor([1.0, 2.0])}}
        right = {"model_state_dict": {"a": torch.tensor([1.0, 1.5])}}
        stats = runner._model_difference(left, right)
        self.assertTrue(stats["finite"])
        self.assertEqual(stats["changed_elements"], 1)
        self.assertAlmostEqual(stats["global_l2"], 0.5)

    def test_cli_has_no_probe_evaluate_report_or_search(self):
        parser = runner.build_parser()
        for forbidden in ("probe", "evaluate", "report", "validation", "search"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([forbidden])
        with self.assertRaises(PermissionError):
            runner.run_training(run_id="all11_baseline", authorized=False)

    def test_json_writer_refuses_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            runner.core.write_new_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                runner.core.write_new_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})


if __name__ == "__main__":
    unittest.main()
