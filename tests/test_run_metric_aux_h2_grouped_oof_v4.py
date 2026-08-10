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
import run_metric_aux_h2_grouped_oof_v3 as public_v3

_PUBLIC_STATE = {
    "v1_protocol": public_v1.PROTOCOL_PATH,
    "v1_output": public_v1.OUTPUT_ROOT,
    "v1_file": Path(public_v1.__file__).resolve(),
    "v2_protocol": public_v2.PROTOCOL_PATH,
    "v2_output": public_v2.OUTPUT_ROOT,
    "v2_file": Path(public_v2.__file__).resolve(),
    "v3_protocol": public_v3.PROTOCOL_PATH,
    "v3_output": public_v3.OUTPUT_ROOT,
    "v3_file": Path(public_v3.__file__).resolve(),
    "v3_core_file": Path(public_v3.core.__file__).resolve(),
}

import run_metric_aux_h2_grouped_oof_v4 as runner


class MetricAuxGroupedOofEvaluationRecoveryV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = runner.load_protocol()
        cls.overlay, cls.overlay_sha = runner.core.load_json_snapshot(
            runner.PROTOCOL_PATH
        )
        cls.specs = runner.evaluation_specs(cls.protocol)

    def test_protocol_hash_status_and_eval_only_scope_are_frozen(self):
        self.assertEqual(self.protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(self.overlay_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(
            self.protocol["status"],
            "frozen_after_v3_probe_and_six_formal_training_before_any_v4_held_evaluation",
        )
        recovery = self.protocol["recovery_amendment_v4"]
        self.assertTrue(recovery["v3_evaluation_attempt_remains_failed"])
        self.assertTrue(recovery["reuse_all_six_v3_formal_training_runs_required"])
        self.assertFalse(recovery["new_probe_allowed"])
        self.assertFalse(recovery["new_training_allowed"])
        self.assertEqual(recovery["new_training_optimizer_steps"], 0)
        self.assertFalse(
            recovery[
                "scientific_candidate_training_evaluation_settings_folds_or_promotion_change"
            ]
        )

    def test_v3_failure_is_immutable_and_proves_zero_held_work(self):
        record = self.overlay["inheritance"]["v3_evaluation_failure_receipt"]
        failure, digest = runner.core.load_json_snapshot(
            runner.core.workspace_path(record["workspace_relative_path"])
        )
        self.assertEqual(digest, runner.V3_EVALUATION_FAILURE_SHA256)
        self.assertEqual(failure["status"], "failed")
        self.assertFalse(failure["passed"])
        disk = failure["disk_and_control_flow_evidence"]
        for key in (
            "held_source_load_count",
            "model_load_count",
            "full_stream_prediction_call_count",
            "score_tensor_count",
            "postprocess_call_count",
            "sufficient_count_call_count",
        ):
            self.assertEqual(disk[key], 0)

    def test_six_v3_formal_results_and_e3_checkpoints_are_exactly_bound(self):
        evidence, pair_sha = runner.require_v3_prerequisites(self.protocol)
        self.assertEqual(pair_sha, runner.V3_PAIR_AUDIT_SHA256)
        self.assertTrue(evidence["v3_probe"]["passed"])
        self.assertTrue(evidence["v3_pair_audit"]["passed"])
        self.assertEqual(set(evidence["formal_results"]), runner._expected_formal_run_ids())
        for run_id, result in evidence["formal_results"].items():
            frozen = self.protocol["v3_formal_training_evidence_v4"][run_id]
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["protocol_sha256"], runner.V3_PROTOCOL_SHA256)
            self.assertEqual(result["runner_sha256"], runner.V3_RUNNER_SHA256)
            self.assertEqual(
                result["checkpoints"]["e3"]["sha256"],
                frozen["e3_checkpoint"]["sha256"],
            )
            self.assertNotEqual(
                result["checkpoints"]["e3"]["sha256"],
                result["checkpoints"]["e1"]["sha256"],
            )

    def test_private_import_does_not_mutate_public_v1_v2_v3_modules(self):
        self.assertIsNot(runner.v3, public_v3)
        self.assertIsNot(runner.core, public_v3.core)
        self.assertEqual(public_v1.PROTOCOL_PATH, _PUBLIC_STATE["v1_protocol"])
        self.assertEqual(public_v1.OUTPUT_ROOT, _PUBLIC_STATE["v1_output"])
        self.assertEqual(Path(public_v1.__file__).resolve(), _PUBLIC_STATE["v1_file"])
        self.assertEqual(public_v2.PROTOCOL_PATH, _PUBLIC_STATE["v2_protocol"])
        self.assertEqual(public_v2.OUTPUT_ROOT, _PUBLIC_STATE["v2_output"])
        self.assertEqual(Path(public_v2.__file__).resolve(), _PUBLIC_STATE["v2_file"])
        self.assertEqual(public_v3.PROTOCOL_PATH, _PUBLIC_STATE["v3_protocol"])
        self.assertEqual(public_v3.OUTPUT_ROOT, _PUBLIC_STATE["v3_output"])
        self.assertEqual(Path(public_v3.__file__).resolve(), _PUBLIC_STATE["v3_file"])
        self.assertEqual(Path(public_v3.core.__file__).resolve(), _PUBLIC_STATE["v3_core_file"])

    def test_core_identity_and_future_outputs_are_patched_to_v4(self):
        self.assertEqual(runner.core.PROTOCOL_PATH, runner.PROTOCOL_PATH)
        self.assertEqual(runner.core.OUTPUT_ROOT, runner.OUTPUT_ROOT)
        self.assertEqual(runner.core.COMMAND_AUDIT_PATH, runner.COMMAND_AUDIT_PATH)
        self.assertEqual(runner.core.EVALUATION_ROOT, runner.EVALUATION_ROOT)
        self.assertEqual(runner.core.REPORT_PATH, runner.REPORT_PATH)
        self.assertEqual(
            runner.core.EXPECTED_PROTOCOL_SHA256, runner.EXPECTED_PROTOCOL_SHA256
        )
        self.assertEqual(Path(runner.core.__file__).resolve(), Path(runner.__file__).resolve())

    def test_evaluation_plan_is_exact_nine_and_uses_v3_e3_not_best(self):
        self.assertEqual(
            [spec["eval_id"] for spec in self.specs],
            [
                "hold_g1_released_m20",
                "hold_g1_baseline",
                "hold_g1_metric_aux",
                "hold_g2_released_m20",
                "hold_g2_baseline",
                "hold_g2_metric_aux",
                "hold_g3_released_m20",
                "hold_g3_baseline",
                "hold_g3_metric_aux",
            ],
        )
        by_id = {spec["eval_id"]: spec for spec in self.specs}
        for run_id in runner._expected_formal_run_ids():
            frozen = self.protocol["v3_formal_training_evidence_v4"][run_id]
            self.assertEqual(
                by_id[run_id]["checkpoint_sha256"],
                frozen["e3_checkpoint"]["sha256"],
            )
            self.assertIn("20260810_metric_aux_h2_grouped_oof_v4", by_id[run_id]["result_path"])
        for spec in self.specs:
            self.assertIn(
                "20260810_metric_aux_h2_grouped_oof_v4", spec["result_path"]
            )

    def test_api_surface_points_to_temporal_frame_module_and_cuda_stays_uninitialized(self):
        import torch
        import utils.temporal_memory_inference as memory_inference

        self.assertFalse(torch.cuda.is_initialized())
        module, helper = runner._verify_api_surface(self.protocol)
        self.assertIs(module, memory_inference)
        self.assertEqual(
            list(__import__("inspect").signature(helper).parameters),
            ["sample", "temporal_bin_size", "whole_t"],
        )
        self.assertFalse(hasattr(module, "temporal_frame_video_from_sample"))
        self.assertFalse(torch.cuda.is_initialized())

    def _fake_base_payload(self, spec):
        return {
            "schema": "ev-uav-metric-aux-held-train-evaluation-v1",
            "eval_id": spec["eval_id"],
            "fold_id": spec["fold_id"],
            "variant": spec["variant"],
            "dataset_split": "train",
            "t32_read_or_combined": False,
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "protocol_sha256": runner.EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": runner.core.sha256_file(Path(runner.__file__).resolve()),
            "records": [],
        }

    def test_temporary_helper_injection_is_visible_to_base_and_restored_on_success(self):
        spec = copy.deepcopy(self.specs[0])
        writes = []

        def sink(path, payload):
            writes.append((Path(path), copy.deepcopy(payload)))

        def fake_base(protocol, received_spec):
            import utils.temporal_frame_inference as frame_inference
            import utils.temporal_memory_inference as memory_inference

            self.assertIs(
                memory_inference.temporal_frame_video_from_sample,
                frame_inference.temporal_frame_video_from_sample,
            )
            payload = self._fake_base_payload(received_spec)
            runner.core.write_new_json(received_spec["result_path"], payload)
            return payload

        with mock.patch.object(runner, "_BASE_EVALUATE_SPEC", side_effect=fake_base), mock.patch.object(
            runner.core, "write_new_json", side_effect=sink
        ):
            payload = runner.evaluate_spec(self.protocol, spec)
        import utils.temporal_memory_inference as memory_inference

        self.assertFalse(hasattr(memory_inference, "temporal_frame_video_from_sample"))
        self.assertEqual(len(writes), 1)
        self.assertEqual(payload["evaluation_recovery_v4"], runner._evaluation_recovery_record())
        self.assertEqual(writes[0][1]["protocol_sha256"], runner.EXPECTED_PROTOCOL_SHA256)

    def test_temporary_helper_injection_is_restored_when_base_raises(self):
        with mock.patch.object(
            runner, "_BASE_EVALUATE_SPEC", side_effect=RuntimeError("synthetic base failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic base failure"):
                runner.evaluate_spec(self.protocol, copy.deepcopy(self.specs[0]))
        import utils.temporal_memory_inference as memory_inference

        self.assertFalse(hasattr(memory_inference, "temporal_frame_video_from_sample"))
        self.assertIs(runner.core.write_new_json, runner._BASE_WRITE_NEW_JSON)

    def test_cli_exposes_no_probe_or_training_route(self):
        parser = runner.build_parser()
        action = next(
            item for item in parser._actions if item.__class__.__name__ == "_SubParsersAction"
        )
        self.assertEqual(
            set(action.choices), {"audit", "evaluate", "report", "all-evaluate-report"}
        )
        for forbidden in ("probe", "train", "audit-training", "all-after-probe"):
            self.assertNotIn(forbidden, action.choices)

    def test_command_audit_payload_has_only_evaluation_commands_and_zero_training(self):
        payload = runner.command_audit_payload(
            self.protocol,
            self.protocol_sha,
            {"synthetic_asset_audit": True},
        )
        self.assertEqual(payload["evaluation_count"], 9)
        self.assertEqual(len(payload["evaluation_commands"]), 9)
        self.assertEqual(payload["new_probe_optimizer_steps"], 0)
        self.assertEqual(payload["new_training_optimizer_steps"], 0)
        self.assertNotIn("formal_commands", payload)
        self.assertNotIn("probe_commands", payload)
        self.assertFalse(payload["gpu_or_cuda_initialized"])

    def test_load_evaluation_result_requires_v4_recovery_provenance(self):
        spec = self.specs[0]
        payload = self._fake_base_payload(spec)
        with mock.patch.object(
            runner,
            "_BASE_LOAD_EVALUATION_RESULT",
            return_value=(payload, "0" * 64),
        ):
            with self.assertRaises(RuntimeError):
                runner.load_evaluation_result(spec)
        payload["evaluation_recovery_v4"] = runner._evaluation_recovery_record()
        with mock.patch.object(
            runner,
            "_BASE_LOAD_EVALUATION_RESULT",
            return_value=(payload, "0" * 64),
        ):
            loaded, _ = runner.load_evaluation_result(spec)
        self.assertEqual(loaded["evaluation_recovery_v4"], runner._evaluation_recovery_record())

    def test_inherited_double_anchor_gate_includes_each_fold_score(self):
        counts = {
            "true_positive_events": 10,
            "false_positive_events": 0,
            "false_negative_events": 0,
            "correct_objects": 1,
            "object_count": 1,
            "false_components": 0,
            "frame_count": 1,
            "event_count": 10,
        }
        metrics = {"score": 1.0, "pd": 1.0, "fa": 0.0, "iou": 1.0}
        gate = runner.core.comparator_gates(counts, metrics, counts, metrics, pooled=False)
        self.assertTrue(gate["checks"]["score_not_lower"])
        self.assertTrue(gate["checks"]["true_positive_events_not_lower"])
        self.assertTrue(gate["checks"]["correct_objects_not_lower"])
        self.assertTrue(gate["checks"]["pd_not_lower"])
        self.assertTrue(gate["checks"]["iou_not_lower"])
        self.assertTrue(gate["checks"]["fa_not_higher"])


if __name__ == "__main__":
    unittest.main()
