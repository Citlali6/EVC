import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_metric_aux_task_arithmetic_validation_v2 as v2


class WFullValidationRecoveryV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.science, cls.science_sha = v2.load_effective_science_protocol()
        cls.base, _ = v2._v1_load_json_snapshot(
            v2.V1_SCIENCE_PATH, v2.V1_SCIENCE_SHA256, "V1 science"
        )

    def test_overlay_hash_schema_and_governance(self):
        self.assertEqual(v2.sha256_file(v2.V2_SCIENCE_PATH), v2.V2_SCIENCE_SHA256)
        self.assertEqual(self.science_sha, v2.V2_SCIENCE_SHA256)
        self.assertEqual(self.science["schema"], v2.EFFECTIVE_SCIENCE_SCHEMA)
        governance = self.science["governance_exception"]
        self.assertTrue(governance["explicitly_approved_by_root"])
        self.assertTrue(governance["v1_attempt_remains_failed_and_consumed"])
        self.assertTrue(governance["v2_is_not_a_v1_resume_or_retry_under_the_v1_contract"])
        self.assertTrue(governance["v2_is_a_new_second_adaptive_attempt_after_an_implementation_failure"])
        self.assertFalse(governance["candidate_performance_observed_before_v2_definition"])
        self.assertFalse(governance["formal_validation_wfull_inference_completed_in_v1"])
        self.assertFalse(governance["independent_held_or_unbiased_claim_allowed"])

    def test_scientific_candidate_route_threshold_c00_and_gates_are_unchanged(self):
        overlay = self.science["recovery_overlay"]
        for name in overlay["inherited_scientific_fields_required_bitwise_equal_to_v1"]:
            self.assertEqual(self.science[name], self.base[name], name)
        for name, value in self.base["train_only_evidence"].items():
            self.assertEqual(self.science["train_only_evidence"][name], value, name)
        self.assertEqual(self.science["candidate_id"], self.base["candidate_id"])
        self.assertEqual(self.science["inference"]["prediction_threshold"], 0.719)
        self.assertEqual(
            self.science["postprocess"]["effective_c00_canonical_sha256"],
            v2.core.EXPECTED_EFFECTIVE_C00_SHA256,
        )

    def test_v1_failure_chain_is_fully_bound_and_cache_absent(self):
        published, published_path, _ = v2._load_published_validation_contract(self.science)
        paths = v2._canonical_inputs(self.science, published, published_path)
        expected = v2._expected_input_sha256(self.science, published)
        result = v2._validate_train_only_evidence(self.science, paths, expected)
        chain = result["v1_failure_chain"]
        self.assertTrue(chain["passed"])
        self.assertTrue(chain["v1_h2_cache_absent"])
        self.assertFalse(chain["formal_validation_candidate_inference_completed"])
        self.assertFalse(chain["candidate_performance_observed"])
        self.assertEqual(chain["v1_claim_sha256"], v2.V1_FAILURE_SHA256["v1_claim"])
        self.assertEqual(
            chain["v1_failure_report_sha256"],
            v2.V1_FAILURE_SHA256["v1_failure_report"],
        )

    def test_top_level_json_load_and_explicit_count_mapping(self):
        raw_counts = {
            "event_true_positives": 63981,
            "event_false_positives": 2396,
            "ground_truth_positive_events": 65506,
            "evaluator_detected_objects": 4649,
            "evaluator_objects": 4762,
            "evaluator_false_components": 1584,
            "evaluator_frames": 3752,
            "event_false_negatives": 1525,
            "events": 1424330,
            "videos": 24,
        }
        payload = {"counts": raw_counts, "metrics": dict(v2.core.GOLDEN_METRICS)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden_fixture.json"
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            digest = v2.sha256_file(path)
            result = v2._validate_golden_report_after_claim(path, digest)
        self.assertEqual(result["counts"], v2.core.GOLDEN_COUNTS)
        self.assertEqual(result["metrics"], v2.core.GOLDEN_METRICS)
        self.assertEqual(
            result["parser"], "top_level_json_load_with_explicit_count_key_mapping_v2"
        )
        self.assertEqual(result["raw_counts_to_golden_counts"], v2.RAW_TO_GOLDEN_COUNT)

    def test_parser_fails_closed_on_count_or_metric_drift(self):
        raw_counts = {
            raw_name: v2.core.GOLDEN_COUNTS[golden_name]
            for raw_name, golden_name in v2.RAW_TO_GOLDEN_COUNT.items()
        }
        for changed_member in ("counts", "metrics"):
            payload = {"counts": dict(raw_counts), "metrics": dict(v2.core.GOLDEN_METRICS)}
            if changed_member == "counts":
                payload["counts"]["event_true_positives"] -= 1
            else:
                payload["metrics"]["score"] -= 0.001
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "drift.json"
                path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    v2._validate_golden_report_after_claim(path, v2.sha256_file(path))

    def test_execution_build_binds_v2_runner_core_and_v1_failure_inputs(self):
        published, published_path, _ = v2._load_published_validation_contract(self.science)
        protocol = v2.build_execution_protocol(
            self.science,
            self.science_sha,
            published,
            published_path,
            v2._code_sha256(),
            {"head": "0" * 40, "clean": True, "status_sha256": "0" * 64},
        )
        v2.validate_execution_protocol(protocol)
        self.assertEqual(protocol["schema"], v2.core.EXECUTION_SCHEMA)
        self.assertEqual(protocol["evidence_class"], v2.EVIDENCE_CLASS)
        self.assertEqual(protocol["science_protocol"]["sha256"], v2.V2_SCIENCE_SHA256)
        for name in v2.V1_FAILURE_INPUT_NAMES:
            self.assertIn(name, protocol["inputs"])
            self.assertFalse(protocol["inputs"][name]["deferred_until_after_claim"])
        self.assertIn(
            "evaluate_metric_aux_task_arithmetic_validation_v2.py",
            protocol["repository"]["code_sha256"],
        )
        self.assertIn(
            "evaluate_metric_aux_task_arithmetic_validation.py",
            protocol["repository"]["code_sha256"],
        )

    def test_v2_output_and_claim_paths_are_new_and_v1_is_untouched(self):
        paths = v2._paths()
        self.assertEqual(paths["claim"].parent, v2.V2_EXPERIMENT_DIRECTORY)
        self.assertEqual(paths["h2_cache"].parent, v2.V2_EXPERIMENT_DIRECTORY)
        self.assertNotIn("wfull_val24_v1", str(paths["claim"]))
        self.assertFalse(v2.V2_EXPERIMENT_DIRECTORY.exists())
        v1_dir = v2.WORKSPACE_ROOT / "experiments" / "20260811_metric_aux_task_arithmetic_wfull_val24_v1"
        self.assertTrue((v1_dir / "validation_attempt_claim.json").is_file())
        self.assertTrue((v1_dir / "frozen_validation_report.json").is_file())
        self.assertFalse((v1_dir / "raw_wfull_full_t160_h2_only.pt").exists())

    def test_runtime_functions_are_inherited_and_only_parser_is_overridden(self):
        self.assertIs(v2.classify_wfull_route, v2.core.classify_wfull_route)
        self.assertIs(v2.choose_candidate_scores, v2.core.choose_candidate_scores)
        self.assertIs(v2.promotion_gate_results, v2.core.promotion_gate_results)
        self.assertIs(v2.run_execution, v2.core.run_execution)
        parser_source = inspect.getsource(v2._validate_golden_report_after_claim)
        self.assertIn("_v1_load_json_snapshot", parser_source)
        self.assertIn("RAW_TO_GOLDEN_COUNT", parser_source)
        self.assertNotIn("_extract_json_member", parser_source)

    def test_cli_has_no_train_and_gpu_commands_remain_authorization_gated(self):
        parsed = v2.parse_args(
            [
                "runtime-preflight",
                "--expected-execution-protocol-sha256",
                "0" * 64,
                "--expected-cpu-preflight-receipt-sha256",
                "1" * 64,
            ]
        )
        self.assertFalse(parsed.authorized_by_root)
        with self.assertRaises(PermissionError):
            v2.runtime_preflight_execution("0" * 64, "1" * 64, False)
        with self.assertRaises(PermissionError):
            v2.run_execution("0" * 64, "1" * 64, "2" * 64, False)
        source = inspect.getsource(v2.core.parse_args)
        self.assertNotIn('"train"', source)

    def test_cpu_strict_load_and_c00_identity_remain_unchanged(self):
        import torch

        self.assertFalse(torch.cuda.is_initialized())
        published, published_path, _ = v2._load_published_validation_contract(self.science)
        paths = v2._canonical_inputs(self.science, published, published_path)
        audit = v2._strict_cpu_wfull_load(paths["wfull_checkpoint"])
        self.assertTrue(audit["strict_load_passed"])
        self.assertEqual(audit["tensor_count"], 89)
        self.assertEqual(audit["parameter_count"], 1_924_716)
        self.assertEqual(
            v2._effective_c00_sha256(), v2.core.EXPECTED_EFFECTIVE_C00_SHA256
        )
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
