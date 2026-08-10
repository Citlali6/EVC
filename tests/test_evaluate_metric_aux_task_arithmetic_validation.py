import ast
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_metric_aux_task_arithmetic_validation as runner


class WFullValidationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.science = json.loads(runner.SCIENCE_PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_science_protocol_hash_and_contract(self):
        self.assertEqual(
            runner.sha256_file(runner.SCIENCE_PROTOCOL_PATH),
            runner.EXPECTED_SCIENCE_PROTOCOL_SHA256,
        )
        self.assertIs(runner.validate_science_protocol(self.science), self.science)
        self.assertEqual(
            self.science["train_only_evidence"]["wfull_checkpoint"]["sha256"],
            runner.WFULL_CHECKPOINT_SHA256,
        )
        self.assertFalse(
            self.science["promotion_gates"]["materiality_report_only"][
                "included_in_safety_pass"
            ]
        )

    def test_route_boundaries_and_exact_minority_cutoff(self):
        import numpy as np

        self.assertEqual(runner.classify_wfull_route(np.zeros(30000), 160).domain, "low")
        self.assertEqual(runner.classify_wfull_route(np.zeros(30001), 160).domain, "middle")
        self.assertEqual(runner.classify_wfull_route(np.zeros(200001), 160).domain, "h1")
        exact = np.r_[np.ones(40001), np.zeros(160004)]
        decision = runner.classify_wfull_route(exact, 160)
        self.assertEqual(decision.polarity_minority_fraction, 0.20)
        self.assertEqual(decision.domain, "h2")
        self.assertEqual(decision.candidate_action, "infer_wfull_full_stream_t160")
        self.assertEqual(decision.prediction_threshold, 0.719)

    def test_route_rejects_non_160_bins_and_malformed_polarity(self):
        import numpy as np

        with self.assertRaises(ValueError):
            runner.classify_wfull_route(np.zeros(10), 159)
        with self.assertRaises(ValueError):
            runner.classify_wfull_route(np.zeros((2, 2)), 160)
        with self.assertRaises(ValueError):
            runner.classify_wfull_route(np.asarray([0.0, 2.0]), 160)

    def test_only_h2_calls_predictor_and_non_h2_keeps_same_object(self):
        sentinel = object()
        calls = []

        def predictor():
            calls.append("called")
            return "candidate"

        for domain in ("low", "middle", "h1"):
            value, preserved = runner.choose_candidate_scores({"domain": domain}, sentinel, predictor)
            self.assertIs(value, sentinel)
            self.assertTrue(preserved)
        value, preserved = runner.choose_candidate_scores({"domain": "h2"}, sentinel, predictor)
        self.assertEqual(value, "candidate")
        self.assertFalse(preserved)
        self.assertEqual(calls, ["called"])
        with self.assertRaises(ValueError):
            runner.choose_candidate_scores({"domain": "unknown"}, sentinel, predictor)

    def test_materiality_is_report_only_and_raw_tp_is_not_a_safety_gate(self):
        baseline_counts = dict(runner.GOLDEN_COUNTS)
        candidate_counts = dict(baseline_counts)
        candidate_counts["true_positive_events"] -= 1
        baseline_metrics = dict(runner.GOLDEN_METRICS)
        candidate_metrics = dict(baseline_metrics)
        candidate_metrics["score"] += 0.000001
        h2_counts = {
            "true_positive_events": 10,
            "false_positive_events": 1,
            "positive_events": 11,
            "detected_target_frames": 2,
            "target_frames": 2,
            "false_components": 1,
            "frame_count": 1,
        }
        h2_metrics = dict(baseline_metrics)
        gates, materiality = runner.promotion_gate_results(
            baseline_counts,
            baseline_metrics,
            candidate_counts,
            candidate_metrics,
            h2_counts,
            h2_metrics,
            dict(h2_counts),
            dict(h2_metrics),
            True,
            1,
            1,
            True,
        )
        self.assertTrue(all(gates.values()))
        self.assertFalse(materiality["met"])
        self.assertFalse(materiality["included_in_safety_pass"])
        self.assertNotIn("true_positive", " ".join(gates))

    def test_gpu_commands_require_explicit_authorization_and_no_train_cli(self):
        parsed = runner.parse_args(
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
            runner.runtime_preflight_execution("0" * 64, "1" * 64, False)
        with self.assertRaises(PermissionError):
            runner.run_execution("0" * 64, "1" * 64, "2" * 64, False)
        parser_source = inspect.getsource(runner.parse_args)
        self.assertNotIn('"train"', parser_source)
        self.assertNotIn('"evaluate"', parser_source)

    def test_full_t160_api_only_and_forbidden_modules_not_bound(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertNotIn("utils.temporal_memory_windowed_inference", imports)
        self.assertNotIn("utils.persistence_component_suppressor", imports)
        self.assertNotIn("predict_temporal_memory_scores_windowed(", source)
        self.assertNotIn("predict_persistence_component_keep_probabilities(", source)
        self.assertIn("predict_temporal_memory_scores(", source)
        self.assertNotIn("utils/temporal_memory_input_router.py", runner.CODE_PATHS)
        self.assertNotIn("utils/temporal_memory_windowed_inference.py", runner.CODE_PATHS)
        self.assertNotIn("utils/persistence_component_suppressor.py", runner.CODE_PATHS)

    def test_claim_is_required_before_deferred_io(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {"claim": Path(directory) / "claim.json"}
            with self.assertRaises(FileNotFoundError):
                runner._require_claim_before_deferred(
                    paths, "1" * 64, "2" * 64, "3" * 64
                )
            claim, claim_sha = runner._atomic_claim(
                paths["claim"], "1" * 64, "2" * 64, "3" * 64
            )
            loaded, loaded_sha = runner._require_claim_before_deferred(
                paths, "1" * 64, "2" * 64, "3" * 64
            )
            self.assertEqual(loaded, claim)
            self.assertEqual(loaded_sha, claim_sha)
            with self.assertRaises(FileExistsError):
                runner._atomic_claim(paths["claim"], "1" * 64, "2" * 64, "3" * 64)

    def test_run_claim_precedes_first_claimed_phase(self):
        run_source = inspect.getsource(runner.run_execution)
        self.assertLess(run_source.index("_atomic_claim("), run_source.index("_run_claimed("))
        claimed_source = inspect.getsource(runner._run_claimed)
        self.assertLess(
            claimed_source.index("_require_claim_before_deferred("),
            claimed_source.index("_validate_validation_files_after_claim("),
        )

    def test_execution_build_binds_deferred_flags_without_reading_them(self):
        published, published_path, _ = runner._load_published_validation_contract(self.science)
        protocol = runner.build_execution_protocol(
            self.science,
            runner.EXPECTED_SCIENCE_PROTOCOL_SHA256,
            published,
            published_path,
            runner._code_sha256(),
            {"head": "0" * 40, "clean": True, "status_sha256": "0" * 64},
        )
        runner.validate_execution_protocol(protocol)
        for name, spec in protocol["inputs"].items():
            self.assertEqual(
                spec["deferred_until_after_claim"], name in runner.DEFERRED_INPUT_NAMES
            )
        self.assertEqual(protocol["inference"]["mode"], "full_stream_t160")
        self.assertIsNone(protocol["inference"]["window_length"])
        self.assertIsNone(protocol["inference"]["stride"])

    def test_train_only_evidence_and_wfull_hash_chain(self):
        published, published_path, _ = runner._load_published_validation_contract(self.science)
        paths = runner._canonical_inputs(self.science, published, published_path)
        hashes = runner._expected_input_sha256(self.science, published)
        result = runner._validate_train_only_evidence(self.science, paths, hashes)
        self.assertTrue(result["v5_passed"])
        self.assertTrue(result["all11_pair_passed"])
        self.assertTrue(result["all11_manifest_passed"])
        self.assertEqual(
            runner.sha256_file(paths["wfull_checkpoint"]), runner.WFULL_CHECKPOINT_SHA256
        )

    def test_cpu_wfull_strict_load_and_c00_hash_without_cuda(self):
        import torch

        self.assertFalse(torch.cuda.is_initialized())
        published, published_path, _ = runner._load_published_validation_contract(self.science)
        paths = runner._canonical_inputs(self.science, published, published_path)
        audit = runner._strict_cpu_wfull_load(paths["wfull_checkpoint"])
        self.assertTrue(audit["strict_load_passed"])
        self.assertEqual(audit["tensor_count"], 89)
        self.assertEqual(audit["parameter_count"], 1_924_716)
        self.assertEqual(runner._effective_c00_sha256(), runner.EXPECTED_EFFECTIVE_C00_SHA256)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
