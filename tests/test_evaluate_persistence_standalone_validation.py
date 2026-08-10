import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_persistence_standalone_validation as frozen


def _science():
    with frozen.SCIENCE_PROTOCOL_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


class PersistenceValidationProtocolTests(unittest.TestCase):
    def test_science_protocol_and_runtime_are_exact(self):
        science = frozen.validate_science_protocol(_science())
        self.assertEqual(science["runtime_contract"], frozen.EXPECTED_RUNTIME)
        self.assertTrue(science["promotion_gates"]["t32_not_read_or_combined"])
        self.assertEqual(
            science["candidate_chain"]["effective_c00_canonical_sha256"],
            frozen._effective_c00_sha256(),
        )
        self.assertFalse(frozen._runtime_identity()["cuda_initialized"])

    def test_science_tampering_fails_closed(self):
        science = _science()
        science["candidate_chain"]["h2_stage_order"][2] = "different topology"
        with self.assertRaisesRegex(ValueError, "stage chain"):
            frozen.validate_science_protocol(science)

        science = _science()
        science["runtime_contract"]["opencv_version"] = "different"
        with self.assertRaisesRegex(ValueError, "runtime contract"):
            frozen.validate_science_protocol(science)

        science = _science()
        science["promotion_gates"]["t32_not_read_or_combined"] = False
        with self.assertRaisesRegex(ValueError, "promotion gates"):
            frozen.validate_science_protocol(science)

    def test_execution_protocol_binds_evidence_preflight_code_and_cache_output(self):
        science = _science()
        protocol = frozen.build_execution_protocol(science, frozen._code_sha256())
        frozen.validate_execution_protocol(protocol)
        self.assertIn("crossfit_component_reranker.py", protocol["code_sha256"])
        self.assertIn("utils/density_threshold.py", protocol["code_sha256"])
        self.assertIn("h2_cache", protocol["outputs"])

        changed = copy.deepcopy(protocol)
        changed["evidence_class"] = "independent_held"
        with self.assertRaisesRegex(ValueError, "evidence class"):
            frozen.validate_execution_protocol(changed)

        changed = copy.deepcopy(protocol)
        changed["preflight_contract"]["validation_cache_read"] = True
        with self.assertRaisesRegex(ValueError, "preflight contract"):
            frozen.validate_execution_protocol(changed)


class PersistenceRoutingAndGateTests(unittest.TestCase):
    def test_full_stream_route_has_exact_boundaries_and_no_t32_metadata(self):
        low = frozen.classify_full_stream_route(30_000, np.zeros(30_000, dtype=np.uint8))
        middle = frozen.classify_full_stream_route(
            30_001, np.zeros(30_001, dtype=np.uint8)
        )
        h1 = frozen.classify_full_stream_route(
            200_001, np.zeros(200_001, dtype=np.uint8)
        )
        h2_polarities = np.zeros(200_010, dtype=np.uint8)
        h2_polarities[:40_002] = 1
        h2 = frozen.classify_full_stream_route(200_010, h2_polarities)
        self.assertEqual(
            [low["domain"], middle["domain"], h1["domain"], h2["domain"]],
            ["low", "middle", "h1", "h2"],
        )
        for route in (low, middle, h1, h2):
            self.assertEqual(route["mode"], "full_stream")
            self.assertFalse(route["t32_read_or_combined"])
            self.assertNotIn("window_length", route)

    def test_non_h2_identity_and_h2_single_call(self):
        scores = torch.tensor([0.1, 0.9], dtype=torch.float32)
        calls = 0

        def predictor():
            nonlocal calls
            calls += 1
            return scores.clone()

        output, preserved = frozen.choose_persistence_scores(
            {"eligible": False}, scores, predictor
        )
        self.assertTrue(preserved)
        self.assertIs(output, scores)
        self.assertTrue(torch.equal(output, scores))
        self.assertEqual(calls, 0)

        output, preserved = frozen.choose_persistence_scores(
            {"eligible": True}, scores, predictor
        )
        self.assertFalse(preserved)
        self.assertEqual(calls, 1)
        self.assertIsNot(output, scores)

    def test_perfect_candidate_has_all_positive_gates_and_score_is_strict(self):
        counts = dict(frozen.GOLDEN_COUNTS)
        boundary = dict(frozen.GOLDEN_METRICS)
        boundary["score"] = (
            frozen.GOLDEN_METRICS["score"] + frozen.MINIMUM_SCORE_DELTA
        )
        gates = frozen._gate_results(
            counts,
            dict(frozen.GOLDEN_METRICS),
            boundary,
            counts,
            True,
            3,
            3,
            True,
        )
        self.assertFalse(
            gates["candidate_score_strictly_greater_than_golden_plus_0p0001"]
        )

        passing = dict(boundary)
        passing["score"] += 1e-12
        gates = frozen._gate_results(
            counts,
            dict(frozen.GOLDEN_METRICS),
            passing,
            counts,
            True,
            3,
            3,
            True,
        )
        self.assertTrue(all(gates.values()))
        self.assertTrue(gates["t32_not_read_or_combined"])

        gates = frozen._gate_results(
            counts,
            dict(frozen.GOLDEN_METRICS),
            passing,
            counts,
            True,
            3,
            3,
            False,
        )
        self.assertFalse(
            gates["each_h2_zero_true_positive_and_correct_target_loss"]
        )


class PersistenceNoEarlyValidationAccessTests(unittest.TestCase):
    def test_freeze_and_preflight_never_hash_deferred_inputs(self):
        science = _science()
        deferred = {
            frozen._workspace_path(item["workspace_relative_path"], name)
            for name, item in science["deferred_validation_inputs"].items()
        }
        original_sha256_file = frozen.sha256_file

        def guarded_sha256(path, *args, **kwargs):
            resolved = Path(path).resolve()
            if resolved in deferred:
                raise AssertionError("deferred validation input was read")
            return original_sha256_file(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory).resolve() / "validation"
            with mock.patch.object(frozen, "EXPERIMENT_DIRECTORY", experiment), mock.patch.object(
                frozen, "sha256_file", side_effect=guarded_sha256
            ):
                frozen_result = frozen.freeze_execution_protocol()
                receipt = frozen.preflight_execution(frozen_result["sha256"])
                paths = frozen._paths()
                self.assertTrue(receipt["passed"])
                self.assertFalse(receipt["runtime"]["cuda_initialized"])
                self.assertGreater(
                    receipt["synthetic_smoke"]["h2_candidate_component_count"], 0
                )
                self.assertFalse(paths["claim"].exists())
                self.assertFalse(paths["h2_cache"].exists())
                self.assertFalse(paths["report"].exists())

                with paths["preflight_receipt"].open("r", encoding="utf-8") as stream:
                    tampered = json.load(stream)
                tampered["synthetic_smoke"]["h2_predictor_calls"] = 0
                with paths["preflight_receipt"].open("w", encoding="utf-8") as stream:
                    json.dump(tampered, stream, sort_keys=True)
                with self.assertRaisesRegex(ValueError, "receipt differs"):
                    frozen._load_preflight_receipt(frozen_result["sha256"])


if __name__ == "__main__":
    unittest.main()
