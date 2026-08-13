import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import run_middle_multiscale_temporal_summary_first_fold as runner


class MiddleFirstFoldFormalTests(unittest.TestCase):
    def test_frozen_hashes(self):
        self.assertEqual(runner.sha256_file(runner.SCIENCE_PATH), runner.EXPECTED_SCIENCE_SHA256)
        self.assertEqual(runner.sha256_file(runner.EXECUTION_PATH), runner.EXPECTED_EXECUTION_SHA256)
        self.assertEqual(runner.sha256_file(runner.CORE_PATH), runner.EXPECTED_CORE_SHA256)
        self.assertEqual(runner.sha256_file(runner.TRAIN_CACHE_MANIFEST_PATH), runner.EXPECTED_MANIFEST_SHA256)

    def test_first_fold_is_exact_and_disjoint(self):
        self.assertEqual(len(runner.FIT_SOURCES), 24)
        self.assertEqual(len(runner.HELD_SOURCES), 15)
        self.assertFalse(set(runner.FIT_SOURCES) & set(runner.HELD_SOURCES))
        self.assertEqual(runner.HELD_SOURCES, tuple("train_{:03d}.npz".format(i) for i in range(15)))
        self.assertEqual(len(runner.FIT_SOURCES) * 2 * 2, 96)

    def test_contract_and_manifest_population(self):
        science, contract = runner.load_frozen_contract()
        sources = contract["source_manifest"]["sources"]
        self.assertEqual(len(sources), 39)
        self.assertEqual(tuple(contract["scope"]["fit_sources"]), runner.FIT_SOURCES)
        self.assertEqual(tuple(contract["scope"]["held_sources"]), runner.HELD_SOURCES)
        self.assertTrue(all(runner.middle_route(value["event_count"]) for value in sources.values()))
        self.assertFalse(science["science_scope"]["validation_read_allowed"])

    def test_input_only_route_boundaries(self):
        self.assertFalse(runner.middle_route(30000))
        self.assertTrue(runner.middle_route(30001))
        self.assertTrue(runner.middle_route(200000))
        self.assertFalse(runner.middle_route(200001))

    def test_private_core_is_bound_to_middle_fold(self):
        self.assertEqual(runner.core.FIT_SOURCES, runner.FIT_SOURCES)
        self.assertEqual(runner.core.HELD_SOURCES, runner.HELD_SOURCES)
        self.assertEqual(runner.core.EXPECTED_STEPS, 96)
        self.assertEqual(runner.core.VIEWS_PER_SOURCE_PER_EPOCH, 2)
        self.assertIs(runner.core.load_input_and_truth, runner.load_middle_input_and_truth)
        self.assertIs(runner.core.safety_gates, runner.middle_safety_gates)

    def test_primary_gate_matches_frozen_policy(self):
        base = {"Score": .90, "IoU": .80, "Pd": .95, "Fa": 1e-5, "TP": 100, "FP": 20, "CO": 100, "FC": 10}
        candidate = {"Score": .91, "IoU": .81, "Pd": .949, "Fa": 9e-6, "TP": 98, "FP": 19, "CO": 100, "FC": 10}
        gates = runner.middle_safety_gates(base, candidate, require_effect_size=True)
        self.assertTrue(all(gates.values()))
        candidate["Score"] = .9049
        self.assertFalse(runner.middle_safety_gates(base, candidate, require_effect_size=True)["Score_gain_at_least_0_01"])

    def test_tp_zero_loss_is_not_a_gate(self):
        base = {"Score": .90, "IoU": .80, "Pd": .95, "Fa": 1e-5, "TP": 100, "FP": 30, "CO": 100, "FC": 15}
        candidate = {"Score": .92, "IoU": .82, "Pd": .949, "Fa": 8e-6, "TP": 99, "FP": 10, "CO": 100, "FC": 5}
        gates = runner.middle_safety_gates(base, candidate, require_effect_size=True)
        self.assertNotIn("TP_not_lower", gates)
        self.assertTrue(all(gates.values()))

    def test_cli_has_only_audit_train_and_evaluate(self):
        parser = runner.build_parser()
        action = next(value for value in parser._actions if value.dest == "command")
        self.assertEqual(set(action.choices), {"audit", "train-first-fold", "evaluate-first-fold"})
        for forbidden in ("train-all", "evaluate-all", "report", "validation", "test"):
            self.assertNotIn(forbidden, action.choices)

    def test_exclusive_json_and_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            digest = runner.write_json_sidecar_exclusive(path, {"passed": True})
            self.assertEqual(digest, runner.verify_sidecar(path))
            with self.assertRaises(FileExistsError):
                runner.write_json_sidecar_exclusive(path, {"passed": True})

    def test_outputs_are_pristine_before_formal(self):
        self.assertFalse(runner.TRAIN_OUTPUT_ROOT.exists())
        self.assertFalse(runner.EVALUATION_ROOT.exists())
        self.assertFalse(runner.MIDDLE_REPORT_PATH.exists())

    def test_cuda_not_initialized_by_cpu_contract_checks(self):
        self.assertFalse(runner.torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
