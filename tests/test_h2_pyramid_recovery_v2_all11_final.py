import inspect
import json
import unittest

import numpy as np

import run_h2_pyramid_component_recovery_v2_all11_final as final
from utils.h2_pyramid_component_recovery import restore_whole_components_bitwise
from utils.target_preserving_residual import use_h2_residual_refiner


class FinalAll11ProtocolTests(unittest.TestCase):
    def test_protocol_hash_and_source_order_are_frozen(self):
        self.assertEqual(final.sha256_file(final.PROTOCOL_PATH), final.EXPECTED_PROTOCOL_SHA256)
        protocol = json.loads(final.PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(tuple(protocol["data_scope"]["all11_sources_in_order"]), final.ALL11)
        self.assertFalse(protocol["data_scope"]["validation_read_allowed"])
        self.assertFalse(protocol["data_scope"]["test_read_allowed"])

    def test_stage_schedules_are_exact_and_group_oof_is_complete(self):
        self.assertEqual(final.STAGE1_STEPS, 2 * 4 * 11)
        self.assertEqual(sum(value[3] for value in final.OOF_FOLDS), 176)
        self.assertEqual(final.FINAL_HEAD_STEPS, 88)
        self.assertEqual(final.TOTAL_RECOVERY_STEPS, 264)
        coverage = tuple(source for _, _, held, _ in final.OOF_FOLDS for source in held)
        self.assertEqual(coverage, final.ALL11)
        self.assertEqual(len(set(coverage)), 11)
        for _, fit, held, steps in final.OOF_FOLDS:
            self.assertFalse(set(fit) & set(held))
            self.assertEqual(set(fit) | set(held), set(final.ALL11))
            self.assertEqual(steps, 8 * len(fit))

    def test_cli_has_no_scoring_or_hyperparameter_knobs(self):
        parser = final.build_parser()
        commands = next(
            action.choices
            for action in parser._actions
            if hasattr(action, "choices") and isinstance(action.choices, dict)
        )
        self.assertEqual(
            set(commands),
            {
                "cpu-audit",
                "train-stage1",
                "extract-all11-features",
                "train-stage2-and-freeze",
                "audit-final-package",
            },
        )
        exposed_options = {
            option
            for command in commands.values()
            for action in command._actions
            for option in action.option_strings
        }
        self.assertLessEqual(exposed_options, {"-h", "--help", final.GPU_FLAG})

    def test_gpu_entrypoints_accept_only_args_object(self):
        for function in (
            final.train_stage1,
            final.extract_all11_features,
            final.train_stage2_and_freeze,
        ):
            self.assertEqual(tuple(inspect.signature(function).parameters), ("args",))


class FinalAll11RuntimeContractTests(unittest.TestCase):
    def test_route_boundaries_are_exact(self):
        below = np.tile(np.asarray([0, 1], dtype=np.uint8), 100000)
        self.assertFalse(use_h2_residual_refiner(200000, below))
        above_balanced = np.tile(np.asarray([0, 1], dtype=np.uint8), 100001)[:200001]
        self.assertTrue(use_h2_residual_refiner(200001, above_balanced))
        minority_below = np.zeros(200001, dtype=np.uint8)
        minority_below[:39999] = 1
        self.assertFalse(use_h2_residual_refiner(200001, minority_below))

    def test_atomic_action_changes_only_complete_selected_components(self):
        stage1 = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        m20 = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        components = (np.asarray([0, 2]), np.asarray([1, 3]))
        output = restore_whole_components_bitwise(
            stage1, m20, components, np.asarray([True, False])
        )
        self.assertTrue(np.array_equal(output, np.asarray([0.9, 0.2, 0.7, 0.4, 0.5], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
