import inspect
import unittest

import numpy as np
import torch

from model.h2_pyramid_component_recovery import (
    H2PyramidComponentRecoveryHead,
    NODE_FEATURE_DIM,
    component_recovery_parameter_count,
)
from utils.h2_pyramid_component_recovery import (
    BreakpointRecord,
    exact_risk_controlled_breakpoint,
    restore_whole_components_bitwise,
)


class AtomicRecoveryTests(unittest.TestCase):
    def test_only_complete_components_change_and_bits_match_m20(self):
        pyramid = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        m20 = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
        components = (np.asarray([0, 2]), np.asarray([3, 5]))
        output = restore_whole_components_bitwise(
            pyramid, m20, components, np.asarray([True, False])
        )
        self.assertTrue(np.array_equal(output[[0, 2]], m20[[0, 2]]))
        self.assertTrue(np.array_equal(output[[1, 3, 4, 5]], pyramid[[1, 3, 4, 5]]))

    def test_overlap_and_event_level_decisions_are_rejected(self):
        pyramid = np.zeros(5, dtype=np.float32)
        m20 = np.ones(5, dtype=np.float32)
        with self.assertRaises(ValueError):
            restore_whole_components_bitwise(
                pyramid,
                m20,
                (np.asarray([0, 1]), np.asarray([1, 2])),
                [True, False],
            )
        with self.assertRaises(ValueError):
            restore_whole_components_bitwise(
                pyramid, m20, (np.asarray([0, 1]),), [True, False]
            )

    def test_exact_breakpoint_is_risk_controlled_without_grid(self):
        records = (
            BreakpointRecord(0.9, 0.021, 101, 100, 10, 10),
            BreakpointRecord(0.7, 0.025, 99, 100, 11, 10),
            BreakpointRecord(0.8, 0.023, 102, 100, 11, 10),
        )
        selected = exact_risk_controlled_breakpoint(records)
        self.assertEqual(selected, records[2])


class RecoveryHeadTests(unittest.TestCase):
    def test_forward_is_finite_and_padding_mask_is_respected(self):
        torch.manual_seed(5)
        model = H2PyramidComponentRecoveryHead()
        features = torch.randn(2, 5, NODE_FEATURE_DIM)
        mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        )
        first = model(features, mask)
        changed_padding = features.clone()
        changed_padding[0, 3:] = 10000.0
        second = model(changed_padding, mask)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first[0], second[0]))

    def test_forward_interface_has_no_provenance_or_truth_argument(self):
        parameters = set(inspect.signature(H2PyramidComponentRecoveryHead.forward).parameters)
        forbidden = {"source", "source_name", "path", "fold", "label", "labels", "target_id"}
        self.assertFalse(parameters & forbidden)

    def test_parameter_budget_is_small(self):
        count = component_recovery_parameter_count(H2PyramidComponentRecoveryHead())
        self.assertGreater(count, 0)
        self.assertLess(count, 20000)


if __name__ == "__main__":
    unittest.main()
