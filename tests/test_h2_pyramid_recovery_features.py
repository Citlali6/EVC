import inspect
import unittest

import numpy as np
import torch

from utils.h2_pyramid_recovery_features import (
    EVENT_FEATURE_DIM,
    EVENT_FEATURE_NAMES,
    MAX_COMPONENTS_PER_BATCH,
    MAX_TEMPORAL_NODES,
    NODE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    RELATIVE_FEATURE_SLICE,
    assemble_component_time_nodes,
    assemble_event_feature_rows,
    build_component_node_features,
    collate_component_nodes,
)


def synthetic_inputs():
    event_count = 8
    decoder = np.arange(event_count * 16, dtype=np.float32).reshape(event_count, 16)
    scales = (
        np.arange(event_count * 4 * 16, dtype=np.float32).reshape(event_count, 4, 16)
        / np.float32(10.0)
    )
    base = np.linspace(-1.0, 1.0, event_count, dtype=np.float32)
    stage1 = base + np.linspace(-0.3, 0.4, event_count, dtype=np.float32)
    centre = np.arange(event_count * 3, dtype=np.float32).reshape(event_count, 3) / 7.0
    retained = np.asarray([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.bool_)
    locations = np.asarray(
        [
            [0, 10, 20, 2],
            [0, 12, 22, 48],
            [0, 14, 23, 51],
            [0, 15, 24, 80],
            [0, 17, 26, 104],
            [0, 18, 27, 129],
            [0, 40, 50, 12],
            [0, 41, 52, 16],
        ],
        dtype=np.int64,
    )
    components = (
        np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
        np.asarray([6, 7], dtype=np.int64),
    )
    return decoder, scales, base, stage1, centre, retained, locations, components


class RecoveryFeatureAssemblyTests(unittest.TestCase):
    def test_exact_96_dimensional_contract_and_padded_collation(self):
        values = synthetic_inputs()
        decoder, scales, base, stage1, centre, retained, locations, components = values
        event_rows = assemble_event_feature_rows(
            decoder, scales, base, stage1, centre
        )
        self.assertEqual(event_rows.shape, (8, EVENT_FEATURE_DIM))
        self.assertEqual(EVENT_FEATURE_DIM, 86)
        self.assertEqual(NODE_FEATURE_DIM, 96)
        self.assertEqual(len(EVENT_FEATURE_NAMES), 86)
        self.assertEqual(len(NODE_FEATURE_NAMES), 96)
        np.testing.assert_array_equal(event_rows[:, :16], decoder)
        np.testing.assert_allclose(event_rows[:, 16:80], scales.reshape(8, 64))
        np.testing.assert_allclose(event_rows[:, 80], base)
        np.testing.assert_allclose(event_rows[:, 81], stage1)
        np.testing.assert_allclose(event_rows[:, 82], stage1 - base)
        np.testing.assert_allclose(event_rows[:, 83:86], centre)

        nodes = assemble_component_time_nodes(
            event_rows, locations, components, retained
        )
        self.assertEqual(tuple(value.shape for value in nodes), ((3, 96), (1, 96)))
        np.testing.assert_allclose(nodes[0][0, :86], event_rows[[0, 1]].mean(axis=0))
        geometry = nodes[0][:, RELATIVE_FEATURE_SLICE]
        np.testing.assert_allclose(geometry[:, 0], np.log1p([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(geometry[:, 1], [1.0 / 3.0] * 3)
        np.testing.assert_allclose(geometry[:, 2], [-0.5, 0.0, 0.5])
        np.testing.assert_allclose(geometry[:, 9], [0.5, 1.0, 0.5])

        padded, mask = collate_component_nodes(nodes)
        self.assertEqual(tuple(padded.shape), (2, 3, 96))
        self.assertEqual(tuple(mask.shape), (2, 3))
        self.assertEqual(padded.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(padded.device.type, "cpu")
        self.assertEqual(mask.device.type, "cpu")
        self.assertTrue(torch.equal(mask, torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)))
        self.assertTrue(torch.equal(padded[0], torch.from_numpy(nodes[0])))
        self.assertTrue(torch.equal(padded[1, 0], torch.from_numpy(nodes[1][0])))
        self.assertTrue(torch.equal(padded[1, 1:], torch.zeros(2, 96)))

    def test_permutation_within_nodes_is_bitwise_invariant(self):
        values = synthetic_inputs()
        decoder, scales, base, stage1, centre, retained, locations, components = values
        first = build_component_node_features(
            decoder, scales, base, stage1, centre, retained, locations, components
        )
        permuted = (
            np.asarray([5, 3, 4, 1, 0, 2], dtype=np.int64),
            np.asarray([7, 6], dtype=np.int64),
        )
        second = build_component_node_features(
            decoder, scales, base, stage1, centre, retained, locations, permuted
        )
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            self.assertTrue(np.array_equal(left, right))

    def test_relative_time_is_bitwise_translation_invariant(self):
        values = synthetic_inputs()
        decoder, scales, base, stage1, centre, retained, locations, components = values
        first = build_component_node_features(
            decoder, scales, base, stage1, centre, retained, locations, components
        )
        translated = locations.copy()
        translated[:, 3] += 37 * 50
        second = build_component_node_features(
            decoder, scales, base, stage1, centre, retained, translated, components
        )
        for left, right in zip(first, second):
            self.assertTrue(np.array_equal(left, right))

    def test_cpu_tensor_inputs_match_numpy_inputs(self):
        values = synthetic_inputs()
        numpy_result = build_component_node_features(*values)
        tensor_values = tuple(
            tuple(torch.from_numpy(row) for row in value)
            if isinstance(value, tuple)
            else torch.from_numpy(value)
            for value in values
        )
        tensor_result = build_component_node_features(*tensor_values)
        for left, right in zip(numpy_result, tensor_result):
            self.assertTrue(np.array_equal(left, right))

    def test_inference_interfaces_accept_no_provenance_or_truth(self):
        forbidden = {
            "source",
            "name",
            "path",
            "hash",
            "index",
            "indices",
            "fold",
            "label",
            "labels",
            "target_id",
            "target_ids",
        }
        inference_functions = (
            assemble_event_feature_rows,
            assemble_component_time_nodes,
            build_component_node_features,
            collate_component_nodes,
        )
        for function in inference_functions:
            parameters = set(inspect.signature(function).parameters)
            self.assertFalse(parameters & forbidden, (function.__name__, parameters))

    def test_frozen_resource_bounds_are_enforced(self):
        long_sequence = np.zeros((MAX_TEMPORAL_NODES + 1, 96), dtype=np.float32)
        with self.assertRaises(ValueError):
            collate_component_nodes((long_sequence,))
        one_node = np.zeros((1, 96), dtype=np.float32)
        with self.assertRaises(ValueError):
            collate_component_nodes(tuple(one_node for _ in range(MAX_COMPONENTS_PER_BATCH + 1)))


if __name__ == "__main__":
    unittest.main()
