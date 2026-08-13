import inspect
import json
from pathlib import Path
import unittest

import numpy as np

import run_allsize_deletion_head_oof as base
import run_frame_set_graph_deletion_h2_smoke as runner
from utils import frame_set_graph_deletion_head as graph


ROOT = Path(__file__).resolve().parents[1]


class FrameSetGraphFeatureTests(unittest.TestCase):
    def synthetic(self):
        scores = np.asarray([0.80, 0.90, 0.76, 0.82, 0.84, 0.78, 0.81], dtype=np.float32)
        locations = np.asarray(
            [
                [0, 10, 10, 10],
                [0, 11, 10, 20],
                [0, 30, 30, 25],
                [0, 11, 10, 60],
                [0, 12, 10, 70],
                [0, 13, 10, 115],
                [0, 60, 60, 118],
            ],
            dtype=np.int64,
        )
        components = (
            np.asarray([0, 1]),
            np.asarray([2]),
            np.asarray([3, 4]),
            np.asarray([5]),
            np.asarray([6]),
        )
        return scores, locations, components

    def test_feature_shape_finite_and_frame_partition(self):
        scores, locations, components = self.synthetic()
        batch = graph.extract_frame_set_graph_features(
            scores, locations, components, scores.size
        )
        self.assertEqual(batch.frame_bins.tolist(), [0, 1, 2])
        self.assertEqual(batch.features.shape, (3, len(graph.FRAME_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(batch.features).all())
        rows = np.concatenate(batch.component_rows)
        self.assertEqual(sorted(rows.tolist()), list(range(len(components))))

    def test_prediction_is_joint_within_each_frame(self):
        scores, locations, components = self.synthetic()
        batch = graph.extract_frame_set_graph_features(
            scores, locations, components, scores.size
        )
        component_values = graph.broadcast_frame_probabilities(
            batch, np.asarray([0.1, 0.6, 0.9]), len(components)
        )
        self.assertEqual(component_values.tolist(), [0.1, 0.1, 0.6, 0.9, 0.9])

    def test_feature_api_has_no_label_source_or_target_input(self):
        parameters = set(inspect.signature(graph.extract_frame_set_graph_features).parameters)
        self.assertFalse(parameters & {"labels", "target_ids", "source_name", "path", "fold"})

    def test_graph_queries_are_bounded(self):
        scores, locations, components = self.synthetic()
        original = graph.cKDTree
        observed_k = []

        class SpyTree:
            def __init__(self, values):
                self.inner = original(values)

            def query(self, *args, **kwargs):
                observed_k.append(int(kwargs.get("k", 1)))
                return self.inner.query(*args, **kwargs)

        graph.cKDTree = SpyTree
        try:
            graph.extract_frame_set_graph_features(
                scores, locations, components, scores.size
            )
        finally:
            graph.cKDTree = original
        self.assertTrue(observed_k)
        self.assertLessEqual(max(observed_k), 2)


class FrameSetGraphRunnerTests(unittest.TestCase):
    def test_protocol_and_runner_hash_agree(self):
        path = ROOT / "protocols" / "frame_set_graph_deletion_f5_smoke_v1.json"
        self.assertEqual(runner.sha256_file(path), runner.EXPECTED_PROTOCOL_SHA256)
        protocol = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["feature_names"], list(graph.FRAME_FEATURE_NAMES))

    def test_five_contiguous_families_are_disjoint_and_complete(self):
        groups = base.SOURCE_GROUPS
        values = [value for members in groups.values() for value in members]
        self.assertEqual(len(values), 54)
        self.assertEqual(len(set(values)), 54)
        self.assertEqual(len(groups[runner.OUTER_HELD_GROUP]), 11)
        self.assertEqual(
            groups["block_059_074"],
            tuple(f"train_{index:03d}.npz" for index in list(range(59, 66)) + list(range(67, 75))),
        )

    def test_official_safety_gates_accept_only_nonharmful_counts(self):
        baseline = base.crossfit.SufficientCounts(
            true_positive_events=100,
            false_positive_events=20,
            false_negative_events=10,
            correct_objects=9,
            object_count=10,
            false_components=10,
            frame_count=10,
            event_count=200,
        )
        candidate = base.crossfit.SufficientCounts(
            true_positive_events=100,
            false_positive_events=19,
            false_negative_events=10,
            correct_objects=9,
            object_count=10,
            false_components=9,
            frame_count=10,
            event_count=200,
        )
        self.assertTrue(all(runner._gates(candidate, baseline).values()))
        harmful = base.crossfit.SufficientCounts(
            true_positive_events=99,
            false_positive_events=19,
            false_negative_events=11,
            correct_objects=8,
            object_count=10,
            false_components=9,
            frame_count=10,
            event_count=200,
        )
        self.assertFalse(all(runner._gates(harmful, baseline).values()))

    def test_cli_has_no_validation_test_or_gpu_input(self):
        destinations = {action.dest for action in runner.build_parser()._actions}
        self.assertFalse(destinations & {"validation", "val", "test", "gpu", "checkpoint"})


if __name__ == "__main__":
    unittest.main()
