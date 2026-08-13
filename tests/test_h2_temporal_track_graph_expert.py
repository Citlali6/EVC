import unittest

import numpy as np
import torch

from model.h2_temporal_track_graph_expert import (
    TemporalTrackGraphExpert,
    balanced_graph_bce,
    graph_expert_parameter_count,
)
from utils.h2_temporal_track_graph import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    TRACK_FEATURE_NAMES,
    aggregate_track_node_features,
    atomic_delete_from_graph,
    derive_zero_observed_target_loss_cutoff,
    extract_temporal_track_graph,
    pure_false_positive_component_targets,
    pure_false_positive_track_targets,
    validate_inference_feature_contract,
)


def synthetic_video(with_decoder=False):
    # Two persistent objects plus input-only neighborhood events.  Locations
    # are [batch,x,y,t], and only the first eight events cross the threshold.
    locations = np.asarray(
        [
            [0, 1, 1, 0],
            [0, 2, 1, 0],
            [0, 10, 10, 0],
            [0, 10, 11, 0],
            [0, 2, 1, 50],
            [0, 3, 1, 50],
            [0, 11, 10, 50],
            [0, 11, 11, 50],
            [0, 5, 5, 0],
            [0, 6, 5, 50],
        ],
        dtype=np.int64,
    )
    scores = np.asarray(
        [0.91, 0.87, 0.80, 0.79, 0.93, 0.88, 0.81, 0.78, 0.1, 0.2],
        dtype=np.float32,
    )
    polarities = np.asarray([0, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32)
    decoder = None
    if with_decoder:
        decoder = np.arange(scores.size * 16, dtype=np.float32).reshape(scores.size, 16)
        decoder /= float(decoder.max())
    return scores, locations, polarities, decoder


class TemporalTrackGraphUtilityTests(unittest.TestCase):
    def test_feature_contract_and_determinism(self):
        self.assertTrue(validate_inference_feature_contract())
        scores, locations, polarities, _ = synthetic_video()
        first = extract_temporal_track_graph(
            scores, locations, polarities, 0.719, scores.size
        )
        second = extract_temporal_track_graph(
            scores, locations, polarities, 0.719, scores.size
        )
        self.assertEqual(first.node_features.shape, (4, len(NODE_FEATURE_NAMES)))
        self.assertEqual(first.edge_features.shape[1], len(EDGE_FEATURE_NAMES))
        self.assertEqual(first.track_features.shape, (2, len(TRACK_FEATURE_NAMES)))
        np.testing.assert_array_equal(first.node_features, second.node_features)
        np.testing.assert_array_equal(first.edge_index, second.edge_index)
        np.testing.assert_array_equal(first.component_to_track, second.component_to_track)
        self.assertTrue(np.isfinite(first.node_features).all())
        self.assertTrue(np.isfinite(first.edge_features).all())

    def test_decoder_summary_is_input_only_and_aligned(self):
        scores, locations, polarities, decoder = synthetic_video(with_decoder=True)
        graph = extract_temporal_track_graph(
            scores,
            locations,
            polarities,
            0.719,
            scores.size,
            decoder_event_features=decoder,
        )
        decoder_available_column = NODE_FEATURE_NAMES.index("decoder_available")
        self.assertTrue(np.all(graph.node_features[:, decoder_available_column] == 1.0))
        first_decoder_mean = NODE_FEATURE_NAMES.index("decoder_mean_00")
        expected = decoder[np.asarray((0, 1)), 0].mean()
        self.assertAlmostEqual(graph.node_features[0, first_decoder_mean], expected)

    def test_truth_is_derived_after_graph_and_atomic_track_edit_is_bitwise(self):
        scores, locations, polarities, _ = synthetic_video()
        graph = extract_temporal_track_graph(
            scores, locations, polarities, 0.719, scores.size
        )
        labels = np.asarray([1, 1, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
        component_targets = pure_false_positive_component_targets(
            graph.event_indices, labels
        )
        track_targets = pure_false_positive_track_targets(graph, component_targets)
        self.assertEqual(sorted(track_targets.tolist()), [0, 1])
        candidate, receipt = atomic_delete_from_graph(
            scores,
            graph,
            cutoff=0.5,
            track_pure_fp_probabilities=track_targets.astype(np.float64),
            mode="track",
        )
        deleted_tracks = np.flatnonzero(track_targets == 1)
        deleted_components = np.concatenate(
            [graph.track_component_rows[index] for index in deleted_tracks]
        )
        deleted_events = np.concatenate(
            [graph.event_indices[index] for index in deleted_components]
        )
        retained = np.ones(scores.size, dtype=bool)
        retained[deleted_events] = False
        self.assertTrue(np.array_equal(candidate[retained], scores[retained]))
        self.assertTrue(np.all(candidate[deleted_events] == 0.0))
        self.assertTrue(receipt.complete_components_only)
        self.assertTrue(receipt.complete_tracks_only)
        self.assertTrue(receipt.retained_scores_bitwise_equal)

    def test_track_pooling_and_analytic_cutoff(self):
        scores, locations, polarities, _ = synthetic_video()
        graph = extract_temporal_track_graph(
            scores, locations, polarities, 0.719, scores.size
        )
        pooled = aggregate_track_node_features(graph)
        self.assertEqual(pooled.shape[0], len(graph.track_component_rows))
        self.assertEqual(
            pooled.shape[1], 4 * len(NODE_FEATURE_NAMES) + len(TRACK_FEATURE_NAMES)
        )
        cutoff, enabled = derive_zero_observed_target_loss_cutoff(
            np.asarray([0.2, 0.8, 0.6, 0.9]),
            np.asarray([0, 1, 0, 1]),
        )
        self.assertGreater(cutoff, 0.6)
        self.assertTrue(enabled)


class TemporalTrackGraphModelTests(unittest.TestCase):
    def _graph_tensors(self):
        scores, locations, polarities, decoder = synthetic_video(with_decoder=True)
        graph = extract_temporal_track_graph(
            scores,
            locations,
            polarities,
            0.719,
            scores.size,
            decoder_event_features=decoder,
        )
        return graph, (
            torch.from_numpy(graph.node_features),
            torch.from_numpy(graph.edge_index),
            torch.from_numpy(graph.edge_features),
            torch.from_numpy(graph.component_to_track),
            torch.from_numpy(graph.track_features),
        )

    def test_forward_shapes_finite_and_all_parameters_receive_gradients(self):
        torch.manual_seed(7)
        graph, tensors = self._graph_tensors()
        model = TemporalTrackGraphExpert()
        output = model(*tensors)
        self.assertEqual(
            output.component_pure_fp_logits.shape, (len(graph.event_indices),)
        )
        self.assertEqual(
            output.track_pure_fp_logits.shape, (len(graph.track_component_rows),)
        )
        component_targets = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        track_targets = torch.tensor([0, 1], dtype=torch.float32)
        losses = balanced_graph_bce(output, component_targets, track_targets)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(graph_expert_parameter_count(model), 0)

    def test_empty_edge_graph_is_supported(self):
        scores, locations, polarities, decoder = synthetic_video(with_decoder=True)
        keep = locations[:, 3] == 0
        graph = extract_temporal_track_graph(
            scores[keep],
            locations[keep],
            polarities[keep],
            0.719,
            int(keep.sum()),
            decoder_event_features=decoder[keep],
        )
        model = TemporalTrackGraphExpert()
        output = model(
            torch.from_numpy(graph.node_features),
            torch.from_numpy(graph.edge_index),
            torch.from_numpy(graph.edge_features),
            torch.from_numpy(graph.component_to_track),
            torch.from_numpy(graph.track_features),
        )
        self.assertTrue(torch.isfinite(output.component_pure_fp_logits).all())


if __name__ == "__main__":
    unittest.main()
