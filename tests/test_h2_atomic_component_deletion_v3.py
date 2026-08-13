import unittest
from types import SimpleNamespace
import tempfile
from pathlib import Path

import numpy as np
import torch

from model.h2_atomic_component_deletion_net import (
    ActivityFirstComponentScorer,
    PATCH_CHANNELS,
    balanced_component_bce,
)
from utils.atomic_component_deletion import (
    atomic_delete_or_identity,
    build_component_patch_queries,
    derive_strict_safe_cutoff,
    extract_atomic_components,
    pure_false_positive_targets,
    use_h2_atomic_deletion,
    verify_atomic_candidate,
)
from run_h2_atomic_component_deletion_v3 import (
    ComponentSequenceDataset,
    component_sequence_collate,
    persist_component_score_artifact,
    persist_source_feature_artifact,
)


class AtomicComponentTopologyTests(unittest.TestCase):
    def test_exact_c00_connectivity_and_partition(self):
        # Events 0/1 link across adjacent bins at dx=2. Event 2 is spatially
        # isolated; event 3 is below threshold and belongs to no component.
        scores = np.asarray([0.8, 0.9, 0.95, 0.2, 0.8], dtype=np.float32)
        locations = np.asarray(
            [
                [0, 10, 10, 1],
                [0, 12, 10, 51],
                [0, 30, 30, 51],
                [0, 10, 10, 101],
                [1, 10, 10, 1],
            ],
            dtype=np.int64,
        )
        components = extract_atomic_components(scores, locations, 0.719)
        self.assertEqual(3, len(components.event_indices))
        self.assertTrue(np.array_equal(components.event_indices[0], [0, 1]))
        self.assertTrue(np.array_equal(components.event_indices[1], [2]))
        self.assertTrue(np.array_equal(components.event_indices[2], [4]))

    def test_component_targets_are_training_only_pure_fp_classes(self):
        components = (np.asarray([0, 1]), np.asarray([2, 3]))
        labels = np.asarray([0, 1, 0, 0], dtype=np.uint8)
        self.assertTrue(
            np.array_equal(pure_false_positive_targets(components, labels), [0, 1])
        )

    def test_patch_queries_are_recentred_and_contiguous(self):
        locations = np.asarray(
            [[0, 9, 10, 1], [0, 11, 10, 49], [0, 12, 11, 51]],
            dtype=np.int64,
        )
        queries = build_component_patch_queries(
            (np.asarray([0, 1, 2]),), locations, patch_radius=2
        )[0]
        self.assertEqual([0, 1], [item.temporal_bin for item in queries])
        self.assertEqual((5, 5), queries[0].component_mask.shape)
        self.assertEqual(2.0, float(queries[0].component_mask.sum()))
        self.assertEqual(1.0, float(queries[1].component_mask.sum()))


class AtomicCalibrationAndActionTests(unittest.TestCase):
    def test_cutoff_is_strict_upper_bound_of_target_bearing_scores(self):
        scores = np.asarray([0.1, 0.4, 0.8, 0.9])
        targets = np.asarray([0, 0, 1, 1])
        cutoff, enabled, diagnostics = derive_strict_safe_cutoff(scores, targets)
        self.assertGreater(cutoff, 0.4)
        self.assertEqual(np.nextafter(np.float64(0.4), np.float64(np.inf)), cutoff)
        self.assertTrue(enabled)
        self.assertEqual(2, diagnostics["safe_pure_fp_component_count"])

    def test_no_safe_pure_fp_means_identity_policy(self):
        cutoff, enabled, diagnostics = derive_strict_safe_cutoff(
            [0.8, 0.7], [0, 1]
        )
        self.assertFalse(enabled)
        self.assertTrue(diagnostics["identity_due_to_no_safe_component"])
        self.assertGreater(cutoff, 0.8)

    def test_atomic_delete_preserves_every_keep_bitwise(self):
        base = np.asarray([0.72, 0.91, 0.73, 0.1], dtype=np.float32)
        components = (np.asarray([0, 1]), np.asarray([2]))
        candidate, receipt = atomic_delete_or_identity(
            base, components, [0.2, 0.9], 0.8, enabled=True
        )
        self.assertTrue(receipt.complete_components_only)
        self.assertTrue(receipt.retained_scores_bitwise_equal)
        self.assertEqual(1, receipt.deleted_component_count)
        self.assertTrue(np.array_equal(candidate[:2].view(np.uint32), base[:2].view(np.uint32)))
        self.assertEqual(np.float32(0.0), candidate[2])
        self.assertEqual(base[3].view(np.uint32), candidate[3].view(np.uint32))

    def test_overlap_fails_closed_to_identity(self):
        base = np.asarray([0.8, 0.9], dtype=np.float32)
        candidate, receipt = atomic_delete_or_identity(
            base,
            (np.asarray([0, 1]), np.asarray([1])),
            [0.9, 0.9],
            0.8,
            enabled=True,
        )
        self.assertFalse(receipt.enabled)
        self.assertFalse(receipt.complete_components_only)
        self.assertTrue(np.array_equal(candidate.view(np.uint32), base.view(np.uint32)))

    def test_partial_component_deletion_is_detected(self):
        base = np.asarray([0.8, 0.9], dtype=np.float32)
        partial = np.asarray([0.0, 0.9], dtype=np.float32)
        retained, complete = verify_atomic_candidate(
            base, partial, (np.asarray([0, 1]),), [True]
        )
        self.assertTrue(retained)
        self.assertFalse(complete)

    def test_route_excludes_h1_and_accepts_h2(self):
        h1 = np.concatenate((np.zeros(270000), np.ones(30000)))
        h2 = np.concatenate((np.zeros(160000), np.ones(140001)))
        self.assertFalse(use_h2_atomic_deletion(len(h1), h1))
        self.assertTrue(use_h2_atomic_deletion(len(h2), h2))


class ActivityFirstNetworkTests(unittest.TestCase):
    @staticmethod
    def patches(batch=2, length=4, size=7):
        values = torch.randn(batch, length, PATCH_CHANNELS, size, size)
        values[:, :, -1].zero_()
        values[:, :, -1, size // 2, size // 2] = 1.0
        return values

    def test_forward_and_both_adapters_receive_gradient(self):
        torch.manual_seed(4)
        model = ActivityFirstComponentScorer()
        patches = self.patches()
        logits = model(patches, torch.tensor([4, 3]))
        self.assertEqual((2,), tuple(logits.shape))
        activity_embedding, fused_embedding = model.component_embeddings(
            patches, torch.tensor([4, 3])
        )
        self.assertEqual((2, 96), tuple(activity_embedding.shape))
        self.assertEqual((2, 128), tuple(fused_embedding.shape))
        loss = balanced_component_bce(logits, [0.0, 1.0], [1.0, 1.0])
        loss.backward()
        activity_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.activity_adapter.parameters()
            if parameter.grad is not None
        )
        semantic_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.semantic_adapter.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(activity_grad, 0.0)
        self.assertGreater(semantic_grad, 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_padding_is_ignored_when_last_valid_patch_is_repeated(self):
        torch.manual_seed(9)
        model = ActivityFirstComponentScorer().eval()
        short = self.patches(batch=1, length=2)
        padded = torch.cat((short, short[:, -1:].repeat(1, 3, 1, 1, 1)), dim=1)
        with torch.no_grad():
            direct = model(short, [2])
            padded_value = model(padded, [2])
        self.assertTrue(torch.allclose(direct, padded_value, atol=1e-6, rtol=0.0))

    def test_invalid_empty_mask_is_rejected(self):
        model = ActivityFirstComponentScorer()
        patches = self.patches(batch=1, length=1)
        patches[:, :, -1].zero_()
        with self.assertRaises(ValueError):
            model(patches, [1])

    def test_frozen_small_architecture_has_1705_parameters(self):
        model = ActivityFirstComponentScorer(
            activity_width=4, semantic_width=4, temporal_width=8
        )
        self.assertEqual(1705, sum(parameter.numel() for parameter in model.parameters()))
        activity, fused = model.component_embeddings(self.patches(), [4, 3])
        self.assertEqual((2, 24), tuple(activity.shape))
        self.assertEqual((2, 32), tuple(fused.shape))


class ComponentDatasetTests(unittest.TestCase):
    @staticmethod
    def patch(length):
        patch = np.zeros((length, PATCH_CHANNELS, 5, 5), dtype=np.float16)
        patch[:, -1, 2, 2] = 1.0
        return patch

    def test_collate_repeats_padding_but_preserves_true_lengths(self):
        batch = component_sequence_collate(
            [
                {"patches": self.patch(1), "target": 0.0, "weight": 1.0},
                {"patches": self.patch(3), "target": 1.0, "weight": 1.0},
            ]
        )
        self.assertEqual((2, 3, PATCH_CHANNELS, 5, 5), tuple(batch["patches"].shape))
        self.assertTrue(torch.equal(batch["lengths"], torch.tensor([1, 3])))
        self.assertGreater(float(batch["patches"][0, 2, -1].sum()), 0.0)

    def test_fit_weights_balance_sources_and_classes_without_model_features(self):
        left = SimpleNamespace(
            component_patches=(self.patch(1), self.patch(1)),
            pure_fp_targets=np.asarray([0, 1], dtype=np.uint8),
        )
        right = SimpleNamespace(
            component_patches=(self.patch(1),) * 4,
            pure_fp_targets=np.asarray([0, 1, 1, 1], dtype=np.uint8),
        )
        dataset = ComponentSequenceDataset([left, right])
        self.assertEqual(6, len(dataset))
        # The dataset returns only patches/class/fit-derived weight.  Source
        # identity is used for weighting bookkeeping and cannot reach forward.
        self.assertEqual({"patches", "target", "weight"}, set(dataset[0]))
        target_mass = float(dataset.weights[dataset.targets == 0].sum())
        fp_mass = float(dataset.weights[dataset.targets == 1].sum())
        self.assertAlmostEqual(target_mass, fp_mass, places=6)


class ImmutableArtifactTests(unittest.TestCase):
    def source(self):
        patches = np.zeros((1, PATCH_CHANNELS, 5, 5), dtype=np.float16)
        patches[:, -1, 2, 2] = 1.0
        return SimpleNamespace(
            event_count=3,
            base_raw_scores=np.asarray([0.8, 0.9, 0.1], dtype=np.float32),
            base_scores=np.asarray([0.8, 0.9, 0.0], dtype=np.float32),
            rich_cache_reference_scores=np.asarray([0.8, 0.9, 0.1], dtype=np.float32),
            locations=np.asarray([[0, 1, 1, 1], [0, 2, 1, 1], [0, 9, 9, 1]], dtype=np.int64),
            component_event_indices=(np.asarray([0, 1], dtype=np.int64),),
            component_patches=(patches,),
            rich_cache_record_sha256="a" * 64,
            rich_cache_comparison={"threshold_disagreement_events": 0},
            pure_fp_targets=np.asarray([1], dtype=np.uint8),
        )

    def test_input_artifact_is_label_free_hashed_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npz"
            receipt = persist_source_feature_artifact(self.source(), path)
            self.assertEqual(64, len(receipt["sha256"]))
            with np.load(path, allow_pickle=False) as archive:
                self.assertIn("activity_polarity_raw_patches", archive.files)
                self.assertIn("event_component_ids", archive.files)
                self.assertNotIn("labels", archive.files)
                self.assertNotIn("target_ids", archive.files)
            with self.assertRaises(FileExistsError):
                persist_source_feature_artifact(self.source(), path)

    def test_held_score_artifact_has_embeddings_but_no_fit_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            record = {
                "probabilities": np.asarray([0.9]),
                "activity_embeddings": np.ones((1, 24), dtype=np.float32),
                "fused_embeddings": np.ones((1, 32), dtype=np.float32),
            }
            persist_component_score_artifact(
                path,
                self.source(),
                model_group_ids=["g1"],
                score_records=[record],
                consensus_probabilities=record["probabilities"],
                cutoff=0.8,
                enabled=True,
                include_fit_targets=False,
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertIn("activity_adapter_embeddings", archive.files)
                self.assertIn("fused_component_embeddings", archive.files)
                self.assertIn("delete_component", archive.files)
                self.assertNotIn("fit_only_pure_fp_targets", archive.files)


if __name__ == "__main__":
    unittest.main()
