import unittest

import numpy as np
import torch

from dataset.temporal_frame import (
    TemporalFrameVideo,
    append_local_density_contrast,
    build_temporal_context_frame,
    temporal_bin_count,
    temporal_event_bins,
    temporal_frame_collate,
    temporal_frame_view_schedule,
)
from model.temporal_frame_net import (
    TemporalFrameNet,
    append_local_contrast_channels,
    build_motion_persistence_channels,
    gather_event_logits,
)
from utils.temporal_frame_inference import (
    TemporalFrameInferenceConfig,
    blend_temporal_frame_scores,
)
from utils.temporal_frame_loss import (
    build_target_center_heatmaps,
    frame_balanced_event_bce,
    target_center_heatmap_loss,
    target_group_coverage_loss,
)


class TemporalFrameTests(unittest.TestCase):
    def test_temporal_bins_are_clipped_to_the_known_range(self):
        locations = np.array(
            [
                [0, 0, 0],
                [0, 0, 50],
                [0, 0, 199],
                [0, 0, 250],
            ],
            dtype=np.int64,
        )
        self.assertEqual(temporal_bin_count(200, 50), 4)
        np.testing.assert_array_equal(
            temporal_event_bins(locations, 50, 4),
            np.array([0, 1, 3, 3], dtype=np.int64),
        )

    def test_context_frame_preserves_temporal_and_polarity_channels(self):
        locations = np.array(
            [
                [1, 2, 0],
                [2, 2, 50],
                [3, 2, 100],
            ],
            dtype=np.int64,
        )
        video = TemporalFrameVideo(
            name='synthetic',
            locations=locations,
            polarities=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            labels=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            target_ids=np.array([0, 1, 0], dtype=np.int64),
            event_bins=np.array([0, 1, 2], dtype=np.int64),
            event_indices_by_bin=(
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([2], dtype=np.int64),
            ),
            positive_bins=np.array([1], dtype=np.int64),
            occupied_bins=np.array([0, 1, 2], dtype=np.int64),
        )
        frame = build_temporal_context_frame(
            video,
            center_bin=1,
            context_bins=3,
            width=5,
            height=4,
            log_count_clip=4.0,
        )
        expected = np.log(2.0) / 4.0
        self.assertEqual(frame.shape, (6, 4, 5))
        self.assertAlmostEqual(frame[0, 2, 1], expected)
        self.assertAlmostEqual(frame[3, 2, 2], expected)
        self.assertAlmostEqual(frame[4, 2, 3], expected)
        self.assertAlmostEqual(float(frame.sum()), expected * 3, places=6)

    def test_collate_and_model_gather_preserve_event_alignment(self):
        samples = [
            {
                'frame': np.zeros((6, 33, 45), dtype=np.float32),
                'event_x': np.array([1, 2], dtype=np.int64),
                'event_y': np.array([3, 4], dtype=np.int64),
                'labels': np.array([1.0, 0.0], dtype=np.float32),
            },
            {
                'frame': np.ones((6, 33, 45), dtype=np.float32),
                'event_x': np.array([5], dtype=np.int64),
                'event_y': np.array([6], dtype=np.int64),
                'labels': np.array([1.0], dtype=np.float32),
            },
        ]
        batch = temporal_frame_collate(samples)
        self.assertEqual(tuple(batch['frames'].shape), (2, 6, 33, 45))
        self.assertEqual(batch['event_batch_indices'].tolist(), [0, 0, 1])

        model = TemporalFrameNet(input_channels=6, width=4)
        logit_maps = model(batch['frames'])
        event_logits = gather_event_logits(
            logit_maps,
            batch['event_batch_indices'],
            batch['event_y'],
            batch['event_x'],
        )
        self.assertEqual(tuple(logit_maps.shape), (2, 1, 33, 45))
        self.assertEqual(tuple(event_logits.shape), (3,))

        loss, diagnostics = frame_balanced_event_bce(
            event_logits,
            batch['labels'],
            batch['event_batch_indices'],
            target_positive_loss_mass=0.2,
            max_positive_weight=8.0,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(diagnostics['positive_fraction'], 0.0)
        self.assertIsNotNone(model.head.weight.grad)

    def test_collate_preserves_optional_fine_detail_frames(self):
        samples = [
            {
                'frame': np.zeros((6, 4, 5), dtype=np.float32),
                'fine_detail_frame': np.ones((10, 4, 5), dtype=np.float32),
                'event_x': np.array([1], dtype=np.int64),
                'event_y': np.array([2], dtype=np.int64),
                'labels': np.array([1.0], dtype=np.float32),
                'target_ids': np.array([3], dtype=np.int64),
            },
            {
                'frame': np.zeros((6, 4, 5), dtype=np.float32),
                'fine_detail_frame': np.zeros((10, 4, 5), dtype=np.float32),
                'event_x': np.array([2], dtype=np.int64),
                'event_y': np.array([1], dtype=np.int64),
                'labels': np.array([0.0], dtype=np.float32),
                'target_ids': np.array([0], dtype=np.int64),
            },
        ]

        batch = temporal_frame_collate(samples)

        self.assertEqual(tuple(batch['fine_detail_frames'].shape), (2, 10, 4, 5))
        self.assertEqual(batch['target_ids'].tolist(), [3, 0])

    def test_local_density_contrast_preserves_raw_channels(self):
        frame = np.ones((2, 5, 7), dtype=np.float32)
        frame[0, 2, 3] = 3.0

        combined = append_local_density_contrast(frame, kernel_size=3)

        self.assertEqual(combined.shape, (4, 5, 7))
        np.testing.assert_array_equal(combined[:2], frame)
        self.assertGreater(combined[2, 2, 3], 0.0)
        self.assertTrue(np.allclose(combined[3], 0.0))
        tensor_combined = append_local_contrast_channels(
            torch.from_numpy(frame).unsqueeze(0),
            kernel_size=3,
        ).squeeze(0).numpy()
        np.testing.assert_allclose(tensor_combined, combined, rtol=0.0, atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'positive odd'):
            append_local_density_contrast(frame, kernel_size=4)

    def test_motion_persistence_matches_shifted_neighbour_activity(self):
        raw = torch.zeros((1, 6, 7, 9))
        # Context order: previous, centre, next, with one channel per polarity.
        raw[0, 2, 3, 4] = 0.75
        raw[0, 0, 3, 2] = 1.00
        raw[0, 5, 3, 6] = 0.50

        features = build_motion_persistence_channels(
            raw,
            context_bins=3,
            spatial_radius_per_bin=2,
        )

        self.assertEqual(tuple(features.shape), (1, 2, 7, 9))
        self.assertAlmostEqual(float(features[0, 0, 3, 4]), 0.75)
        self.assertAlmostEqual(float(features[0, 1, 3, 4]), 0.50)
        no_shift_features = build_motion_persistence_channels(
            raw,
            context_bins=3,
            spatial_radius_per_bin=0,
        )
        self.assertEqual(float(no_shift_features[0, 0, 3, 4]), 0.0)
        with self.assertRaisesRegex(ValueError, 'raw temporal channels'):
            build_motion_persistence_channels(raw[:, :5], context_bins=3)

    def test_dense_view_schedule_repeats_only_dense_training_videos(self):
        schedule = temporal_frame_view_schedule(
            [100, 200, 300],
            views_per_video=2,
            dense_sampling_enabled=True,
            dense_event_count_cutoff=200,
            dense_view_multiplier=2,
        )

        self.assertEqual(len(schedule), 10)
        self.assertEqual(schedule[:2], ((0, 0), (0, 1)))
        self.assertEqual(schedule[2:6], ((1, 0), (1, 1), (1, 2), (1, 3)))
        self.assertEqual(schedule[6:], ((2, 0), (2, 1), (2, 2), (2, 3)))

    def test_zero_initialized_contrast_adapter_preserves_p23_output(self):
        torch.manual_seed(37)
        base_model = TemporalFrameNet(input_channels=6, width=4)
        contrast_model = TemporalFrameNet(
            input_channels=6,
            width=4,
            local_contrast_channels=6,
        )
        incompatible = contrast_model.load_state_dict(
            base_model.state_dict(),
            strict=False,
        )
        self.assertEqual(
            set(incompatible.missing_keys),
            {
                'local_contrast_adapter.weight',
                'local_contrast_adapter.bias',
            },
        )
        self.assertEqual(incompatible.unexpected_keys, [])

        raw_frame = torch.rand((2, 6, 33, 45))
        contrast_frame = torch.randn((2, 6, 33, 45))
        with torch.no_grad():
            base_logits = base_model(raw_frame)
            contrast_logits = contrast_model(
                torch.cat((raw_frame, contrast_frame), dim=1)
            )
        torch.testing.assert_close(base_logits, contrast_logits, rtol=0.0, atol=0.0)

    def test_zero_initialized_motion_adapter_preserves_p23_output(self):
        torch.manual_seed(37)
        base_model = TemporalFrameNet(input_channels=6, width=4)
        motion_model = TemporalFrameNet(
            input_channels=6,
            width=4,
            motion_persistence_channels=2,
        )
        incompatible = motion_model.load_state_dict(
            base_model.state_dict(),
            strict=False,
        )
        self.assertEqual(
            set(incompatible.missing_keys),
            {
                'motion_persistence_adapter.weight',
                'motion_persistence_adapter.bias',
            },
        )
        self.assertEqual(incompatible.unexpected_keys, [])

        raw_frame = torch.rand((2, 6, 33, 45))
        motion_features = build_motion_persistence_channels(
            raw_frame,
            context_bins=3,
            spatial_radius_per_bin=2,
        )
        with torch.no_grad():
            base_logits = base_model(raw_frame)
            motion_logits = motion_model(
                torch.cat((raw_frame, motion_features), dim=1)
            )
        torch.testing.assert_close(base_logits, motion_logits, rtol=0.0, atol=0.0)

    def test_zero_initialized_fine_detail_adapter_preserves_p23_output(self):
        torch.manual_seed(37)
        base_model = TemporalFrameNet(input_channels=6, width=4)
        fine_model = TemporalFrameNet(
            input_channels=6,
            width=4,
            fine_detail_channels=10,
        )
        incompatible = fine_model.load_state_dict(
            base_model.state_dict(),
            strict=False,
        )
        self.assertEqual(
            set(incompatible.missing_keys),
            {
                'fine_detail_adapter.weight',
                'fine_detail_adapter.bias',
            },
        )
        self.assertEqual(incompatible.unexpected_keys, [])

        raw_frame = torch.rand((2, 6, 33, 45))
        fine_detail_frame = torch.rand((2, 10, 33, 45))
        with torch.no_grad():
            base_logits = base_model(raw_frame)
            fine_logits = fine_model(
                torch.cat((raw_frame, fine_detail_frame), dim=1)
            )
        torch.testing.assert_close(base_logits, fine_logits, rtol=0.0, atol=0.0)

    def test_target_center_heatmap_uses_target_id_centroids(self):
        event_x = torch.tensor([2, 4, 9, 11, 1])
        event_y = torch.tensor([3, 3, 8, 8, 1])
        labels = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        target_ids = torch.tensor([1, 1, 2, 2, 0])
        event_batches = torch.tensor([0, 0, 0, 0, 0])

        heatmaps = build_target_center_heatmaps(
            event_x,
            event_y,
            labels,
            target_ids,
            event_batches,
            batch_size=1,
            height=12,
            width=14,
            sigma=2.5,
            radius=6,
        )

        self.assertEqual(tuple(heatmaps.shape), (1, 1, 12, 14))
        self.assertAlmostEqual(float(heatmaps[0, 0, 3, 3]), 1.0)
        self.assertAlmostEqual(float(heatmaps[0, 0, 8, 10]), 1.0)
        self.assertEqual(float(heatmaps[0, 0, 11, 0]), 0.0)

        logits = torch.zeros_like(heatmaps, requires_grad=True)
        loss, diagnostics = target_center_heatmap_loss(
            logits,
            heatmaps,
            target_positive_loss_mass=0.20,
            max_positive_weight=512.0,
            empty_loss_weight=0.10,
        )
        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(diagnostics['nonempty_view_fraction'], 1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_zero_initialized_target_center_branch_preserves_p23_output(self):
        torch.manual_seed(37)
        base_model = TemporalFrameNet(input_channels=6, width=4)
        center_model = TemporalFrameNet(
            input_channels=6,
            width=4,
            target_center_enabled=True,
        )
        incompatible = center_model.load_state_dict(
            base_model.state_dict(),
            strict=False,
        )
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                name.startswith('target_center_')
                for name in incompatible.missing_keys
            )
        )
        self.assertEqual(incompatible.unexpected_keys, [])

        raw_frame = torch.rand((2, 6, 33, 45))
        with torch.no_grad():
            base_logits = base_model(raw_frame)
            center_logits = center_model(raw_frame)
            returned_logits, target_center_logits = center_model(
                raw_frame,
                return_target_center_logits=True,
            )
        torch.testing.assert_close(base_logits, center_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(base_logits, returned_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            target_center_logits,
            torch.zeros_like(target_center_logits),
            rtol=0.0,
            atol=0.0,
        )

    def test_loss_rejects_invalid_balance_parameters(self):
        logits = torch.tensor([0.0])
        labels = torch.tensor([1.0])
        batches = torch.tensor([0])
        with self.assertRaisesRegex(ValueError, 'target_positive_loss_mass'):
            frame_balanced_event_bce(
                logits,
                labels,
                batches,
                target_positive_loss_mass=1.0,
            )

    def test_target_group_coverage_pushes_uncovered_groups(self):
        logits = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
        labels = torch.ones(3)
        target_ids = torch.tensor([1, 1, 2])
        event_batches = torch.zeros(3, dtype=torch.long)

        loss, diagnostics = target_group_coverage_loss(
            logits,
            labels,
            target_ids,
            event_batches,
            score_floor=0.70,
            correct_fraction=0.0001,
        )

        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(diagnostics['target_group_count'], 2)
        self.assertEqual(diagnostics['uncovered_group_count'], 1)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[2]), 0.0)

    def test_target_group_coverage_has_zero_gradient_without_targets(self):
        logits = torch.tensor([0.2, -0.4], requires_grad=True)
        labels = torch.zeros(2)
        target_ids = torch.zeros(2, dtype=torch.long)
        event_batches = torch.zeros(2, dtype=torch.long)

        loss, diagnostics = target_group_coverage_loss(
            logits,
            labels,
            target_ids,
            event_batches,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(diagnostics['target_group_count'], 0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_inference_config_defaults_to_a_disabled_expert(self):
        config = TemporalFrameInferenceConfig.from_cfg(object())

        self.assertFalse(config.enabled)
        self.assertFalse(config.frame_only)
        self.assertEqual(config.describe(), 'disabled')

    def test_inference_config_requires_a_checkpoint_when_enabled(self):
        with self.assertRaisesRegex(ValueError, 'model_path'):
            TemporalFrameInferenceConfig(enabled=True)
        with self.assertRaisesRegex(ValueError, 'sparse_weight'):
            TemporalFrameInferenceConfig(sparse_weight=1.1)
        self.assertTrue(
            TemporalFrameInferenceConfig(
                enabled=True,
                model_path='temporal_frame.pt',
                sparse_weight=0.0,
                local_contrast_enabled=True,
                local_contrast_kernel_size=9,
            ).frame_only
        )
        with self.assertRaisesRegex(ValueError, 'positive odd'):
            TemporalFrameInferenceConfig(local_contrast_kernel_size=2)
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            TemporalFrameInferenceConfig(motion_persistence_radius_per_bin=-1)
        with self.assertRaisesRegex(ValueError, 'positive odd'):
            TemporalFrameInferenceConfig(
                fine_detail_enabled=True,
                fine_context_bins=8,
            )

    def test_sparse_and_temporal_scores_are_blended_in_event_order(self):
        sparse_scores = torch.tensor([0.90, 0.50, 0.10])
        frame_scores = torch.tensor([0.70, 0.90, 0.80])

        scores = blend_temporal_frame_scores(
            sparse_scores,
            frame_scores,
            sparse_weight=0.75,
        )

        torch.testing.assert_close(
            scores,
            torch.tensor([0.85, 0.60, 0.275]),
            rtol=0.0,
            atol=1e-6,
        )
        with self.assertRaisesRegex(ValueError, 'shapes do not match'):
            blend_temporal_frame_scores(
                sparse_scores,
                frame_scores[:2],
                sparse_weight=0.5,
            )


if __name__ == '__main__':
    unittest.main()
