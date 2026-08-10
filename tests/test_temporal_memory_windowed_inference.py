import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch

from dataset.temporal_frame import temporal_frame_video_from_events
from diagnose_temporal_memory_windowing import discover_train_files
from utils.temporal_memory_inference import predict_temporal_memory_scores
from utils.temporal_memory_windowed_inference import (
    predict_temporal_memory_scores_windowed,
    stitch_temporal_window_tensors,
    temporal_center_stitch_plan,
)


class _SequenceSensitiveModel:
    """Tiny deterministic stand-in with sequence-boundary-sensitive memory."""

    confidence_head_enabled = False

    def encode_bottleneck(self, frames):
        return frames[:, :1]

    def temporal_residual(self, bottlenecks):
        if bottlenecks.ndim != 4:
            raise ValueError('Expected [T, C, H, W].')
        forward = torch.cumsum(bottlenecks, dim=0)
        backward = torch.flip(
            torch.cumsum(torch.flip(bottlenecks, dims=(0,)), dim=0),
            dims=(0,),
        )
        return (forward + backward) * 0.125

    def decode_with_residual(
        self,
        frames,
        residual,
        return_confidence_logits=False,
    ):
        if return_confidence_logits:
            raise AssertionError('The fake model has no confidence head.')
        return frames[:, :1] + residual


def _tiny_video():
    locations = np.asarray(
        [
            [0, 0, 1],
            [1, 0, 11],
            [2, 0, 21],
            [0, 1, 31],
            [1, 1, 41],
            [2, 1, 51],
            [0, 0, 61],
            [1, 0, 71],
        ],
        dtype=np.int64,
    )
    return temporal_frame_video_from_events(
        name='train_synthetic',
        locations=locations,
        polarities=np.zeros(locations.shape[0], dtype=np.float32),
        temporal_bin_size=10,
        whole_t=80,
    )


class TemporalCenterStitchPlanTests(unittest.TestCase):
    def test_t16_half_stride_covers_160_bins_once(self):
        plan = temporal_center_stitch_plan(160, 16)

        self.assertEqual(len(plan), 19)
        self.assertEqual((plan[0].window_start, plan[0].window_stop), (0, 16))
        self.assertEqual((plan[-1].window_start, plan[-1].window_stop), (144, 160))
        coverage = np.zeros(160, dtype=np.int64)
        for item in plan:
            self.assertLessEqual(item.window_start, item.keep_start)
            self.assertLessEqual(item.keep_stop, item.window_stop)
            coverage[item.keep_start:item.keep_stop] += 1
        np.testing.assert_array_equal(coverage, np.ones(160, dtype=np.int64))

    def test_non_aligned_tail_has_no_gap_or_duplicate(self):
        plan = temporal_center_stitch_plan(161, 16, stride=8)

        self.assertEqual(plan[-1].window_stop, 161)
        retained = np.concatenate(
            [np.arange(item.keep_start, item.keep_stop) for item in plan]
        )
        np.testing.assert_array_equal(retained, np.arange(161))

    def test_full_length_plan_is_identity(self):
        plan = temporal_center_stitch_plan(8, 8)

        self.assertEqual(len(plan), 1)
        self.assertEqual(
            (
                plan[0].window_start,
                plan[0].window_stop,
                plan[0].keep_start,
                plan[0].keep_stop,
            ),
            (0, 8, 0, 8),
        )

    def test_invalid_window_and_stride_fail_closed(self):
        for args in ((0, 1, None), (8, 0, None), (8, 9, None), (8, 4, 5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                temporal_center_stitch_plan(*args)

    def test_stitch_uses_only_each_windows_retained_interval(self):
        plan = temporal_center_stitch_plan(10, 4, stride=2)
        windows = tuple(
            torch.full((item.window_length,), index, dtype=torch.int64)
            for index, item in enumerate(plan)
        )

        stitched = stitch_temporal_window_tensors(windows, plan, 10)

        for index, item in enumerate(plan):
            self.assertTrue(
                torch.equal(
                    stitched[item.keep_start:item.keep_stop],
                    torch.full((item.keep_length,), index, dtype=torch.int64),
                )
            )


class TemporalWindowedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.video = _tiny_video()
        self.model = _SequenceSensitiveModel()
        self.arguments = {
            'model': self.model,
            'video': self.video,
            'device': torch.device('cpu'),
            'context_bins': 1,
            'width': 3,
            'height': 2,
            'inference_batch_size': 3,
            'log_count_clip': 4.0,
        }

    def test_full_window_is_bitwise_identical_to_full_stream_helper(self):
        full_stream = predict_temporal_memory_scores(**self.arguments)
        full_window = predict_temporal_memory_scores_windowed(
            **self.arguments,
            window_length=8,
        )

        self.assertTrue(torch.equal(full_stream, full_window))

    def test_short_windows_change_sequence_sensitive_model_but_cover_events(self):
        full_stream = predict_temporal_memory_scores(**self.arguments)
        windowed = predict_temporal_memory_scores_windowed(
            **self.arguments,
            window_length=4,
            stride=2,
        )

        self.assertEqual(windowed.shape, (self.video.locations.shape[0],))
        self.assertTrue(torch.isfinite(windowed).all())
        self.assertFalse(torch.equal(full_stream, windowed))

    def test_window_inference_rejects_bad_context_and_oversized_window(self):
        with self.assertRaises(ValueError):
            predict_temporal_memory_scores_windowed(
                **{**self.arguments, 'context_bins': 2},
                window_length=4,
            )
        with self.assertRaises(ValueError):
            predict_temporal_memory_scores_windowed(
                **self.arguments,
                window_length=9,
            )


class TrainOnlyInputGuardTests(unittest.TestCase):
    def test_directory_with_validation_name_is_rejected_before_npz_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'val_000.npz').touch()

            with self.assertRaisesRegex(ValueError, 'non-train inputs'):
                discover_train_files(directory, [], 0, 0)

    def test_requested_validation_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'train_000.npz').touch()

            with self.assertRaisesRegex(ValueError, 'requested names'):
                discover_train_files(directory, ['val_000'], 0, 0)


if __name__ == '__main__':
    unittest.main()
