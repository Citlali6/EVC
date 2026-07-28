import unittest

import numpy as np
import torch

from dataset.temporal_frame import TemporalFrameVideo
from dataset.temporal_memory import temporal_sequence_start
from model.temporal_frame_net import TemporalFrameNet
from model.temporal_memory_net import BidirectionalTemporalMemoryNet
from utils.temporal_memory_inference import (
    TemporalMemoryInferenceConfig,
    predict_temporal_memory_scores,
)


class TemporalMemoryTests(unittest.TestCase):
    def test_sequence_start_stays_within_video_boundaries(self):
        self.assertEqual(temporal_sequence_start(0, 12, 5), 0)
        self.assertEqual(temporal_sequence_start(6, 12, 5), 4)
        self.assertEqual(temporal_sequence_start(11, 12, 5), 7)
        with self.assertRaisesRegex(ValueError, 'must not exceed'):
            temporal_sequence_start(0, 4, 5)
        with self.assertRaisesRegex(ValueError, 'outside'):
            temporal_sequence_start(12, 12, 5)

    def test_zero_initialized_memory_matches_p23_logits(self):
        torch.manual_seed(37)
        base = TemporalFrameNet(input_channels=6, width=4).eval()
        memory = BidirectionalTemporalMemoryNet(
            input_channels=6,
            width=4,
        ).eval()
        memory.base.load_state_dict(base.state_dict(), strict=True)
        sequence = torch.rand((2, 3, 6, 33, 45))

        with torch.no_grad():
            expected = base(sequence.reshape(6, 6, 33, 45)).reshape(
                2,
                3,
                1,
                33,
                45,
            )
            actual = memory(sequence)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_memory_model_keeps_sequence_shape_and_backpropagates(self):
        model = BidirectionalTemporalMemoryNet(input_channels=6, width=4)
        sequence = torch.rand((1, 4, 6, 33, 45))
        logits = model(sequence)
        self.assertEqual(tuple(logits.shape), (1, 4, 1, 33, 45))
        loss = logits.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.memory_projection.weight.grad)

    def test_full_video_inference_preserves_original_event_order(self):
        video = TemporalFrameVideo(
            name='synthetic',
            locations=np.array(
                [[1, 2, 0], [2, 2, 50], [3, 2, 100], [4, 2, 100]],
                dtype=np.int64,
            ),
            polarities=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            labels=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            target_ids=np.array([0, 1, 0, 2], dtype=np.int64),
            event_bins=np.array([0, 1, 2, 2], dtype=np.int64),
            event_indices_by_bin=(
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([2, 3], dtype=np.int64),
            ),
            positive_bins=np.array([1, 2], dtype=np.int64),
            occupied_bins=np.array([0, 1, 2], dtype=np.int64),
        )
        model = BidirectionalTemporalMemoryNet(input_channels=6, width=4).eval()
        scores = predict_temporal_memory_scores(
            model,
            video,
            torch.device('cpu'),
            context_bins=3,
            width=9,
            height=7,
            inference_batch_size=2,
        )

        self.assertEqual(tuple(scores.shape), (4,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.all((scores >= 0.0) & (scores <= 1.0)))

    def test_inference_config_requires_checkpoint_only_when_enabled(self):
        config = TemporalMemoryInferenceConfig.from_cfg(object())
        self.assertFalse(config.enabled)
        self.assertEqual(config.describe(), 'disabled')
        with self.assertRaisesRegex(ValueError, 'model_path'):
            TemporalMemoryInferenceConfig(enabled=True)
        self.assertEqual(
            TemporalMemoryInferenceConfig(
                enabled=True,
                model_path='memory.pt',
            ).describe(),
            'enabled (model=memory.pt)',
        )


if __name__ == '__main__':
    unittest.main()
