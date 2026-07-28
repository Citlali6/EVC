import math
import unittest

import torch

from utils.target_frame_balanced import target_frame_balanced_positive_loss


class TargetFrameBalancedLossTests(unittest.TestCase):
    def test_each_target_frame_has_equal_weight(self):
        predictions = torch.tensor([0.9, 0.9, 0.5], requires_grad=True)
        labels = torch.ones(3)
        target_ids = torch.tensor([1, 1, 2])
        locations = torch.tensor([
            [0, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 2, 0, 1],
        ])

        loss, group_count = target_frame_balanced_positive_loss(
            predictions,
            labels,
            target_ids,
            locations,
            temporal_bin_size=50,
        )

        expected = (-math.log(0.9 + 1e-5) - math.log(0.5 + 1e-5)) / 2
        self.assertEqual(group_count, 2)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())

    def test_boundary_events_and_background_ids_are_excluded(self):
        predictions = torch.tensor([0.1, 0.2, 0.8], requires_grad=True)
        labels = torch.ones(3)
        target_ids = torch.tensor([1, 1, 0])
        locations = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 0, 50],
            [0, 2, 0, 1],
        ])

        loss, group_count = target_frame_balanced_positive_loss(
            predictions,
            labels,
            target_ids,
            locations,
            temporal_bin_size=50,
        )

        self.assertEqual(group_count, 0)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(predictions.grad, torch.zeros_like(predictions)))

    def test_groups_are_separated_by_batch_and_time(self):
        predictions = torch.tensor([0.8, 0.6, 0.4], requires_grad=True)
        labels = torch.ones(3)
        target_ids = torch.tensor([1, 1, 1])
        locations = torch.tensor([
            [0, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 2, 0, 51],
        ])

        loss, group_count = target_frame_balanced_positive_loss(
            predictions,
            labels,
            target_ids,
            locations,
            temporal_bin_size=50,
        )

        expected = sum(-math.log(value + 1e-5) for value in (0.8, 0.6, 0.4)) / 3
        self.assertEqual(group_count, 3)
        self.assertAlmostEqual(loss.item(), expected, places=6)


if __name__ == '__main__':
    unittest.main()
