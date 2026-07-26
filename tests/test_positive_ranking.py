import unittest

import torch

from utils.positive_ranking import positive_hard_ranking_loss


class PositiveRankingTests(unittest.TestCase):
    def test_penalizes_overlapping_confidence_tails(self):
        loss, positive_count, background_count = positive_hard_ranking_loss(
            torch.tensor([0.55, 0.95, 0.80, 0.20]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            ratio=0.5,
            margin=0.1,
        )

        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(positive_count, 1)
        self.assertEqual(background_count, 1)

    def test_zero_when_margin_is_satisfied(self):
        loss, _, _ = positive_hard_ranking_loss(
            torch.tensor([0.95, 0.90, 0.10, 0.20]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            ratio=1.0,
            margin=0.1,
        )

        self.assertEqual(loss.item(), 0.0)

    def test_empty_class_returns_differentiable_zero(self):
        predictions = torch.tensor([0.2, 0.3], requires_grad=True)
        loss, positive_count, background_count = positive_hard_ranking_loss(
            predictions,
            torch.zeros(2),
            ratio=0.1,
            margin=0.1,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual((positive_count, background_count), (0, 0))
        loss.backward()


if __name__ == '__main__':
    unittest.main()
