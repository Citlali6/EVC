import math
import unittest

try:
    import torch
    from utils.hard_negative import top_ratio_background_bce
except ModuleNotFoundError:
    torch = None
    top_ratio_background_bce = None


@unittest.skipUnless(
    torch is not None and top_ratio_background_bce is not None,
    'PyTorch is required for hard-negative loss tests.',
)
class HardNegativeLossTests(unittest.TestCase):
    def test_only_highest_scoring_background_events_contribute(self):
        predictions = torch.tensor([0.10, 0.90, 0.80, 0.99])
        labels = torch.tensor([0.0, 0.0, 0.0, 1.0])

        loss, hard_count = top_ratio_background_bce(
            predictions,
            labels,
            ratio=0.5,
        )

        expected = (-math.log(1 - 0.90 + 1e-5) - math.log(1 - 0.80 + 1e-5)) / 2
        self.assertEqual(hard_count, 2)
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_empty_background_returns_a_differentiable_zero(self):
        predictions = torch.tensor([0.70, 0.80], requires_grad=True)
        labels = torch.tensor([1.0, 1.0])

        loss, hard_count = top_ratio_background_bce(
            predictions,
            labels,
            ratio=0.01,
        )
        loss.backward()

        self.assertEqual(hard_count, 0)
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(predictions.grad, torch.zeros_like(predictions)))


if __name__ == '__main__':
    unittest.main()
