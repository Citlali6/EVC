import unittest
from types import SimpleNamespace

try:
    import torch
    from utils.eval import evalute
except ModuleNotFoundError:
    torch = None
    evalute = None


@unittest.skipUnless(
    torch is not None and evalute is not None,
    'PyTorch and evaluator dependencies are required for evaluator tests.',
)
class SegmentationAccuracyTests(unittest.TestCase):
    def test_accuracy_binarizes_predictions_without_cuda(self):
        evaluator = evalute(SimpleNamespace(roc=False))
        evaluator.matches = {
            '0': {
                'seg_gt': torch.tensor([1.0, 1.0, 0.0, 1.0]),
                'seg_pred': torch.tensor([0.95, 0.20, 0.99, 0.90]),
            }
        }

        accuracy = evaluator.evaluate_semantic_segmantation_accuracy(thresh=0.9)

        self.assertAlmostEqual(accuracy.item(), 2.0 / 3.0)


if __name__ == '__main__':
    unittest.main()
