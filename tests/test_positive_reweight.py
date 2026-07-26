import unittest

try:
    import torch
    from utils.positive_reweight import apply_positive_stc_floor
except ModuleNotFoundError:
    torch = None
    apply_positive_stc_floor = None


@unittest.skipUnless(
    torch is not None and apply_positive_stc_floor is not None,
    'PyTorch is required for positive reweight tests.',
)
class PositiveReweightTests(unittest.TestCase):
    def test_only_low_support_positive_events_are_raised(self):
        weights = torch.tensor([0.10, 0.80, 0.20, 0.90])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

        adjusted, boosted_count = apply_positive_stc_floor(
            weights,
            labels,
            floor=0.35,
        )

        self.assertEqual(boosted_count, 1)
        self.assertTrue(
            torch.equal(adjusted, torch.tensor([0.35, 0.80, 0.20, 0.90]))
        )

    def test_invalid_floor_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_positive_stc_floor(
                torch.tensor([0.5]),
                torch.tensor([1.0]),
                floor=1.5,
            )


if __name__ == '__main__':
    unittest.main()
