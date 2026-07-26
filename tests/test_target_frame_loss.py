import unittest

try:
    import torch
    from utils.target_frame_loss import target_frame_detection_loss
except ModuleNotFoundError:
    torch = None
    target_frame_detection_loss = None


@unittest.skipUnless(
    torch is not None and target_frame_detection_loss is not None,
    'PyTorch is required for target-frame loss tests.',
)
class TargetFrameLossTests(unittest.TestCase):
    def test_groups_are_separated_by_target_and_time_bin(self):
        predictions = torch.tensor([0.95, 0.10, 0.50, 0.20], requires_grad=True)
        labels = torch.ones(4)
        target_ids = torch.tensor([1, 1, 1, 2])
        locations = torch.tensor([
            [0, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 2, 0, 51],
            [0, 3, 0, 51],
        ])

        loss, group_count, missed_groups = target_frame_detection_loss(
            predictions,
            labels,
            target_ids,
            locations,
            prediction_threshold=0.9,
            correct_threshold=0.0001,
            temporal_bin_size=50,
        )

        self.assertEqual(group_count, 3)
        self.assertEqual(missed_groups, 2)
        self.assertAlmostEqual(loss.item(), (0.0 + 0.4 + 0.7) / 3, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())

    def test_required_hits_follow_official_fraction_rule(self):
        predictions = torch.zeros(10001, requires_grad=True)
        with torch.no_grad():
            predictions[0] = 0.95
            predictions[1] = 0.10
        labels = torch.ones(10001)
        target_ids = torch.ones(10001, dtype=torch.long)
        locations = torch.zeros((10001, 4), dtype=torch.long)
        locations[:, 3] = 1

        loss, group_count, missed_groups = target_frame_detection_loss(
            predictions,
            labels,
            target_ids,
            locations,
            prediction_threshold=0.9,
            correct_threshold=0.0001,
            temporal_bin_size=50,
        )

        self.assertEqual(group_count, 1)
        self.assertEqual(missed_groups, 1)
        self.assertAlmostEqual(loss.item(), 0.4, places=6)

    def test_frame_boundary_events_are_ignored_like_official_pd(self):
        predictions = torch.tensor([0.95, 0.95, 0.10], requires_grad=True)
        labels = torch.ones(3)
        target_ids = torch.ones(3, dtype=torch.long)
        locations = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 0, 50],
            [0, 2, 0, 1],
        ])

        loss, group_count, missed_groups = target_frame_detection_loss(
            predictions,
            labels,
            target_ids,
            locations,
            prediction_threshold=0.9,
            correct_threshold=0.0001,
            temporal_bin_size=50,
        )

        self.assertEqual(group_count, 1)
        self.assertEqual(missed_groups, 1)
        self.assertAlmostEqual(loss.item(), 0.8, places=6)


if __name__ == '__main__':
    unittest.main()
