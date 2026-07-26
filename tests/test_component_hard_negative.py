import unittest

import torch

from utils.component_hard_negative import (
    component_hard_negative_loss,
    target_frame_activation_loss,
)


class ComponentHardNegativeLossTests(unittest.TestCase):
    def call_loss(self, predictions, labels, locations, **overrides):
        options = {
            'spatial_cell_size': 3,
            'temporal_bin_size': 50,
            'min_cell_events': 2,
            'ratio': 1.0,
            'activation_threshold': 0.45,
            'activation_temperature': 0.10,
        }
        options.update(overrides)
        return component_hard_negative_loss(
            predictions,
            labels,
            locations,
            **options,
        )

    def test_mines_background_cells_but_excludes_target_cells(self):
        predictions = torch.tensor(
            [0.99, 0.99, 0.85, 0.80],
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
        locations = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 3, 0, 0],
            [0, 4, 0, 0],
        ])

        loss, candidate_count, hard_count = self.call_loss(
            predictions,
            labels,
            locations,
        )

        self.assertEqual(candidate_count, 1)
        self.assertEqual(hard_count, 1)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertAlmostEqual(float(predictions.grad[0]), 0.0, places=7)
        self.assertAlmostEqual(float(predictions.grad[1]), 0.0, places=7)
        self.assertGreater(float(predictions.grad[2]), 0.0)
        self.assertGreater(float(predictions.grad[3]), 0.0)

    def test_higher_background_scores_increase_loss(self):
        labels = torch.zeros(2)
        locations = torch.tensor([
            [0, 3, 0, 0],
            [0, 4, 0, 0],
        ])
        low_loss, _, _ = self.call_loss(
            torch.tensor([0.10, 0.10]),
            labels,
            locations,
        )
        high_loss, _, _ = self.call_loss(
            torch.tensor([0.90, 0.90]),
            labels,
            locations,
        )

        self.assertGreater(float(high_loss), float(low_loss))

    def test_cells_below_event_count_are_ignored(self):
        loss, candidate_count, hard_count = self.call_loss(
            torch.tensor([0.99]),
            torch.zeros(1),
            torch.tensor([[0, 3, 0, 0]]),
        )

        self.assertEqual(candidate_count, 0)
        self.assertEqual(hard_count, 0)
        self.assertEqual(float(loss), 0.0)

    def test_target_frame_loss_rewards_one_confident_target_event(self):
        labels = torch.ones(2)
        target_ids = torch.tensor([7, 7])
        locations = torch.tensor([
            [0, 0, 0, 1],
            [0, 1, 0, 2],
        ])
        low_loss, low_count, low_missed = target_frame_activation_loss(
            torch.tensor([0.10, 0.10]),
            labels,
            target_ids,
            locations,
            temporal_bin_size=50,
            activation_threshold=0.45,
            activation_temperature=0.10,
        )
        high_predictions = torch.tensor(
            [0.90, 0.10],
            requires_grad=True,
        )
        high_loss, high_count, high_missed = target_frame_activation_loss(
            high_predictions,
            labels,
            target_ids,
            locations,
            temporal_bin_size=50,
            activation_threshold=0.45,
            activation_temperature=0.10,
        )

        self.assertEqual((low_count, high_count), (1, 1))
        self.assertEqual(low_missed, 1)
        self.assertEqual(high_missed, 0)
        self.assertLess(float(high_loss), float(low_loss))
        high_loss.backward()
        self.assertLess(float(high_predictions.grad[0]), 0.0)

    def test_rejects_mismatched_inputs(self):
        with self.assertRaisesRegex(ValueError, 'counts must match'):
            self.call_loss(
                torch.tensor([0.5, 0.5]),
                torch.tensor([0.0]),
                torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]]),
            )


if __name__ == '__main__':
    unittest.main()
