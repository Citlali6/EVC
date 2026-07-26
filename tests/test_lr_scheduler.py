import unittest

import torch

from utils.lr_scheduler import build_lr_scheduler, describe_lr_scheduler


class LearningRateSchedulerTests(unittest.TestCase):
    def make_optimizer(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        return torch.optim.Adam([parameter], lr=1e-3)

    def test_step_scheduler_matches_original_schedule_type(self):
        scheduler = build_lr_scheduler(
            self.make_optimizer(),
            'step',
            total_epochs=50,
            step_size=10,
            gamma=0.1,
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertEqual(describe_lr_scheduler('step', 50, 10, 0.1, 1e-6),
                         'step (step_size=10, gamma=0.1)')

    def test_cosine_scheduler_uses_requested_epoch_count_and_floor(self):
        scheduler = build_lr_scheduler(
            self.make_optimizer(),
            'cosine',
            total_epochs=100,
            min_lr=1e-5,
        )

        self.assertIsInstance(
            scheduler,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )
        self.assertEqual(scheduler.T_max, 100)
        self.assertEqual(scheduler.eta_min, 1e-5)
        self.assertEqual(describe_lr_scheduler('cosine', 100, 10, 0.1, 1e-5),
                         'cosine (epochs=100, min_lr=1e-05)')

    def test_cosine_scheduler_can_keep_a_longer_horizon_for_a_short_screen(self):
        scheduler = build_lr_scheduler(
            self.make_optimizer(),
            'cosine',
            total_epochs=50,
            min_lr=1e-5,
            cosine_t_max=100,
        )

        self.assertEqual(scheduler.T_max, 100)
        self.assertEqual(
            describe_lr_scheduler(
                'cosine',
                50,
                10,
                0.1,
                1e-5,
                cosine_t_max=100,
            ),
            'cosine (epochs=50, T_max=100, min_lr=1e-05)',
        )

    def test_unknown_scheduler_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown TRAIN.scheduler'):
            build_lr_scheduler(self.make_optimizer(), 'invalid', total_epochs=50)


if __name__ == '__main__':
    unittest.main()
