"""CPU-only regression tests for temporal-memory training resume state."""

import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'evisseg_evuav.yaml'
ORIGINAL_ARGV = sys.argv
sys.argv = ['resume-test', '--config', str(CONFIG_PATH)]

from train_temporal_memory import (  # noqa: E402
    RESUME_CHECKPOINT_FORMAT_VERSION,
    build_training_checkpoint,
    capture_rng_state,
    is_resumable_checkpoint,
    load_training_resume,
    restore_rng_state,
    validate_resume_config,
)

sys.argv = ORIGINAL_ARGV


def make_config():
    resolved_config = {
        'DATA': {'root': 'dataset', 'whole_t': 8000, 'res': [346, 260]},
        'TRAIN': {
            'seed': 49,
            'epochs': 8,
            'lr': 5e-7,
            'scheduler': 'cosine',
            'scheduler_t_max': None,
            'scheduler_min_lr': 5e-8,
            'model_save_root': 'runs/original',
            'resume_checkpoint': '',
        },
        'SAMPLING': {
            'density_buckets': [30000, 80000, 200000],
            'density_view_multipliers': [1, 2, 2, 4],
        },
        'LOSS': {
            'frame_balanced_weight': 1.0,
            'metric_target_weight': 0.005,
        },
        'TEST': {'challenge_output_dir': 'submission/original'},
        'TEMPORAL_MEMORY': {
            'temporal_memory_sequence_length': 16,
            'temporal_memory_train_views_per_video': 2,
        },
    }
    return SimpleNamespace(
        temporal_memory_bin_size=50,
        temporal_memory_context_bins=5,
        temporal_memory_width=16,
        temporal_memory_sequence_length=16,
        temporal_memory_log_count_clip=4.0,
        temporal_frame_density_calibration_enabled=False,
        temporal_frame_trajectory_extrapolation_enabled=False,
        temporal_frame_confidence_head_enabled=False,
        temporal_memory_confidence_only_enabled=False,
        temporal_memory_temporal_attention_enabled=True,
        resolved_config=resolved_config,
        config_overrides=['TRAIN.epochs=8'],
    )


class TemporalMemoryResumeTests(unittest.TestCase):
    def test_default_resume_checkpoint_is_opt_in(self):
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
        self.assertIn('resume_checkpoint', config['TRAIN'])
        self.assertEqual(config['TRAIN']['resume_checkpoint'], '')

    def test_identical_config_and_output_path_changes_are_accepted(self):
        saved_config = make_config()
        checkpoint = {
            'provenance': {
                'resolved_config': copy.deepcopy(saved_config.resolved_config)
            }
        }
        validate_resume_config(checkpoint, make_config())

        moved_output = make_config()
        moved_output.resolved_config['TRAIN']['resume_checkpoint'] = 'last.pt'
        moved_output.resolved_config['TRAIN']['model_save_root'] = 'runs/resumed'
        moved_output.resolved_config['TEST'][
            'challenge_output_dir'
        ] = 'submission/resumed'
        validate_resume_config(checkpoint, moved_output)

    def test_training_sensitive_config_changes_are_rejected(self):
        saved_config = make_config()
        checkpoint = {
            'provenance': {
                'resolved_config': copy.deepcopy(saved_config.resolved_config)
            }
        }
        changes = (
            ('TRAIN.seed', 'TRAIN', 'seed', 50),
            ('SAMPLING.density_view_multipliers', 'SAMPLING',
             'density_view_multipliers', [1, 1, 1, 1]),
            ('LOSS.metric_target_weight', 'LOSS',
             'metric_target_weight', 0.01),
            ('TRAIN.epochs', 'TRAIN', 'epochs', 9),
            ('TRAIN.scheduler_t_max', 'TRAIN', 'scheduler_t_max', 8),
        )
        for expected_path, section, key, value in changes:
            with self.subTest(path=expected_path):
                changed = make_config()
                changed.resolved_config[section][key] = value
                with self.assertRaisesRegex(ValueError, expected_path):
                    validate_resume_config(checkpoint, changed)

    def test_cpu_rng_round_trip(self):
        random.seed(112)
        np.random.seed(113)
        torch.manual_seed(114)
        state = capture_rng_state(include_cuda=False)

        expected = (
            random.random(),
            np.random.random(4),
            torch.rand(4),
        )
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        restore_rng_state(state)

        self.assertEqual(random.random(), expected[0])
        np.testing.assert_array_equal(np.random.random(4), expected[1])
        torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)

    def test_checkpoint_restores_training_state_and_next_epoch(self):
        random.seed(212)
        np.random.seed(213)
        torch.manual_seed(214)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.5
        )
        loss = model(torch.ones(2, 3)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        checkpoint = build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            epoch_loss=0.125,
            best_loss=0.1,
            best_epoch=2,
            config=make_config(),
            initialized_from='parent/model.pt',
            resume_parent_checkpoint='parent/last.pt',
            run_start_epoch=1,
            best_loss_checkpoint='parent/best.pt',
            git_provenance={'commit': 'abc123', 'dirty': False},
            include_cuda_rng=False,
        )
        self.assertTrue(is_resumable_checkpoint(checkpoint))
        self.assertEqual(
            checkpoint['checkpoint_format_version'],
            RESUME_CHECKPOINT_FORMAT_VERSION,
        )
        self.assertEqual(checkpoint['next_epoch'], 4)
        self.assertEqual(checkpoint['start_epoch'], 1)
        self.assertEqual(checkpoint['best_epoch'], 2)
        self.assertEqual(checkpoint['provenance']['git']['commit'], 'abc123')
        self.assertEqual(
            checkpoint['provenance']['config_overrides'], ['TRAIN.epochs=8']
        )

        expected_python = random.random()
        expected_numpy = np.random.random()
        expected_torch = torch.rand(3)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'resume.pt'
            torch.save(checkpoint, path)

            torch.manual_seed(888)
            restored_model = torch.nn.Linear(3, 2)
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=0.3
            )
            restored_scheduler = torch.optim.lr_scheduler.StepLR(
                restored_optimizer, step_size=9, gamma=0.1
            )
            restored, start_epoch = load_training_resume(
                path,
                restored_model,
                restored_optimizer,
                restored_scheduler,
                current_config=make_config(),
            )

        self.assertEqual(start_epoch, 4)
        self.assertEqual(restored['best_loss'], 0.1)
        for expected_parameter, actual_parameter in zip(
            model.parameters(), restored_model.parameters()
        ):
            torch.testing.assert_close(
                actual_parameter, expected_parameter, rtol=0, atol=0
            )
        self.assertEqual(
            restored_optimizer.state_dict()['param_groups'],
            optimizer.state_dict()['param_groups'],
        )
        self.assertEqual(
            restored_scheduler.state_dict(), scheduler.state_dict()
        )
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        torch.testing.assert_close(
            torch.rand(3), expected_torch, rtol=0, atol=0
        )

    def test_legacy_checkpoint_is_initialization_only(self):
        legacy = {
            'model_state_dict': torch.nn.Linear(1, 1).state_dict(),
            'epoch': 7,
            'temporal_memory': {'context_bins': 5},
        }
        self.assertFalse(is_resumable_checkpoint(legacy))
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'legacy.pt'
            torch.save(legacy, path)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
            with self.assertRaisesRegex(
                ValueError,
                'temporal_memory_init_model_path',
            ):
                load_training_resume(
                    path,
                    model,
                    optimizer,
                    scheduler,
                    current_config=make_config(),
                )

    def test_optimizer_group_name_or_order_change_is_rejected(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(
            [{'name': 'base', 'params': model.parameters()}],
            lr=0.01,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
        checkpoint = build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            epoch_loss=0.2,
            best_loss=0.2,
            best_epoch=0,
            config=make_config(),
            initialized_from='parent.pt',
            include_cuda_rng=False,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'resume.pt'
            torch.save(checkpoint, path)
            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.AdamW(
                [{'name': 'memory', 'params': restored_model.parameters()}],
                lr=0.01,
            )
            restored_scheduler = torch.optim.lr_scheduler.StepLR(
                restored_optimizer, 2
            )
            with self.assertRaisesRegex(ValueError, 'names/order'):
                load_training_resume(
                    path,
                    restored_model,
                    restored_optimizer,
                    restored_scheduler,
                    current_config=make_config(),
                )


if __name__ == '__main__':
    unittest.main()
