"""CPU-only tests for the M24 target-time coverage loss."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset.temporal_memory import (
    TemporalMemoryTrainDataset,
    temporal_memory_collate,
)
from utils.temporal_frame_loss import (
    frame_balanced_event_bce,
    target_time_group_coverage_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'evisseg_evuav.yaml'
ORIGINAL_ARGV = sys.argv
sys.argv = ['coverage-test', '--config', str(CONFIG_PATH)]

from train_temporal_memory import apply_target_coverage_loss  # noqa: E402

sys.argv = ORIGINAL_ARGV


SCORE_FLOOR = 0.719
FLOOR_LOGIT = math.log(SCORE_FLOOR / (1.0 - SCORE_FLOOR))


def coverage_loss(logits, labels, target_ids, batch_ids, timestamps):
    return target_time_group_coverage_loss(
        logits,
        labels,
        target_ids,
        batch_ids,
        timestamps,
        temporal_bin_size=50,
        score_floor=SCORE_FLOOR,
        correct_fraction=0.0001,
    )


class TargetTimeGroupCoverageLossTest(unittest.TestCase):
    def test_same_target_in_two_time_bins_remains_two_groups(self):
        logits = torch.tensor(
            [FLOOR_LOGIT + 0.5, FLOOR_LOGIT - 0.4],
            requires_grad=True,
        )
        loss, stats = coverage_loss(
            logits,
            torch.ones(2),
            torch.ones(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            torch.tensor([1, 51]),
        )

        self.assertEqual(stats['target_group_count'], 2)
        self.assertEqual(stats['uncovered_group_count'], 1)
        self.assertAlmostEqual(loss.item(), 0.2, places=6)
        loss.backward()
        torch.testing.assert_close(
            logits.grad,
            torch.tensor([0.0, -0.5]),
            rtol=0,
            atol=1e-7,
        )

    def test_open_intervals_exclude_boundary_events(self):
        logits = torch.tensor(
            [FLOOR_LOGIT + 5.0, FLOOR_LOGIT - 0.3, FLOOR_LOGIT + 5.0],
            requires_grad=True,
        )
        loss, stats = coverage_loss(
            logits,
            torch.ones(3),
            torch.tensor([1, 1, 2]),
            torch.zeros(3, dtype=torch.long),
            torch.tensor([50, 51, 100]),
        )

        self.assertEqual(stats['target_group_count'], 1)
        self.assertEqual(stats['uncovered_group_count'], 1)
        self.assertAlmostEqual(loss.item(), 0.3, places=6)
        loss.backward()
        torch.testing.assert_close(
            logits.grad,
            torch.tensor([0.0, -1.0, 0.0]),
            rtol=0,
            atol=1e-7,
        )

    def test_correct_fraction_uses_kth_logit_and_sparse_gradient(self):
        logits = torch.full((10001,), FLOOR_LOGIT - 0.25)
        logits[0] = FLOOR_LOGIT + 0.5
        logits[1] = FLOOR_LOGIT - 0.1
        logits.requires_grad_()
        loss, stats = coverage_loss(
            logits,
            torch.ones_like(logits),
            torch.ones(logits.numel(), dtype=torch.long),
            torch.zeros(logits.numel(), dtype=torch.long),
            torch.ones(logits.numel(), dtype=torch.long),
        )

        self.assertEqual(stats['target_group_count'], 1)
        self.assertEqual(stats['uncovered_group_count'], 1)
        self.assertAlmostEqual(loss.item(), 0.1, places=6)
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 1)
        self.assertAlmostEqual(logits.grad[1].item(), -1.0, places=7)

        covered_logits = torch.full((10000,), FLOOR_LOGIT - 0.25)
        covered_logits[0] = FLOOR_LOGIT + 0.5
        covered_loss, covered_stats = coverage_loss(
            covered_logits,
            torch.ones_like(covered_logits),
            torch.ones(covered_logits.numel(), dtype=torch.long),
            torch.zeros(covered_logits.numel(), dtype=torch.long),
            torch.ones(covered_logits.numel(), dtype=torch.long),
        )
        self.assertEqual(covered_stats['uncovered_group_count'], 0)
        self.assertEqual(covered_loss.item(), 0.0)

    def test_no_target_returns_graph_connected_zero(self):
        logits = torch.tensor([0.2, -0.4, 1.0], requires_grad=True)
        loss, stats = coverage_loss(
            logits,
            torch.tensor([0.0, 1.0, 1.0]),
            torch.tensor([0, 3, 4]),
            torch.zeros(3, dtype=torch.long),
            torch.tensor([1, 50, 100]),
        )

        self.assertEqual(
            stats,
            {'target_group_count': 0, 'uncovered_group_count': 0},
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(
            logits.grad,
            torch.zeros_like(logits),
            rtol=0,
            atol=0,
        )

    def test_identical_target_and_time_are_isolated_by_batch(self):
        logits = torch.tensor(
            [FLOOR_LOGIT + 0.2, FLOOR_LOGIT - 0.4],
            requires_grad=True,
        )
        loss, stats = coverage_loss(
            logits,
            torch.ones(2),
            torch.tensor([7, 7]),
            torch.tensor([0, 1]),
            torch.tensor([1, 1]),
        )

        self.assertEqual(stats['target_group_count'], 2)
        self.assertEqual(stats['uncovered_group_count'], 1)
        self.assertAlmostEqual(loss.item(), 0.2, places=6)
        loss.backward()
        torch.testing.assert_close(
            logits.grad,
            torch.tensor([0.0, -0.5]),
            rtol=0,
            atol=1e-7,
        )


class TemporalMemoryTimestampAlignmentTest(unittest.TestCase):
    def test_dataset_and_collate_preserve_raw_timestamp_alignment(self):
        locations = np.asarray(
            [
                [0, 0, 51],
                [1, 0, 0],
                [2, 0, 49],
                [3, 0, 50],
                [0, 1, 1],
                [1, 1, 799],
            ],
            dtype=np.int64,
        )
        labels = np.asarray([1, 0, 1, 1, 0, 1], dtype=np.float32)
        target_ids = np.asarray([1, 0, 2, 3, 0, 4], dtype=np.int64)
        evs_norm = np.zeros((locations.shape[0], 6), dtype=np.float32)
        evs_norm[:, 3] = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.float32)
        evs_norm[:, 4] = labels
        evs_norm[:, 5] = target_ids

        with tempfile.TemporaryDirectory() as directory:
            np.savez(
                Path(directory) / 'train_000.npz',
                ev_loc=locations,
                evs_norm=evs_norm,
            )
            dataset = TemporalMemoryTrainDataset(
                root=directory,
                whole_t=800,
                temporal_bin_size=50,
                context_bins=5,
                sequence_length=16,
                width=4,
                height=3,
                views_per_video=1,
                positive_frame_probability=0.0,
                random_seed=49,
                cache_all_videos=False,
                cache_video_count=1,
            )
            sample = dataset[0]

        expected_order = np.asarray([1, 2, 4, 0, 3, 5])
        np.testing.assert_array_equal(
            sample['event_timestamps'],
            locations[expected_order, 2],
        )
        np.testing.assert_array_equal(
            sample['event_time_indices'],
            np.asarray([0, 0, 0, 1, 1, 15]),
        )
        np.testing.assert_array_equal(sample['event_x'], locations[expected_order, 0])
        np.testing.assert_array_equal(sample['event_y'], locations[expected_order, 1])
        np.testing.assert_array_equal(sample['labels'], labels[expected_order])
        np.testing.assert_array_equal(sample['target_ids'], target_ids[expected_order])

        batch = temporal_memory_collate([sample])
        for key in (
            'event_timestamps',
            'event_time_indices',
            'event_x',
            'event_y',
            'target_ids',
        ):
            torch.testing.assert_close(
                batch[key],
                torch.from_numpy(sample[key]).long(),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            batch['labels'],
            torch.from_numpy(sample['labels']).float(),
            rtol=0,
            atol=0,
        )


class TargetCoverageIntegrationTest(unittest.TestCase):
    def test_default_config_is_opt_in_with_audited_m24_values(self):
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
        memory = config['TEMPORAL_MEMORY']
        self.assertFalse(memory['temporal_memory_target_coverage_enabled'])
        self.assertEqual(memory['temporal_memory_target_coverage_weight'], 0.005)
        self.assertEqual(
            memory['temporal_memory_target_coverage_warmup_epochs'],
            1,
        )
        self.assertEqual(
            memory['temporal_memory_target_coverage_score_floor'],
            0.719,
        )
        self.assertEqual(
            memory['temporal_memory_target_coverage_correct_fraction'],
            0.0001,
        )

    def test_disabled_and_epoch_zero_paths_preserve_bce_loss_and_gradient(self):
        initial_logits = torch.tensor([0.1, -0.4, 0.8, -1.2])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        time_indices = torch.tensor([0, 0, 1, 1])
        target_ids = torch.tensor([1, 0, 1, 0])
        batch_ids = torch.zeros(4, dtype=torch.long)
        timestamps = torch.tensor([1, 2, 51, 52])

        reference_logits = initial_logits.clone().requires_grad_()
        reference_loss, _ = frame_balanced_event_bce(
            reference_logits,
            labels,
            time_indices,
            target_positive_loss_mass=0.2,
            max_positive_weight=16.0,
        )
        reference_loss.backward()

        for enabled, epoch in ((False, 3), (True, 0)):
            with self.subTest(enabled=enabled, epoch=epoch):
                logits = initial_logits.clone().requires_grad_()
                base_loss, _ = frame_balanced_event_bce(
                    logits,
                    labels,
                    time_indices,
                    target_positive_loss_mass=0.2,
                    max_positive_weight=16.0,
                )
                auxiliary_loss, _ = coverage_loss(
                    logits,
                    labels,
                    target_ids,
                    batch_ids,
                    timestamps,
                )
                combined_loss, applied_weight = apply_target_coverage_loss(
                    base_loss,
                    auxiliary_loss,
                    enabled=enabled,
                    epoch=epoch,
                    warmup_epochs=1,
                    weight=0.005,
                )
                self.assertIs(combined_loss, base_loss)
                self.assertEqual(applied_weight, 0.0)
                torch.testing.assert_close(
                    combined_loss,
                    reference_loss.detach(),
                    rtol=0,
                    atol=0,
                )
                combined_loss.backward()
                torch.testing.assert_close(
                    logits.grad,
                    reference_logits.grad,
                    rtol=0,
                    atol=0,
                )


if __name__ == '__main__':
    unittest.main()
