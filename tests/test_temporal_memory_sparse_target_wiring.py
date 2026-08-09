"""CPU-only wiring tests for sparse-target-support training sampling."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = tuple(sorted((PROJECT_ROOT / 'configs').glob('evisseg_evuav*.yaml')))
OFFICIAL_DATA_ROOT = (
    PROJECT_ROOT.parent / 'datasets' / 'EV-UAV-Challenge2'
)
ORIGINAL_ARGV = sys.argv
sys.argv = ['sparse-target-wiring-test', '--config', str(CONFIG_PATHS[0])]

from train_temporal_memory import (  # noqa: E402
    build_temporal_memory_train_dataset,
    validate_resume_config,
)

sys.argv = ORIGINAL_ARGV


def write_video(path):
    groups = (
        (0, 1, 1, 1.0),
        (1, 1, 3, 1.0),
        (2, 1, 4, 1.0),
        (3, 0, 2, 1.0),
        (4, 2, 2, 1.0),
    )
    locations = []
    rows = []
    event_index = 0
    for temporal_bin, target_id, count, label in groups:
        for _ in range(count):
            locations.append(
                (event_index % 16, event_index % 12, temporal_bin * 50 + 1)
            )
            row = np.zeros(6, dtype=np.float32)
            row[3] = 1.0
            row[4] = label
            row[5] = target_id
            rows.append(row)
            event_index += 1
    np.savez(
        path,
        ev_loc=np.asarray(locations, dtype=np.int64),
        evs_norm=np.stack(rows),
    )


def make_config(root, **overrides):
    options = {
        'root': str(root),
        'whole_t': 400,
        'res': [16, 12],
        'seed': 49,
        'temporal_memory_bin_size': 50,
        'temporal_memory_context_bins': 5,
        'temporal_memory_sequence_length': 2,
        'temporal_memory_train_views_per_video': 1,
        'temporal_memory_positive_frame_probability': 0.75,
        'temporal_memory_log_count_clip': 4.0,
        'temporal_memory_cache_all_videos': False,
        'temporal_memory_cache_video_count': 1,
        'temporal_memory_dense_sampling_enabled': False,
        'temporal_memory_dense_event_count_cutoff': 200000,
        'temporal_memory_dense_view_multiplier': 2,
        'temporal_memory_density_bucket_boundaries': [],
        'temporal_memory_density_bucket_views': [],
        'temporal_memory_train_min_event_count_exclusive': None,
        'temporal_memory_sparse_target_support_sampling_enabled': True,
        'temporal_memory_sparse_target_support_max_events': 3,
        'temporal_memory_sparse_target_support_probability': 0.75,
    }
    options.update(overrides)
    return SimpleNamespace(**options)


class SparseTargetSupportWiringTest(unittest.TestCase):
    def test_all_shipped_configs_are_default_off(self):
        self.assertEqual(len(CONFIG_PATHS), 5)
        for path in CONFIG_PATHS:
            with self.subTest(path=path.name):
                config = yaml.safe_load(path.read_text(encoding='utf-8'))
                temporal_memory = config['TEMPORAL_MEMORY']
                self.assertIs(
                    temporal_memory[
                        'temporal_memory_sparse_target_support_sampling_enabled'
                    ],
                    False,
                )
                self.assertEqual(
                    temporal_memory[
                        'temporal_memory_sparse_target_support_max_events'
                    ],
                    3,
                )
                self.assertEqual(
                    temporal_memory[
                        'temporal_memory_sparse_target_support_probability'
                    ],
                    0.75,
                )

    def test_training_dataset_receives_resolved_sampler_options(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / 'train').mkdir()
            write_video(root / 'train' / 'train_000.npz')
            dataset = build_temporal_memory_train_dataset(make_config(root))

        summary = dataset.sampling_summary()
        self.assertTrue(dataset.sparse_target_support_sampling_enabled)
        self.assertEqual(dataset.sparse_target_support_max_events, 3)
        self.assertEqual(dataset.sparse_target_support_probability, 0.75)
        self.assertEqual(summary['sparse_target_support_video_count'], 1)
        self.assertEqual(summary['sparse_target_support_bin_count'], 3)

    def test_training_dataset_uses_legacy_defaults_when_keys_are_absent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / 'train').mkdir()
            write_video(root / 'train' / 'train_000.npz')
            config = make_config(root)
            del config.temporal_memory_sparse_target_support_sampling_enabled
            del config.temporal_memory_sparse_target_support_max_events
            del config.temporal_memory_sparse_target_support_probability
            dataset = build_temporal_memory_train_dataset(config)

        self.assertFalse(dataset.sparse_target_support_sampling_enabled)
        self.assertEqual(dataset.sparse_target_support_max_events, 3)
        self.assertEqual(dataset.sparse_target_support_probability, 0.75)
        self.assertNotIn(
            'sparse_target_support_sampling_enabled',
            dataset.sampling_summary(),
        )

    def test_legacy_resume_accepts_only_missing_default_values(self):
        defaults = {
            'temporal_memory_sparse_target_support_sampling_enabled': False,
            'temporal_memory_sparse_target_support_max_events': 3,
            'temporal_memory_sparse_target_support_probability': 0.75,
        }
        checkpoint = {
            'provenance': {'resolved_config': {'TEMPORAL_MEMORY': {}}}
        }
        current = SimpleNamespace(
            resolved_config={'TEMPORAL_MEMORY': copy.deepcopy(defaults)}
        )
        validate_resume_config(checkpoint, current)

        changed_values = {
            'temporal_memory_sparse_target_support_sampling_enabled': True,
            'temporal_memory_sparse_target_support_max_events': 2,
            'temporal_memory_sparse_target_support_probability': 1.0,
        }
        for key, value in changed_values.items():
            with self.subTest(key=key):
                changed = copy.deepcopy(defaults)
                changed[key] = value
                config = SimpleNamespace(
                    resolved_config={'TEMPORAL_MEMORY': changed}
                )
                with self.assertRaisesRegex(ValueError, key):
                    validate_resume_config(checkpoint, config)

    @unittest.skipUnless(
        (OFFICIAL_DATA_ROOT / 'train').is_dir(),
        'official EV-UAV Challenge 2 train split is unavailable',
    )
    def test_official_h17_route_has_expected_sparse_pool_summary(self):
        dataset = build_temporal_memory_train_dataset(
            make_config(
                OFFICIAL_DATA_ROOT,
                whole_t=8000,
                res=[346, 260],
                temporal_memory_sequence_length=16,
                temporal_memory_cache_video_count=2,
                temporal_memory_train_min_event_count_exclusive=30000,
            )
        )
        summary = dataset.sampling_summary()

        self.assertEqual(summary['source_video_count'], 99)
        self.assertEqual(summary['video_count'], 54)
        self.assertEqual(summary['excluded_video_count'], 45)
        self.assertEqual(summary['sequence_count'], 54)
        self.assertEqual(summary['sparse_target_support_video_count'], 54)
        self.assertEqual(summary['sparse_target_support_bin_count'], 1343)


if __name__ == '__main__':
    unittest.main()
