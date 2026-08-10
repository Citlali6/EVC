"""CPU-only tests for the init-only T16 -> T32 warm-start contract."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'evisseg_evuav.yaml'
CONFIG_PATHS = sorted((PROJECT_ROOT / 'configs').glob('*.yaml'))
M20_CHECKPOINT_PATH = (
    PROJECT_ROOT / 'checkpoints' / 'm20_attn_dense_views8_epoch_003_seed48.pt'
)
ORIGINAL_ARGV = sys.argv
sys.argv = ['sequence-length-warm-start-test', '--config', str(CONFIG_PATH)]

from model.temporal_memory_net import BidirectionalTemporalMemoryNet  # noqa: E402
from train_temporal_memory import (  # noqa: E402
    ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
    LEGACY_RESUME_CONFIG_DEFAULTS,
    SEQUENCE_LENGTH_WARM_START_MIGRATION,
    build_training_checkpoint,
    configure_temporal_memory_trainable_parameters,
    frozen_model_state_sha256,
    load_checkpoint_file,
    load_p23_base_weights,
    load_training_resume,
    sha256_file,
    validate_resume_config,
)
from utils.temporal_memory_inference import (  # noqa: E402
    load_temporal_memory_model,
)

sys.argv = ORIGINAL_ARGV


def new_m20_model():
    return BidirectionalTemporalMemoryNet(
        input_channels=10,
        width=16,
        density_calibration_enabled=True,
        confidence_head_enabled=False,
        temporal_attention_enabled=True,
    )


def checkpoint_config(warm_start_enabled):
    resolved = {
        'TEMPORAL_MEMORY': {
            'temporal_memory_sequence_length': 32,
            'temporal_memory_init_sequence_length_warm_start_enabled': bool(
                warm_start_enabled
            ),
            'temporal_memory_attention_projection_only_enabled': True,
        }
    }
    return SimpleNamespace(
        temporal_memory_bin_size=50,
        temporal_memory_context_bins=5,
        temporal_memory_width=16,
        temporal_memory_sequence_length=32,
        temporal_memory_log_count_clip=4.0,
        temporal_frame_density_calibration_enabled=True,
        temporal_frame_trajectory_extrapolation_enabled=False,
        temporal_frame_confidence_head_enabled=False,
        temporal_memory_confidence_only_enabled=False,
        temporal_memory_freeze_base_enabled=False,
        temporal_memory_head_only_enabled=False,
        temporal_memory_dacc_v2_enabled=False,
        temporal_memory_dacc_v2_only_enabled=False,
        temporal_memory_attention_projection_only_enabled=True,
        temporal_memory_temporal_attention_enabled=True,
        temporal_memory_init_sequence_length_warm_start_enabled=bool(
            warm_start_enabled
        ),
        resolved_config=resolved,
        config_overrides=[],
    )


class TemporalMemorySequenceLengthWarmStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = load_checkpoint_file(M20_CHECKPOINT_PATH, map_location='cpu')
        cls.parent_sha256 = sha256_file(M20_CHECKPOINT_PATH)

    def load_parent(
        self,
        model,
        *,
        enabled,
        target_sequence_length,
        temporal_bin_size=50,
        log_count_clip=4.0,
        trajectory_extrapolation_enabled=False,
        attention_projection_only_enabled=True,
        checkpoint_path=M20_CHECKPOINT_PATH,
    ):
        migrations = []
        loaded = load_p23_base_weights(
            model,
            checkpoint_path,
            context_bins=5,
            width=16,
            density_calibration_enabled=True,
            confidence_head_enabled=False,
            density_calibration_v2_enabled=False,
            initialization_migrations=migrations,
            target_sequence_length=target_sequence_length,
            sequence_length_warm_start_enabled=enabled,
            temporal_bin_size=temporal_bin_size,
            log_count_clip=log_count_clip,
            trajectory_extrapolation_enabled=(
                trajectory_extrapolation_enabled
            ),
            attention_projection_only_enabled=(
                attention_projection_only_enabled
            ),
        )
        return loaded, migrations

    def assert_parent_state_identical(self, model):
        actual = model.state_dict()
        expected = self.parent['model_state_dict']
        self.assertEqual(set(actual), set(expected))
        for name in sorted(expected):
            self.assertEqual(actual[name].dtype, expected[name].dtype, name)
            self.assertEqual(tuple(actual[name].shape), tuple(expected[name].shape), name)
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_all_defaults_are_off_and_legacy_resume_default_is_off(self):
        for path in CONFIG_PATHS:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text(encoding='utf-8'))
                self.assertIs(
                    config['TEMPORAL_MEMORY'][
                        'temporal_memory_init_sequence_length_warm_start_enabled'
                    ],
                    False,
                )
        self.assertIs(
            LEGACY_RESUME_CONFIG_DEFAULTS[
                (
                    'TEMPORAL_MEMORY',
                    'temporal_memory_init_sequence_length_warm_start_enabled',
                )
            ],
            False,
        )

    def test_default_rejects_t16_to_t32_and_preserves_t16_path(self):
        with self.assertRaisesRegex(ValueError, 'sequence_length=16'):
            self.load_parent(
                new_m20_model(),
                enabled=False,
                target_sequence_length=32,
            )
        model = new_m20_model()
        _, migrations = self.load_parent(
            model,
            enabled=False,
            target_sequence_length=16,
        )
        self.assertEqual(migrations, [])
        self.assert_parent_state_identical(model)

    def test_explicit_t16_to_t32_is_strict_and_audited(self):
        model = new_m20_model()
        loaded, migrations = self.load_parent(
            model,
            enabled=True,
            target_sequence_length=32,
        )
        self.assertEqual(loaded.resolve(), M20_CHECKPOINT_PATH.resolve())
        self.assert_parent_state_identical(model)
        self.assertEqual(len(migrations), 1)
        migration = migrations[0]
        self.assertEqual(migration['name'], SEQUENCE_LENGTH_WARM_START_MIGRATION)
        self.assertEqual(migration['source_sequence_length'], 16)
        self.assertEqual(migration['target_sequence_length'], 32)
        self.assertEqual(migration['metadata_difference_allowlist'], ['sequence_length'])
        self.assertIs(migration['state_dict_strict'], True)
        self.assertEqual(migration['parent_checkpoint_sha256'], self.parent_sha256)
        self.assertEqual(
            migration['source_model_state_sha256'],
            migration['loaded_model_state_sha256'],
        )

    def test_warm_start_rejects_other_metadata_or_state_differences(self):
        with self.assertRaisesRegex(
            ValueError,
            'authorized only for temporal_attention_projection_only',
        ):
            self.load_parent(
                new_m20_model(),
                enabled=True,
                target_sequence_length=32,
                attention_projection_only_enabled=False,
            )
        cases = (
            {'temporal_bin_size': 51},
            {'log_count_clip': 5.0},
            {'trajectory_extrapolation_enabled': True},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, 'only sequence_length'):
                    self.load_parent(
                        new_m20_model(),
                        enabled=True,
                        target_sequence_length=32,
                        **overrides,
                    )
        with self.assertRaisesRegex(ValueError, 'restricted to T16 -> T32'):
            self.load_parent(
                new_m20_model(),
                enabled=True,
                target_sequence_length=24,
            )

        tampered = copy.deepcopy(self.parent)
        tampered['model_state_dict'].pop('memory_projection.bias')
        with tempfile.TemporaryDirectory() as temporary_directory:
            tampered_path = Path(temporary_directory) / 'missing-state.pt'
            torch.save(tampered, tampered_path)
            with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                self.load_parent(
                    new_m20_model(),
                    enabled=True,
                    target_sequence_length=32,
                    checkpoint_path=tampered_path,
                )

    def test_t32_checkpoint_provenance_attention_scope_and_inference_are_strict(self):
        model = new_m20_model()
        _, migrations = self.load_parent(
            model,
            enabled=True,
            target_sequence_length=32,
        )
        configure_temporal_memory_trainable_parameters(
            model,
            attention_projection_only_enabled=True,
        )
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(trainable, ATTENTION_PROJECTION_MUTABLE_STATE_KEYS)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            9312,
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        config = checkpoint_config(True)
        checkpoint = build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            epoch_loss=0.1,
            best_loss=0.1,
            best_epoch=0,
            config=config,
            initialized_from=M20_CHECKPOINT_PATH,
            initialized_from_sha256=self.parent_sha256,
            initialization_migrations=migrations,
            include_cuda_rng=False,
            frozen_state_reference_sha256=frozen_model_state_sha256(
                model,
                mutable_state_keys=ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
            ),
        )
        self.assertEqual(checkpoint['temporal_memory']['sequence_length'], 32)
        self.assertIs(
            checkpoint['temporal_memory'][
                'init_sequence_length_warm_start_enabled'
            ],
            True,
        )
        recorded = checkpoint['provenance']['initialization_migrations'][0]
        self.assertEqual(recorded['source_sequence_length'], 16)
        self.assertEqual(recorded['target_sequence_length'], 32)
        self.assertEqual(recorded['parent_checkpoint_sha256'], self.parent_sha256)

        with tempfile.TemporaryDirectory() as temporary_directory:
            t32_path = Path(temporary_directory) / 't32.pt'
            torch.save(checkpoint, t32_path)
            restored, _ = load_temporal_memory_model(
                t32_path,
                device=torch.device('cpu'),
                context_bins=5,
                width=16,
                sequence_length=32,
            )
            self.assertEqual(set(restored.state_dict()), set(model.state_dict()))
            with self.assertRaisesRegex(ValueError, 'must both be 32'):
                load_temporal_memory_model(
                    t32_path,
                    device=torch.device('cpu'),
                    context_bins=5,
                    width=16,
                    sequence_length=16,
                )

            missing_sequence = copy.deepcopy(checkpoint)
            missing_sequence['temporal_memory'].pop('sequence_length')
            missing_sequence_path = (
                Path(temporary_directory) / 't32-missing-sequence.pt'
            )
            torch.save(missing_sequence, missing_sequence_path)
            with self.assertRaisesRegex(ValueError, 'missing required sequence_length'):
                load_temporal_memory_model(
                    missing_sequence_path,
                    device=torch.device('cpu'),
                    context_bins=5,
                    width=16,
                    sequence_length=32,
                )

            stripped_marker = copy.deepcopy(missing_sequence)
            stripped_marker['temporal_memory'].pop(
                'init_sequence_length_warm_start_enabled'
            )
            stripped_marker_path = (
                Path(temporary_directory) / 't32-stripped-marker.pt'
            )
            torch.save(stripped_marker, stripped_marker_path)
            with self.assertRaisesRegex(ValueError, 'metadata marker is missing'):
                load_temporal_memory_model(
                    stripped_marker_path,
                    device=torch.device('cpu'),
                    context_bins=5,
                    width=16,
                    sequence_length=32,
                )

            resolved_config_only = copy.deepcopy(stripped_marker)
            resolved_config_only['provenance']['initialization_migrations'] = []
            resolved_config_only_path = (
                Path(temporary_directory) / 't32-resolved-config-only.pt'
            )
            torch.save(resolved_config_only, resolved_config_only_path)
            with self.assertRaisesRegex(ValueError, 'metadata marker is missing'):
                load_temporal_memory_model(
                    resolved_config_only_path,
                    device=torch.device('cpu'),
                    context_bins=5,
                    width=16,
                    sequence_length=32,
                )

            migration_only = copy.deepcopy(stripped_marker)
            migration_only['provenance']['resolved_config']['TEMPORAL_MEMORY'][
                'temporal_memory_init_sequence_length_warm_start_enabled'
            ] = False
            migration_only_path = (
                Path(temporary_directory) / 't32-migration-only.pt'
            )
            torch.save(migration_only, migration_only_path)
            with self.assertRaisesRegex(ValueError, 'metadata marker is missing'):
                load_temporal_memory_model(
                    migration_only_path,
                    device=torch.device('cpu'),
                    context_bins=5,
                    width=16,
                    sequence_length=32,
                )

            restored_model = new_m20_model()
            configure_temporal_memory_trainable_parameters(
                restored_model,
                attention_projection_only_enabled=True,
            )
            restored_optimizer = torch.optim.AdamW(
                (
                    parameter
                    for parameter in restored_model.parameters()
                    if parameter.requires_grad
                ),
                lr=1e-6,
            )
            restored_scheduler = torch.optim.lr_scheduler.StepLR(
                restored_optimizer,
                step_size=1,
            )
            with self.assertRaisesRegex(ValueError, 'missing required sequence_length'):
                load_training_resume(
                    missing_sequence_path,
                    restored_model,
                    restored_optimizer,
                    restored_scheduler,
                    current_config=config,
                    restore_rng=False,
                )

            tampered_migration = copy.deepcopy(checkpoint)
            tampered_migration['provenance']['initialization_migrations'][0][
                'parent_checkpoint_sha256'
            ] = '0' * 64
            tampered_migration_path = (
                Path(temporary_directory) / 't32-tampered-migration.pt'
            )
            torch.save(tampered_migration, tampered_migration_path)
            with self.assertRaisesRegex(ValueError, 'parent checkpoint SHA-256'):
                load_training_resume(
                    tampered_migration_path,
                    restored_model,
                    restored_optimizer,
                    restored_scheduler,
                    current_config=config,
                    restore_rng=False,
                )

            extra_migration = copy.deepcopy(checkpoint)
            extra_migration['provenance']['initialization_migrations'].append(
                {'name': 'unauthorized_extra_migration'}
            )
            extra_migration_path = (
                Path(temporary_directory) / 't32-extra-migration.pt'
            )
            torch.save(extra_migration, extra_migration_path)
            with self.assertRaisesRegex(ValueError, 'sole T16 -> T32'):
                load_temporal_memory_model(
                    extra_migration_path,
                    device=torch.device('cpu'),
                    context_bins=5,
                    width=16,
                    sequence_length=32,
                )
            with self.assertRaisesRegex(ValueError, 'sole initialization migration'):
                load_training_resume(
                    extra_migration_path,
                    restored_model,
                    restored_optimizer,
                    restored_scheduler,
                    current_config=config,
                    restore_rng=False,
                )

    def test_resume_remains_strict_but_accepts_missing_legacy_false_key(self):
        old_resolved = {'TEMPORAL_MEMORY': {'temporal_memory_sequence_length': 32}}
        checkpoint = {'provenance': {'resolved_config': old_resolved}}
        current_false = SimpleNamespace(
            resolved_config={
                'TEMPORAL_MEMORY': {
                    'temporal_memory_sequence_length': 32,
                    'temporal_memory_init_sequence_length_warm_start_enabled': False,
                }
            }
        )
        validate_resume_config(checkpoint, current_false)
        current_true = copy.deepcopy(current_false)
        current_true.resolved_config['TEMPORAL_MEMORY'][
            'temporal_memory_init_sequence_length_warm_start_enabled'
        ] = True
        with self.assertRaisesRegex(
            ValueError,
            'temporal_memory_init_sequence_length_warm_start_enabled',
        ):
            validate_resume_config(checkpoint, current_true)


if __name__ == '__main__':
    unittest.main()
