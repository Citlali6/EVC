"""CPU-only authenticity and compatibility tests for DACC-v2."""

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
M20_PATH = (
    PROJECT_ROOT
    / 'checkpoints'
    / 'm20_attn_dense_views8_epoch_003_seed48.pt'
)
ORIGINAL_ARGV = sys.argv
sys.argv = ['dacc-v2-test', '--config', str(CONFIG_PATH)]

from model.temporal_frame_net import (  # noqa: E402
    DENSITY_CALIBRATION_V2_BASIS,
    DENSITY_CALIBRATION_V2_RESIDUAL_SCALE,
    validate_density_calibration_metadata,
)
from model.temporal_memory_net import (  # noqa: E402
    BidirectionalTemporalMemoryNet,
)
from create_dacc_v2_identity_checkpoint import (  # noqa: E402
    PROJECTION_KEY,
    atomic_torch_save,
    build_identity_checkpoint,
)
from train_temporal_memory import (  # noqa: E402
    DACC_V2_MUTABLE_STATE_KEYS,
    assert_frozen_model_state_unchanged,
    build_optimizer,
    build_training_checkpoint,
    configure_temporal_memory_trainable_parameters,
    frozen_model_state_sha256,
    load_p23_base_weights,
    load_training_resume,
    resolve_temporal_memory_training_scope,
    set_temporal_memory_training_mode,
    snapshot_frozen_model_state,
    validate_dacc_v2_training_config,
)
from utils.temporal_memory_inference import (  # noqa: E402
    load_temporal_memory_model,
)

sys.argv = ORIGINAL_ARGV


def new_model(v2_enabled):
    return BidirectionalTemporalMemoryNet(
        input_channels=10,
        width=16,
        density_calibration_enabled=True,
        density_calibration_v2_enabled=v2_enabled,
        confidence_head_enabled=False,
        temporal_attention_enabled=True,
    )


def dacc_config():
    resolved = {
        'TRAIN': {'epochs': 3, 'lr': 1e-4},
        'TEMPORAL_MEMORY': {
            'temporal_memory_dacc_v2_enabled': True,
            'temporal_memory_dacc_v2_only_enabled': True,
            'temporal_memory_dacc_v2_lr_multiplier': 1.0,
            'temporal_memory_train_min_event_count_exclusive': 30000,
        },
    }
    return SimpleNamespace(
        lr=1e-4,
        temporal_memory_base_lr_multiplier=1.0,
        temporal_memory_memory_lr_multiplier=1.0,
        temporal_memory_confidence_lr_multiplier=1.0,
        temporal_memory_dacc_v2_lr_multiplier=1.0,
        temporal_memory_bin_size=50,
        temporal_memory_context_bins=5,
        temporal_memory_width=16,
        temporal_memory_sequence_length=16,
        temporal_memory_log_count_clip=4.0,
        temporal_memory_train_views_per_video=2,
        temporal_memory_train_min_event_count_exclusive=30000,
        temporal_memory_dense_sampling_enabled=False,
        temporal_memory_density_bucket_boundaries=[],
        temporal_memory_density_bucket_views=[],
        temporal_frame_density_calibration_enabled=True,
        temporal_frame_trajectory_extrapolation_enabled=False,
        temporal_frame_confidence_head_enabled=False,
        temporal_memory_confidence_only_enabled=False,
        temporal_memory_freeze_base_enabled=False,
        temporal_memory_head_only_enabled=False,
        temporal_memory_dacc_v2_enabled=True,
        temporal_memory_dacc_v2_only_enabled=True,
        temporal_memory_temporal_attention_enabled=True,
        temporal_memory_metric_aux_enabled=False,
        temporal_memory_target_coverage_enabled=False,
        resolved_config=resolved,
        config_overrides=[
            'TEMPORAL_MEMORY.temporal_memory_dacc_v2_enabled=true',
            'TEMPORAL_MEMORY.temporal_memory_dacc_v2_only_enabled=true',
        ],
    )


class TemporalMemoryDaccV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m20 = torch.load(M20_PATH, map_location='cpu')

    def migrated_model(self):
        model = new_model(True)
        result = model.load_state_dict(self.m20['model_state_dict'], strict=False)
        self.assertEqual(set(result.missing_keys), DACC_V2_MUTABLE_STATE_KEYS)
        self.assertEqual(result.unexpected_keys, [])
        return model

    def test_all_configs_default_v2_off(self):
        for path in CONFIG_PATHS:
            with self.subTest(config=path.name):
                memory = yaml.safe_load(path.read_text(encoding='utf-8'))[
                    'TEMPORAL_MEMORY'
                ]
                self.assertIs(memory['temporal_memory_dacc_v2_enabled'], False)
                self.assertIs(
                    memory['temporal_memory_dacc_v2_only_enabled'],
                    False,
                )
                self.assertEqual(
                    memory['temporal_memory_dacc_v2_lr_multiplier'],
                    1.0,
                )

    def test_metadata_is_explicit_and_legacy_fallback_is_v1(self):
        self.assertEqual(
            validate_density_calibration_metadata(
                {'density_calibration_enabled': True}
            ),
            1,
        )
        v2 = {
            'density_calibration_enabled': True,
            'density_calibration_version': 2,
            'density_calibration_v2_basis': DENSITY_CALIBRATION_V2_BASIS,
            'density_calibration_v2_residual_scale': (
                DENSITY_CALIBRATION_V2_RESIDUAL_SCALE
            ),
        }
        self.assertEqual(validate_density_calibration_metadata(v2), 2)
        for key, value in (
            ('density_calibration_version', 1),
            ('density_calibration_v2_basis', 'unknown'),
            ('density_calibration_v2_residual_scale', 0.25),
        ):
            with self.subTest(key=key):
                invalid = dict(v2)
                invalid[key] = value
                with self.assertRaises(ValueError):
                    validate_density_calibration_metadata(invalid)

    def test_zero_projection_is_bitwise_identical_to_released_m20(self):
        torch.manual_seed(901)
        released, released_metadata = load_temporal_memory_model(
            M20_PATH,
            'cpu',
            5,
            16,
            16,
        )
        self.assertFalse(released.density_calibration_v2_enabled)
        self.assertNotIn('density_calibration_version', released_metadata)
        v1 = new_model(False)
        v1.load_state_dict(self.m20['model_state_dict'], strict=True)
        v2 = self.migrated_model()
        v1.eval()
        v2.eval()
        frames = torch.rand(1, 1, 10, 32, 32)
        with torch.no_grad():
            expected = v1(frames)
            actual = v2(frames)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(
            torch.count_nonzero(
                v2.base.density_calibrator.residual_projection.weight
            ).item(),
            0,
        )

    def test_identity_artifact_is_explicitly_untrained_and_strict(self):
        identity, context_bins, width, sequence_length = (
            build_identity_checkpoint(M20_PATH)
        )
        audit = identity['identity_upgrade']
        self.assertEqual(
            audit['artifact_kind'],
            'untrained_dacc_v2_identity_upgrade',
        )
        self.assertIs(audit['trained_after_upgrade'], False)
        self.assertEqual(audit['trained_epochs_after_upgrade'], 0)
        self.assertEqual(audit['optimizer_steps_after_upgrade'], 0)
        self.assertEqual(audit['added_state_keys'], [PROJECTION_KEY])
        self.assertEqual(audit['added_parameter_count'], 32)
        self.assertEqual(audit['added_nonzero_count'], 0)
        self.assertEqual(
            audit['source_model_state_sha256'],
            audit['migrated_inherited_state_sha256'],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity.pt'
            torch.save(identity, path)
            model, loaded_checkpoint = load_temporal_memory_model(
                path,
                'cpu',
                context_bins,
                width,
                sequence_length,
            )
        self.assertTrue(model.density_calibration_v2_enabled)
        self.assertEqual(
            validate_density_calibration_metadata(
                loaded_checkpoint['temporal_memory']
            ),
            2,
        )

    def test_identity_atomic_save_does_not_promote_failed_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity.pt'
            path.write_bytes(b'known-good-existing-file')

            def reject(_temporary_path):
                raise RuntimeError('strict validation failed')

            with self.assertRaisesRegex(RuntimeError, 'strict validation failed'):
                atomic_torch_save(
                    {'not': 'a checkpoint'},
                    path,
                    validator=reject,
                    force=True,
                )
            self.assertEqual(path.read_bytes(), b'known-good-existing-file')

    def test_checkpoint_save_rejects_metadata_state_mismatch(self):
        model = new_model(False)
        model.load_state_dict(self.m20['model_state_dict'], strict=True)
        config = dacc_config()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        with self.assertRaisesRegex(ValueError, 'model version'):
            build_training_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=0,
                epoch_loss=0.1,
                best_loss=0.1,
                best_epoch=0,
                config=config,
                initialized_from=M20_PATH,
                include_cuda_rng=False,
                frozen_state_reference_sha256='not-used-before-rejection',
            )

    def test_only_init_loader_permits_exact_v1_to_v2_migration(self):
        model = new_model(True)
        migrations = []
        loaded = load_p23_base_weights(
            model,
            M20_PATH,
            context_bins=5,
            width=16,
            density_calibration_enabled=True,
            confidence_head_enabled=False,
            density_calibration_v2_enabled=True,
            initialization_migrations=migrations,
        )
        self.assertEqual(loaded, M20_PATH)
        self.assertEqual(
            migrations[0]['missing_keys'],
            sorted(DACC_V2_MUTABLE_STATE_KEYS),
        )
        self.assertEqual(
            migrations[0]['source_model_state_sha256'],
            migrations[0]['migrated_frozen_state_sha256'],
        )

        damaged = copy.deepcopy(self.m20)
        damaged['model_state_dict'].pop('memory_projection.bias')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'damaged.pt'
            torch.save(damaged, path)
            with self.assertRaisesRegex(RuntimeError, 'Only newly attached'):
                load_p23_base_weights(
                    new_model(True),
                    path,
                    5,
                    16,
                    density_calibration_enabled=True,
                    density_calibration_v2_enabled=True,
                )

    def test_projection_only_step_has_signal_and_preserves_frozen_bytes(self):
        torch.manual_seed(902)
        model = self.migrated_model()
        configure_temporal_memory_trainable_parameters(
            model,
            dacc_v2_only_enabled=True,
        )
        set_temporal_memory_training_mode(model, dacc_v2_only_enabled=True)
        optimizer = build_optimizer(
            model,
            dacc_config(),
            dacc_v2_only_enabled=True,
        )
        self.assertEqual([g['name'] for g in optimizer.param_groups], ['dacc_v2'])
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.0)
        self.assertEqual(sum(p.numel() for p in optimizer.param_groups[0]['params']), 32)
        frozen = snapshot_frozen_model_state(
            model,
            mutable_state_keys=DACC_V2_MUTABLE_STATE_KEYS,
        )
        frozen_hash = frozen_model_state_sha256(
            model,
            mutable_state_keys=DACC_V2_MUTABLE_STATE_KEYS,
        )
        projection = model.base.density_calibrator.residual_projection.weight
        before = projection.detach().clone()
        frames = torch.rand(1, 1, 10, 32, 32)
        target = torch.randint(0, 2, (1, 1, 1, 32, 32)).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(frames),
            target,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assertIsNotNone(projection.grad)
        self.assertTrue(torch.isfinite(projection.grad).all())
        self.assertGreater(torch.count_nonzero(projection.grad).item(), 0)
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if name not in DACC_V2_MUTABLE_STATE_KEYS
            )
        )
        optimizer.step()
        self.assertFalse(torch.equal(before, projection))
        assert_frozen_model_state_unchanged(
            model,
            frozen,
            mutable_state_keys=DACC_V2_MUTABLE_STATE_KEYS,
        )
        self.assertEqual(
            frozen_model_state_sha256(
                model,
                mutable_state_keys=DACC_V2_MUTABLE_STATE_KEYS,
            ),
            frozen_hash,
        )

    def test_v2_inference_and_resume_are_strict(self):
        model = self.migrated_model()
        config = dacc_config()
        configure_temporal_memory_trainable_parameters(
            model,
            dacc_v2_only_enabled=True,
        )
        optimizer = build_optimizer(model, config, dacc_v2_only_enabled=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        frozen_hash = frozen_model_state_sha256(
            model,
            mutable_state_keys=DACC_V2_MUTABLE_STATE_KEYS,
        )
        checkpoint = build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            epoch_loss=0.1,
            best_loss=0.1,
            best_epoch=0,
            config=config,
            initialized_from=M20_PATH,
            initialized_from_sha256='parent-sha256',
            initialization_migrations=[{'name': 'v1-to-v2'}],
            include_cuda_rng=False,
            frozen_state_reference_sha256=frozen_hash,
        )
        self.assertEqual(checkpoint['temporal_memory']['density_calibration_version'], 2)
        self.assertEqual(
            checkpoint['temporal_memory']['density_calibration_v2_basis'],
            DENSITY_CALIBRATION_V2_BASIS,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v2.pt'
            torch.save(checkpoint, path)
            inferred, _ = load_temporal_memory_model(path, 'cpu', 5, 16, 16)
            self.assertTrue(inferred.density_calibration_v2_enabled)

            restored = new_model(True)
            configure_temporal_memory_trainable_parameters(
                restored,
                dacc_v2_only_enabled=True,
            )
            restored_optimizer = build_optimizer(
                restored,
                config,
                dacc_v2_only_enabled=True,
            )
            restored_scheduler = torch.optim.lr_scheduler.StepLR(
                restored_optimizer,
                1,
            )
            _, start_epoch = load_training_resume(
                path,
                restored,
                restored_optimizer,
                restored_scheduler,
                current_config=config,
                restore_rng=False,
            )
            self.assertEqual(start_epoch, 1)

            missing = copy.deepcopy(checkpoint)
            missing['model_state_dict'].pop(
                'base.density_calibrator.residual_projection.weight'
            )
            missing_path = Path(directory) / 'missing.pt'
            torch.save(missing, missing_path)
            with self.assertRaises(RuntimeError):
                load_temporal_memory_model(missing_path, 'cpu', 5, 16, 16)

            wrong_metadata = copy.deepcopy(checkpoint)
            wrong_metadata['temporal_memory'][
                'density_calibration_v2_basis'
            ] = 'wrong'
            wrong_path = Path(directory) / 'wrong-metadata.pt'
            torch.save(wrong_metadata, wrong_path)
            with self.assertRaises(ValueError):
                load_temporal_memory_model(wrong_path, 'cpu', 5, 16, 16)

    def test_scope_and_route_contract(self):
        config = dacc_config()
        self.assertEqual(
            resolve_temporal_memory_training_scope(
                dacc_v2_only_enabled=True
            ),
            'dacc_v2_projection_only',
        )
        validate_dacc_v2_training_config(
            config,
            'dacc_v2_projection_only',
        )
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            resolve_temporal_memory_training_scope(
                head_only_enabled=True,
                dacc_v2_only_enabled=True,
            )
        config.temporal_memory_train_min_event_count_exclusive = None
        with self.assertRaisesRegex(ValueError, 'event-count filter'):
            validate_dacc_v2_training_config(
                config,
                'dacc_v2_projection_only',
            )
        for attribute, value, message in (
            ('temporal_memory_train_min_event_count_exclusive', 30001, '30000'),
            ('temporal_memory_train_views_per_video', 1, 'two views'),
            ('temporal_memory_metric_aux_enabled', True, 'metric auxiliary'),
            ('temporal_memory_target_coverage_enabled', True, 'coverage'),
            (
                'temporal_frame_trajectory_extrapolation_enabled',
                True,
                'trajectory',
            ),
            ('temporal_frame_confidence_head_enabled', True, 'confidence'),
        ):
            with self.subTest(attribute=attribute):
                candidate = dacc_config()
                setattr(candidate, attribute, value)
                with self.assertRaisesRegex(ValueError, message):
                    validate_dacc_v2_training_config(
                        candidate,
                        'dacc_v2_projection_only',
                    )


if __name__ == '__main__':
    unittest.main()
