"""CPU-only tests for temporal-attention output-projection tuning."""

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
    PROJECT_ROOT
    / 'checkpoints'
    / 'm20_attn_dense_views8_epoch_003_seed48.pt'
)
ORIGINAL_ARGV = sys.argv
sys.argv = ['attention-projection-test', '--config', str(CONFIG_PATH)]

from model.temporal_memory_net import (  # noqa: E402
    BidirectionalTemporalMemoryNet,
)
from train_temporal_memory import (  # noqa: E402
    ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
    assert_frozen_model_state_unchanged,
    build_optimizer,
    build_training_checkpoint,
    configure_temporal_memory_trainable_parameters,
    frozen_model_state_sha256,
    load_checkpoint_file,
    load_training_resume,
    resolve_temporal_memory_training_scope,
    set_temporal_memory_training_mode,
    snapshot_frozen_model_state,
    validate_attention_projection_initialization_checkpoint,
    validate_attention_projection_training_config,
)

sys.argv = ORIGINAL_ARGV


def optimizer_config():
    return SimpleNamespace(
        lr=1e-6,
        temporal_memory_base_lr_multiplier=1.0,
        temporal_memory_memory_lr_multiplier=1.0,
        temporal_memory_confidence_lr_multiplier=10.0,
        temporal_memory_dacc_v2_lr_multiplier=1.0,
    )


def attention_projection_config():
    config = optimizer_config()
    config.temporal_memory_bin_size = 50
    config.temporal_memory_context_bins = 5
    config.temporal_memory_width = 16
    config.temporal_memory_sequence_length = 16
    config.temporal_memory_log_count_clip = 4.0
    config.temporal_frame_density_calibration_enabled = True
    config.temporal_frame_trajectory_extrapolation_enabled = False
    config.temporal_frame_confidence_head_enabled = False
    config.temporal_memory_confidence_only_enabled = False
    config.temporal_memory_freeze_base_enabled = False
    config.temporal_memory_head_only_enabled = False
    config.temporal_memory_dacc_v2_enabled = False
    config.temporal_memory_dacc_v2_only_enabled = False
    config.temporal_memory_attention_projection_only_enabled = True
    config.temporal_memory_temporal_attention_enabled = True
    config.resolved_config = {
        'TEMPORAL_MEMORY': {
            'temporal_memory_attention_projection_only_enabled': True,
            'temporal_memory_temporal_attention_enabled': True,
        },
    }
    config.config_overrides = [
        'TEMPORAL_MEMORY.temporal_memory_attention_projection_only_enabled=true',
    ]
    return config


class TemporalMemoryAttentionProjectionOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = load_checkpoint_file(M20_CHECKPOINT_PATH, map_location='cpu')
        cls.m20_state_dict = checkpoint['model_state_dict']

    def new_m20_model(self):
        model = BidirectionalTemporalMemoryNet(
            input_channels=10,
            width=16,
            density_calibration_enabled=True,
            confidence_head_enabled=False,
            temporal_attention_enabled=True,
        )
        incompatible = model.load_state_dict(self.m20_state_dict, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        return model

    def test_all_configs_default_scope_off(self):
        for path in CONFIG_PATHS:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text(encoding='utf-8'))
                self.assertIs(
                    config['TEMPORAL_MEMORY'][
                        'temporal_memory_attention_projection_only_enabled'
                    ],
                    False,
                )

    def test_scope_is_mutually_exclusive_with_every_narrow_scope(self):
        other_scopes = (
            'confidence_only_enabled',
            'freeze_base_enabled',
            'head_only_enabled',
            'dacc_v2_only_enabled',
        )
        for other_scope in other_scopes:
            with self.subTest(other_scope=other_scope):
                arguments = {
                    'attention_projection_only_enabled': True,
                    other_scope: True,
                }
                with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    resolve_temporal_memory_training_scope(**arguments)

    def test_scope_requires_temporal_attention(self):
        config = attention_projection_config()
        validate_attention_projection_training_config(
            config,
            'temporal_attention_projection_only',
        )
        config.temporal_memory_temporal_attention_enabled = False
        with self.assertRaisesRegex(ValueError, 'temporal_attention_enabled=true'):
            validate_attention_projection_training_config(
                config,
                'temporal_attention_projection_only',
            )

        model = BidirectionalTemporalMemoryNet(
            input_channels=10,
            width=16,
            density_calibration_enabled=True,
            temporal_attention_enabled=False,
        )
        with self.assertRaisesRegex(ValueError, 'requires temporal attention'):
            configure_temporal_memory_trainable_parameters(
                model,
                attention_projection_only_enabled=True,
            )

    def test_initialization_requires_complete_attention_checkpoint(self):
        validate_attention_projection_initialization_checkpoint(
            M20_CHECKPOINT_PATH
        )
        with self.assertRaisesRegex(
            ValueError,
            'complete temporal-attention checkpoint',
        ):
            validate_attention_projection_initialization_checkpoint(
                PROJECT_ROOT / 'checkpoints' / 'p23_baseline_5ep_seed42.pt'
            )

    def test_exact_projection_tensors_are_trainable_and_optimized(self):
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            attention_projection_only_enabled=True,
        )
        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            set(trainable),
            ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
        )
        self.assertEqual(sum(p.numel() for p in trainable.values()), 9312)

        optimizer = build_optimizer(
            model,
            optimizer_config(),
            attention_projection_only_enabled=True,
        )
        self.assertEqual(
            [group['name'] for group in optimizer.param_groups],
            ['temporal_attention_projection'],
        )
        optimizer_parameters = optimizer.param_groups[0]['params']
        self.assertEqual(len(optimizer_parameters), 2)
        self.assertEqual(sum(p.numel() for p in optimizer_parameters), 9312)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.0)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-6)
        self.assertEqual(
            {id(parameter) for parameter in optimizer_parameters},
            {id(parameter) for parameter in trainable.values()},
        )

    def test_training_mode_keeps_everything_except_projection_in_eval(self):
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            attention_projection_only_enabled=True,
        )
        set_temporal_memory_training_mode(
            model,
            attention_projection_only_enabled=True,
        )
        self.assertFalse(model.training)
        self.assertFalse(model.base.training)
        self.assertFalse(model.forward_memory.training)
        self.assertFalse(model.backward_memory.training)
        self.assertFalse(model.temporal_attn.training)
        self.assertFalse(model.temporal_attn.attention.training)
        self.assertTrue(model.temporal_attn.output_projection.training)

    def test_real_step_changes_projection_and_preserves_frozen_bytes(self):
        torch.manual_seed(17)
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            attention_projection_only_enabled=True,
        )
        set_temporal_memory_training_mode(
            model,
            attention_projection_only_enabled=True,
        )
        optimizer = build_optimizer(
            model,
            optimizer_config(),
            attention_projection_only_enabled=True,
        )
        frozen_reference = snapshot_frozen_model_state(
            model,
            mutable_state_keys=ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
        )
        frozen_hash = frozen_model_state_sha256(
            model,
            mutable_state_keys=ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
        )
        projection_before = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
            if name in ATTENTION_PROJECTION_MUTABLE_STATE_KEYS
        }

        frames = torch.randn(1, 2, 10, 32, 32)
        labels = torch.randint(
            0,
            2,
            (1, 2, 1, 32, 32),
            dtype=torch.float32,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(frames),
            labels,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        for name, parameter in model.named_parameters():
            if name in ATTENTION_PROJECTION_MUTABLE_STATE_KEYS:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)
            else:
                self.assertIsNone(parameter.grad, name)
        optimizer.step()

        self.assertTrue(
            any(
                not torch.equal(model.state_dict()[name], previous)
                for name, previous in projection_before.items()
            )
        )
        assert_frozen_model_state_unchanged(
            model,
            frozen_reference,
            mutable_state_keys=ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
        )
        self.assertEqual(
            frozen_model_state_sha256(
                model,
                mutable_state_keys=ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
            ),
            frozen_hash,
        )

    def test_frozen_audit_detects_non_projection_change(self):
        model = self.new_m20_model()
        frozen_reference = snapshot_frozen_model_state(
            model,
            mutable_state_keys=ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
        )
        with torch.no_grad():
            model.temporal_attn.attention.out_proj.bias[0].add_(1.0)
        with self.assertRaisesRegex(RuntimeError, 'attention.out_proj.bias'):
            assert_frozen_model_state_unchanged(
                model,
                frozen_reference,
                mutable_state_keys=ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
            )

    def build_checkpoint(self):
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            attention_projection_only_enabled=True,
        )
        config = attention_projection_config()
        optimizer = build_optimizer(
            model,
            config,
            attention_projection_only_enabled=True,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        reference_sha256 = frozen_model_state_sha256(
            model,
            mutable_state_keys=ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
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
            initialized_from=M20_CHECKPOINT_PATH,
            include_cuda_rng=False,
            frozen_state_reference_sha256=reference_sha256,
        )
        return model, config, checkpoint, reference_sha256

    def test_checkpoint_records_scope_and_strictly_round_trips(self):
        _, _, checkpoint, reference_sha256 = self.build_checkpoint()
        self.assertTrue(
            checkpoint['temporal_memory'][
                'attention_projection_only_enabled'
            ]
        )
        self.assertEqual(
            checkpoint['provenance']['training_scope'],
            {
                'name': 'temporal_attention_projection_only',
                'trainable_parameter_count': 9312,
                'frozen_parameter_count': 1915404,
                'mutable_state_keys': sorted(
                    ATTENTION_PROJECTION_MUTABLE_STATE_KEYS
                ),
                'frozen_state_reference_sha256': reference_sha256,
            },
        )
        restored_model = self.new_m20_model()
        incompatible = restored_model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=True,
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_resume_verifies_scope_and_frozen_reference_hash(self):
        _, config, checkpoint, reference_sha256 = self.build_checkpoint()
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / 'attention.pt'
            torch.save(checkpoint, checkpoint_path)
            restored_model = self.new_m20_model()
            configure_temporal_memory_trainable_parameters(
                restored_model,
                attention_projection_only_enabled=True,
            )
            restored_optimizer = build_optimizer(
                restored_model,
                config,
                attention_projection_only_enabled=True,
            )
            restored_scheduler = torch.optim.lr_scheduler.StepLR(
                restored_optimizer,
                step_size=1,
            )
            _, start_epoch = load_training_resume(
                checkpoint_path,
                restored_model,
                restored_optimizer,
                restored_scheduler,
                current_config=config,
                restore_rng=False,
            )
            self.assertEqual(start_epoch, 1)
            self.assertEqual(
                frozen_model_state_sha256(
                    restored_model,
                    mutable_state_keys=ATTENTION_PROJECTION_MUTABLE_STATE_KEYS,
                ),
                reference_sha256,
            )

            tampered = copy.deepcopy(checkpoint)
            tampered['model_state_dict']['memory_projection.bias'][0].add_(1.0)
            tampered_path = Path(temporary_directory) / 'tampered.pt'
            torch.save(tampered, tampered_path)
            with self.assertRaisesRegex(ValueError, 'frozen state'):
                load_training_resume(
                    tampered_path,
                    restored_model,
                    restored_optimizer,
                    restored_scheduler,
                    current_config=config,
                    restore_rng=False,
                )


if __name__ == '__main__':
    unittest.main()
