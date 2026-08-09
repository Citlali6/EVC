"""CPU-only regression tests for the H17 event-head training scope."""

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
sys.argv = ['head-only-test', '--config', str(CONFIG_PATH)]

from model.temporal_memory_net import (  # noqa: E402
    BidirectionalTemporalMemoryNet,
)
from train_temporal_memory import (  # noqa: E402
    HEAD_ONLY_MUTABLE_STATE_KEYS,
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
    validate_head_only_initialization_checkpoint,
    validate_head_only_training_config,
)

sys.argv = ORIGINAL_ARGV


def optimizer_config():
    return SimpleNamespace(
        lr=2e-6,
        temporal_memory_base_lr_multiplier=1.0,
        temporal_memory_memory_lr_multiplier=1.0,
        temporal_memory_confidence_lr_multiplier=10.0,
    )


def head_only_config():
    config = optimizer_config()
    config.temporal_memory_bin_size = 50
    config.temporal_memory_context_bins = 5
    config.temporal_memory_width = 16
    config.temporal_memory_sequence_length = 16
    config.temporal_memory_train_views_per_video = 1
    config.temporal_memory_train_min_event_count_exclusive = 30000
    config.temporal_memory_log_count_clip = 4.0
    config.temporal_memory_dense_sampling_enabled = False
    config.temporal_memory_density_bucket_boundaries = []
    config.temporal_memory_density_bucket_views = []
    config.temporal_memory_metric_aux_enabled = False
    config.temporal_frame_density_calibration_enabled = True
    config.temporal_frame_trajectory_extrapolation_enabled = False
    config.temporal_frame_confidence_head_enabled = False
    config.temporal_memory_confidence_only_enabled = False
    config.temporal_memory_freeze_base_enabled = False
    config.temporal_memory_head_only_enabled = True
    config.temporal_memory_temporal_attention_enabled = True
    config.resolved_config = {
        'TEMPORAL_MEMORY': {
            'temporal_memory_freeze_base_enabled': False,
            'temporal_memory_head_only_enabled': True,
            'temporal_memory_train_min_event_count_exclusive': 30000,
        },
    }
    config.config_overrides = [
        'TEMPORAL_MEMORY.temporal_memory_head_only_enabled=true',
    ]
    return config


class TemporalMemoryHeadOnlyTests(unittest.TestCase):
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
        model.load_state_dict(self.m20_state_dict, strict=True)
        return model

    def test_all_temporal_configs_default_new_scopes_off(self):
        for path in CONFIG_PATHS:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text(encoding='utf-8'))
                memory = config['TEMPORAL_MEMORY']
                self.assertIs(
                    memory['temporal_memory_freeze_base_enabled'],
                    False,
                )
                self.assertIs(
                    memory['temporal_memory_head_only_enabled'],
                    False,
                )
                self.assertIsNone(
                    memory[
                        'temporal_memory_train_min_event_count_exclusive'
                    ]
                )

    def test_specialized_scopes_are_pairwise_mutually_exclusive(self):
        combinations = (
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        )
        for confidence_only, freeze_base, head_only in combinations:
            with self.subTest(
                confidence_only=confidence_only,
                freeze_base=freeze_base,
                head_only=head_only,
            ):
                with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    resolve_temporal_memory_training_scope(
                        confidence_only,
                        freeze_base,
                        head_only,
                    )

    def test_head_only_config_contract(self):
        config = head_only_config()
        validate_head_only_training_config(config, 'event_head_only')
        invalid = (
            ('temporal_memory_train_views_per_video', 2, 'one view'),
            (
                'temporal_memory_train_min_event_count_exclusive',
                None,
                'event-count filter',
            ),
            ('temporal_memory_dense_sampling_enabled', True, 'dense multiplier'),
            ('temporal_memory_metric_aux_enabled', True, 'metric auxiliary'),
            (
                'temporal_frame_trajectory_extrapolation_enabled',
                True,
                'trajectory',
            ),
            ('temporal_frame_confidence_head_enabled', True, 'confidence_head'),
        )
        for attribute, value, message in invalid:
            with self.subTest(attribute=attribute):
                candidate = head_only_config()
                setattr(candidate, attribute, value)
                with self.assertRaisesRegex(ValueError, message):
                    validate_head_only_training_config(
                        candidate,
                        'event_head_only',
                    )

    def test_only_two_event_head_tensors_are_trainable_and_optimized(self):
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            head_only_enabled=True,
        )
        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(set(trainable), HEAD_ONLY_MUTABLE_STATE_KEYS)
        self.assertEqual(sum(p.numel() for p in trainable.values()), 17)

        optimizer = build_optimizer(
            model,
            optimizer_config(),
            head_only_enabled=True,
        )
        self.assertEqual([group['name'] for group in optimizer.param_groups], ['event_head'])
        optimizer_parameters = optimizer.param_groups[0]['params']
        self.assertEqual(len(optimizer_parameters), 2)
        self.assertEqual(sum(p.numel() for p in optimizer_parameters), 17)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.0)
        self.assertEqual(
            {id(parameter) for parameter in optimizer_parameters},
            {id(parameter) for parameter in trainable.values()},
        )

    def test_training_mode_is_eval_except_for_event_head(self):
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            head_only_enabled=True,
        )
        set_temporal_memory_training_mode(model, head_only_enabled=True)
        self.assertFalse(model.training)
        self.assertFalse(model.base.training)
        self.assertTrue(model.base.head.training)
        self.assertFalse(model.forward_memory.training)
        self.assertFalse(model.temporal_attn.training)

    def test_real_step_changes_head_and_preserves_frozen_state_bytes(self):
        torch.manual_seed(11)
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            head_only_enabled=True,
        )
        set_temporal_memory_training_mode(model, head_only_enabled=True)
        optimizer = build_optimizer(
            model,
            optimizer_config(),
            head_only_enabled=True,
        )
        frozen_reference = snapshot_frozen_model_state(model)
        frozen_hash = frozen_model_state_sha256(model)
        head_before = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
            if name in HEAD_ONLY_MUTABLE_STATE_KEYS
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
        optimizer.step()

        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if name not in HEAD_ONLY_MUTABLE_STATE_KEYS
            )
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if name in HEAD_ONLY_MUTABLE_STATE_KEYS
            )
        )
        self.assertTrue(
            any(
                not torch.equal(model.state_dict()[name], previous)
                for name, previous in head_before.items()
            )
        )
        assert_frozen_model_state_unchanged(model, frozen_reference)
        self.assertEqual(frozen_model_state_sha256(model), frozen_hash)

    def test_frozen_state_audit_reports_modified_tensor(self):
        model = self.new_m20_model()
        frozen_reference = snapshot_frozen_model_state(model)
        with torch.no_grad():
            model.memory_projection.bias[0].add_(1.0)
        with self.assertRaisesRegex(
            RuntimeError,
            'memory_projection.bias',
        ):
            assert_frozen_model_state_unchanged(model, frozen_reference)

    def test_checkpoint_records_head_scope_and_frozen_hash(self):
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            head_only_enabled=True,
        )
        reference_sha256 = frozen_model_state_sha256(model)
        config = head_only_config()
        optimizer = build_optimizer(
            model,
            config,
            head_only_enabled=True,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
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
        self.assertTrue(checkpoint['temporal_memory']['head_only_enabled'])
        self.assertEqual(
            checkpoint['provenance']['training_scope'],
            {
                'name': 'event_head_only',
                'trainable_parameter_count': 17,
                'frozen_parameter_count': 1924699,
                'mutable_state_keys': sorted(HEAD_ONLY_MUTABLE_STATE_KEYS),
                'frozen_state_reference_sha256': reference_sha256,
            },
        )

    def test_head_only_resume_verifies_frozen_reference_hash(self):
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            head_only_enabled=True,
        )
        reference_sha256 = frozen_model_state_sha256(model)
        config = head_only_config()
        optimizer = build_optimizer(
            model,
            config,
            head_only_enabled=True,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
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

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / 'head-only.pt'
            torch.save(checkpoint, checkpoint_path)
            restored_model = self.new_m20_model()
            configure_temporal_memory_trainable_parameters(
                restored_model,
                head_only_enabled=True,
            )
            restored_optimizer = build_optimizer(
                restored_model,
                config,
                head_only_enabled=True,
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
                frozen_model_state_sha256(restored_model),
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

    def test_head_only_requires_complete_temporal_checkpoint(self):
        validate_head_only_initialization_checkpoint(M20_CHECKPOINT_PATH)
        with self.assertRaisesRegex(ValueError, 'complete temporal-memory'):
            validate_head_only_initialization_checkpoint(
                PROJECT_ROOT / 'checkpoints' / 'p23_baseline_5ep_seed42.pt'
            )


if __name__ == '__main__':
    unittest.main()
