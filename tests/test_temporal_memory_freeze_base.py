"""CPU-only regression tests for the M25 memory-only training scope."""

import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'evisseg_evuav.yaml'
M20_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / 'checkpoints'
    / 'm20_attn_dense_views8_epoch_003_seed48.pt'
)
ORIGINAL_ARGV = sys.argv
sys.argv = ['freeze-base-test', '--config', str(CONFIG_PATH)]

from model.temporal_memory_net import (  # noqa: E402
    BidirectionalTemporalMemoryNet,
)
from train_temporal_memory import (  # noqa: E402
    build_optimizer,
    build_training_checkpoint,
    configure_temporal_memory_trainable_parameters,
    load_checkpoint_file,
    set_temporal_memory_training_mode,
)

sys.argv = ORIGINAL_ARGV


MEMORY_PREFIXES = (
    'forward_memory.',
    'backward_memory.',
    'memory_projection.',
    'temporal_attn.',
)


def optimizer_config():
    return SimpleNamespace(
        lr=5e-7,
        temporal_memory_base_lr_multiplier=1.0,
        temporal_memory_memory_lr_multiplier=1.0,
        temporal_memory_confidence_lr_multiplier=10.0,
    )


def checkpoint_config():
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
    config.temporal_memory_freeze_base_enabled = True
    config.temporal_memory_temporal_attention_enabled = True
    config.resolved_config = {
        'TEMPORAL_MEMORY': {
            'temporal_memory_freeze_base_enabled': True,
        },
    }
    config.config_overrides = [
        'TEMPORAL_MEMORY.temporal_memory_freeze_base_enabled=true',
    ]
    return config


def module_state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(contiguous.dtype).encode('ascii'))
        digest.update(str(tuple(contiguous.shape)).encode('ascii'))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


class TemporalMemoryFreezeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = load_checkpoint_file(
            M20_CHECKPOINT_PATH,
            map_location='cpu',
        )
        cls.m20_state_dict = checkpoint['model_state_dict']
        cls.m20_metadata = checkpoint['temporal_memory']

    def new_m20_model(self):
        model = BidirectionalTemporalMemoryNet(
            input_channels=10,
            width=16,
            density_calibration_enabled=True,
            confidence_head_enabled=False,
            temporal_attention_enabled=True,
        )
        incompatible = model.load_state_dict(
            self.m20_state_dict,
            strict=True,
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        return model

    def test_default_config_keeps_freeze_base_opt_in(self):
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
        self.assertIs(
            config['TEMPORAL_MEMORY'][
                'temporal_memory_freeze_base_enabled'
            ],
            False,
        )

    def test_released_m20_strict_load_and_memory_only_optimizer(self):
        self.assertTrue(self.m20_metadata['density_calibration_enabled'])
        self.assertTrue(self.m20_metadata['temporal_attention_enabled'])
        model = self.new_m20_model()

        configure_temporal_memory_trainable_parameters(
            model,
            freeze_base_enabled=True,
        )

        base_parameters = list(model.base.parameters())
        expected_memory = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(MEMORY_PREFIXES)
        ]
        self.assertTrue(base_parameters)
        self.assertTrue(all(not parameter.requires_grad for parameter in base_parameters))
        self.assertTrue(all(parameter.requires_grad for parameter in expected_memory))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.base.head.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.base.density_calibrator.parameters()
            )
        )

        optimizer = build_optimizer(model, optimizer_config())
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group['params']
        ]
        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        expected_ids = {id(parameter) for parameter in expected_memory}

        self.assertEqual(
            [group['name'] for group in optimizer.param_groups],
            ['memory'],
        )
        self.assertEqual(len(optimizer_ids), len(set(optimizer_ids)))
        self.assertEqual(set(optimizer_ids), expected_ids)
        self.assertEqual(len(optimizer_parameters), 16)
        self.assertEqual(
            sum(parameter.numel() for parameter in optimizer_parameters),
            1060992,
        )
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 1924716)
        self.assertEqual(sum(parameter.numel() for parameter in base_parameters), 863724)

    def test_disabled_scope_preserves_original_optimizer_groups(self):
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            freeze_base_enabled=False,
        )
        optimizer = build_optimizer(model, optimizer_config())
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group['params']
        ]

        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(
            [group['name'] for group in optimizer.param_groups],
            ['base', 'memory'],
        )
        self.assertEqual(
            {id(parameter) for parameter in optimizer_parameters},
            {id(parameter) for parameter in model.parameters()},
        )
        self.assertTrue(all(group['params'] for group in optimizer.param_groups))

    def test_training_mode_keeps_only_base_in_eval(self):
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            freeze_base_enabled=True,
        )
        set_temporal_memory_training_mode(
            model,
            freeze_base_enabled=True,
        )

        self.assertTrue(model.training)
        self.assertFalse(model.base.training)
        self.assertTrue(model.forward_memory.training)
        self.assertTrue(model.backward_memory.training)
        self.assertTrue(model.memory_projection.training)
        self.assertTrue(model.temporal_attn.training)

    def test_checkpoint_records_scope_and_strictly_round_trips(self):
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            freeze_base_enabled=True,
        )
        config = checkpoint_config()
        optimizer = build_optimizer(model, config)
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
        )

        self.assertTrue(
            checkpoint['temporal_memory']['freeze_base_enabled']
        )
        self.assertEqual(
            checkpoint['provenance']['training_scope'],
            {
                'name': 'memory_only',
                'trainable_parameter_count': 1060992,
                'frozen_parameter_count': 863724,
            },
        )
        restored_model = BidirectionalTemporalMemoryNet(
            input_channels=10,
            width=16,
            density_calibration_enabled=True,
            confidence_head_enabled=False,
            temporal_attention_enabled=True,
        )
        incompatible = restored_model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=True,
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_first_step_has_memory_gradients_and_byte_stable_base(self):
        torch.manual_seed(7)
        model = self.new_m20_model()
        configure_temporal_memory_trainable_parameters(
            model,
            freeze_base_enabled=True,
        )
        set_temporal_memory_training_mode(
            model,
            freeze_base_enabled=True,
        )
        optimizer = build_optimizer(model, optimizer_config())
        base_hash_before = module_state_hash(model.base)
        projection_before = model.memory_projection.weight.detach().clone()

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

        self.assertTrue(
            all(parameter.grad is None for parameter in model.base.parameters())
        )
        memory_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(MEMORY_PREFIXES)
        ]
        self.assertTrue(all(parameter.grad is not None for parameter in memory_parameters))
        self.assertTrue(
            all(torch.isfinite(parameter.grad).all() for parameter in memory_parameters)
        )
        self.assertTrue(
            all(torch.count_nonzero(parameter.grad).item() > 0 for parameter in memory_parameters)
        )

        optimizer.step()
        self.assertEqual(module_state_hash(model.base), base_hash_before)
        self.assertFalse(
            torch.equal(model.memory_projection.weight, projection_before)
        )

    def test_confidence_only_and_freeze_base_are_mutually_exclusive(self):
        model = BidirectionalTemporalMemoryNet(
            input_channels=10,
            width=16,
            density_calibration_enabled=True,
            confidence_head_enabled=True,
            temporal_attention_enabled=True,
        )
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            configure_temporal_memory_trainable_parameters(
                model,
                confidence_only_enabled=True,
                freeze_base_enabled=True,
            )


if __name__ == '__main__':
    unittest.main()
