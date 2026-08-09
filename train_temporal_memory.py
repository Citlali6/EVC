"""Train a bidirectional full-stream temporal-memory event segmentation model."""

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import tqdm
import yaml

from configs.configs import cfg
from dataset.temporal_memory import (
    TemporalMemoryTrainDataset,
    temporal_memory_collate,
)
from model.modules.confidence_head import confidence_calibration_loss
from model.temporal_frame_net import (
    DENSITY_CALIBRATION_LEGACY_VERSION,
    DENSITY_CALIBRATION_V2_BASIS,
    DENSITY_CALIBRATION_V2_RESIDUAL_SCALE,
    DENSITY_CALIBRATION_V2_VERSION,
    density_calibration_version,
    validate_density_calibration_metadata,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet
from utils.component_hard_negative import (
    component_hard_negative_loss,
    target_frame_activation_loss,
)
from utils.temporal_frame_loss import (
    frame_balanced_event_bce,
    target_time_group_coverage_loss,
    trajectory_extrapolation_loss_memory,
)


RESUME_CHECKPOINT_FORMAT_VERSION = 2
ALLOWED_RESUME_CONFIG_DIFFERENCES = frozenset(
    {
        ('TRAIN', 'resume_checkpoint'),
        ('TRAIN', 'model_save_root'),
        ('TEST', 'challenge_output_dir'),
    }
)
LEGACY_RESUME_CONFIG_DEFAULTS = {
    (
        'TEMPORAL_MEMORY',
        'temporal_memory_freeze_base_enabled',
    ): False,
    (
        'TEMPORAL_MEMORY',
        'temporal_memory_head_only_enabled',
    ): False,
    (
        'TEMPORAL_MEMORY',
        'temporal_memory_train_min_event_count_exclusive',
    ): None,
    (
        'TEMPORAL_MEMORY',
        'temporal_memory_dacc_v2_enabled',
    ): False,
    (
        'TEMPORAL_MEMORY',
        'temporal_memory_dacc_v2_only_enabled',
    ): False,
    (
        'TEMPORAL_MEMORY',
        'temporal_memory_dacc_v2_lr_multiplier',
    ): 1.0,
}
HEAD_ONLY_MUTABLE_STATE_KEYS = frozenset(
    {
        'base.head.weight',
        'base.head.bias',
    }
)
DACC_V2_MUTABLE_STATE_KEYS = frozenset(
    {
        'base.density_calibrator.residual_projection.weight',
    }
)


def load_checkpoint_file(path, map_location='cpu'):
    """Load trusted local checkpoints across PyTorch's weights-only change."""
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        # PyTorch releases predating the ``weights_only`` argument.
        return torch.load(path, map_location=map_location)


def sha256_file(path, chunk_size=1024 * 1024):
    """Hash a checkpoint file without loading another copy into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_temporal_memory_training_scope(
    confidence_only_enabled=False,
    freeze_base_enabled=False,
    head_only_enabled=False,
    dacc_v2_only_enabled=False,
):
    """Resolve the mutually exclusive temporal-memory training scope."""
    enabled = [
        name
        for name, active in (
            ('confidence_only', confidence_only_enabled),
            ('memory_only', freeze_base_enabled),
            ('event_head_only', head_only_enabled),
            ('dacc_v2_projection_only', dacc_v2_only_enabled),
        )
        if bool(active)
    ]
    if len(enabled) > 1:
        raise ValueError(
            'Temporal-memory confidence-only, freeze-base, head-only, and '
            'DACC-v2-only '
            'modes are mutually exclusive: {}.'.format(', '.join(enabled))
        )
    return enabled[0] if enabled else 'all'


def validate_head_only_training_config(config, training_scope):
    """Reject settings that would make the H17 probe ambiguous."""
    if training_scope != 'event_head_only':
        return
    if bool(getattr(config, 'temporal_frame_confidence_head_enabled', False)):
        raise ValueError(
            'Head-only mode requires TEMPORAL_FRAME.confidence_head_enabled=false.'
        )
    if int(config.temporal_memory_train_views_per_video) != 1:
        raise ValueError('Head-only mode requires exactly one view per video.')
    if getattr(
        config,
        'temporal_memory_train_min_event_count_exclusive',
        None,
    ) is None:
        raise ValueError(
            'Head-only mode requires a strict train event-count filter.'
        )
    if bool(
        getattr(config, 'temporal_memory_dense_sampling_enabled', False)
    ):
        raise ValueError('Head-only mode does not allow dense multiplier sampling.')
    if list(
        getattr(config, 'temporal_memory_density_bucket_boundaries', [])
    ) or list(getattr(config, 'temporal_memory_density_bucket_views', [])):
        raise ValueError('Head-only mode does not allow density bucket sampling.')
    if bool(getattr(config, 'temporal_memory_metric_aux_enabled', False)):
        raise ValueError('Head-only mode requires metric auxiliary loss disabled.')
    if bool(
        getattr(
            config,
            'temporal_frame_trajectory_extrapolation_enabled',
            False,
        )
    ):
        raise ValueError('Head-only mode requires trajectory loss disabled.')


def validate_dacc_v2_training_config(config, training_scope):
    """Require an explicit legacy-DACC parent for projection-only training."""
    dacc_v2_enabled = bool(
        getattr(config, 'temporal_memory_dacc_v2_enabled', False)
    )
    density_enabled = bool(
        getattr(config, 'temporal_frame_density_calibration_enabled', False)
    )
    if dacc_v2_enabled and not density_enabled:
        raise ValueError(
            'DACC-v2 requires TEMPORAL_FRAME.density_calibration_enabled=true.'
        )
    if dacc_v2_enabled and training_scope != 'dacc_v2_projection_only':
        raise ValueError(
            'DACC-v2 training requires the projection-only scope.'
        )
    if training_scope != 'dacc_v2_projection_only':
        return
    if not dacc_v2_enabled:
        raise ValueError(
            'DACC-v2-only mode requires '
            'TEMPORAL_MEMORY.temporal_memory_dacc_v2_enabled=true.'
        )
    min_event_count = getattr(
        config,
        'temporal_memory_train_min_event_count_exclusive',
        None,
    )
    if min_event_count is None:
        raise ValueError(
            'DACC-v2-only mode requires a strict train event-count filter.'
        )
    if int(min_event_count) != 30000:
        raise ValueError(
            'The controlled DACC-v2 experiment requires event_count > 30000.'
        )
    if int(config.temporal_memory_train_views_per_video) != 2:
        raise ValueError(
            'The controlled DACC-v2 experiment requires two views per video.'
        )
    if bool(
        getattr(config, 'temporal_memory_dense_sampling_enabled', False)
    ):
        raise ValueError(
            'DACC-v2-only mode does not allow dense multiplier sampling.'
        )
    if list(
        getattr(config, 'temporal_memory_density_bucket_boundaries', [])
    ) or list(getattr(config, 'temporal_memory_density_bucket_views', [])):
        raise ValueError(
            'DACC-v2-only mode does not allow density bucket sampling.'
        )
    if bool(getattr(config, 'temporal_memory_metric_aux_enabled', False)):
        raise ValueError(
            'The controlled DACC-v2 experiment requires metric auxiliary loss off.'
        )
    if bool(getattr(config, 'temporal_memory_target_coverage_enabled', False)):
        raise ValueError(
            'The controlled DACC-v2 experiment requires target coverage loss off.'
        )
    if bool(
        getattr(
            config,
            'temporal_frame_trajectory_extrapolation_enabled',
            False,
        )
    ):
        raise ValueError(
            'The controlled DACC-v2 experiment requires trajectory loss off.'
        )
    if bool(getattr(config, 'temporal_frame_confidence_head_enabled', False)):
        raise ValueError(
            'The controlled DACC-v2 experiment requires confidence head off.'
        )


def _state_tensor_byte_view(tensor):
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
    )


def snapshot_frozen_model_state(
    model,
    mutable_state_keys=HEAD_ONLY_MUTABLE_STATE_KEYS,
):
    """Clone every state tensor outside the explicitly mutable set."""
    mutable_state_keys = frozenset(mutable_state_keys)
    state = model.state_dict()
    missing = sorted(mutable_state_keys.difference(state))
    if missing:
        raise ValueError(
            'Mutable state keys are missing from the model: {}.'.format(missing)
        )
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in state.items()
        if name not in mutable_state_keys
    }


def frozen_model_state_sha256(
    model_or_state,
    mutable_state_keys=HEAD_ONLY_MUTABLE_STATE_KEYS,
):
    """Hash frozen state names, metadata, and exact tensor bytes."""
    state = (
        model_or_state.state_dict()
        if hasattr(model_or_state, 'state_dict')
        else model_or_state
    )
    mutable_state_keys = frozenset(mutable_state_keys)
    digest = hashlib.sha256()
    for name in sorted(state):
        if name in mutable_state_keys:
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(_state_tensor_byte_view(tensor).numpy().tobytes())
    return digest.hexdigest()


def assert_frozen_model_state_unchanged(
    model,
    reference_state,
    mutable_state_keys=HEAD_ONLY_MUTABLE_STATE_KEYS,
):
    """Fail before checkpointing if scoped training changed frozen state."""
    mutable_state_keys = frozenset(mutable_state_keys)
    current = model.state_dict()
    expected_names = set(reference_state)
    current_names = set(current).difference(mutable_state_keys)
    if current_names != expected_names:
        raise RuntimeError(
            'Frozen model state keys changed: missing={}, extra={}.'.format(
                sorted(expected_names.difference(current_names)),
                sorted(current_names.difference(expected_names)),
            )
        )
    for name in sorted(reference_state):
        expected = reference_state[name]
        actual = current[name]
        if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(
            expected.shape
        ):
            raise RuntimeError(
                'Scoped training changed frozen state metadata: {}.'.format(name)
            )
        if not torch.equal(
            _state_tensor_byte_view(actual),
            _state_tensor_byte_view(expected),
        ):
            raise RuntimeError(
                'Scoped training modified frozen state tensor: {}.'.format(name)
            )


def capture_rng_state(include_cuda=True):
    """Capture every global RNG used by the training process.

    CUDA state collection is best-effort so checkpointing is still usable on
    CPU-only machines and in environments where CUDA was compiled in but
    cannot be initialized.
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if include_cuda and torch.cuda.is_available():
        try:
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
        except (RuntimeError, AssertionError):
            state['torch_cuda'] = None
    return state


def restore_rng_state(state):
    """Restore a state produced by :func:`capture_rng_state`."""
    if not isinstance(state, dict):
        raise ValueError('Resume checkpoint RNG state must be a mapping.')
    required = {'python', 'numpy', 'torch_cpu'}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(
            'Resume checkpoint RNG state is missing: {}.'.format(
                ', '.join(missing)
            )
        )
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    cuda_state = state.get('torch_cuda')
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except (RuntimeError, AssertionError) as error:
            raise RuntimeError(
                'Could not restore CUDA RNG state from resume checkpoint.'
            ) from error


def collect_git_provenance(repository_root=None):
    """Return the current Git commit and dirty flag without requiring Git."""
    root = Path(repository_root or Path(__file__).resolve().parent)
    provenance = {'commit': None, 'dirty': None}
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return provenance
    provenance['commit'] = commit or None
    provenance['dirty'] = bool(status.strip())
    return provenance


def is_resumable_checkpoint(checkpoint):
    """Distinguish complete training snapshots from model-only checkpoints."""
    if not isinstance(checkpoint, dict):
        return False
    if int(checkpoint.get('checkpoint_format_version', 0)) < 2:
        return False
    required = {
        'model_state_dict',
        'optimizer_state_dict',
        'optimizer_class',
        'scheduler_state_dict',
        'scheduler_class',
        'rng_state',
        'start_epoch',
        'next_epoch',
        'best_loss',
        'best_epoch',
    }
    return required.issubset(checkpoint)


def _config_difference_paths(saved, current, path=()):
    """Return leaf paths that differ between two resolved configurations."""
    if path in ALLOWED_RESUME_CONFIG_DIFFERENCES:
        return []
    if isinstance(saved, dict) and isinstance(current, dict):
        differences = []
        for key in sorted(set(saved).union(current)):
            child_path = path + (str(key),)
            if key not in saved or key not in current:
                legacy_default = LEGACY_RESUME_CONFIG_DEFAULTS.get(child_path)
                present_value = (
                    current[key] if key in current else saved[key]
                )
                if (
                    child_path in LEGACY_RESUME_CONFIG_DEFAULTS
                    and type(present_value) is type(legacy_default)
                    and present_value == legacy_default
                ):
                    continue
                if child_path not in ALLOWED_RESUME_CONFIG_DIFFERENCES:
                    differences.append(child_path)
                continue
            differences.extend(
                _config_difference_paths(
                    saved[key],
                    current[key],
                    child_path,
                )
            )
        return differences
    if saved != current:
        return [path]
    return []


def validate_resume_config(checkpoint, current_config):
    """Enforce resume as an exact continuation, not a fine-tuning shortcut."""
    provenance = checkpoint.get('provenance')
    saved_config = (
        provenance.get('resolved_config')
        if isinstance(provenance, dict)
        else None
    )
    current_resolved = getattr(
        current_config,
        'resolved_config',
        current_config,
    )
    if not isinstance(saved_config, dict):
        raise ValueError(
            'Resume checkpoint is missing its resolved training configuration.'
        )
    if not isinstance(current_resolved, dict):
        raise TypeError('Current resolved training configuration must be a mapping.')
    differences = _config_difference_paths(saved_config, current_resolved)
    if differences:
        rendered = ', '.join(
            '.'.join(path) if path else '<root>'
            for path in differences[:12]
        )
        if len(differences) > 12:
            rendered += ', ... ({} total)'.format(len(differences))
        raise ValueError(
            'Resume requires the identical resolved training configuration, '
            'including TRAIN.epochs and scheduler horizon. Only '
            'TRAIN.resume_checkpoint, TRAIN.model_save_root, and the '
            'submission output directory may differ. Changed: {}.'.format(
                rendered
            )
        )


def build_training_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    epoch_loss,
    best_loss,
    best_epoch,
    config,
    initialized_from,
    resume_parent_checkpoint=None,
    run_start_epoch=0,
    best_loss_checkpoint=None,
    git_provenance=None,
    include_cuda_rng=True,
    frozen_state_reference_sha256=None,
    initialized_from_sha256=None,
    initialization_migrations=None,
):
    """Build an epoch-boundary snapshot that can resume deterministically."""
    freeze_base_enabled = bool(
        getattr(config, 'temporal_memory_freeze_base_enabled', False)
    )
    confidence_only_enabled = bool(
        getattr(config, 'temporal_memory_confidence_only_enabled', False)
    )
    head_only_enabled = bool(
        getattr(config, 'temporal_memory_head_only_enabled', False)
    )
    dacc_v2_enabled = bool(
        getattr(config, 'temporal_memory_dacc_v2_enabled', False)
    )
    dacc_v2_only_enabled = bool(
        getattr(config, 'temporal_memory_dacc_v2_only_enabled', False)
    )
    density_calibration_enabled = bool(
        getattr(
            config,
            'temporal_frame_density_calibration_enabled',
            False,
        )
    )
    configured_density_version = density_calibration_version(
        density_calibration_enabled,
        dacc_v2_enabled,
    )
    model_state_keys = set(model.state_dict())
    projection_state_keys = {
        name
        for name in model_state_keys
        if name.startswith(
            'base.density_calibrator.residual_projection.'
        )
    }
    if hasattr(model, 'base') and hasattr(
        model.base,
        'density_calibration_enabled',
    ):
        model_density_version = density_calibration_version(
            model.base.density_calibration_enabled,
            getattr(model, 'density_calibration_v2_enabled', False),
        )
        if model_density_version != configured_density_version:
            raise ValueError(
                'Checkpoint config density calibration version {} does not '
                'match model version {}.'.format(
                    configured_density_version,
                    model_density_version,
                )
            )
    expected_projection_keys = (
        DACC_V2_MUTABLE_STATE_KEYS
        if configured_density_version == DENSITY_CALIBRATION_V2_VERSION
        else frozenset()
    )
    if projection_state_keys != expected_projection_keys:
        raise ValueError(
            'Checkpoint density metadata and projection state keys disagree: '
            'expected {}, got {}.'.format(
                sorted(expected_projection_keys),
                sorted(projection_state_keys),
            )
        )
    training_scope = resolve_temporal_memory_training_scope(
        confidence_only_enabled=confidence_only_enabled,
        freeze_base_enabled=freeze_base_enabled,
        head_only_enabled=head_only_enabled,
        dacc_v2_only_enabled=dacc_v2_only_enabled,
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    training_scope_metadata = {
        'name': training_scope,
        'trainable_parameter_count': int(trainable_parameter_count),
        'frozen_parameter_count': int(frozen_parameter_count),
    }
    audited_mutable_keys = None
    if training_scope == 'event_head_only':
        audited_mutable_keys = HEAD_ONLY_MUTABLE_STATE_KEYS
    elif training_scope == 'dacc_v2_projection_only':
        audited_mutable_keys = DACC_V2_MUTABLE_STATE_KEYS
    if audited_mutable_keys is not None:
        if not frozen_state_reference_sha256:
            raise ValueError(
                'Audited scoped checkpoints require a frozen-state reference hash.'
            )
        training_scope_metadata.update(
            {
                'mutable_state_keys': sorted(audited_mutable_keys),
                'frozen_state_reference_sha256': str(
                    frozen_state_reference_sha256
                ),
            }
        )
    temporal_memory_metadata = {
        'temporal_bin_size': int(config.temporal_memory_bin_size),
        'context_bins': int(config.temporal_memory_context_bins),
        'width': int(config.temporal_memory_width),
        'sequence_length': int(config.temporal_memory_sequence_length),
        'log_count_clip': float(config.temporal_memory_log_count_clip),
        'density_calibration_enabled': density_calibration_enabled,
        'density_calibration_version': int(configured_density_version),
        'trajectory_extrapolation_enabled': bool(
            getattr(
                config,
                'temporal_frame_trajectory_extrapolation_enabled',
                False,
            )
        ),
        'confidence_head_enabled': bool(
            getattr(config, 'temporal_frame_confidence_head_enabled', False)
        ),
        'confidence_only_enabled': bool(confidence_only_enabled),
        'freeze_base_enabled': bool(freeze_base_enabled),
        'head_only_enabled': bool(head_only_enabled),
        'dacc_v2_only_enabled': bool(dacc_v2_only_enabled),
        'temporal_attention_enabled': bool(
            getattr(
                config,
                'temporal_memory_temporal_attention_enabled',
                False,
            )
        ),
    }
    if configured_density_version == DENSITY_CALIBRATION_V2_VERSION:
        temporal_memory_metadata.update(
            {
                'density_calibration_v2_basis': DENSITY_CALIBRATION_V2_BASIS,
                'density_calibration_v2_residual_scale': (
                    DENSITY_CALIBRATION_V2_RESIDUAL_SCALE
                ),
            }
        )
    return {
        'checkpoint_format_version': RESUME_CHECKPOINT_FORMAT_VERSION,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_class': optimizer.__class__.__qualname__,
        'scheduler_state_dict': scheduler.state_dict(),
        'scheduler_class': scheduler.__class__.__qualname__,
        'rng_state': capture_rng_state(include_cuda=include_cuda_rng),
        # ``epoch`` is the completed zero-based epoch. ``next_epoch`` is the
        # exact dataset/scheduler epoch at which a resumed process must start.
        'start_epoch': int(run_start_epoch),
        'epoch': int(epoch),
        'next_epoch': int(epoch) + 1,
        'loss': float(epoch_loss),
        'best_loss': float(best_loss),
        'best_epoch': None if best_epoch is None else int(best_epoch),
        'best_loss_checkpoint': (
            None
            if best_loss_checkpoint is None
            else str(Path(best_loss_checkpoint).resolve())
        ),
        'temporal_memory': temporal_memory_metadata,
        'provenance': {
            'saved_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'resolved_config': config.resolved_config,
            'config_overrides': list(config.config_overrides),
            'initialized_from': (
                None
                if initialized_from is None
                else str(Path(initialized_from).resolve())
            ),
            'initialized_from_sha256': (
                None
                if initialized_from_sha256 is None
                else str(initialized_from_sha256)
            ),
            'initialization_migrations': list(initialization_migrations or []),
            'resume_parent_checkpoint': (
                None
                if resume_parent_checkpoint is None
                else str(Path(resume_parent_checkpoint).resolve())
            ),
            'git': git_provenance or {'commit': None, 'dirty': None},
            'training_scope': training_scope_metadata,
        },
    }


def load_training_resume(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    current_config,
    restore_rng=True,
):
    """Restore a complete training snapshot and return its epoch metadata.

    Historical checkpoints intentionally remain model-only initialization
    inputs. They should be passed through
    ``TEMPORAL_MEMORY.temporal_memory_init_model_path`` rather than silently
    pretending that optimizer, scheduler, and RNG state were recoverable.
    """
    checkpoint_path = Path(str(checkpoint_path).strip())
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'Resume checkpoint not found: {}'.format(checkpoint_path)
        )
    checkpoint = load_checkpoint_file(checkpoint_path, map_location='cpu')
    if not is_resumable_checkpoint(checkpoint):
        raise ValueError(
            '{} is a legacy/model-only checkpoint and cannot be resumed. '
            'Use TEMPORAL_MEMORY.temporal_memory_init_model_path to use it '
            'as initialization instead.'.format(checkpoint_path)
        )
    validate_resume_config(checkpoint, current_config)
    start_epoch = int(checkpoint['next_epoch'])
    if start_epoch < 0:
        raise ValueError('Resume checkpoint next_epoch must be non-negative.')
    completed_epoch = int(checkpoint.get('epoch', start_epoch - 1))
    if start_epoch != completed_epoch + 1:
        raise ValueError(
            'Resume checkpoint has inconsistent epoch={} and next_epoch={}.'.format(
                completed_epoch, start_epoch
            )
        )
    if checkpoint['optimizer_class'] != optimizer.__class__.__qualname__:
        raise ValueError(
            'Resume optimizer class {} does not match configured {}.'.format(
                checkpoint['optimizer_class'], optimizer.__class__.__qualname__
            )
        )
    if checkpoint['scheduler_class'] != scheduler.__class__.__qualname__:
        raise ValueError(
            'Resume scheduler class {} does not match configured {}.'.format(
                checkpoint['scheduler_class'], scheduler.__class__.__qualname__
            )
        )
    saved_groups = checkpoint['optimizer_state_dict'].get('param_groups', [])
    current_groups = optimizer.state_dict().get('param_groups', [])
    saved_group_names = tuple(group.get('name') for group in saved_groups)
    current_group_names = tuple(group.get('name') for group in current_groups)
    if saved_group_names != current_group_names:
        raise ValueError(
            'Resume optimizer parameter-group names/order {} do not match '
            'configured {}.'.format(saved_group_names, current_group_names)
        )
    saved_memory = checkpoint.get('temporal_memory')
    if not isinstance(saved_memory, dict):
        raise ValueError('Resume checkpoint is missing temporal-memory metadata.')
    saved_density_version = validate_density_calibration_metadata(saved_memory)
    if hasattr(model, 'base') and hasattr(
        model.base,
        'density_calibration_enabled',
    ):
        model_density_version = density_calibration_version(
            model.base.density_calibration_enabled,
            getattr(model, 'density_calibration_v2_enabled', False),
        )
        if saved_density_version != model_density_version:
            raise ValueError(
                'Resume checkpoint density calibration version {} does not match '
                'configured {}.'.format(
                    saved_density_version,
                    model_density_version,
                )
            )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    expected_scope = None
    expected_mutable_keys = None
    if bool(getattr(current_config, 'temporal_memory_head_only_enabled', False)):
        expected_scope = 'event_head_only'
        expected_mutable_keys = HEAD_ONLY_MUTABLE_STATE_KEYS
    elif bool(
        getattr(current_config, 'temporal_memory_dacc_v2_only_enabled', False)
    ):
        expected_scope = 'dacc_v2_projection_only'
        expected_mutable_keys = DACC_V2_MUTABLE_STATE_KEYS
    if expected_scope is not None:
        training_scope = checkpoint.get('provenance', {}).get(
            'training_scope',
            {},
        )
        if training_scope.get('name') != expected_scope:
            raise ValueError(
                'Audited resume checkpoint is missing its training scope.'
            )
        mutable_state_keys = frozenset(
            training_scope.get('mutable_state_keys', [])
        )
        if mutable_state_keys != expected_mutable_keys:
            raise ValueError(
                'Audited resume checkpoint has unexpected mutable state keys.'
            )
        reference_sha256 = training_scope.get(
            'frozen_state_reference_sha256'
        )
        if not reference_sha256:
            raise ValueError(
                'Audited resume checkpoint is missing its frozen-state hash.'
            )
        actual_sha256 = frozen_model_state_sha256(
            model,
            mutable_state_keys=expected_mutable_keys,
        )
        if actual_sha256 != reference_sha256:
            raise ValueError(
                'Audited resume checkpoint frozen state does not match its '
                'recorded reference hash.'
            )
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except (ValueError, KeyError) as error:
        raise ValueError(
            'Resume optimizer state is incompatible with the configured '
            'model or parameter groups.'
        ) from error
    try:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    except (ValueError, KeyError) as error:
        raise ValueError(
            'Resume scheduler state is incompatible with the configured scheduler.'
        ) from error
    if restore_rng:
        restore_rng_state(checkpoint['rng_state'])
    return checkpoint, start_epoch


def setup_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_directory(config):
    started_at = datetime.now().astimezone()
    run_name = '{}_seed{}_pid{}'.format(
        started_at.strftime('%Y%m%d-%H%M%S'),
        int(config.seed),
        os.getpid(),
    )
    run_dir = Path(config.model_save_root) / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / 'config.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(
            config.resolved_config,
            stream,
            allow_unicode=True,
            sort_keys=False,
        )
    return run_dir, started_at


def save_checkpoint(checkpoint, path):
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def build_scheduler(optimizer, config):
    scheduler_name = str(config.scheduler).lower()
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.epochs),
            eta_min=float(config.scheduler_min_lr),
        )
    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.scheduler_step_size),
            gamma=float(config.scheduler_gamma),
        )
    raise ValueError('Unsupported scheduler: {}'.format(config.scheduler))


def load_p23_base_weights(
    model,
    checkpoint_path,
    context_bins,
    width,
    density_calibration_enabled=False,
    confidence_head_enabled=False,
    density_calibration_v2_enabled=False,
    initialization_migrations=None,
):
    checkpoint_path = Path(str(checkpoint_path).strip())
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'P23 initialization checkpoint not found: {}'.format(checkpoint_path)
        )
    checkpoint = load_checkpoint_file(checkpoint_path, map_location='cpu')
    saved_memory = checkpoint.get('temporal_memory')
    if saved_memory is not None:
        if not isinstance(saved_memory, dict):
            raise ValueError('Temporal-memory checkpoint metadata must be a mapping.')
        saved_context_bins = saved_memory.get('context_bins')
        saved_width = saved_memory.get('width')
        saved_sequence_length = saved_memory.get('sequence_length')
        if (
            saved_context_bins is not None
            and int(saved_context_bins) != int(context_bins)
        ):
            raise ValueError(
                'M5 context_bins={} does not match {}.'.format(
                    saved_context_bins, context_bins
                )
            )
        if saved_width is not None and int(saved_width) != int(width):
            raise ValueError(
                'M5 width={} does not match {}.'.format(saved_width, width)
            )
        if (
            saved_sequence_length is not None
            and int(saved_sequence_length) != int(cfg.temporal_memory_sequence_length)
        ):
            raise ValueError(
                'M5 sequence_length={} does not match {}.'.format(
                    saved_sequence_length, cfg.temporal_memory_sequence_length
                )
            )
        saved_density_calibration = bool(
            saved_memory.get('density_calibration_enabled', False)
        )
        saved_density_version = validate_density_calibration_metadata(
            saved_memory
        )
        configured_density_version = density_calibration_version(
            density_calibration_enabled,
            density_calibration_v2_enabled,
        )
        adding_dacc_v2 = (
            saved_density_version == DENSITY_CALIBRATION_LEGACY_VERSION
            and configured_density_version == DENSITY_CALIBRATION_V2_VERSION
        )
        if (
            saved_density_version != configured_density_version
            and not adding_dacc_v2
        ):
            raise ValueError(
                'M5 density calibration version={} does not match configured '
                '{}.'.format(
                    saved_density_version,
                    configured_density_version,
                )
            )
        saved_confidence_head = bool(
            saved_memory.get('confidence_head_enabled', False)
        )
        saved_temporal_attention = bool(
            saved_memory.get('temporal_attention_enabled', False)
        )
        if saved_density_calibration != bool(density_calibration_enabled):
            raise ValueError(
                'M5 density calibration={} does not match configured {}.'.format(
                    saved_density_calibration, density_calibration_enabled
                )
            )
        adding_confidence_head = (
            bool(confidence_head_enabled) and not saved_confidence_head
        )
        if (
            saved_confidence_head != bool(confidence_head_enabled)
            and not adding_confidence_head
        ):
            raise ValueError(
                'M5 confidence head={} does not match configured {}.'.format(
                    saved_confidence_head, confidence_head_enabled
                )
            )
        configured_temporal_attention = bool(
            getattr(model, 'temporal_attention_enabled', False)
        )
        adding_temporal_attention = (
            configured_temporal_attention and not saved_temporal_attention
        )
        if saved_temporal_attention and not configured_temporal_attention:
            raise ValueError(
                'M5 temporal attention={} does not match configured {}.'.format(
                    saved_temporal_attention, configured_temporal_attention
                )
            )
        if adding_dacc_v2 and (
            adding_confidence_head or adding_temporal_attention
        ):
            raise ValueError(
                'DACC-v2 migration must add only the residual projection; '
                'confidence and attention branches must already match the parent.'
            )
        # Every permitted addition is zero initialized and its exact missing
        # keys are audited below. No inference or resume path uses this escape.
        adding_branch = (
            adding_confidence_head
            or adding_temporal_attention
            or adding_dacc_v2
        )
        load_result = model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=not adding_branch,
        )
        if adding_branch:
            expected_missing = set()
            if adding_confidence_head:
                expected_missing.update(
                    'base.confidence_head.' + name
                    for name in model.base.confidence_head.state_dict()
                )
            if adding_temporal_attention:
                expected_missing.update(
                    'temporal_attn.' + name
                    for name in model.temporal_attn.state_dict()
                )
            if adding_dacc_v2:
                expected_missing.update(DACC_V2_MUTABLE_STATE_KEYS)
            if (
                set(load_result.missing_keys) != expected_missing
                or load_result.unexpected_keys
            ):
                raise RuntimeError(
                    'Only newly attached branches may be missing when '
                    'initializing from a complete temporal-memory checkpoint. '
                    'Missing={}, unexpected={}.'.format(
                        load_result.missing_keys,
                        load_result.unexpected_keys,
                    )
                )
        if adding_dacc_v2:
            model_state = model.state_dict()
            source_state = checkpoint['model_state_dict']
            for name, expected in source_state.items():
                actual = model_state[name].detach().cpu()
                if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(
                    expected.shape
                ) or not torch.equal(
                    _state_tensor_byte_view(actual),
                    _state_tensor_byte_view(expected),
                ):
                    raise RuntimeError(
                        'DACC-v2 migration changed inherited tensor: {}.'.format(
                            name
                        )
                    )
            projection = model_state[
                'base.density_calibrator.residual_projection.weight'
            ]
            if torch.count_nonzero(projection).item() != 0:
                raise RuntimeError(
                    'DACC-v2 residual projection must remain zero after migration.'
                )
            source_state_sha256 = frozen_model_state_sha256(
                source_state,
                mutable_state_keys=DACC_V2_MUTABLE_STATE_KEYS,
            )
            migrated_state_sha256 = frozen_model_state_sha256(
                model,
                mutable_state_keys=DACC_V2_MUTABLE_STATE_KEYS,
            )
            if source_state_sha256 != migrated_state_sha256:
                raise RuntimeError(
                    'DACC-v2 migrated frozen state does not match its parent.'
                )
            if initialization_migrations is not None:
                initialization_migrations.append(
                    {
                        'name': 'density_calibration_v1_to_v2_zero_residual',
                        'missing_keys': sorted(DACC_V2_MUTABLE_STATE_KEYS),
                        'source_model_state_sha256': source_state_sha256,
                        'migrated_frozen_state_sha256': migrated_state_sha256,
                    }
                )
        return checkpoint_path

    saved = checkpoint.get('temporal_frame', {})
    if density_calibration_v2_enabled:
        raise ValueError(
            'DACC-v2 requires a legacy temporal-memory checkpoint parent; '
            'direct temporal-frame initialization is not allowed.'
        )
    if saved.get('context_bins') is not None and int(
        saved['context_bins']
    ) != int(context_bins):
        raise ValueError(
            'P23 context_bins={} does not match {}.'.format(
                saved['context_bins'], context_bins
            )
        )
    if saved.get('width') is not None and int(saved['width']) != int(width):
        raise ValueError(
            'P23 width={} does not match {}.'.format(saved['width'], width)
        )
    # A pure-P23 checkpoint has no density-calibrator or confidence-head
    # keys; leave those at their safe identity/zero init instead.
    model.base.load_state_dict(
        checkpoint['model_state_dict'],
        strict=not bool(
            density_calibration_enabled or confidence_head_enabled
        ),
    )
    return checkpoint_path


def validate_head_only_initialization_checkpoint(checkpoint_path):
    """Require a complete temporal-memory parent for the H17 probe."""
    checkpoint = load_checkpoint_file(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint.get('temporal_memory'), dict):
        raise ValueError(
            'Head-only mode requires a complete temporal-memory checkpoint.'
        )
    state = checkpoint.get('model_state_dict', {})
    missing = sorted(HEAD_ONLY_MUTABLE_STATE_KEYS.difference(state))
    if missing:
        raise ValueError(
            'Head-only initialization checkpoint is missing: {}.'.format(missing)
        )


def configure_temporal_memory_trainable_parameters(
    model,
    confidence_only_enabled=False,
    freeze_base_enabled=False,
    head_only_enabled=False,
    dacc_v2_only_enabled=False,
):
    """Apply an explicit, checkpoint-neutral temporal training scope."""
    training_scope = resolve_temporal_memory_training_scope(
        confidence_only_enabled=confidence_only_enabled,
        freeze_base_enabled=freeze_base_enabled,
        head_only_enabled=head_only_enabled,
        dacc_v2_only_enabled=dacc_v2_only_enabled,
    )
    if training_scope == 'confidence_only':
        if not model.confidence_head_enabled:
            raise ValueError(
                'Confidence-only mode requires the confidence head to be enabled.'
            )
        confidence_parameter_ids = {
            id(parameter) for parameter in model.base.confidence_head.parameters()
        }
        for parameter in model.parameters():
            parameter.requires_grad_(id(parameter) in confidence_parameter_ids)
    elif training_scope == 'memory_only':
        model.base.requires_grad_(False)
    elif training_scope == 'event_head_only':
        head_parameter_ids = {
            id(parameter) for parameter in model.base.head.parameters()
        }
        for parameter in model.parameters():
            parameter.requires_grad_(id(parameter) in head_parameter_ids)
        trainable_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if trainable_names != HEAD_ONLY_MUTABLE_STATE_KEYS:
            raise RuntimeError(
                'Head-only trainable parameters are {} instead of {}.'.format(
                    sorted(trainable_names),
                    sorted(HEAD_ONLY_MUTABLE_STATE_KEYS),
                )
            )
        trainable_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if trainable_count != 17:
            raise RuntimeError(
                'Head-only mode expected 17 trainable parameters, got {}.'.format(
                    trainable_count
                )
            )
    elif training_scope == 'dacc_v2_projection_only':
        calibrator = model.base.density_calibrator
        projection = (
            None if calibrator is None else calibrator.residual_projection
        )
        if projection is None or not getattr(
            model,
            'density_calibration_v2_enabled',
            False,
        ):
            raise ValueError(
                'DACC-v2-only mode requires a DACC-v2 model.'
            )
        projection_parameter_ids = {
            id(parameter) for parameter in projection.parameters()
        }
        for parameter in model.parameters():
            parameter.requires_grad_(
                id(parameter) in projection_parameter_ids
            )
        trainable_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if trainable_names != DACC_V2_MUTABLE_STATE_KEYS:
            raise RuntimeError(
                'DACC-v2 trainable parameters are {} instead of {}.'.format(
                    sorted(trainable_names),
                    sorted(DACC_V2_MUTABLE_STATE_KEYS),
                )
            )
        trainable_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if trainable_count != 32:
            raise RuntimeError(
                'DACC-v2-only mode expected 32 trainable parameters, got {}.'.format(
                    trainable_count
                )
            )


def set_temporal_memory_training_mode(
    model,
    confidence_only_enabled=False,
    freeze_base_enabled=False,
    head_only_enabled=False,
    dacc_v2_only_enabled=False,
):
    """Set module modes without re-enabling a frozen base at each epoch."""
    training_scope = resolve_temporal_memory_training_scope(
        confidence_only_enabled=confidence_only_enabled,
        freeze_base_enabled=freeze_base_enabled,
        head_only_enabled=head_only_enabled,
        dacc_v2_only_enabled=dacc_v2_only_enabled,
    )
    if training_scope == 'confidence_only':
        model.eval()
        model.base.confidence_head.train()
        return
    if training_scope == 'event_head_only':
        model.eval()
        model.base.head.train()
        return
    if training_scope == 'dacc_v2_projection_only':
        model.eval()
        model.base.density_calibrator.residual_projection.train()
        return
    model.train()
    if training_scope == 'memory_only':
        model.base.eval()


def build_optimizer(
    model,
    config,
    confidence_only_enabled=False,
    head_only_enabled=False,
    dacc_v2_only_enabled=False,
):
    base_multiplier = float(config.temporal_memory_base_lr_multiplier)
    memory_multiplier = float(config.temporal_memory_memory_lr_multiplier)
    confidence_multiplier = float(
        getattr(config, 'temporal_memory_confidence_lr_multiplier', 1.0)
    )
    dacc_v2_multiplier = float(
        getattr(config, 'temporal_memory_dacc_v2_lr_multiplier', 1.0)
    )
    if (
        base_multiplier <= 0.0
        or memory_multiplier <= 0.0
        or confidence_multiplier <= 0.0
        or dacc_v2_multiplier <= 0.0
    ):
        raise ValueError('Temporal-memory learning-rate multipliers must be positive.')
    resolve_temporal_memory_training_scope(
        confidence_only_enabled=confidence_only_enabled,
        head_only_enabled=head_only_enabled,
        dacc_v2_only_enabled=dacc_v2_only_enabled,
    )
    if bool(head_only_enabled):
        head_parameters = [
            parameter
            for parameter in model.base.head.parameters()
            if parameter.requires_grad
        ]
        if len(head_parameters) != 2 or sum(
            parameter.numel() for parameter in head_parameters
        ) != 17:
            raise RuntimeError(
                'Head-only optimizer requires exactly two tensors and 17 parameters.'
            )
        return optim.AdamW(
            [
                {
                    'name': 'event_head',
                    'params': head_parameters,
                    'lr': float(config.lr),
                }
            ],
            weight_decay=0.0,
        )
    if bool(dacc_v2_only_enabled):
        projection = model.base.density_calibrator.residual_projection
        projection_parameters = [
            parameter
            for parameter in projection.parameters()
            if parameter.requires_grad
        ]
        if len(projection_parameters) != 1 or sum(
            parameter.numel() for parameter in projection_parameters
        ) != 32:
            raise RuntimeError(
                'DACC-v2 optimizer requires exactly one tensor and 32 parameters.'
            )
        return optim.AdamW(
            [
                {
                    'name': 'dacc_v2',
                    'params': projection_parameters,
                    'lr': float(config.lr) * dacc_v2_multiplier,
                }
            ],
            weight_decay=0.0,
        )
    confidence_parameters = []
    if model.confidence_head_enabled:
        confidence_parameters = [
            parameter
            for parameter in model.base.confidence_head.parameters()
            if parameter.requires_grad
        ]
    confidence_parameter_ids = {id(parameter) for parameter in confidence_parameters}
    if confidence_only_enabled:
        if not confidence_parameters:
            raise ValueError(
                'Confidence-only mode requires the confidence head to be enabled.'
            )
        return optim.AdamW(
            [
                {
                    'name': 'confidence',
                    'params': confidence_parameters,
                    'lr': float(config.lr) * confidence_multiplier,
                }
            ],
            weight_decay=1e-4,
        )
    base_parameters = [
        parameter
        for parameter in model.base.parameters()
        if parameter.requires_grad
        and id(parameter) not in confidence_parameter_ids
    ]
    memory_parameters = [
        parameter
        for parameter in model.forward_memory.parameters()
        if parameter.requires_grad
    ]
    memory_parameters += [
        parameter
        for parameter in model.backward_memory.parameters()
        if parameter.requires_grad
    ]
    memory_parameters += [
        parameter
        for parameter in model.memory_projection.parameters()
        if parameter.requires_grad
    ]
    if getattr(model, 'temporal_attention_enabled', False):
        memory_parameters += [
            parameter
            for parameter in model.temporal_attn.parameters()
            if parameter.requires_grad
        ]
    parameter_groups = []
    if base_parameters:
        parameter_groups.append(
            {
                'name': 'base',
                'params': base_parameters,
                'lr': float(config.lr) * base_multiplier,
            }
        )
    if confidence_parameters:
        parameter_groups.append(
            {
                'name': 'confidence',
                'params': confidence_parameters,
                'lr': float(config.lr) * confidence_multiplier,
            }
        )
    if memory_parameters:
        parameter_groups.append(
            {
                'name': 'memory',
                'params': memory_parameters,
                'lr': float(config.lr) * memory_multiplier,
            }
        )
    if not parameter_groups:
        raise ValueError('Temporal-memory optimizer has no trainable parameters.')
    return optim.AdamW(parameter_groups, weight_decay=1e-4)


def memory_config_summary(config):
    return (
        'enabled (bin_size={}, context_bins={}, width={}, sequence_length={}, '
        'views_per_video={}, positive_frame_probability={}, '
        'target_positive_loss_mass={}, max_positive_weight={}, '
        'base_lr_multiplier={}, memory_lr_multiplier={}, '
        'confidence_lr_multiplier={}, '
        'attention_enabled={}, freeze_base_enabled={}, head_only_enabled={}, '
        'dacc_v2_enabled={}, dacc_v2_only_enabled={}, '
        'min_event_count_exclusive={})'
    ).format(
        config.temporal_memory_bin_size,
        config.temporal_memory_context_bins,
        config.temporal_memory_width,
        config.temporal_memory_sequence_length,
        config.temporal_memory_train_views_per_video,
        config.temporal_memory_positive_frame_probability,
        config.temporal_memory_target_positive_loss_mass,
        config.temporal_memory_max_positive_weight,
        config.temporal_memory_base_lr_multiplier,
        config.temporal_memory_memory_lr_multiplier,
        getattr(config, 'temporal_memory_confidence_lr_multiplier', 1.0),
        bool(getattr(config, 'temporal_memory_temporal_attention_enabled', False)),
        bool(getattr(config, 'temporal_memory_freeze_base_enabled', False)),
        bool(getattr(config, 'temporal_memory_head_only_enabled', False)),
        bool(getattr(config, 'temporal_memory_dacc_v2_enabled', False)),
        bool(getattr(config, 'temporal_memory_dacc_v2_only_enabled', False)),
        getattr(
            config,
            'temporal_memory_train_min_event_count_exclusive',
            None,
        ),
    )


def apply_target_coverage_loss(
    base_loss,
    coverage_loss,
    enabled,
    epoch,
    warmup_epochs,
    weight,
):
    """Add coverage only after warmup and preserve the original loss otherwise."""
    if bool(enabled) and int(epoch) >= int(warmup_epochs):
        return base_loss + float(weight) * coverage_loss, float(weight)
    return base_loss, 0.0


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for temporal-memory training.')
    if not bool(cfg.temporal_memory_enabled):
        raise ValueError('Set TEMPORAL_MEMORY.temporal_memory_enabled=true.')
    if int(cfg.temporal_memory_context_bins) % 2 == 0:
        raise ValueError('TEMPORAL_MEMORY.context_bins must be odd.')
    if int(cfg.temporal_memory_sequence_length) <= 1:
        raise ValueError('TEMPORAL_MEMORY.sequence_length must exceed one.')
    if int(cfg.temporal_memory_train_workers) != 0 and bool(
        cfg.temporal_memory_cache_all_videos
    ):
        raise ValueError(
            'Use TEMPORAL_MEMORY.train_workers=0 when cache_all_videos=true.'
        )
    if int(cfg.epochs) <= 0:
        raise ValueError('TRAIN.epochs must be positive.')

    confidence_head_enabled = bool(
        getattr(cfg, 'temporal_frame_confidence_head_enabled', False)
    )
    confidence_only_enabled = bool(
        getattr(cfg, 'temporal_memory_confidence_only_enabled', False)
    )
    freeze_base_enabled = bool(
        getattr(cfg, 'temporal_memory_freeze_base_enabled', False)
    )
    head_only_enabled = bool(
        getattr(cfg, 'temporal_memory_head_only_enabled', False)
    )
    dacc_v2_enabled = bool(
        getattr(cfg, 'temporal_memory_dacc_v2_enabled', False)
    )
    dacc_v2_only_enabled = bool(
        getattr(cfg, 'temporal_memory_dacc_v2_only_enabled', False)
    )
    training_scope = resolve_temporal_memory_training_scope(
        confidence_only_enabled=confidence_only_enabled,
        freeze_base_enabled=freeze_base_enabled,
        head_only_enabled=head_only_enabled,
        dacc_v2_only_enabled=dacc_v2_only_enabled,
    )
    if confidence_only_enabled and not confidence_head_enabled:
        raise ValueError(
            'TEMPORAL_MEMORY.confidence_only_enabled requires '
            'TEMPORAL_FRAME.confidence_head_enabled=true.'
        )
    validate_head_only_training_config(cfg, training_scope)
    validate_dacc_v2_training_config(cfg, training_scope)

    resume_checkpoint_value = str(
        getattr(cfg, 'resume_checkpoint', '')
    ).strip()
    resume_parent_checkpoint = (
        Path(resume_checkpoint_value).resolve()
        if resume_checkpoint_value
        else None
    )
    git_provenance = collect_git_provenance()
    setup_seed(cfg.seed)
    device = torch.device('cuda:0')
    run_dir, started_at = create_run_directory(cfg)
    dataset = TemporalMemoryTrainDataset(
        root=Path(cfg.root) / 'train',
        whole_t=cfg.whole_t,
        temporal_bin_size=cfg.temporal_memory_bin_size,
        context_bins=cfg.temporal_memory_context_bins,
        sequence_length=cfg.temporal_memory_sequence_length,
        width=cfg.res[0],
        height=cfg.res[1],
        views_per_video=cfg.temporal_memory_train_views_per_video,
        positive_frame_probability=cfg.temporal_memory_positive_frame_probability,
        random_seed=cfg.seed,
        log_count_clip=cfg.temporal_memory_log_count_clip,
        cache_all_videos=cfg.temporal_memory_cache_all_videos,
        cache_video_count=cfg.temporal_memory_cache_video_count,
        dense_sampling_enabled=getattr(
            cfg,
            'temporal_memory_dense_sampling_enabled',
            False,
        ),
        dense_event_count_cutoff=getattr(
            cfg,
            'temporal_memory_dense_event_count_cutoff',
            200000,
        ),
        dense_view_multiplier=getattr(
            cfg,
            'temporal_memory_dense_view_multiplier',
            2,
        ),
        density_bucket_boundaries=getattr(
            cfg,
            'temporal_memory_density_bucket_boundaries',
            [],
        ),
        density_bucket_views=getattr(
            cfg,
            'temporal_memory_density_bucket_views',
            [],
        ),
        min_event_count_exclusive=getattr(
            cfg,
            'temporal_memory_train_min_event_count_exclusive',
            None,
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.temporal_memory_train_workers),
        collate_fn=temporal_memory_collate,
        pin_memory=True,
    )
    density_calibration_enabled = bool(
        getattr(cfg, 'temporal_frame_density_calibration_enabled', False)
    )
    trajectory_enabled = bool(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_enabled', False)
    )
    trajectory_weight = float(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_weight', 0.05)
    )
    trajectory_margin = float(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_margin_logit', 1.0)
    )
    trajectory_min_points = int(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_min_points', 3)
    )
    trajectory_warmup_epochs = int(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_warmup_epochs', 3)
    )
    if trajectory_enabled:
        if trajectory_weight <= 0.0:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_weight must be positive.'
            )
        if trajectory_min_points < 2:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_min_points must be at least 2.'
            )
        if trajectory_warmup_epochs < 0:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_warmup_epochs must be non-negative.'
            )
    target_coverage_enabled = bool(
        getattr(cfg, 'temporal_memory_target_coverage_enabled', False)
    )
    target_coverage_weight = float(
        getattr(cfg, 'temporal_memory_target_coverage_weight', 0.005)
    )
    target_coverage_warmup_epochs = int(
        getattr(cfg, 'temporal_memory_target_coverage_warmup_epochs', 1)
    )
    target_coverage_score_floor = float(
        getattr(cfg, 'temporal_memory_target_coverage_score_floor', 0.719)
    )
    target_coverage_correct_fraction = float(
        getattr(
            cfg,
            'temporal_memory_target_coverage_correct_fraction',
            0.0001,
        )
    )
    if target_coverage_enabled:
        if target_coverage_weight <= 0.0:
            raise ValueError(
                'TEMPORAL_MEMORY.target_coverage_weight must be positive.'
            )
        if target_coverage_warmup_epochs < 0:
            raise ValueError(
                'TEMPORAL_MEMORY.target_coverage_warmup_epochs must be '
                'non-negative.'
            )
        if not 0.0 < target_coverage_score_floor < 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.target_coverage_score_floor must be in (0, 1).'
            )
        if not 0.0 < target_coverage_correct_fraction <= 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.target_coverage_correct_fraction must be '
                'in (0, 1].'
            )
    metric_aux_enabled = bool(
        getattr(cfg, 'temporal_memory_metric_aux_enabled', False)
    )
    metric_target_weight = float(
        getattr(cfg, 'temporal_memory_metric_target_weight', 0.01)
    )
    metric_component_weight = float(
        getattr(cfg, 'temporal_memory_metric_component_weight', 0.002)
    )
    metric_warmup_epochs = int(
        getattr(cfg, 'temporal_memory_metric_warmup_epochs', 5)
    )
    metric_spatial_cell_size = int(
        getattr(cfg, 'temporal_memory_metric_spatial_cell_size', 3)
    )
    metric_min_cell_events = int(
        getattr(cfg, 'temporal_memory_metric_min_cell_events', 2)
    )
    metric_component_ratio = float(
        getattr(cfg, 'temporal_memory_metric_component_ratio', 0.01)
    )
    metric_activation_threshold = float(
        getattr(cfg, 'temporal_memory_metric_activation_threshold', 0.70)
    )
    metric_activation_temperature = float(
        getattr(cfg, 'temporal_memory_metric_activation_temperature', 0.10)
    )
    if metric_aux_enabled:
        if metric_target_weight < 0.0 or metric_component_weight < 0.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric target/component weights must be non-negative.'
            )
        if metric_warmup_epochs < 0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_warmup_epochs must be non-negative.'
            )
        if metric_spatial_cell_size <= 0 or metric_min_cell_events <= 0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric cell size and minimum events must be positive.'
            )
        if not 0.0 < metric_component_ratio <= 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_component_ratio must be in (0, 1].'
            )
        if not 0.0 < metric_activation_threshold < 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_activation_threshold must be in (0, 1).'
            )
        if metric_activation_temperature <= 0.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_activation_temperature must be positive.'
            )
    checkpoint_interval = int(getattr(cfg, 'checkpoint_interval', 0))
    if checkpoint_interval < 0:
        raise ValueError('TRAIN.checkpoint_interval must be non-negative.')
    confidence_calibration_weight = float(
        getattr(cfg, 'temporal_frame_confidence_calibration_weight', 0.1)
    )
    if confidence_head_enabled and confidence_calibration_weight <= 0.0:
        raise ValueError(
            'TEMPORAL_FRAME.confidence_calibration_weight must be positive.'
        )
    temporal_attention_enabled = bool(
        getattr(cfg, 'temporal_memory_temporal_attention_enabled', False)
    )
    model = BidirectionalTemporalMemoryNet(
        input_channels=int(cfg.temporal_memory_context_bins) * 2,
        width=int(cfg.temporal_memory_width),
        density_calibration_enabled=density_calibration_enabled,
        density_calibration_v2_enabled=dacc_v2_enabled,
        confidence_head_enabled=confidence_head_enabled,
        temporal_attention_enabled=temporal_attention_enabled,
    ).to(device)
    initialized_from = None
    initialized_from_sha256 = None
    initialization_migrations = []
    if resume_parent_checkpoint is None:
        initialized_from = load_p23_base_weights(
            model,
            cfg.temporal_memory_init_model_path,
            cfg.temporal_memory_context_bins,
            cfg.temporal_memory_width,
            density_calibration_enabled=density_calibration_enabled,
            density_calibration_v2_enabled=dacc_v2_enabled,
            confidence_head_enabled=confidence_head_enabled,
            initialization_migrations=initialization_migrations,
        )
        initialized_from_sha256 = sha256_file(initialized_from)
        if head_only_enabled:
            validate_head_only_initialization_checkpoint(initialized_from)
    configure_temporal_memory_trainable_parameters(
        model,
        confidence_only_enabled=confidence_only_enabled,
        freeze_base_enabled=freeze_base_enabled,
        head_only_enabled=head_only_enabled,
        dacc_v2_only_enabled=dacc_v2_only_enabled,
    )
    optimizer = build_optimizer(
        model,
        cfg,
        confidence_only_enabled=confidence_only_enabled,
        head_only_enabled=head_only_enabled,
        dacc_v2_only_enabled=dacc_v2_only_enabled,
    )
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group['params']
    ]
    scheduler = build_scheduler(optimizer, cfg)
    start_epoch = 0
    best_loss = float('inf')
    best_epoch = None
    best_loss_checkpoint = None
    frozen_state_reference = None
    frozen_state_reference_sha256 = None
    if resume_parent_checkpoint is not None:
        resumed, start_epoch = load_training_resume(
            resume_parent_checkpoint,
            model,
            optimizer,
            scheduler,
            current_config=cfg,
        )
        best_loss = float(resumed['best_loss'])
        best_epoch = resumed['best_epoch']
        if best_epoch is not None:
            best_epoch = int(best_epoch)
        best_loss_checkpoint = resumed.get('best_loss_checkpoint')
        initialized_from = resumed.get('provenance', {}).get(
            'initialized_from'
        )
        initialized_from_sha256 = resumed.get('provenance', {}).get(
            'initialized_from_sha256'
        )
        initialization_migrations = list(
            resumed.get('provenance', {}).get('initialization_migrations', [])
        )
        if start_epoch > int(cfg.epochs):
            raise ValueError(
                'Resume checkpoint starts at epoch {}, beyond configured '
                'TRAIN.epochs={}.'.format(start_epoch, cfg.epochs)
            )
    audited_mutable_state_keys = None
    if head_only_enabled:
        audited_mutable_state_keys = HEAD_ONLY_MUTABLE_STATE_KEYS
    elif dacc_v2_only_enabled:
        audited_mutable_state_keys = DACC_V2_MUTABLE_STATE_KEYS
    if audited_mutable_state_keys is not None:
        if resume_parent_checkpoint is None:
            frozen_state_reference_sha256 = frozen_model_state_sha256(
                model,
                mutable_state_keys=audited_mutable_state_keys,
            )
        else:
            frozen_state_reference_sha256 = resumed['provenance'][
                'training_scope'
            ]['frozen_state_reference_sha256']
        frozen_state_reference = snapshot_frozen_model_state(
            model,
            mutable_state_keys=audited_mutable_state_keys,
        )
        if (
            frozen_model_state_sha256(frozen_state_reference, mutable_state_keys=())
            != frozen_state_reference_sha256
        ):
            raise RuntimeError(
                'Audited frozen-state snapshot does not match its reference hash.'
            )

    print('random seed:{}'.format(cfg.seed))
    print('run directory:', run_dir)
    print('config overrides:', ', '.join(cfg.config_overrides) or '(none)')
    print('temporal-memory model:', memory_config_summary(cfg))
    print('training videos:', len(dataset.file_paths))
    print('training sequences per epoch:', len(dataset))
    print(
        'sampling summary:',
        json.dumps(dataset.sampling_summary(), sort_keys=True),
    )
    print(
        'dense sequence sampling: enabled={}, cutoff={}, multiplier={}, '
        'dense_videos={}, extra_views={}'.format(
            dataset.dense_sampling_enabled,
            dataset.dense_event_count_cutoff,
            dataset.dense_view_multiplier,
            dataset.dense_video_count,
            dataset.extra_dense_views,
        )
    )
    if resume_parent_checkpoint is not None:
        print('resumed complete training state from:', resume_parent_checkpoint)
        print('resume start epoch:', start_epoch)
    elif 'temporal_memory' in load_checkpoint_file(
        initialized_from, map_location='cpu'
    ):
        print('initialized full temporal-memory weights from:', initialized_from)
    else:
        print('initialized P23 base weights from:', initialized_from)
    print('learning-rate scheduler:', cfg.scheduler)
    if target_coverage_enabled:
        print(
            'target-time coverage loss: enabled (weight={}, '
            'warmup_epochs={}, score_floor={}, correct_fraction={})'.format(
                target_coverage_weight,
                target_coverage_warmup_epochs,
                target_coverage_score_floor,
                target_coverage_correct_fraction,
            )
        )
    else:
        print('target-time coverage loss: disabled')
    if confidence_only_enabled:
        print('confidence calibration mode: backbone and memory frozen')
    if head_only_enabled:
        print(
            'event-head-only mode: 17 parameters trainable; frozen state sha256:',
            frozen_state_reference_sha256,
        )
    if dacc_v2_only_enabled:
        print(
            'DACC-v2 projection-only mode: 32 parameters trainable; '
            'frozen state sha256:',
            frozen_state_reference_sha256,
        )

    last_checkpoint_path = (
        resume_parent_checkpoint
        if start_epoch >= int(cfg.epochs)
        else run_dir / 'last_seed{}.pt'.format(cfg.seed)
    )
    for epoch in range(start_epoch, int(cfg.epochs)):
        dataset.set_epoch(epoch)
        set_temporal_memory_training_mode(
            model,
            confidence_only_enabled=confidence_only_enabled,
            freeze_base_enabled=freeze_base_enabled,
            head_only_enabled=head_only_enabled,
            dacc_v2_only_enabled=dacc_v2_only_enabled,
        )
        loss_sum = 0.0
        positive_fraction_sum = 0.0
        positive_weight_sum = 0.0
        trajectory_loss_sum = 0.0
        confidence_loss_sum = 0.0
        target_coverage_loss_sum = 0.0
        target_coverage_weighted_loss_sum = 0.0
        target_coverage_group_count = 0
        target_coverage_uncovered_count = 0
        metric_target_loss_sum = 0.0
        metric_component_loss_sum = 0.0
        batch_count = 0
        pbar = tqdm.tqdm(
            dataloader,
            desc='Epoch: {}'.format(epoch),
            unit='Sequence',
            position=0,
            leave=True,
        )
        for batch in pbar:
            frames = batch['frames'].to(device, non_blocking=True).unsqueeze(0)
            event_time_indices = batch['event_time_indices'].to(
                device,
                non_blocking=True,
            )
            event_y = batch['event_y'].to(device, non_blocking=True)
            event_x = batch['event_x'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            target_ids = batch['target_ids'].to(device, non_blocking=True)
            event_timestamps = None
            if target_coverage_enabled:
                event_timestamps = batch['event_timestamps'].to(
                    device,
                    non_blocking=True,
                )

            optimizer.zero_grad(set_to_none=True)
            model_output = model(frames)
            if confidence_head_enabled:
                logit_maps, confidence_logit_maps = model_output
                logit_maps = logit_maps.squeeze(0)
                confidence_logit_maps = confidence_logit_maps.squeeze(0)
            else:
                logit_maps = model_output.squeeze(0)
            event_logits = logit_maps[
                event_time_indices,
                0,
                event_y,
                event_x,
            ]
            loss, diagnostics = frame_balanced_event_bce(
                event_logits,
                labels,
                event_time_indices,
                target_positive_loss_mass=(
                    cfg.temporal_memory_target_positive_loss_mass
                ),
                max_positive_weight=cfg.temporal_memory_max_positive_weight,
            )
            target_coverage_loss = None
            target_coverage_stats = {
                'target_group_count': 0,
                'uncovered_group_count': 0,
            }
            target_coverage_applied_weight = 0.0
            if target_coverage_enabled:
                target_coverage_loss, target_coverage_stats = (
                    target_time_group_coverage_loss(
                        event_logits,
                        labels,
                        target_ids,
                        torch.zeros_like(event_time_indices),
                        event_timestamps,
                        temporal_bin_size=int(cfg.temporal_memory_bin_size),
                        score_floor=target_coverage_score_floor,
                        correct_fraction=target_coverage_correct_fraction,
                    )
                )
                loss, target_coverage_applied_weight = (
                    apply_target_coverage_loss(
                        loss,
                        target_coverage_loss,
                        target_coverage_enabled,
                        epoch,
                        target_coverage_warmup_epochs,
                        target_coverage_weight,
                    )
                )
            confidence_loss = event_logits.sum() * 0.0
            if confidence_head_enabled:
                event_confidence_logits = confidence_logit_maps[
                    event_time_indices,
                    0,
                    event_y,
                    event_x,
                ]
                confidence_loss = confidence_calibration_loss(
                    event_confidence_logits,
                    event_logits,
                    labels,
                    hard_target=True,
                )
                loss = loss + confidence_calibration_weight * confidence_loss
            metric_target_loss = event_logits.sum() * 0.0
            metric_component_loss = event_logits.sum() * 0.0
            if metric_aux_enabled and epoch >= metric_warmup_epochs:
                event_scores = torch.sigmoid(event_logits)
                event_locations = torch.stack(
                    (
                        torch.zeros_like(event_x),
                        event_x,
                        event_y,
                        event_time_indices * int(cfg.temporal_memory_bin_size) + 1,
                    ),
                    dim=1,
                )
                if metric_target_weight > 0.0:
                    metric_target_loss, _, _ = target_frame_activation_loss(
                        event_scores,
                        labels,
                        target_ids,
                        event_locations,
                        int(cfg.temporal_memory_bin_size),
                        metric_activation_threshold,
                        metric_activation_temperature,
                    )
                    loss = loss + metric_target_weight * metric_target_loss
                if metric_component_weight > 0.0:
                    metric_component_loss, _, _ = component_hard_negative_loss(
                        event_scores,
                        labels,
                        event_locations,
                        metric_spatial_cell_size,
                        int(cfg.temporal_memory_bin_size),
                        metric_min_cell_events,
                        metric_component_ratio,
                        metric_activation_threshold,
                        metric_activation_temperature,
                    )
                    loss = loss + metric_component_weight * metric_component_loss
            trajectory_loss = event_logits.sum() * 0.0
            if trajectory_enabled and epoch >= trajectory_warmup_epochs:
                trajectory_loss, trajectory_stats = (
                    trajectory_extrapolation_loss_memory(
                        logit_maps,
                        event_time_indices,
                        event_x,
                        event_y,
                        labels,
                        target_ids,
                        min_known_points=trajectory_min_points,
                        margin_logit=trajectory_margin,
                    )
                )
                loss = loss + trajectory_weight * trajectory_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_parameters, max_norm=5.0)
            optimizer.step()

            loss_sum += float(loss.detach().item())
            positive_fraction_sum += diagnostics['positive_fraction']
            positive_weight_sum += diagnostics['mean_positive_weight']
            trajectory_loss_sum += float(trajectory_loss.detach().item())
            confidence_loss_sum += float(confidence_loss.detach().item())
            target_coverage_loss_value = (
                0.0
                if target_coverage_loss is None
                else float(target_coverage_loss.detach().item())
            )
            target_coverage_loss_sum += target_coverage_loss_value
            target_coverage_weighted_loss_sum += (
                target_coverage_applied_weight * target_coverage_loss_value
            )
            target_coverage_group_count += int(
                target_coverage_stats['target_group_count']
            )
            target_coverage_uncovered_count += int(
                target_coverage_stats['uncovered_group_count']
            )
            metric_target_loss_sum += float(metric_target_loss.detach().item())
            metric_component_loss_sum += float(metric_component_loss.detach().item())
            batch_count += 1
            pbar.set_postfix(
                loss='{:.5f}'.format(loss_sum / batch_count),
                pos='{:.4f}'.format(positive_fraction_sum / batch_count),
                pos_w='{:.2f}'.format(positive_weight_sum / batch_count),
                traj='{:.5f}'.format(trajectory_loss_sum / batch_count),
                conf='{:.5f}'.format(confidence_loss_sum / batch_count),
                coverage='{:.5f}'.format(
                    target_coverage_loss_sum / batch_count
                ),
                coverage_w='{:.6f}'.format(
                    target_coverage_weighted_loss_sum / batch_count
                ),
                coverage_miss='{:.4f}'.format(
                    target_coverage_uncovered_count
                    / max(target_coverage_group_count, 1)
                ),
                metric_pd='{:.5f}'.format(metric_target_loss_sum / batch_count),
                metric_fa='{:.5f}'.format(metric_component_loss_sum / batch_count),
            )
        pbar.close()
        scheduler.step()

        epoch_loss = loss_sum / max(batch_count, 1)
        if audited_mutable_state_keys is not None:
            assert_frozen_model_state_unchanged(
                model,
                frozen_state_reference,
                mutable_state_keys=audited_mutable_state_keys,
            )
            if frozen_model_state_sha256(
                model,
                mutable_state_keys=audited_mutable_state_keys,
            ) != frozen_state_reference_sha256:
                raise RuntimeError(
                    'Audited frozen-state hash changed before checkpointing.'
                )
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
            best_epoch = epoch
            best_loss_checkpoint = (
                run_dir / 'best_loss_seed{}.pt'.format(cfg.seed)
            )
        checkpoint = build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            epoch_loss=epoch_loss,
            best_loss=best_loss,
            best_epoch=best_epoch,
            config=cfg,
            initialized_from=initialized_from,
            resume_parent_checkpoint=resume_parent_checkpoint,
            run_start_epoch=start_epoch,
            best_loss_checkpoint=best_loss_checkpoint,
            git_provenance=git_provenance,
            frozen_state_reference_sha256=frozen_state_reference_sha256,
            initialized_from_sha256=initialized_from_sha256,
            initialization_migrations=initialization_migrations,
        )
        if is_best:
            save_checkpoint(
                checkpoint,
                best_loss_checkpoint,
            )
        last_checkpoint_path = run_dir / 'last_seed{}.pt'.format(cfg.seed)
        save_checkpoint(checkpoint, last_checkpoint_path)
        if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint,
                run_dir / 'epoch_{:03d}_seed{}.pt'.format(epoch + 1, cfg.seed),
            )
        learning_rates = ', '.join(
            'lr_{}={:.8f}'.format(group['name'], group['lr'])
            for group in optimizer.param_groups
        )
        print(
            'epoch {}: loss={:.6f}, {}, best_loss={:.6f}'.format(
                epoch,
                epoch_loss,
                learning_rates,
                best_loss,
            )
        )
        if target_coverage_enabled:
            print(
                'epoch {} coverage: raw_loss={:.6f}, weighted_loss={:.6f}, '
                'groups={}, uncovered={}, miss_rate={:.6f}'.format(
                    epoch,
                    target_coverage_loss_sum / max(batch_count, 1),
                    target_coverage_weighted_loss_sum / max(batch_count, 1),
                    target_coverage_group_count,
                    target_coverage_uncovered_count,
                    target_coverage_uncovered_count
                    / max(target_coverage_group_count, 1),
                )
            )

    if audited_mutable_state_keys is not None:
        assert_frozen_model_state_unchanged(
            model,
            frozen_state_reference,
            mutable_state_keys=audited_mutable_state_keys,
        )
    summary = {
        'started_at': started_at.isoformat(timespec='seconds'),
        'seed': int(cfg.seed),
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'best_loss_checkpoint': (
            None
            if best_loss_checkpoint is None
            else str(best_loss_checkpoint)
        ),
        'last_checkpoint': (
            None
            if last_checkpoint_path is None
            else str(last_checkpoint_path)
        ),
        'start_epoch': start_epoch,
        'next_epoch': int(cfg.epochs),
        'initialized_from': (
            None if initialized_from is None else str(initialized_from)
        ),
        'initialized_from_sha256': initialized_from_sha256,
        'initialization_migrations': initialization_migrations,
        'resume_parent_checkpoint': (
            None
            if resume_parent_checkpoint is None
            else str(resume_parent_checkpoint)
        ),
        'git': git_provenance,
        'training_scope': training_scope,
        'frozen_state_reference_sha256': frozen_state_reference_sha256,
        'sampling': dataset.sampling_summary(),
        'resolved_config': cfg.resolved_config,
        'config_overrides': list(cfg.config_overrides),
    }
    with (run_dir / 'run_summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    print('best loss checkpoint:', summary['best_loss_checkpoint'])
    print('last checkpoint:', summary['last_checkpoint'])
