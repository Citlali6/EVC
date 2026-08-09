"""Create an auditable, untrained DACC-v2 identity upgrade from a v1 checkpoint.

This utility only adds the zero-initialized DACC-v2 projection.  It does not
run an optimizer step and records that fact explicitly in the output.
"""

import argparse
import copy
import hashlib
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import torch

from model.temporal_frame_net import (
    DENSITY_CALIBRATION_LEGACY_VERSION,
    DENSITY_CALIBRATION_V2_BASIS,
    DENSITY_CALIBRATION_V2_RESIDUAL_SCALE,
    DENSITY_CALIBRATION_V2_VERSION,
    validate_density_calibration_metadata,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet
from utils.temporal_memory_inference import load_temporal_memory_model


PROJECTION_KEY = 'base.density_calibrator.residual_projection.weight'
MIGRATION_NAME = 'density_calibration_v1_to_v2_zero_residual'


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state, excluded_keys=()):
    excluded_keys = frozenset(excluded_keys)
    digest = hashlib.sha256()
    for name in sorted(state):
        if name in excluded_keys:
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def git_provenance():
    repository = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(repository),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ['git', 'status', '--porcelain'],
                cwd=str(repository),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {'commit': commit, 'dirty': dirty}


def build_identity_checkpoint(source_path):
    source_path = Path(source_path).resolve()
    checkpoint = load_checkpoint(source_path)
    metadata = checkpoint.get('temporal_memory')
    state = checkpoint.get('model_state_dict')
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError('Source must be a complete temporal-memory checkpoint.')
    source_version = validate_density_calibration_metadata(metadata)
    if source_version != DENSITY_CALIBRATION_LEGACY_VERSION:
        raise ValueError(
            'Identity upgrade requires a legacy DACC-v1 checkpoint, got v{}.'.format(
                source_version
            )
        )

    context_bins = int(metadata['context_bins'])
    width = int(metadata['width'])
    sequence_length = int(metadata['sequence_length'])
    model = BidirectionalTemporalMemoryNet(
        input_channels=context_bins * 2,
        width=width,
        density_calibration_enabled=True,
        density_calibration_v2_enabled=True,
        confidence_head_enabled=bool(metadata.get('confidence_head_enabled', False)),
        temporal_attention_enabled=bool(
            metadata.get('temporal_attention_enabled', False)
        ),
    )
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != {PROJECTION_KEY} or result.unexpected_keys:
        raise RuntimeError(
            'Expected only {} to be newly added; missing={}, unexpected={}.'.format(
                PROJECTION_KEY,
                result.missing_keys,
                result.unexpected_keys,
            )
        )
    migrated_state = model.state_dict()
    for name, expected in state.items():
        actual = migrated_state[name]
        if (
            actual.dtype != expected.dtype
            or tuple(actual.shape) != tuple(expected.shape)
            or not torch.equal(actual.detach().cpu(), expected.detach().cpu())
        ):
            raise RuntimeError('Identity upgrade changed inherited tensor {}.'.format(name))
    projection = migrated_state[PROJECTION_KEY]
    if tuple(projection.shape) != (width, 2):
        raise RuntimeError('Unexpected DACC-v2 projection shape {}.'.format(tuple(projection.shape)))
    if torch.count_nonzero(projection).item() != 0:
        raise RuntimeError('DACC-v2 identity projection is not exactly zero.')

    source_state_hash = state_sha256(state)
    inherited_state_hash = state_sha256(
        migrated_state,
        excluded_keys={PROJECTION_KEY},
    )
    if source_state_hash != inherited_state_hash:
        raise RuntimeError('Inherited model-state hash changed during identity upgrade.')

    output = copy.deepcopy(checkpoint)
    output['model_state_dict'] = migrated_state
    output_metadata = dict(metadata)
    output_metadata.update(
        {
            'density_calibration_version': DENSITY_CALIBRATION_V2_VERSION,
            'density_calibration_v2_basis': DENSITY_CALIBRATION_V2_BASIS,
            'density_calibration_v2_residual_scale': (
                DENSITY_CALIBRATION_V2_RESIDUAL_SCALE
            ),
        }
    )
    output['temporal_memory'] = output_metadata
    output['identity_upgrade'] = {
        'schema_version': 1,
        'artifact_kind': 'untrained_dacc_v2_identity_upgrade',
        'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'trained_after_upgrade': False,
        'trained_epochs_after_upgrade': 0,
        'optimizer_steps_after_upgrade': 0,
        'migration': MIGRATION_NAME,
        'source_checkpoint': str(source_path),
        'source_checkpoint_sha256': sha256_file(source_path),
        'source_epoch': checkpoint.get('epoch'),
        'source_model_state_sha256': source_state_hash,
        'migrated_inherited_state_sha256': inherited_state_hash,
        'added_state_keys': [PROJECTION_KEY],
        'added_parameter_count': int(projection.numel()),
        'added_nonzero_count': int(torch.count_nonzero(projection).item()),
        'git': git_provenance(),
    }
    return output, context_bins, width, sequence_length


def atomic_torch_save(value, output_path, validator=None, force=False):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(
            'Output already exists; pass --force to replace it: {}'.format(output_path)
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + '.',
        suffix='.tmp',
        dir=str(output_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        if validator is not None:
            validator(temporary_path)
        os.replace(str(temporary_path), str(output_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-source-sha256')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('Source and output checkpoint paths must differ.')
    source_sha256 = sha256_file(args.source)
    if (
        args.expected_source_sha256
        and source_sha256.lower() != args.expected_source_sha256.lower()
    ):
        raise ValueError(
            'Source SHA-256 {} does not match expected {}.'.format(
                source_sha256,
                args.expected_source_sha256,
            )
        )
    checkpoint, context_bins, width, sequence_length = build_identity_checkpoint(
        args.source
    )

    def validate_temporary_checkpoint(path):
        loaded_model, loaded_checkpoint = load_temporal_memory_model(
            path,
            torch.device('cpu'),
            context_bins,
            width,
            sequence_length,
        )
        loaded_state = loaded_model.state_dict()
        if set(loaded_state) != set(checkpoint['model_state_dict']):
            raise RuntimeError('Strict production reload changed model-state keys.')
        for name, expected in checkpoint['model_state_dict'].items():
            if not torch.equal(
                loaded_state[name].detach().cpu(),
                expected.detach().cpu(),
            ):
                raise RuntimeError(
                    'Strict production reload changed tensor {}.'.format(name)
                )
        if validate_density_calibration_metadata(
            loaded_checkpoint['temporal_memory']
        ) != 2:
            raise RuntimeError(
                'Strict production reload did not preserve DACC-v2 metadata.'
            )

    output_path = atomic_torch_save(
        checkpoint,
        args.output,
        validator=validate_temporary_checkpoint,
        force=args.force,
    )
    print('identity checkpoint:', output_path)
    print('sha256:', sha256_file(output_path))
    print('trained_after_upgrade: false')
    print('optimizer_steps_after_upgrade: 0')
    added_parameters = checkpoint['identity_upgrade']['added_parameter_count']
    print('added parameters: {} (all exactly zero)'.format(added_parameters))


if __name__ == '__main__':
    main()
