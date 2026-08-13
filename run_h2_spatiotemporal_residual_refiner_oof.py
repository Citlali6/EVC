"""Train-only grouped-OOF runner for the frozen-M20 H2 residual refiner.

No command accepts a validation/test path, threshold, learning rate, loss
weight, architecture choice or source list.  The only GPU commands consume the
single frozen protocol and require an explicit root authorization flag.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset.temporal_frame import (
    build_temporal_context_frame,
    load_temporal_frame_video,
    temporal_frame_video_from_events,
)
from dataset.temporal_memory import temporal_memory_collate, temporal_sequence_start
from model.h2_spatiotemporal_residual_refiner import (
    FrozenM20ResidualRefiner,
    refiner_parameter_count,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet


EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
PROTOCOL_PATH = (
    EVC_ROOT / 'protocols' / 'h2_spatiotemporal_residual_refiner_oof_science_v1.json'
)
TRAIN_ROOT = WORKSPACE_ROOT / 'datasets' / 'EV-UAV-Challenge2' / 'train'
OUTPUT_ROOT = (
    WORKSPACE_ROOT / 'experiments' / '20260811_h2_spatiotemporal_residual_refiner_oof_v1'
)
M20_PATH = EVC_ROOT / 'checkpoints' / 'm20_attn_dense_views8_epoch_003_seed48.pt'
GPU_AUTHORIZATION_FLAG = '--root-authorized-gpu'
EXPECTED_SCHEMA = 'ev-uav-frozen-m20-h2-spatiotemporal-residual-refiner-oof-v1'

WHOLE_T = 8000
TEMPORAL_BIN_SIZE = 50
CONTEXT_BINS = 5
SEQUENCE_LENGTH = 16
WIDTH = 346
HEIGHT = 260
LOG_COUNT_CLIP = 4.0
INFERENCE_BATCH_SIZE = 16
H2_EVENT_COUNT_CUTOFF = 200000

C00_OVERRIDES = [
    'TEST.prediction_threshold=0.719',
    'TEMPORAL_FRAME.temporal_frame_enabled=false',
    'TEMPORAL_MEMORY.temporal_memory_enabled=true',
    'TEMPORAL_MEMORY.temporal_memory_sequence_length=16',
    'TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0',
    'TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true',
    'POSTPROCESS.p0_enabled=true',
    'POSTPROCESS.p0_spatial_radius=2',
    'POSTPROCESS.p0_temporal_bin_size=50',
    'POSTPROCESS.p0_temporal_radius_bins=1',
    'POSTPROCESS.p0_min_cluster_events=3',
    'POSTPROCESS.p0_min_duration_bins=5',
    'POSTPROCESS.p0c_high_confidence_recovery_enabled=true',
    'POSTPROCESS.p0c_retain_min_score=0.95',
    'POSTPROCESS.p0c_density_retain_enabled=false',
    'POSTPROCESS.p0c_density_event_count_cutoff=100000',
    'POSTPROCESS.p0c_density_retain_min_score=0.97',
    'POSTPROCESS.p0b_enabled=false',
    'POSTPROCESS.p18_score_track_recovery_enabled=true',
    'POSTPROCESS.p18_event_count_cutoff=1',
    'POSTPROCESS.p18_max_event_count=35000',
    'POSTPROCESS.p18_candidate_floor=0.53',
    'POSTPROCESS.p18_spatial_radius=5',
    'POSTPROCESS.p18_temporal_bin_size=50',
    'POSTPROCESS.p18_max_link_distance=8.0',
    'POSTPROCESS.p18_max_gap_bins=1',
    'POSTPROCESS.p18_min_track_bins=4',
    'POSTPROCESS.p18_restore_mode=best',
    'POSTPROCESS.p18_max_restore_events_per_component=0',
    'POSTPROCESS.p6_density_threshold_enabled=true',
    'POSTPROCESS.p6_event_count_cutoff=30000',
    'POSTPROCESS.p6_low_density_threshold=0.718',
    'POSTPROCESS.p6_high_density_threshold=0.719',
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(chunk_size), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256_float32(values):
    values = np.asarray(values, dtype='<f4').reshape(-1)
    return hashlib.sha256(values.tobytes(order='C')).hexdigest()


def state_sha256(state_dict):
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(np.asarray(value.shape, dtype='<i8').tobytes())
        digest.update(value.numpy().tobytes(order='C'))
    return digest.hexdigest()


def load_protocol():
    with PROTOCOL_PATH.open('r', encoding='utf-8') as stream:
        protocol = json.load(stream)
    if protocol.get('schema') != EXPECTED_SCHEMA:
        raise RuntimeError('Unexpected residual-refiner protocol schema.')
    if protocol.get('status') != 'frozen_before_any_gpu_probe_training_or_held_fold_prediction':
        raise RuntimeError('Protocol is not in its frozen pre-execution state.')
    return protocol


def fold_spec(protocol, fold_id):
    matches = [item for item in protocol['folds'] if item['fold_id'] == fold_id]
    if len(matches) != 1:
        raise ValueError('Unknown or duplicate fold: {}'.format(fold_id))
    return matches[0]


def load_checkpoint_file(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_released_m20(device):
    payload = load_checkpoint_file(M20_PATH, map_location='cpu')
    metadata = payload.get('temporal_memory', {})
    required = {
        'temporal_bin_size': TEMPORAL_BIN_SIZE,
        'context_bins': CONTEXT_BINS,
        'width': 16,
        'sequence_length': SEQUENCE_LENGTH,
    }
    for key, expected in required.items():
        if int(metadata.get(key, -1)) != expected:
            raise RuntimeError('Released M20 metadata differs for {}.'.format(key))
    model = BidirectionalTemporalMemoryNet(
        input_channels=CONTEXT_BINS * 2,
        width=16,
        density_calibration_enabled=bool(
            metadata.get('density_calibration_enabled', False)
        ),
        density_calibration_v2_enabled=bool(
            metadata.get('density_calibration_v2_enabled', False)
        ),
        confidence_head_enabled=bool(metadata.get('confidence_head_enabled', False)),
        temporal_attention_enabled=bool(
            metadata.get('temporal_attention_enabled', False)
        ),
    )
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.to(device).eval()
    return model, payload


class ExactSourceSequenceDataset(Dataset):
    """Deterministic temporal views from an exact fit-source allowlist."""

    def __init__(self, source_paths, views_per_source, positive_probability, seed):
        self.source_paths = tuple(Path(path).resolve() for path in source_paths)
        self.views_per_source = int(views_per_source)
        self.positive_probability = float(positive_probability)
        self.seed = int(seed)
        self.epoch = 0
        if not self.source_paths or self.views_per_source <= 0:
            raise ValueError('Exact-source dataset requires paths and positive views.')
        if not 0.0 <= self.positive_probability <= 1.0:
            raise ValueError('positive_probability must be in [0,1].')
        self._videos = {
            index: load_temporal_frame_video(
                path, TEMPORAL_BIN_SIZE, WHOLE_T,
            )
            for index, path in enumerate(self.source_paths)
        }

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.source_paths) * self.views_per_source

    def __getitem__(self, index):
        index = int(index)
        video_index = index // self.views_per_source
        view_index = index % self.views_per_source
        video = self._videos[video_index]
        rng = np.random.default_rng(
            self.seed + 1000003 * self.epoch + 1009 * video_index + view_index
        )
        use_positive = (
            video.positive_bins.size > 0
            and rng.random() < self.positive_probability
        )
        candidates = video.positive_bins if use_positive else video.occupied_bins
        center_bin = int(candidates[rng.integers(candidates.size)])
        start_bin = temporal_sequence_start(
            center_bin, len(video.event_indices_by_bin), SEQUENCE_LENGTH,
        )
        frames = []
        time_indices = []
        timestamps = []
        event_x = []
        event_y = []
        labels = []
        target_ids = []
        for local_time, temporal_bin in enumerate(
            range(start_bin, start_bin + SEQUENCE_LENGTH)
        ):
            frames.append(build_temporal_context_frame(
                video,
                temporal_bin,
                CONTEXT_BINS,
                WIDTH,
                HEIGHT,
                LOG_COUNT_CLIP,
            ))
            indices = video.event_indices_by_bin[temporal_bin]
            if indices.size == 0:
                continue
            locations = video.locations[indices]
            time_indices.append(np.full(indices.size, local_time, dtype=np.int64))
            timestamps.append(locations[:, 2].astype(np.int64, copy=False))
            event_x.append(locations[:, 0].astype(np.int64, copy=False))
            event_y.append(locations[:, 1].astype(np.int64, copy=False))
            labels.append(video.labels[indices].astype(np.float32, copy=False))
            target_ids.append(video.target_ids[indices].astype(np.int64, copy=False))
        return {
            'frames': np.stack(frames, axis=0),
            'event_time_indices': np.concatenate(time_indices),
            'event_timestamps': np.concatenate(timestamps),
            'event_x': np.concatenate(event_x),
            'event_y': np.concatenate(event_y),
            'labels': np.concatenate(labels),
            'target_ids': np.concatenate(target_ids),
        }


def target_group_retention_loss(refined, base, labels, target_ids, times):
    positive_target = (labels > 0.5) & (target_ids > 0)
    if not bool(torch.any(positive_target)):
        return refined.sum() * 0.0, 0
    selected_ids = target_ids[positive_target]
    selected_times = times[positive_target]
    multiplier = torch.max(selected_ids) + 1
    keys = selected_times * multiplier + selected_ids
    refined_positive = refined[positive_target]
    base_positive = base[positive_target]
    losses = []
    for key in torch.unique(keys):
        mask = keys == key
        losses.append(F.relu(torch.max(base_positive[mask]) - torch.max(refined_positive[mask])).square())
    return torch.stack(losses).mean(), len(losses)


def residual_fit_loss(refined_maps, base_maps, batch):
    times = batch['event_time_indices'].to(refined_maps.device, non_blocking=True)
    event_x = batch['event_x'].to(refined_maps.device, non_blocking=True)
    event_y = batch['event_y'].to(refined_maps.device, non_blocking=True)
    labels = batch['labels'].to(refined_maps.device, non_blocking=True)
    target_ids = batch['target_ids'].to(refined_maps.device, non_blocking=True)
    interior = (times > 0) & (times < refined_maps.shape[1] - 1)
    if not bool(torch.any(interior)):
        raise RuntimeError('Sample contains no interior-bin events.')
    times = times[interior]
    event_x = event_x[interior]
    event_y = event_y[interior]
    labels = labels[interior]
    target_ids = target_ids[interior]
    refined = refined_maps[0, times, 0, event_y, event_x]
    base = base_maps[0, times, 0, event_y, event_x].detach()
    positive = labels > 0.5
    negative = ~positive

    zero = refined.sum() * 0.0
    positive_term = F.softplus(-refined[positive]).mean() if bool(torch.any(positive)) else zero
    if bool(torch.any(negative)):
        negative_weight = torch.sigmoid(base[negative]).detach()
        negative_term = (
            negative_weight * F.softplus(refined[negative])
        ).sum() / negative_weight.sum().clamp_min(torch.finfo(refined.dtype).eps)
    else:
        negative_term = zero
    classification = 0.5 * (positive_term + negative_term)
    event_retention = (
        F.relu(base[positive] - refined[positive]).square().mean()
        if bool(torch.any(positive)) else zero
    )
    group_retention, group_count = target_group_retention_loss(
        refined, base, labels, target_ids, times,
    )
    loss = classification + event_retention + group_retention
    diagnostics = {
        'loss': float(loss.detach()),
        'classification': float(classification.detach()),
        'positive_event_retention': float(event_retention.detach()),
        'target_group_retention': float(group_retention.detach()),
        'event_count': int(refined.numel()),
        'positive_count': int(torch.count_nonzero(positive)),
        'target_group_count': int(group_count),
        'mean_abs_residual': float(torch.mean(torch.abs(refined - base)).detach()),
    }
    return loss, diagnostics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_gpu_authorization(args):
    if not bool(getattr(args, 'root_authorized_gpu', False)):
        raise RuntimeError('GPU execution requires {}.'.format(GPU_AUTHORIZATION_FLAG))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')


def source_paths(protocol, names):
    expected = set(protocol['h2_sources'])
    if not set(names).issubset(expected):
        raise RuntimeError('A requested source is outside the frozen H2 set.')
    paths = [TRAIN_ROOT / name for name in names]
    if any(path.parent.resolve() != TRAIN_ROOT.resolve() for path in paths):
        raise RuntimeError('Source path escaped the official train root.')
    return paths


def audit_protocol(run_tests=True):
    protocol = load_protocol()
    if protocol['evaluation'].get('fixed_config_overrides') != C00_OVERRIDES:
        raise RuntimeError('Runner C00 overrides differ from the frozen protocol.')
    if sha256_file(M20_PATH) != protocol['released_m20']['sha256']:
        raise RuntimeError('Released M20 SHA-256 mismatch.')
    h2_names = set(protocol['h2_sources'])
    held_union = set()
    for fold in protocol['folds']:
        held = set(fold['held'])
        fit = set(fold['fit'])
        if held & fit or held | fit != h2_names:
            raise RuntimeError('Fold partition is not an exact H2 split.')
        held_union.update(held)
    if held_union != h2_names:
        raise RuntimeError('Every H2 source must be held exactly once.')
    for name, metadata in protocol['h2_sources'].items():
        path = TRAIN_ROOT / name
        if not path.is_file() or sha256_file(path) != metadata['sha256']:
            raise RuntimeError('Frozen train-source identity mismatch: {}'.format(name))
        if int(metadata['event_count']) <= H2_EVENT_COUNT_CUTOFF:
            raise RuntimeError('Frozen H2 route mismatch: {}'.format(name))

    base, payload = build_released_m20(torch.device('cpu'))
    if len(payload['model_state_dict']) != int(protocol['released_m20']['state_tensor_count']):
        raise RuntimeError('Released M20 state tensor count mismatch.')
    if sum(p.numel() for p in base.parameters()) != int(protocol['released_m20']['parameter_count']):
        raise RuntimeError('Released M20 parameter count mismatch.')
    wrapper = FrozenM20ResidualRefiner(
        base,
        context_bins=CONTEXT_BINS,
        hidden_channels=int(protocol['architecture']['hidden_channels']),
    )
    if torch.count_nonzero(wrapper.refiner.output_projection.weight).item() != 0:
        raise RuntimeError('Residual output projection is not zero initialized.')
    if torch.count_nonzero(wrapper.refiner.output_projection.bias).item() != 0:
        raise RuntimeError('Residual output bias is not zero initialized.')
    if any(parameter.requires_grad for parameter in wrapper.released_m20.parameters()):
        raise RuntimeError('A released M20 parameter remains trainable.')

    test_result = None
    if run_tests:
        command = [
            sys.executable,
            '-m',
            'unittest',
            'discover',
            '-s',
            str(EVC_ROOT / 'tests'),
            '-p',
            'test_h2_spatiotemporal_residual_refiner.py',
            '-v',
        ]
        completed = subprocess.run(command, cwd=EVC_ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError('CPU residual-refiner tests failed:\n{}'.format(
                completed.stdout + completed.stderr
            ))
        test_result = (completed.stdout + completed.stderr).strip()
    result = {
        'schema': 'ev-uav-h2-residual-refiner-cpu-audit-v1',
        'created_utc': utc_now(),
        'protocol_sha256': sha256_file(PROTOCOL_PATH),
        'runner_sha256': sha256_file(Path(__file__)),
        'module_sha256': sha256_file(
            EVC_ROOT / 'model' / 'h2_spatiotemporal_residual_refiner.py'
        ),
        'tests_sha256': sha256_file(
            EVC_ROOT / 'tests' / 'test_h2_spatiotemporal_residual_refiner.py'
        ),
        'released_m20_sha256': sha256_file(M20_PATH),
        'released_m20_state_sha256': state_sha256(base.state_dict()),
        'released_m20_state_tensor_count': len(base.state_dict()),
        'released_m20_parameter_count': sum(p.numel() for p in base.parameters()),
        'refiner_parameter_count': refiner_parameter_count(wrapper),
        'refiner_zero_identity': True,
        'fold_count': len(protocol['folds']),
        'h2_source_count': len(h2_names),
        'validation_or_test_read': False,
        'gpu_used': False,
        'cpu_tests': test_result,
    }
    return result


def write_json_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def save_torch_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def make_training_objects(protocol, fold_id, device):
    fold = fold_spec(protocol, fold_id)
    training = protocol['training']
    seed_everything(int(training['seed']))
    base, _ = build_released_m20(device)
    wrapper = FrozenM20ResidualRefiner(
        base,
        context_bins=CONTEXT_BINS,
        hidden_channels=int(protocol['architecture']['hidden_channels']),
    ).to(device).train()
    dataset = ExactSourceSequenceDataset(
        source_paths(protocol, fold['fit']),
        views_per_source=int(training['views_per_fit_source_per_epoch']),
        positive_probability=float(training['positive_frame_probability']),
        seed=int(training['seed']),
    )
    generator = torch.Generator().manual_seed(int(training['seed']))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=temporal_memory_collate,
        generator=generator,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        wrapper.trainable_parameters(),
        lr=float(training['learning_rate']),
        weight_decay=float(training['weight_decay']),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(training['mixed_precision'])
    )
    return fold, wrapper, dataset, loader, optimizer, scaler


def train_steps(protocol, fold_id, max_steps=None):
    device = torch.device('cuda:0')
    fold, wrapper, dataset, loader, optimizer, scaler = make_training_objects(
        protocol, fold_id, device,
    )
    training = protocol['training']
    frozen_before = state_sha256(wrapper.frozen_state_dict())
    started = time.perf_counter()
    records = []
    step = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(int(training['epochs'])):
        dataset.set_epoch(epoch)
        wrapper.train()
        for batch in loader:
            frames = batch['frames'].to(device, non_blocking=True).unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16,
                enabled=bool(training['mixed_precision']),
            ):
                refined, base = wrapper(frames, return_base_logits=True)
                loss, diagnostics = residual_fit_loss(refined, base, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                wrapper.trainable_parameters(),
                max_norm=float(training['gradient_clip_norm']),
            )
            scaler.step(optimizer)
            scaler.update()
            step += 1
            diagnostics.update({
                'step': step,
                'epoch': epoch,
                'gradient_norm': float(grad_norm.detach()),
            })
            records.append(diagnostics)
            if not np.isfinite(diagnostics['loss']):
                raise RuntimeError('Non-finite residual-refiner loss.')
            if max_steps is not None and step >= int(max_steps):
                break
        if max_steps is not None and step >= int(max_steps):
            break
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    frozen_after = state_sha256(wrapper.frozen_state_dict())
    if frozen_after != frozen_before:
        raise RuntimeError('Released M20 state changed during residual training.')
    return {
        'fold': fold,
        'wrapper': wrapper,
        'records': records,
        'step_count': step,
        'elapsed_seconds': elapsed,
        'peak_cuda_bytes': int(torch.cuda.max_memory_allocated(device)),
        'frozen_m20_state_sha256_before': frozen_before,
        'frozen_m20_state_sha256_after': frozen_after,
    }


def run_probe(args):
    require_gpu_authorization(args)
    protocol = load_protocol()
    expected_steps = int(protocol['probe']['optimizer_steps'])
    result_path = OUTPUT_ROOT / 'resource_probe' / 'eight_step_probe.json'
    if result_path.exists():
        raise FileExistsError('Refusing to overwrite probe receipt: {}'.format(result_path))
    result = train_steps(protocol, protocol['probe']['fold'], max_steps=expected_steps)
    if result['step_count'] != expected_steps:
        raise RuntimeError('Probe optimizer-step count mismatch.')
    payload = {
        'schema': 'ev-uav-h2-residual-refiner-eight-step-probe-v1',
        'created_utc': utc_now(),
        'protocol_sha256': sha256_file(PROTOCOL_PATH),
        'fold_id': result['fold']['fold_id'],
        'fit_sources': result['fold']['fit'],
        'held_source_arrays_read': False,
        'optimizer_steps': result['step_count'],
        'elapsed_seconds': result['elapsed_seconds'],
        'seconds_per_step': result['elapsed_seconds'] / result['step_count'],
        'peak_cuda_bytes': result['peak_cuda_bytes'],
        'peak_cuda_mib': result['peak_cuda_bytes'] / (1024.0 ** 2),
        'loss_first': result['records'][0],
        'loss_last': result['records'][-1],
        'frozen_m20_state_sha256_before': result['frozen_m20_state_sha256_before'],
        'frozen_m20_state_sha256_after': result['frozen_m20_state_sha256_after'],
        'validation_or_test_read': False,
    }
    write_json_exclusive(result_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_train_fold(args):
    require_gpu_authorization(args)
    protocol = load_protocol()
    fold = fold_spec(protocol, args.fold)
    fold_root = OUTPUT_ROOT / 'formal_training' / fold['fold_id']
    checkpoint_path = fold_root / 'final_refiner.pt'
    result_path = fold_root / 'training_result.json'
    if checkpoint_path.exists() or result_path.exists() or fold_root.exists():
        raise FileExistsError('Refusing to overwrite formal fold output: {}'.format(fold_root))
    result = train_steps(protocol, fold['fold_id'], max_steps=None)
    expected_steps = (
        int(protocol['training']['epochs'])
        * int(protocol['training']['views_per_fit_source_per_epoch'])
        * len(fold['fit'])
    )
    if result['step_count'] != expected_steps:
        raise RuntimeError('Formal optimizer-step count mismatch.')
    checkpoint = {
        'schema': 'ev-uav-h2-residual-refiner-checkpoint-v1',
        'created_utc': utc_now(),
        'fold_id': fold['fold_id'],
        'fit_sources': fold['fit'],
        'released_m20_path': str(M20_PATH.resolve()),
        'released_m20_sha256': sha256_file(M20_PATH),
        'protocol_path': str(PROTOCOL_PATH.resolve()),
        'protocol_sha256': sha256_file(PROTOCOL_PATH),
        'architecture': protocol['architecture'],
        'training': protocol['training'],
        'optimizer_steps': result['step_count'],
        'refiner_state_dict': {
            name: value.detach().cpu()
            for name, value in result['wrapper'].refiner.state_dict().items()
        },
        'frozen_m20_state_sha256': result['frozen_m20_state_sha256_after'],
    }
    save_torch_exclusive(checkpoint_path, checkpoint)
    training_payload = {
        'schema': 'ev-uav-h2-residual-refiner-training-result-v1',
        'created_utc': utc_now(),
        'fold_id': fold['fold_id'],
        'fit_sources': fold['fit'],
        'held_source_arrays_read': False,
        'checkpoint': str(checkpoint_path.resolve()),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'optimizer_steps': result['step_count'],
        'elapsed_seconds': result['elapsed_seconds'],
        'peak_cuda_mib': result['peak_cuda_bytes'] / (1024.0 ** 2),
        'first_step': result['records'][0],
        'last_step': result['records'][-1],
        'mean_loss': float(np.mean([item['loss'] for item in result['records']])),
        'frozen_m20_state_sha256_before': result['frozen_m20_state_sha256_before'],
        'frozen_m20_state_sha256_after': result['frozen_m20_state_sha256_after'],
        'validation_or_test_read': False,
    }
    write_json_exclusive(result_path, training_payload)
    print(json.dumps(training_payload, indent=2, ensure_ascii=False))


def _frame_tensor(video, bins, device):
    frames = np.stack([
        build_temporal_context_frame(
            video, temporal_bin, CONTEXT_BINS, WIDTH, HEIGHT, LOG_COUNT_CLIP,
        )
        for temporal_bin in bins
    ], axis=0)
    return torch.from_numpy(frames).float().to(device)


def predict_paired_full_stream(wrapper, video, device):
    temporal_count = len(video.event_indices_by_bin)
    bottlenecks = []
    with torch.no_grad():
        for start in range(0, temporal_count, INFERENCE_BATCH_SIZE):
            bins = list(range(start, min(start + INFERENCE_BATCH_SIZE, temporal_count)))
            frames = _frame_tensor(video, bins, device)
            bottlenecks.append(wrapper.encode_bottleneck(frames))
        memory = wrapper.temporal_residual(torch.cat(bottlenecks, dim=0))
    base_scores = np.empty(video.locations.shape[0], dtype=np.float32)
    candidate_scores = np.empty_like(base_scores)
    with torch.no_grad():
        for core_start in range(0, temporal_count, INFERENCE_BATCH_SIZE):
            core_stop = min(core_start + INFERENCE_BATCH_SIZE, temporal_count)
            ext_start = max(0, core_start - 1)
            ext_stop = min(temporal_count, core_stop + 1)
            bins = list(range(ext_start, ext_stop))
            frames = _frame_tensor(video, bins, device)
            features, logits = wrapper._decode_frozen_features(
                frames, memory[ext_start:ext_stop],
            )
            refined = wrapper.refine_decoded_sequence(frames, features, logits)
            base_prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            candidate_prob = torch.sigmoid(refined).squeeze(1).cpu().numpy()
            for temporal_bin in range(core_start, core_stop):
                event_indices = video.event_indices_by_bin[temporal_bin]
                if event_indices.size == 0:
                    continue
                local = temporal_bin - ext_start
                locations = video.locations[event_indices]
                ys = locations[:, 1]
                xs = locations[:, 0]
                base_scores[event_indices] = base_prob[local, ys, xs]
                candidate_scores[event_indices] = candidate_prob[local, ys, xs]
    if not np.isfinite(base_scores).all() or not np.isfinite(candidate_scores).all():
        raise RuntimeError('Non-finite full-stream score.')
    return base_scores, candidate_scores


def load_unlabelled_video_and_truth(path):
    with np.load(path, allow_pickle=False) as archive:
        evs_norm = np.asarray(archive['evs_norm'])
        locations = np.asarray(archive['ev_loc']).astype(np.int64, copy=False)
    labels = evs_norm[:, 4].astype(np.uint8, copy=False)
    target_ids = evs_norm[:, 5].astype(np.int64, copy=False)
    # The inference object deliberately receives neither labels nor target ids.
    video = temporal_frame_video_from_events(
        name='',
        locations=locations,
        polarities=evs_norm[:, 3],
        temporal_bin_size=TEMPORAL_BIN_SIZE,
        whole_t=WHOLE_T,
        labels=None,
        target_ids=None,
    )
    return video, labels, target_ids, locations


def evaluate_fold(args):
    require_gpu_authorization(args)
    protocol = load_protocol()
    fold = fold_spec(protocol, args.fold)
    checkpoint_path = OUTPUT_ROOT / 'formal_training' / fold['fold_id'] / 'final_refiner.pt'
    result_path = OUTPUT_ROOT / 'held_train_evaluation' / fold['fold_id'] / 'paired_evaluation.json'
    if result_path.exists() or result_path.parent.exists():
        raise FileExistsError('Refusing to overwrite held evaluation.')
    checkpoint = load_checkpoint_file(checkpoint_path, map_location='cpu')
    if checkpoint.get('fold_id') != fold['fold_id'] or checkpoint.get('fit_sources') != fold['fit']:
        raise RuntimeError('Checkpoint fold provenance mismatch.')
    if checkpoint.get('protocol_sha256') != sha256_file(PROTOCOL_PATH):
        raise RuntimeError('Checkpoint protocol provenance mismatch.')

    import crossfit_component_reranker as component_crossfit
    import replay_temporal_memory_validation as replay
    from crossfit_component_reranker import (
        SufficientCounts,
        metrics_from_counts,
        sufficient_counts_for_video,
    )
    from utils.postprocess import ChallengePostprocessor

    cfg = replay.load_flat_config(
        EVC_ROOT / 'configs' / 'evisseg_evuav.yaml', C00_OVERRIDES,
    )
    c00 = component_crossfit.validate_c00_config(
        cfg, float(protocol['evaluation']['prediction_threshold'])
    )
    if component_crossfit.sha256_json(c00) != protocol['evaluation']['effective_c00_canonical_sha256']:
        raise RuntimeError('C00 contract SHA-256 mismatch.')

    device = torch.device('cuda:0')
    base, _ = build_released_m20(device)
    wrapper = FrozenM20ResidualRefiner(
        base,
        context_bins=CONTEXT_BINS,
        hidden_channels=int(protocol['architecture']['hidden_channels']),
    ).to(device).eval()
    wrapper.refiner.load_state_dict(checkpoint['refiner_state_dict'], strict=True)
    threshold = float(protocol['evaluation']['prediction_threshold'])
    pooled_base = SufficientCounts()
    pooled_candidate = SufficientCounts()
    records = []
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for name in fold['held']:
        path = TRAIN_ROOT / name
        if sha256_file(path) != protocol['h2_sources'][name]['sha256']:
            raise RuntimeError('Held source identity mismatch: {}'.format(name))
        video, labels, target_ids, locations3 = load_unlabelled_video_and_truth(path)
        if len(labels) <= H2_EVENT_COUNT_CUTOFF:
            raise RuntimeError('Held source does not satisfy the input-only H2 route.')
        base_raw, candidate_raw = predict_paired_full_stream(wrapper, video, device)
        locations4 = np.column_stack((
            np.zeros(len(locations3), dtype=np.int64), locations3,
        ))
        location_tensor = torch.from_numpy(locations4).long().contiguous()
        base_processed, base_stats = ChallengePostprocessor.from_cfg(
            cfg, threshold, event_count=len(labels),
        ).apply(torch.from_numpy(base_raw.copy()), location_tensor)
        candidate_processed, candidate_stats = ChallengePostprocessor.from_cfg(
            cfg, threshold, event_count=len(labels),
        ).apply(torch.from_numpy(candidate_raw.copy()), location_tensor)
        base_counts = sufficient_counts_for_video(
            base_processed.numpy(), labels, target_ids, locations4, threshold,
        )
        candidate_counts = sufficient_counts_for_video(
            candidate_processed.numpy(), labels, target_ids, locations4, threshold,
        )
        pooled_base = pooled_base + base_counts
        pooled_candidate = pooled_candidate + candidate_counts
        records.append({
            'source_name': name,
            'event_count': len(labels),
            'base_raw_scores_sha256': sha256_float32(base_raw),
            'candidate_raw_scores_sha256': sha256_float32(candidate_raw),
            'base_postprocess': asdict(base_stats),
            'candidate_postprocess': asdict(candidate_stats),
            'base_counts': base_counts.to_dict(),
            'candidate_counts': candidate_counts.to_dict(),
            'base_metrics': metrics_from_counts(base_counts),
            'candidate_metrics': metrics_from_counts(candidate_counts),
        })
        torch.cuda.empty_cache()
        print('held paired:', name, flush=True)
    torch.cuda.synchronize(device)
    base_metrics = metrics_from_counts(pooled_base)
    candidate_metrics = metrics_from_counts(pooled_candidate)
    count_delta = {
        key: int(candidate_counts_value - getattr(pooled_base, key))
        for key, candidate_counts_value in pooled_candidate.to_dict().items()
    }
    metric_delta = {
        key: float(candidate_metrics[key] - base_metrics[key])
        for key in base_metrics
    }
    gates = {
        'score_not_lower': candidate_metrics['score'] >= base_metrics['score'],
        'iou_not_lower': candidate_metrics['iou'] >= base_metrics['iou'],
        'pd_not_lower': candidate_metrics['pd'] >= base_metrics['pd'],
        'fa_not_higher': candidate_metrics['fa'] <= base_metrics['fa'],
        'correct_objects_not_lower': pooled_candidate.correct_objects >= pooled_base.correct_objects,
        'true_positive_events_not_lower': pooled_candidate.true_positive_events >= pooled_base.true_positive_events,
        'h2_score_gain_at_least_0_02': metric_delta['score'] >= 0.02,
    }
    payload = {
        'schema': 'ev-uav-h2-residual-refiner-held-paired-evaluation-v1',
        'created_utc': utc_now(),
        'fold_id': fold['fold_id'],
        'held_sources': fold['held'],
        'checkpoint': str(checkpoint_path.resolve()),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'prediction_threshold': threshold,
        'effective_c00_contract': c00,
        'effective_c00_canonical_sha256': component_crossfit.sha256_json(c00),
        'records': records,
        'pooled_base_counts': pooled_base.to_dict(),
        'pooled_candidate_counts': pooled_candidate.to_dict(),
        'pooled_count_delta': count_delta,
        'pooled_base_metrics': base_metrics,
        'pooled_candidate_metrics': candidate_metrics,
        'pooled_metric_delta': metric_delta,
        'gates': gates,
        'all_safety_gates_passed': all(
            value for key, value in gates.items() if key != 'h2_score_gain_at_least_0_02'
        ),
        'continuation_effect_size_gate_passed': gates['h2_score_gain_at_least_0_02'],
        'elapsed_seconds': time.perf_counter() - started,
        'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        'validation_or_test_read': False,
    }
    write_json_exclusive(result_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_audit(_args):
    payload = audit_protocol(run_tests=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    audit = subparsers.add_parser('audit')
    audit.set_defaults(func=run_audit)
    probe = subparsers.add_parser('probe')
    probe.add_argument(GPU_AUTHORIZATION_FLAG, dest='root_authorized_gpu', action='store_true')
    probe.set_defaults(func=run_probe)
    train = subparsers.add_parser('train-fold')
    train.add_argument('--fold', required=True, choices=('hold_g1', 'hold_g2', 'hold_g3'))
    train.add_argument(GPU_AUTHORIZATION_FLAG, dest='root_authorized_gpu', action='store_true')
    train.set_defaults(func=run_train_fold)
    evaluate = subparsers.add_parser('evaluate-fold')
    evaluate.add_argument('--fold', required=True, choices=('hold_g1', 'hold_g2', 'hold_g3'))
    evaluate.add_argument(GPU_AUTHORIZATION_FLAG, dest='root_authorized_gpu', action='store_true')
    evaluate.set_defaults(func=evaluate_fold)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
