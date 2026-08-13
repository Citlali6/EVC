"""Frozen single-candidate G2 runner for target-preserving residual V2."""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

import run_h2_spatiotemporal_residual_refiner_oof as core
from dataset.temporal_memory import temporal_memory_collate
from model.h2_target_preserving_residual_refiner import (
    FrozenM20TargetPreservingRefiner,
    target_preserving_parameter_count,
)
from utils.target_preserving_residual import (
    H2_EVENT_COUNT_CUTOFF,
    H2_POLARITY_MINORITY_CUTOFF,
    TargetRetentionDualState,
    complete_input_polarity_minority_fraction,
    input_only_routed_scores,
    target_preserving_event_loss,
    use_h2_residual_refiner,
    validate_all_step_diagnostics,
)


EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
PROTOCOL_PATH = (
    EVC_ROOT / 'protocols' / 'h2_target_preserving_residual_refiner_g2_science_v2.json'
)
V1_PROTOCOL_PATH = (
    EVC_ROOT / 'protocols' / 'h2_spatiotemporal_residual_refiner_oof_science_v1.json'
)
ROUTE_EVIDENCE_PROTOCOL_PATH = (
    EVC_ROOT / 'protocols' / 'high_density_dual_expert_grouped_oof_science_v1.json'
)
OUTPUT_ROOT = (
    WORKSPACE_ROOT / 'experiments' / '20260811_h2_target_preserving_residual_refiner_g2_v2'
)
EXPECTED_SCHEMA = 'ev-uav-frozen-m20-h2-target-preserving-residual-refiner-g2-v2'
GPU_AUTHORIZATION_FLAG = '--root-authorized-gpu'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_protocol():
    with PROTOCOL_PATH.open('r', encoding='utf-8') as stream:
        protocol = json.load(stream)
    if protocol.get('schema') != EXPECTED_SCHEMA:
        raise RuntimeError('Unexpected V2 protocol schema.')
    if protocol.get('status') != (
        'frozen_after_g1_development_evidence_before_any_v2_gpu_or_g2_held_prediction'
    ):
        raise RuntimeError('V2 protocol is not frozen at the GPU gate.')
    if core.sha256_file(V1_PROTOCOL_PATH) != protocol[
        'source_manifest_inheritance'
    ]['sha256']:
        raise RuntimeError('Inherited V1 source manifest changed.')
    return protocol


def load_source_manifest():
    with V1_PROTOCOL_PATH.open('r', encoding='utf-8') as stream:
        return json.load(stream)['h2_sources']


def source_paths(protocol, names):
    manifest = load_source_manifest()
    if not set(names).issubset(manifest):
        raise RuntimeError('A V2 source is outside the frozen H2 manifest.')
    paths = [core.TRAIN_ROOT / name for name in names]
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            polarities = np.asarray(archive['evs_norm'])[:, 3]
        if not use_h2_residual_refiner(len(polarities), polarities):
            raise RuntimeError(
                'A fit source does not satisfy the complete-input H2 route: {}'.format(
                    path.name
                )
            )
    return paths


def require_gpu_authorization(args):
    if not bool(getattr(args, 'root_authorized_gpu', False)):
        raise RuntimeError('GPU execution requires {}.'.format(GPU_AUTHORIZATION_FLAG))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _development_evidence_paths(protocol):
    paths = {}
    for key in ('training_result', 'checkpoint', 'paired_evaluation'):
        record = protocol['development_evidence'][key]
        paths[key] = WORKSPACE_ROOT / record['workspace_relative_path']
    return paths


def audit_protocol(run_tests=True):
    protocol = load_protocol()
    route_evidence = protocol['input_only_route']['evidence_protocol']
    if (
        core.sha256_file(ROUTE_EVIDENCE_PROTOCOL_PATH) != route_evidence['sha256']
        or float(protocol['input_only_route']['polarity_minority_cutoff'])
        != H2_POLARITY_MINORITY_CUTOFF
        or int(protocol['input_only_route']['event_count_cutoff_exclusive'])
        != H2_EVENT_COUNT_CUTOFF
    ):
        raise RuntimeError('Complete-input H2 route evidence or cutoffs changed.')
    evidence_paths = _development_evidence_paths(protocol)
    for key, path in evidence_paths.items():
        expected = protocol['development_evidence'][key]['sha256']
        if not path.is_file() or core.sha256_file(path) != expected:
            raise RuntimeError('G1 development evidence changed: {}'.format(key))
    if core.sha256_file(core.M20_PATH) != protocol[
        'source_manifest_inheritance'
    ]['released_M20_sha256']:
        raise RuntimeError('Released M20 identity mismatch.')

    manifest = load_source_manifest()
    fold = protocol['single_unseen_fold']
    fit = set(fold['fit'])
    held = set(fold['held'])
    if fit & held or fit | held != set(manifest):
        raise RuntimeError('V2 G2 fit/held split is not an exact H2 partition.')
    if fold['fold_id'] != 'hold_g2' or held != {
        'train_092.npz', 'train_093.npz', 'train_094.npz'
    }:
        raise RuntimeError('V2 must use only the frozen unseen G2 held block.')
    for name, metadata in manifest.items():
        path = core.TRAIN_ROOT / name
        if not path.is_file() or core.sha256_file(path) != metadata['sha256']:
            raise RuntimeError('H2 source identity mismatch: {}'.format(name))
        with np.load(path, allow_pickle=False) as archive:
            polarities = np.asarray(archive['evs_norm'])[:, 3]
        minority = complete_input_polarity_minority_fraction(polarities)
        if not use_h2_residual_refiner(len(polarities), polarities):
            raise RuntimeError('Frozen H2 source no longer routes to H2: {}'.format(name))
        expected_minorities = protocol['input_only_route'][
            'frozen_H2_polarity_minority_fraction'
        ]
        if abs(minority - float(expected_minorities[name])) > 1e-15:
            raise RuntimeError('Frozen H2 polarity fraction changed: {}'.format(name))

    base, payload = core.build_released_m20(torch.device('cpu'))
    wrapper = FrozenM20TargetPreservingRefiner(
        base,
        context_bins=core.CONTEXT_BINS,
        hidden_channels=int(protocol['architecture']['hidden_channels_each_branch']),
    ).eval()
    if target_preserving_parameter_count(wrapper) != int(
        protocol['architecture']['trainable_parameter_count']
    ):
        raise RuntimeError('V2 trainable parameter count mismatch.')
    if any(parameter.requires_grad for parameter in wrapper.released_m20.parameters()):
        raise RuntimeError('A released M20 parameter is trainable in V2.')
    protection_gate, suppression_gate = wrapper.refiner.gate_values()
    if float(protection_gate) != 0.0 or float(suppression_gate) != 0.0:
        raise RuntimeError('V2 gates are not exact zero at initialization.')

    test_output = None
    if run_tests:
        command = [
            sys.executable,
            '-m',
            'unittest',
            'discover',
            '-s',
            str(EVC_ROOT / 'tests'),
            '-p',
            'test_h2_target_preserving_residual_refiner.py',
            '-v',
        ]
        completed = subprocess.run(command, cwd=EVC_ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError('V2 CPU tests failed:\n{}'.format(
                completed.stdout + completed.stderr
            ))
        test_output = (completed.stdout + completed.stderr).strip()
    return {
        'schema': 'ev-uav-h2-target-preserving-residual-v2-cpu-audit',
        'created_utc': utc_now(),
        'protocol_sha256': core.sha256_file(PROTOCOL_PATH),
        'runner_sha256': core.sha256_file(Path(__file__)),
        'model_sha256': core.sha256_file(
            EVC_ROOT / 'model' / 'h2_target_preserving_residual_refiner.py'
        ),
        'loss_utils_sha256': core.sha256_file(
            EVC_ROOT / 'utils' / 'target_preserving_residual.py'
        ),
        'tests_sha256': core.sha256_file(
            EVC_ROOT / 'tests' / 'test_h2_target_preserving_residual_refiner.py'
        ),
        'released_m20_state_tensor_count': len(payload['model_state_dict']),
        'released_m20_state_sha256': core.state_sha256(base.state_dict()),
        'v2_trainable_parameter_count': target_preserving_parameter_count(wrapper),
        'zero_gate_bitwise_identity': True,
        'fit_source_count': len(fit),
        'held_source_count': len(held),
        'G1_reprediction': False,
        'G2_held_prediction': False,
        'validation_or_test_read': False,
        'gpu_used': False,
        'cpu_tests': test_output,
    }


def make_training_objects(protocol, device):
    training = protocol['training']
    seed_everything(int(training['seed']))
    base, _ = core.build_released_m20(device)
    wrapper = FrozenM20TargetPreservingRefiner(
        base,
        context_bins=core.CONTEXT_BINS,
        hidden_channels=int(protocol['architecture']['hidden_channels_each_branch']),
    ).to(device).train()
    dataset = core.ExactSourceSequenceDataset(
        source_paths(protocol, protocol['single_unseen_fold']['fit']),
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
    scaler = torch.cuda.amp.GradScaler(enabled=bool(training['mixed_precision']))
    dual = TargetRetentionDualState(**training['loss']['dynamic_multipliers_initial'])
    return wrapper, dataset, loader, optimizer, scaler, dual


def _gather_interior_event_tensors(refined_maps, base_maps, parts, batch):
    device = refined_maps.device
    times = batch['event_time_indices'].to(device, non_blocking=True)
    xs = batch['event_x'].to(device, non_blocking=True)
    ys = batch['event_y'].to(device, non_blocking=True)
    labels = batch['labels'].to(device, non_blocking=True)
    target_ids = batch['target_ids'].to(device, non_blocking=True)
    interior = (times > 0) & (times < refined_maps.shape[1] - 1)
    if not bool(torch.any(interior)):
        raise RuntimeError('V2 sample has no interior events.')
    times = times[interior]
    xs = xs[interior]
    ys = ys[interior]
    labels = labels[interior]
    target_ids = target_ids[interior]
    event_refined = refined_maps[0, times, 0, ys, xs]
    event_base = base_maps[0, times, 0, ys, xs].detach()
    event_protection = parts['protection'][0, times, 0, ys, xs]
    event_suppression = parts['suppression'][0, times, 0, ys, xs]
    return (
        event_refined,
        event_base,
        event_protection,
        event_suppression,
        labels,
        target_ids,
        times,
    )


def train_steps(protocol, max_steps=None):
    device = torch.device('cuda:0')
    wrapper, dataset, loader, optimizer, scaler, dual = make_training_objects(
        protocol, device,
    )
    training = protocol['training']
    frozen_before = core.state_sha256(wrapper.frozen_state_dict())
    records = []
    step = 0
    started = time.perf_counter()
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
                features, base_maps = wrapper.frozen_forward_parts(frames)
                flat_frames = frames.reshape(
                    frames.shape[0] * frames.shape[1],
                    frames.shape[2], frames.shape[3], frames.shape[4],
                )
                centre = wrapper._centre_inputs(flat_frames).reshape(
                    frames.shape[0], frames.shape[1], 3,
                    frames.shape[3], frames.shape[4],
                )
                parts = wrapper.refiner(
                    features, base_maps, centre, return_parts=True,
                )
                refined_maps = base_maps + parts['residual']
                gathered = _gather_interior_event_tensors(
                    refined_maps, base_maps, parts, batch,
                )
                (
                    event_refined, event_base, event_protection,
                    event_suppression, labels, target_ids, times,
                ) = gathered
                loss, event_constraint, group_constraint, diagnostics = (
                    target_preserving_event_loss(
                        event_refined,
                        event_base,
                        labels,
                        target_ids,
                        times,
                        dual,
                    )
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                wrapper.trainable_parameters(),
                max_norm=float(training['gradient_clip_norm']),
            )
            if not bool(torch.isfinite(grad_norm)):
                raise RuntimeError('V2 produced a non-finite gradient norm.')
            scaler.step(optimizer)
            scaler.update()
            wrapper.project_gates_()
            dual.update(event_constraint, group_constraint)
            protection_gate, suppression_gate = wrapper.refiner.gate_values()
            step += 1
            diagnostics.update({
                'step': step,
                'epoch': epoch,
                'gradient_norm': float(grad_norm.detach()),
                'event_count': int(event_refined.numel()),
                'mean_abs_residual': float(
                    torch.mean(torch.abs(event_refined - event_base)).detach()
                ),
                'mean_protection': float(torch.mean(event_protection).detach()),
                'mean_suppression': float(torch.mean(event_suppression).detach()),
                'protection_gate_after': float(protection_gate.detach()),
                'suppression_gate_after': float(suppression_gate.detach()),
                'dual_positive_event_after': float(dual.positive_event),
                'dual_target_group_after': float(dual.target_group),
            })
            if not all(
                np.isfinite(value)
                for key, value in diagnostics.items()
                if isinstance(value, float)
            ):
                raise RuntimeError('V2 produced non-finite diagnostics.')
            records.append(diagnostics)
            if max_steps is not None and step >= int(max_steps):
                break
        if max_steps is not None and step >= int(max_steps):
            break
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    frozen_after = core.state_sha256(wrapper.frozen_state_dict())
    if frozen_after != frozen_before:
        raise RuntimeError('Released M20 changed during V2 training.')
    return {
        'wrapper': wrapper,
        'dual': dual,
        'records': records,
        'step_count': step,
        'elapsed_seconds': elapsed,
        'peak_cuda_bytes': int(torch.cuda.max_memory_allocated(device)),
        'frozen_before': frozen_before,
        'frozen_after': frozen_after,
    }


def run_audit(_args):
    print(json.dumps(audit_protocol(run_tests=True), indent=2, ensure_ascii=False))


def run_probe(args):
    require_gpu_authorization(args)
    protocol = load_protocol()
    result_path = OUTPUT_ROOT / 'resource_probe' / 'eight_step_probe.json'
    if result_path.exists():
        raise FileExistsError('Refusing to overwrite V2 probe receipt.')
    expected_steps = int(protocol['probe']['optimizer_steps'])
    result = train_steps(protocol, max_steps=expected_steps)
    if result['step_count'] != expected_steps:
        raise RuntimeError('V2 probe step count mismatch.')
    validate_all_step_diagnostics(result['records'], expected_steps)
    payload = {
        'schema': 'ev-uav-h2-target-preserving-residual-v2-probe',
        'created_utc': utc_now(),
        'protocol_sha256': core.sha256_file(PROTOCOL_PATH),
        'fit_sources': protocol['single_unseen_fold']['fit'],
        'G2_held_arrays_read': False,
        'optimizer_steps': result['step_count'],
        'elapsed_seconds': result['elapsed_seconds'],
        'seconds_per_step': result['elapsed_seconds'] / result['step_count'],
        'peak_cuda_mib': result['peak_cuda_bytes'] / (1024.0 ** 2),
        'all_step_diagnostics': result['records'],
        'dual_final': result['dual'].to_dict(),
        'frozen_m20_state_sha256_before': result['frozen_before'],
        'frozen_m20_state_sha256_after': result['frozen_after'],
        'G1_reprediction': False,
        'validation_or_test_read': False,
    }
    write_json_exclusive(result_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_train_g2(args):
    require_gpu_authorization(args)
    protocol = load_protocol()
    root = OUTPUT_ROOT / 'formal_training' / 'hold_g2'
    checkpoint_path = root / 'final_refiner.pt'
    result_path = root / 'training_result.json'
    if root.exists():
        raise FileExistsError('Refusing to overwrite V2 formal G2 training.')
    result = train_steps(protocol, max_steps=None)
    if result['step_count'] != int(protocol['training']['optimizer_steps']):
        raise RuntimeError('V2 formal optimizer-step count mismatch.')
    validate_all_step_diagnostics(
        result['records'], int(protocol['training']['optimizer_steps'])
    )
    checkpoint = {
        'schema': 'ev-uav-h2-target-preserving-residual-v2-checkpoint',
        'created_utc': utc_now(),
        'fold_id': 'hold_g2',
        'fit_sources': protocol['single_unseen_fold']['fit'],
        'protocol_sha256': core.sha256_file(PROTOCOL_PATH),
        'released_m20_sha256': core.sha256_file(core.M20_PATH),
        'optimizer_steps': result['step_count'],
        'dual_state': result['dual'].to_dict(),
        'refiner_state_dict': {
            key: value.detach().cpu()
            for key, value in result['wrapper'].refiner.state_dict().items()
        },
        'frozen_m20_state_sha256': result['frozen_after'],
    }
    save_torch_exclusive(checkpoint_path, checkpoint)
    payload = {
        'schema': 'ev-uav-h2-target-preserving-residual-v2-training-result',
        'created_utc': utc_now(),
        'fold_id': 'hold_g2',
        'fit_sources': protocol['single_unseen_fold']['fit'],
        'G2_held_arrays_read': False,
        'checkpoint': str(checkpoint_path.resolve()),
        'checkpoint_sha256': core.sha256_file(checkpoint_path),
        'optimizer_steps': result['step_count'],
        'elapsed_seconds': result['elapsed_seconds'],
        'peak_cuda_mib': result['peak_cuda_bytes'] / (1024.0 ** 2),
        'all_step_diagnostics': result['records'],
        'dual_final': result['dual'].to_dict(),
        'frozen_m20_state_sha256_before': result['frozen_before'],
        'frozen_m20_state_sha256_after': result['frozen_after'],
        'G1_reprediction': False,
        'validation_or_test_read': False,
    }
    write_json_exclusive(result_path, payload)
    print(json.dumps({
        key: value for key, value in payload.items()
        if key != 'all_step_diagnostics'
    }, indent=2, ensure_ascii=False))


def evaluate_g2(args):
    require_gpu_authorization(args)
    protocol = load_protocol()
    checkpoint_path = (
        OUTPUT_ROOT / 'formal_training' / 'hold_g2' / 'final_refiner.pt'
    )
    result_path = (
        OUTPUT_ROOT / 'held_train_evaluation' / 'hold_g2' / 'paired_evaluation.json'
    )
    if result_path.exists() or result_path.parent.exists():
        raise FileExistsError('Refusing to overwrite V2 G2 held evaluation.')
    checkpoint = core.load_checkpoint_file(checkpoint_path, map_location='cpu')
    if checkpoint.get('fold_id') != 'hold_g2':
        raise RuntimeError('V2 checkpoint fold mismatch.')
    if checkpoint.get('protocol_sha256') != core.sha256_file(PROTOCOL_PATH):
        raise RuntimeError('V2 checkpoint protocol mismatch.')

    import crossfit_component_reranker as component_crossfit
    import replay_temporal_memory_validation as replay
    from crossfit_component_reranker import (
        SufficientCounts, metrics_from_counts, sufficient_counts_for_video,
    )
    from utils.postprocess import ChallengePostprocessor

    cfg = replay.load_flat_config(
        EVC_ROOT / 'configs' / 'evisseg_evuav.yaml', core.C00_OVERRIDES,
    )
    threshold = float(protocol['evaluation']['prediction_threshold'])
    c00 = component_crossfit.validate_c00_config(cfg, threshold)
    if component_crossfit.sha256_json(c00) != protocol[
        'evaluation'
    ]['effective_C00_sha256']:
        raise RuntimeError('V2 C00 contract mismatch.')

    device = torch.device('cuda:0')
    base, _ = core.build_released_m20(device)
    wrapper = FrozenM20TargetPreservingRefiner(
        base,
        context_bins=core.CONTEXT_BINS,
        hidden_channels=int(protocol['architecture']['hidden_channels_each_branch']),
    ).to(device).eval()
    wrapper.refiner.load_state_dict(checkpoint['refiner_state_dict'], strict=True)
    wrapper.project_gates_()
    manifest = load_source_manifest()
    pooled_base = SufficientCounts()
    pooled_candidate = SufficientCounts()
    records = []
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for name in protocol['single_unseen_fold']['held']:
        path = core.TRAIN_ROOT / name
        if core.sha256_file(path) != manifest[name]['sha256']:
            raise RuntimeError('V2 G2 held source identity mismatch.')
        video, labels, target_ids, locations3 = core.load_unlabelled_video_and_truth(path)
        base_raw, candidate_raw = core.predict_paired_full_stream(
            wrapper, video, device,
        )
        minority = complete_input_polarity_minority_fraction(video.polarities)
        routed_candidate_raw = input_only_routed_scores(
            base_raw, candidate_raw, video.polarities,
        )
        if not use_h2_residual_refiner(len(labels), video.polarities):
            if not np.array_equal(routed_candidate_raw, base_raw):
                raise RuntimeError('Non-H2 route did not preserve M20 bitwise.')
        candidate_raw = routed_candidate_raw
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
            'input_only_route': {
                'event_count': int(len(labels)),
                'polarity_minority_fraction': minority,
                'candidate': 'h2_target_preserving_residual'
                if use_h2_residual_refiner(len(labels), video.polarities)
                else 'released_m20_identity',
            },
            'base_raw_scores_sha256': core.sha256_float32(base_raw),
            'candidate_raw_scores_sha256': core.sha256_float32(candidate_raw),
            'base_postprocess': asdict(base_stats),
            'candidate_postprocess': asdict(candidate_stats),
            'base_counts': base_counts.to_dict(),
            'candidate_counts': candidate_counts.to_dict(),
            'base_metrics': metrics_from_counts(base_counts),
            'candidate_metrics': metrics_from_counts(candidate_counts),
        })
        torch.cuda.empty_cache()
        print('V2 G2 held paired:', name, flush=True)
    torch.cuda.synchronize(device)
    base_metrics = metrics_from_counts(pooled_base)
    candidate_metrics = metrics_from_counts(pooled_candidate)
    count_delta = {
        key: int(value - getattr(pooled_base, key))
        for key, value in pooled_candidate.to_dict().items()
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
        'H2_score_gain_at_least_0_02': metric_delta['score'] >= 0.02,
    }
    payload = {
        'schema': 'ev-uav-h2-target-preserving-residual-v2-g2-evaluation',
        'created_utc': utc_now(),
        'fold_id': 'hold_g2',
        'held_sources': protocol['single_unseen_fold']['held'],
        'checkpoint': str(checkpoint_path.resolve()),
        'checkpoint_sha256': core.sha256_file(checkpoint_path),
        'prediction_threshold': threshold,
        'effective_C00_contract': c00,
        'records': records,
        'pooled_base_counts': pooled_base.to_dict(),
        'pooled_candidate_counts': pooled_candidate.to_dict(),
        'pooled_count_delta': count_delta,
        'pooled_base_metrics': base_metrics,
        'pooled_candidate_metrics': candidate_metrics,
        'pooled_metric_delta': metric_delta,
        'gates': gates,
        'all_safety_gates_passed': all(
            value for key, value in gates.items()
            if key != 'H2_score_gain_at_least_0_02'
        ),
        'continuation_effect_size_gate_passed': gates['H2_score_gain_at_least_0_02'],
        'elapsed_seconds': time.perf_counter() - started,
        'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        'G1_reprediction': False,
        'validation_or_test_read': False,
    }
    write_json_exclusive(result_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    audit = subparsers.add_parser('audit')
    audit.set_defaults(func=run_audit)
    probe = subparsers.add_parser('probe')
    probe.add_argument(GPU_AUTHORIZATION_FLAG, dest='root_authorized_gpu', action='store_true')
    probe.set_defaults(func=run_probe)
    train = subparsers.add_parser('train-g2')
    train.add_argument(GPU_AUTHORIZATION_FLAG, dest='root_authorized_gpu', action='store_true')
    train.set_defaults(func=run_train_g2)
    evaluate = subparsers.add_parser('evaluate-g2')
    evaluate.add_argument(GPU_AUTHORIZATION_FLAG, dest='root_authorized_gpu', action='store_true')
    evaluate.set_defaults(func=evaluate_g2)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
