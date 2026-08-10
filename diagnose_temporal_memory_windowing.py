"""Compare full-stream and training-aligned temporal-memory inference on train.

This diagnostic deliberately rejects non-``train_*.npz`` inputs.  It uses one
fixed decision threshold and the released C00 postprocess; there is no
threshold, checkpoint, or postprocess search.  The released inference helper
and core evaluation entry points remain unchanged.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from dataset.temporal_frame import load_temporal_frame_video
from dataset.temporal_memory import npz_event_count
from utils.challenge_eval import evaluate_challenge_metrics
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor
from utils.temporal_memory_inference import (
    load_temporal_memory_model,
    predict_temporal_memory_scores,
)
from utils.temporal_memory_windowed_inference import (
    predict_temporal_memory_scores_windowed,
    temporal_center_stitch_plan,
)


SCHEMA_VERSION = 'temporal-memory-window-train-diagnostic-v1'
TRAIN_NAME_PATTERN = re.compile(r'^train_[0-9]+$')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def checkpoint_identity(path):
    path = Path(path).resolve()
    payload = _torch_load_cpu(path)
    memory = payload.get('temporal_memory', {})
    required = ('context_bins', 'width', 'sequence_length')
    missing = [key for key in required if memory.get(key) is None]
    if missing:
        raise ValueError(
            'Checkpoint is missing temporal-memory metadata: {}'.format(
                ', '.join(missing)
            )
        )
    return {
        'path': str(path),
        'size_bytes': path.stat().st_size,
        'sha256': sha256_file(path),
        'context_bins': int(memory['context_bins']),
        'width': int(memory['width']),
        'sequence_length': int(memory['sequence_length']),
        'temporal_attention_enabled': bool(
            memory.get('temporal_attention_enabled', False)
        ),
        'confidence_head_enabled': bool(memory.get('confidence_head_enabled', False)),
        'density_calibration_enabled': bool(
            memory.get('density_calibration_enabled', False)
        ),
    }


def discover_train_files(
    data_root,
    video_names,
    min_event_count_exclusive,
    max_videos,
):
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError('Train directory not found: {}'.format(data_root))
    all_files = sorted(data_root.glob('*.npz'))
    if not all_files:
        raise RuntimeError('No npz files found in {}'.format(data_root))
    rejected = [path.name for path in all_files if not TRAIN_NAME_PATTERN.fullmatch(path.stem)]
    if rejected:
        raise ValueError(
            'Train-only guard rejected non-train inputs: {}'.format(
                ', '.join(rejected[:5])
            )
        )

    by_name = {path.stem: path for path in all_files}
    if video_names:
        normalized_names = [Path(name).stem for name in video_names]
        invalid = [name for name in normalized_names if not TRAIN_NAME_PATTERN.fullmatch(name)]
        if invalid:
            raise ValueError(
                'Train-only guard rejected requested names: {}'.format(', '.join(invalid))
            )
        missing = [name for name in normalized_names if name not in by_name]
        if missing:
            raise FileNotFoundError(
                'Requested train videos are missing: {}'.format(', '.join(missing))
            )
        selected = [by_name[name] for name in normalized_names]
    else:
        selected = all_files

    event_counts = {path: npz_event_count(path) for path in selected}
    selected = [
        path
        for path in selected
        if event_counts[path] > int(min_event_count_exclusive)
    ]
    if max_videos:
        selected = selected[:int(max_videos)]
    if not selected:
        raise RuntimeError(
            'No train videos remain after the strict event_count > {} filter.'.format(
                min_event_count_exclusive
            )
        )
    return data_root, selected, event_counts


def c00_config():
    """Return the frozen released M20/C00 postprocess configuration."""

    return SimpleNamespace(
        pd_detT=50,
        p0_enabled=True,
        p0_spatial_radius=2,
        p0_temporal_bin_size=50,
        p0_temporal_radius_bins=1,
        p0_min_cluster_events=3,
        p0_min_duration_bins=5,
        p0c_high_confidence_recovery_enabled=True,
        p0c_retain_min_score=0.95,
        p0c_density_retain_enabled=False,
        p0b_enabled=False,
        p18_score_track_recovery_enabled=True,
        p18_event_count_cutoff=1,
        p18_max_event_count=35000,
        p18_candidate_floor=0.53,
        p18_spatial_radius=5,
        p18_temporal_bin_size=50,
        p18_max_link_distance=8.0,
        p18_max_gap_bins=1,
        p18_min_track_bins=4,
        p18_restore_mode='best',
        p18_max_restore_events_per_component=0,
        component_reranker_enabled=False,
    )


def evaluator_config():
    return SimpleNamespace(roc=True, pd_detT=50, correct_thresh=0.0001)


def event_locations_with_batch(video):
    return torch.from_numpy(
        np.column_stack(
            (
                np.zeros(video.locations.shape[0], dtype=np.int64),
                video.locations.astype(np.int64, copy=False),
            )
        )
    ).long()


def add_video_to_evaluator(evaluator, video, scores, sample_number, threshold):
    scores = scores.detach().cpu().float().reshape(-1)
    labels = torch.from_numpy(video.labels.astype(np.float32, copy=False)).float()
    locations = event_locations_with_batch(video)
    evaluator.matches[str(sample_number)] = {
        'seg_pred': scores,
        'seg_gt': labels,
    }
    evaluator.roc_update(
        locations[:, 3],
        scores.clone(),
        video.target_ids.astype(np.int64, copy=False),
        labels,
        locations,
        thresh=float(threshold),
    )


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed_inference(callback, device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    result = callback()
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == 'cuda'
        else None
    )
    return result, elapsed, peak_memory


def confusion_counts(labels, scores, threshold):
    labels = np.asarray(labels).reshape(-1) > 0.5
    predicted = np.asarray(scores).reshape(-1) >= float(threshold)
    return {
        'true_positive_events': int(np.count_nonzero(predicted & labels)),
        'false_positive_events': int(np.count_nonzero(predicted & ~labels)),
        'false_negative_events': int(np.count_nonzero(~predicted & labels)),
        'true_negative_events': int(np.count_nonzero(~predicted & ~labels)),
    }


def add_counts(target, source):
    for key, value in source.items():
        target[key] += int(value)


def score_difference(reference, candidate, labels, threshold):
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1) > 0.5
    if reference.shape != candidate.shape or labels.shape != reference.shape:
        raise ValueError('Scores and labels must be aligned.')
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(delta)
    reference_positive = reference >= float(threshold)
    candidate_positive = candidate >= float(threshold)
    return {
        'event_count': int(reference.size),
        'changed_score_events': int(np.count_nonzero(reference != candidate)),
        'mean_absolute_score_delta': float(absolute.mean()),
        'root_mean_square_score_delta': float(np.sqrt(np.mean(delta * delta))),
        'max_absolute_score_delta': float(absolute.max(initial=0.0)),
        'mean_signed_score_delta_all': float(delta.mean()),
        'mean_signed_score_delta_positive_label': (
            float(delta[labels].mean()) if np.any(labels) else None
        ),
        'mean_signed_score_delta_negative_label': (
            float(delta[~labels].mean()) if np.any(~labels) else None
        ),
        'full_positive_window_negative': int(
            np.count_nonzero(reference_positive & ~candidate_positive)
        ),
        'full_negative_window_positive': int(
            np.count_nonzero(~reference_positive & candidate_positive)
        ),
    }


def boundary_difference(video, reference, candidate, plan, threshold):
    margin_by_bin = np.empty(len(video.event_indices_by_bin), dtype=np.int64)
    for item in plan:
        bins = np.arange(item.keep_start, item.keep_stop, dtype=np.int64)
        margin_by_bin[bins] = np.minimum(
            bins - item.window_start,
            item.window_stop - 1 - bins,
        )
    event_margins = margin_by_bin[video.event_bins]
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    absolute = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    flips = (reference >= float(threshold)) != (candidate >= float(threshold))
    groups = {}
    for name, mask in (
        ('margin_0_1', event_margins <= 1),
        ('margin_2_3', (event_margins >= 2) & (event_margins <= 3)),
        ('margin_4_plus', event_margins >= 4),
    ):
        groups[name] = {
            'event_count': int(np.count_nonzero(mask)),
            'mean_absolute_score_delta': (
                float(absolute[mask].mean()) if np.any(mask) else None
            ),
            'threshold_flip_events': int(np.count_nonzero(flips & mask)),
        }
    return groups


def metrics_and_counts(evaluator, threshold, confusion):
    metrics = evaluate_challenge_metrics(evaluator, threshold).to_dict()
    return {
        'metrics': metrics,
        'counts': {
            **{key: int(value) for key, value in confusion.items()},
            'correct_target_groups': int(evaluator.correct_num),
            'target_groups': int(evaluator.obj_num),
            'false_components': int(evaluator.false_num),
            'frame_count': int(evaluator.frame_num),
        },
    }


def evaluate_one_video(video, scores, threshold, confusion):
    evaluator = evalute(evaluator_config())
    add_video_to_evaluator(
        evaluator,
        video,
        scores,
        sample_number=0,
        threshold=threshold,
    )
    return metrics_and_counts(evaluator, threshold, confusion)


def finite_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError('Refusing to write a non-finite JSON value.')
    if isinstance(value, dict):
        return {key: finite_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json_value(item) for item in value]
    return value


def atomic_json_write(path, payload, force=False):
    path = Path(path).resolve()
    if path.exists() and not force:
        raise FileExistsError(
            'Output already exists; pass --force to replace it: {}'.format(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(finite_json_value(payload), indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def run(args):
    if not 0.0 <= args.prediction_threshold <= 1.0:
        raise ValueError('prediction_threshold must be in [0, 1].')
    if args.min_event_count_exclusive < 0:
        raise ValueError('min_event_count_exclusive must be non-negative.')
    if args.max_videos < 0 or args.identity_videos < 0:
        raise ValueError('max_videos and identity_videos must be non-negative.')
    window_lengths = tuple(dict.fromkeys(int(value) for value in args.window_lengths))
    if any(value <= 0 for value in window_lengths):
        raise ValueError('window_lengths must contain positive integers.')

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    checkpoint = checkpoint_identity(args.checkpoint)
    if checkpoint['context_bins'] != args.context_bins:
        raise ValueError('Configured context_bins does not match checkpoint metadata.')
    if checkpoint['width'] != args.model_width:
        raise ValueError('Configured model_width does not match checkpoint metadata.')

    data_root, files, event_counts = discover_train_files(
        args.data_root,
        args.video_names,
        args.min_event_count_exclusive,
        args.max_videos,
    )
    model, _ = load_temporal_memory_model(
        args.checkpoint,
        device,
        args.context_bins,
        args.model_width,
        checkpoint['sequence_length'],
    )

    mode_names = ['full_stream'] + [
        'window_t{}'.format(length) for length in window_lengths
    ]
    raw_evaluators = {name: evalute(evaluator_config()) for name in mode_names}
    c00_evaluators = {name: evalute(evaluator_config()) for name in mode_names}
    raw_confusions = {name: defaultdict(int) for name in mode_names}
    c00_confusions = {name: defaultdict(int) for name in mode_names}
    inference_seconds = defaultdict(float)
    peak_memory_bytes = defaultdict(int)
    aggregate_differences = {
        name: {
            'event_count': 0,
            'changed_score_events': 0,
            'absolute_score_delta_sum': 0.0,
            'squared_score_delta_sum': 0.0,
            'max_absolute_score_delta': 0.0,
            'full_positive_window_negative': 0,
            'full_negative_window_positive': 0,
        }
        for name in mode_names[1:]
    }
    per_video = []
    identity_records = []

    for sample_number, path in enumerate(files):
        print(
            '[{}/{}] {} events={}'.format(
                sample_number + 1,
                len(files),
                path.name,
                event_counts[path],
            ),
            flush=True,
        )
        video = load_temporal_frame_video(
            path,
            args.temporal_bin_size,
            args.whole_t,
        )
        temporal_bin_count = len(video.event_indices_by_bin)
        oversized = [length for length in window_lengths if length > temporal_bin_count]
        if oversized:
            raise ValueError(
                '{} has {} bins, smaller than windows {}.'.format(
                    path.name,
                    temporal_bin_count,
                    oversized,
                )
            )
        common = dict(
            model=model,
            video=video,
            device=device,
            context_bins=args.context_bins,
            width=args.width,
            height=args.height,
            inference_batch_size=args.inference_batch_size,
            log_count_clip=args.log_count_clip,
        )
        full_scores, elapsed, peak = timed_inference(
            lambda: predict_temporal_memory_scores(**common),
            device,
        )
        inference_seconds['full_stream'] += elapsed
        if peak is not None:
            peak_memory_bytes['full_stream'] = max(
                peak_memory_bytes['full_stream'],
                peak,
            )
        scores_by_mode = {'full_stream': full_scores}
        video_inference_seconds = {'full_stream': elapsed}
        video_peak_memory_bytes = {'full_stream': peak}

        if sample_number < args.identity_videos:
            identity_scores, identity_elapsed, identity_peak = timed_inference(
                lambda: predict_temporal_memory_scores_windowed(
                    **common,
                    window_length=temporal_bin_count,
                ),
                device,
            )
            exact = torch.equal(full_scores, identity_scores)
            max_delta = float(
                torch.max(torch.abs(full_scores - identity_scores)).item()
            )
            identity_records.append(
                {
                    'video': path.name,
                    'temporal_bin_count': temporal_bin_count,
                    'bitwise_equal': bool(exact),
                    'max_absolute_score_delta': max_delta,
                    'inference_seconds': identity_elapsed,
                    'peak_cuda_memory_bytes': identity_peak,
                }
            )
            if not exact:
                raise RuntimeError(
                    'Full-length window identity failed for {} (max delta {}).'.format(
                        path.name,
                        max_delta,
                    )
                )

        plans = {}
        for window_length in window_lengths:
            mode = 'window_t{}'.format(window_length)
            plan = temporal_center_stitch_plan(temporal_bin_count, window_length)
            plans[mode] = plan
            scores, elapsed, peak = timed_inference(
                lambda window_length=window_length: (
                    predict_temporal_memory_scores_windowed(
                        **common,
                        window_length=window_length,
                    )
                ),
                device,
            )
            scores_by_mode[mode] = scores
            video_inference_seconds[mode] = elapsed
            video_peak_memory_bytes[mode] = peak
            inference_seconds[mode] += elapsed
            if peak is not None:
                peak_memory_bytes[mode] = max(peak_memory_bytes[mode], peak)

        locations = event_locations_with_batch(video)
        labels_numpy = video.labels.astype(np.float32, copy=False)
        video_record = {
            'name': path.name,
            'path': str(path.resolve()),
            'size_bytes': path.stat().st_size,
            'sha256': sha256_file(path),
            'event_count': int(video.locations.shape[0]),
            'positive_event_count': int(np.count_nonzero(labels_numpy > 0.5)),
            'temporal_bin_count': temporal_bin_count,
            'modes': {},
        }
        for mode, scores in scores_by_mode.items():
            raw_scores_numpy = scores.numpy()
            raw_counts = confusion_counts(
                labels_numpy,
                raw_scores_numpy,
                args.prediction_threshold,
            )
            add_counts(raw_confusions[mode], raw_counts)
            add_video_to_evaluator(
                raw_evaluators[mode],
                video,
                scores,
                sample_number,
                args.prediction_threshold,
            )

            postprocessor = ChallengePostprocessor.from_cfg(
                c00_config(),
                args.prediction_threshold,
                event_count=int(video.locations.shape[0]),
            )
            c00_scores, postprocess_stats = postprocessor.apply(
                scores.clone(),
                locations,
            )
            c00_scores_numpy = c00_scores.numpy()
            c00_counts = confusion_counts(
                labels_numpy,
                c00_scores_numpy,
                args.prediction_threshold,
            )
            add_counts(c00_confusions[mode], c00_counts)
            add_video_to_evaluator(
                c00_evaluators[mode],
                video,
                c00_scores,
                sample_number,
                args.prediction_threshold,
            )
            mode_record = {
                'inference_seconds': video_inference_seconds[mode],
                'peak_cuda_memory_bytes': video_peak_memory_bytes[mode],
                'raw_confusion': raw_counts,
                'c00_confusion': c00_counts,
                'raw_evaluation': evaluate_one_video(
                    video,
                    scores,
                    args.prediction_threshold,
                    raw_counts,
                ),
                'c00_evaluation': evaluate_one_video(
                    video,
                    c00_scores,
                    args.prediction_threshold,
                    c00_counts,
                ),
                'c00_postprocess': postprocess_stats.summary(),
            }
            if mode != 'full_stream':
                difference = score_difference(
                    full_scores.numpy(),
                    raw_scores_numpy,
                    labels_numpy,
                    args.prediction_threshold,
                )
                mode_record['versus_full_stream'] = difference
                mode_record['boundary_groups'] = boundary_difference(
                    video,
                    full_scores.numpy(),
                    raw_scores_numpy,
                    plans[mode],
                    args.prediction_threshold,
                )
                mode_record['stitch_plan'] = {
                    'window_length': plans[mode][0].window_length,
                    'default_stride': max(1, plans[mode][0].window_length // 2),
                    'window_count': len(plans[mode]),
                    'kept_bins_per_window': [item.keep_length for item in plans[mode]],
                }
                aggregate = aggregate_differences[mode]
                reference = full_scores.numpy().astype(np.float64)
                candidate = raw_scores_numpy.astype(np.float64)
                delta = candidate - reference
                aggregate['event_count'] += int(delta.size)
                aggregate['changed_score_events'] += difference['changed_score_events']
                aggregate['absolute_score_delta_sum'] += float(np.abs(delta).sum())
                aggregate['squared_score_delta_sum'] += float(np.square(delta).sum())
                aggregate['max_absolute_score_delta'] = max(
                    aggregate['max_absolute_score_delta'],
                    difference['max_absolute_score_delta'],
                )
                aggregate['full_positive_window_negative'] += difference[
                    'full_positive_window_negative'
                ]
                aggregate['full_negative_window_positive'] += difference[
                    'full_negative_window_positive'
                ]
            video_record['modes'][mode] = mode_record
        per_video.append(video_record)

    aggregate = {}
    for mode in mode_names:
        aggregate[mode] = {
            'inference_seconds': inference_seconds[mode],
            'peak_cuda_memory_bytes': (
                peak_memory_bytes[mode] if device.type == 'cuda' else None
            ),
            'raw': metrics_and_counts(
                raw_evaluators[mode],
                args.prediction_threshold,
                raw_confusions[mode],
            ),
            'c00': metrics_and_counts(
                c00_evaluators[mode],
                args.prediction_threshold,
                c00_confusions[mode],
            ),
        }
        if mode != 'full_stream':
            difference = aggregate_differences[mode]
            event_count = difference.pop('event_count')
            absolute_sum = difference.pop('absolute_score_delta_sum')
            squared_sum = difference.pop('squared_score_delta_sum')
            difference['event_count'] = event_count
            difference['mean_absolute_score_delta'] = absolute_sum / event_count
            difference['root_mean_square_score_delta'] = math.sqrt(
                squared_sum / event_count
            )
            difference['raw_metric_delta'] = {
                key: (
                    aggregate[mode]['raw']['metrics'][key]
                    - aggregate['full_stream']['raw']['metrics'][key]
                )
                for key in aggregate[mode]['raw']['metrics']
            }
            difference['c00_metric_delta'] = {
                key: (
                    aggregate[mode]['c00']['metrics'][key]
                    - aggregate['full_stream']['c00']['metrics'][key]
                )
                for key in aggregate[mode]['c00']['metrics']
            }
            aggregate[mode]['versus_full_stream'] = difference

    report = {
        'schema_version': SCHEMA_VERSION,
        'created_utc': utc_now(),
        'evidence_class': 'train_only_fixed_diagnostic_not_validation_or_oof',
        'guardrails': {
            'accepted_input_pattern': 'train_[0-9]+.npz only',
            'validation_inputs_permitted': False,
            'threshold_search': False,
            'checkpoint_search': False,
            'postprocess_search': False,
            'prediction_threshold': args.prediction_threshold,
            'postprocess_profile': 'released_M20_C00_fixed',
            'promotion_claim': False,
        },
        'runtime': {
            'command': ' '.join(sys.argv),
            'python': sys.version,
            'platform': platform.platform(),
            'torch': torch.__version__,
            'numpy': np.__version__,
            'device': str(device),
            'device_name': (
                torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
            ),
        },
        'checkpoint': checkpoint,
        'data': {
            'root': str(data_root),
            'selection_rule': 'event_count > {}'.format(
                args.min_event_count_exclusive
            ),
            'selected_video_count': len(files),
            'selected_event_count': int(sum(event_counts[path] for path in files)),
            'selected_files': [path.name for path in files],
        },
        'inference': {
            'whole_t': args.whole_t,
            'temporal_bin_size': args.temporal_bin_size,
            'context_bins': args.context_bins,
            'frame_resolution': [args.width, args.height],
            'inference_batch_size': args.inference_batch_size,
            'window_lengths': list(window_lengths),
            'window_stride_policy': 'floor(window_length / 2), minimum 1',
            'stitch_policy': 'nearest_window_center_ties_to_earlier_window',
        },
        'identity_checks': {
            'requested_video_count': args.identity_videos,
            'completed_video_count': len(identity_records),
            'all_bitwise_equal': all(
                record['bitwise_equal'] for record in identity_records
            ),
            'records': identity_records,
        },
        'aggregate': aggregate,
        'per_video': per_video,
    }
    atomic_json_write(args.output, report, force=args.force)
    print('report:', Path(args.output).resolve())
    for mode in mode_names:
        metrics = aggregate[mode]['c00']['metrics']
        print(
            '{} C00 Score={:.10f} Pd={:.10f} IoU={:.10f} Acc={:.10f} Fa={:.10e}'.format(
                mode,
                metrics['score'],
                metrics['pd'],
                metrics['iou'],
                metrics['acc'],
                metrics['fa'],
            )
        )
    return report


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Train-only comparison of full T160 inference against fixed-window '
            'nearest-centre stitching.'
        )
    )
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--video-names', nargs='*', default=[])
    parser.add_argument('--min-event-count-exclusive', type=int, default=30000)
    parser.add_argument('--max-videos', type=int, default=0)
    parser.add_argument('--window-lengths', nargs='+', type=int, default=[16, 32])
    parser.add_argument('--identity-videos', type=int, default=1)
    parser.add_argument('--prediction-threshold', type=float, default=0.719)
    parser.add_argument('--whole-t', type=int, default=8000)
    parser.add_argument('--temporal-bin-size', type=int, default=50)
    parser.add_argument('--context-bins', type=int, default=5)
    parser.add_argument('--model-width', type=int, default=16)
    parser.add_argument('--width', type=int, default=346)
    parser.add_argument('--height', type=int, default=260)
    parser.add_argument('--inference-batch-size', type=int, default=8)
    parser.add_argument('--log-count-clip', type=float, default=4.0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--force', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
