"""Replay focused post-processing candidates on cached temporal-memory scores.

The cache follows the full-stream path used by ``test2.py``: the primary
checkpoint scores every video, while the secondary checkpoint replaces scores
for videos at or below an event-count cutoff.  Candidate policies can only use
that observable event count; labels are used solely by the official validation
evaluator.

The optional ``blend`` phase combines two caches without a second inference
pass.  It is intended for probes where both caches share an identical
observable routing policy, such as M15/M16 with M10 retained for low-density
videos.
"""

import argparse
import hashlib
import time
from pathlib import Path

import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.eval import evalute
from utils.inference_chunks import evaluation_batch_from_sample
from utils.postprocess import (
    ChallengePostprocessor,
    P0ClusterFilter,
    P0ClusterFilterConfig,
    P18ScoreTrackRecovery,
    P18ScoreTrackRecoveryConfig,
)
from utils.temporal_frame_inference import temporal_frame_video_from_sample
from utils.temporal_memory_inference import (
    load_temporal_memory_model,
    predict_temporal_memory_scores,
)


DEFAULT_PRIMARY_CHECKPOINT = 'checkpoints/m15_e3_low_lr_epoch_008_seed43.pt'
DEFAULT_SECONDARY_CHECKPOINT = 'checkpoints/m10_dense_views2_epoch_002_seed42.pt'
DEFAULT_CACHE = 'log/analysis/m15_e8_m10low30000_raw_scores.pt'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sweep focused P0 and density-threshold rules for temporal memory.'
    )
    parser.add_argument('--primary-checkpoint', default=DEFAULT_PRIMARY_CHECKPOINT)
    parser.add_argument('--secondary-checkpoint', default=DEFAULT_SECONDARY_CHECKPOINT)
    parser.add_argument('--secondary-max-events', type=int, default=30000)
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    parser.add_argument(
        '--blend-cache',
        default='',
        help='Aligned routed raw-score cache to blend with --cache in blend phase.',
    )
    parser.add_argument(
        '--phase',
        choices=('focused', 'p18', 'p18_refine', 'p18_final', 'blend'),
        default='focused',
    )
    args, _ = parser.parse_known_args()
    if args.secondary_max_events < 0:
        parser.error('--secondary-max-events must be non-negative.')
    if args.phase == 'blend' and not args.blend_cache:
        parser.error('--phase blend requires --blend-cache.')
    if args.phase != 'blend' and args.blend_cache:
        parser.error('--blend-cache is only valid with --phase blend.')
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cache_metadata(primary_path, secondary_path, secondary_max_events):
    return {
        'primary_checkpoint': str(primary_path),
        'primary_sha256': sha256(primary_path),
        'secondary_checkpoint': str(secondary_path) if secondary_path else '',
        'secondary_sha256': sha256(secondary_path) if secondary_path else '',
        'secondary_max_events': int(secondary_max_events),
        'dataset_root': str(Path(cfg.root).resolve()),
        'temporal_bin_size': int(cfg.temporal_memory_bin_size),
    }


def build_or_load_cache(args):
    """Return CPU score records for the configured routed temporal-memory model."""
    primary_path = Path(args.primary_checkpoint).resolve()
    secondary_path = (
        Path(args.secondary_checkpoint).resolve()
        if args.secondary_checkpoint else None
    )
    cache_path = Path(args.cache).resolve()
    if not primary_path.is_file():
        raise FileNotFoundError('Primary checkpoint does not exist: {}'.format(primary_path))
    if secondary_path is not None and not secondary_path.is_file():
        raise FileNotFoundError(
            'Secondary checkpoint does not exist: {}'.format(secondary_path)
        )

    metadata = cache_metadata(
        primary_path,
        secondary_path,
        args.secondary_max_events,
    )
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location='cpu')
        if cache.get('metadata') == metadata:
            print('loaded raw-score cache:', cache_path)
            return cache['records']
        print('raw-score cache metadata differs; rebuilding:', cache_path)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to build a temporal-memory score cache.')
    if int(cfg.temporal_memory_context_bins) % 2 == 0:
        raise ValueError('TEMPORAL_MEMORY.temporal_memory_context_bins must be odd.')
    if int(cfg.temporal_memory_sequence_length) <= 1:
        raise ValueError('TEMPORAL_MEMORY.temporal_memory_sequence_length must exceed one.')

    device = torch.device('cuda:0')
    primary_model, _ = load_temporal_memory_model(
        str(primary_path),
        device,
        cfg.temporal_memory_context_bins,
        cfg.temporal_memory_width,
        cfg.temporal_memory_sequence_length,
    )
    secondary_model = None
    if secondary_path is not None and args.secondary_max_events > 0:
        secondary_model, _ = load_temporal_memory_model(
            str(secondary_path),
            device,
            cfg.temporal_memory_context_bins,
            cfg.temporal_memory_width,
            cfg.temporal_memory_sequence_length,
        )

    dataset = EvUAV(cfg, mode='val')
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError('No validation files found in: {}'.format(dataset.root))

    records = []
    for video_index in tqdm.trange(len(dataset), desc='temporal-memory cache', unit='video'):
        sample = dataset[video_index]
        batch = evaluation_batch_from_sample(sample)
        event_count = len(sample['ev_loc'])
        frame_video = temporal_frame_video_from_sample(
            sample,
            cfg.temporal_memory_bin_size,
            cfg.whole_t,
        )
        scores = predict_temporal_memory_scores(
            primary_model,
            frame_video,
            device,
            cfg.temporal_memory_context_bins,
            cfg.res[0],
            cfg.res[1],
            cfg.temporal_memory_inference_batch_size,
            cfg.temporal_memory_log_count_clip,
        )
        routed_to_secondary = (
            secondary_model is not None
            and event_count <= args.secondary_max_events
        )
        if routed_to_secondary:
            scores = predict_temporal_memory_scores(
                secondary_model,
                frame_video,
                device,
                cfg.temporal_memory_context_bins,
                cfg.res[0],
                cfg.res[1],
                cfg.temporal_memory_inference_batch_size,
                cfg.temporal_memory_log_count_clip,
            )
        scores = scores.reshape(-1).cpu().contiguous()
        if scores.numel() != batch['locs'].shape[0]:
            raise RuntimeError(
                'Prediction/event count mismatch for {}'.format(dataset.file_list[video_index])
            )
        records.append({
            'name': dataset.file_list[video_index],
            'event_count': int(event_count),
            'routed_to_secondary': bool(routed_to_secondary),
            'scores': scores,
            'seg_label': batch['seg_label'].cpu().contiguous(),
            'locs': batch['locs'].cpu().contiguous(),
            'idx_label': batch['idx_label'],
        })
        del sample, batch, frame_video, scores
        torch.cuda.empty_cache()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'metadata': metadata, 'records': records}, cache_path)
    print('wrote raw-score cache:', cache_path)
    return records


def load_blend_records(path):
    """Load an existing CPU raw-score cache used as the blend counterpart."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError('Blend cache does not exist: {}'.format(path))
    cache = torch.load(path, map_location='cpu')
    records = cache.get('records')
    if not isinstance(records, list) or not records:
        raise ValueError('Blend cache has no usable records: {}'.format(path))
    print('loaded blend raw-score cache:', path)
    return records


def blend_records(primary_records, secondary_records, primary_weight):
    """Return primary-shaped records with score vectors blended in probability space."""
    if len(primary_records) != len(secondary_records):
        raise ValueError(
            'Raw-score cache length mismatch: {} vs {}.'.format(
                len(primary_records), len(secondary_records)
            )
        )
    primary_weight = float(primary_weight)
    blended_records = []
    for primary, secondary in zip(primary_records, secondary_records):
        for field in ('name', 'event_count', 'routed_to_secondary'):
            if primary[field] != secondary[field]:
                raise ValueError(
                    'Raw-score cache mismatch for {} field {!r}.'.format(
                        primary.get('name', '<unknown>'), field
                    )
                )
        if primary['scores'].shape != secondary['scores'].shape:
            raise ValueError(
                'Score shape mismatch for {}: {} vs {}.'.format(
                    primary['name'],
                    tuple(primary['scores'].shape),
                    tuple(secondary['scores'].shape),
                )
            )
        record = dict(primary)
        record['scores'] = (
            primary['scores'] * primary_weight
            + secondary['scores'] * (1.0 - primary_weight)
        )
        blended_records.append(record)
    return blended_records


def make_postprocessor(
    threshold,
    min_cluster_events=3,
    min_duration_bins=5,
    p18_candidate_floor=0.53,
    p18_max_event_count=30000,
    p18_overrides=None,
):
    """Return the fixed M15 P0c/P18 stack with one focused P0 variation."""
    p0_filter = P0ClusterFilter(
        P0ClusterFilterConfig(
            enabled=True,
            spatial_radius=2,
            temporal_bin_size=50,
            temporal_radius_bins=1,
            min_cluster_events=min_cluster_events,
            min_duration_bins=min_duration_bins,
            high_confidence_recovery_enabled=True,
            retain_min_score=0.95,
        ),
        threshold,
    )
    p18_options = {
        'enabled': True,
        'event_count_cutoff': 1,
        'max_event_count': p18_max_event_count,
        'candidate_floor': p18_candidate_floor,
        'spatial_radius': 2,
        'temporal_bin_size': 50,
        'max_link_distance': 6.0,
        'max_gap_bins': 1,
        'min_track_bins': 2,
        'restore_mode': 'best',
    }
    if p18_overrides:
        p18_options.update(p18_overrides)
    p18_recovery = P18ScoreTrackRecovery(
        P18ScoreTrackRecoveryConfig(**p18_options),
        threshold,
    )
    return ChallengePostprocessor(p0_filter, p18_recovery)


def candidate_threshold(record, static_threshold, density_rule):
    if density_rule is None:
        return static_threshold
    cutoff, low_threshold, high_threshold = density_rule
    if record['event_count'] > cutoff:
        return high_threshold
    return low_threshold


def evaluate_candidate(
    records,
    name,
    static_threshold=0.719,
    density_rule=None,
    min_cluster_events=3,
    min_duration_bins=5,
    p18_candidate_floor=0.53,
    p18_max_event_count=30000,
    p18_overrides=None,
):
    """Replay a candidate exactly, applying a binary output before aggregation."""
    postprocessors = {}
    evaluator = evalute(cfg)
    sample_number = 0
    started = time.monotonic()

    for record in records:
        threshold = candidate_threshold(record, static_threshold, density_rule)
        key = (
            threshold,
            min_cluster_events,
            min_duration_bins,
            p18_candidate_floor,
            p18_max_event_count,
            tuple(sorted((p18_overrides or {}).items())),
        )
        postprocessor = postprocessors.get(key)
        if postprocessor is None:
            postprocessor = make_postprocessor(
                threshold,
                min_cluster_events=min_cluster_events,
                min_duration_bins=min_duration_bins,
                p18_candidate_floor=p18_candidate_floor,
                p18_max_event_count=p18_max_event_count,
                p18_overrides=p18_overrides,
            )
            postprocessors[key] = postprocessor
        predictions, _ = postprocessor.apply(
            record['scores'].clone(),
            record['locs'],
        )
        # With a per-video threshold, a binary vector lets the aggregate IoU
        # and accuracy use one fixed threshold without changing any decision.
        predictions = (predictions >= threshold).to(predictions.dtype)
        sample_number = add_batch_to_evaluator(
            evaluator,
            {
                'seg_label': record['seg_label'],
                'locs': record['locs'],
                'idx_label': record['idx_label'],
            },
            predictions,
            sample_number,
            prediction_threshold=0.5,
        )

    metrics = evaluate_challenge_metrics(evaluator, 0.5)
    elapsed = time.monotonic() - started
    print(
        '{:<44} Score={:.10f} Pd={:.10f} IoU={:.10f} Acc={:.10f} '
        'Fa={:.10e} time={:.1f}s'.format(
            name,
            metrics.score,
            metrics.pd,
            metrics.iou,
            metrics.acc,
            metrics.fa,
            elapsed,
        )
    )
    return metrics


def focused_candidates(phase):
    """Return a compact set that can finish after one raw-score inference pass."""
    if phase == 'p18_final':
        candidates = []
        for name, overrides in (
            (
                'p18_s5_track4_link7',
                {
                    'spatial_radius': 5,
                    'min_track_bins': 4,
                    'max_link_distance': 7.0,
                },
            ),
            (
                'p18_s5_track4_link8',
                {
                    'spatial_radius': 5,
                    'min_track_bins': 4,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_s5_track4_link9',
                {
                    'spatial_radius': 5,
                    'min_track_bins': 4,
                    'max_link_distance': 9.0,
                },
            ),
            (
                'p18_s5_track5_link8',
                {
                    'spatial_radius': 5,
                    'min_track_bins': 5,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_s5_track5_link9',
                {
                    'spatial_radius': 5,
                    'min_track_bins': 5,
                    'max_link_distance': 9.0,
                },
            ),
        ):
            candidates.append((
                name,
                0.719,
                (30000, 0.718, 0.719),
                3,
                5,
                0.53,
                35000,
                overrides,
            ))
        return candidates

    if phase == 'p18_refine':
        candidates = []
        for name, overrides in (
            (
                'p18_s3_track3_link7',
                {
                    'spatial_radius': 3,
                    'min_track_bins': 3,
                    'max_link_distance': 7.0,
                },
            ),
            (
                'p18_s3_track3_link8',
                {
                    'spatial_radius': 3,
                    'min_track_bins': 3,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_s3_track3_link9',
                {
                    'spatial_radius': 3,
                    'min_track_bins': 3,
                    'max_link_distance': 9.0,
                },
            ),
            (
                'p18_s3_track3_link10',
                {
                    'spatial_radius': 3,
                    'min_track_bins': 3,
                    'max_link_distance': 10.0,
                },
            ),
            (
                'p18_s2_track3_link8',
                {
                    'spatial_radius': 2,
                    'min_track_bins': 3,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_s4_track3_link8',
                {
                    'spatial_radius': 4,
                    'min_track_bins': 3,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_s3_track4_link8',
                {
                    'spatial_radius': 3,
                    'min_track_bins': 4,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_s4_track3_link7',
                {
                    'spatial_radius': 4,
                    'min_track_bins': 3,
                    'max_link_distance': 7.0,
                },
            ),
            (
                'p18_s4_track3_link9',
                {
                    'spatial_radius': 4,
                    'min_track_bins': 3,
                    'max_link_distance': 9.0,
                },
            ),
            (
                'p18_s4_track3_link10',
                {
                    'spatial_radius': 4,
                    'min_track_bins': 3,
                    'max_link_distance': 10.0,
                },
            ),
            (
                'p18_s4_track4_link8',
                {
                    'spatial_radius': 4,
                    'min_track_bins': 4,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_s5_track3_link8',
                {
                    'spatial_radius': 5,
                    'min_track_bins': 3,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_s6_track3_link8',
                {
                    'spatial_radius': 6,
                    'min_track_bins': 3,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_s5_track2_link8',
                {
                    'spatial_radius': 5,
                    'min_track_bins': 2,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_s5_track3_link7',
                {
                    'spatial_radius': 5,
                    'min_track_bins': 3,
                    'max_link_distance': 7.0,
                },
            ),
            (
                'p18_s5_track3_link9',
                {
                    'spatial_radius': 5,
                    'min_track_bins': 3,
                    'max_link_distance': 9.0,
                },
            ),
            (
                'p18_s5_track4_link8',
                {
                    'spatial_radius': 5,
                    'min_track_bins': 4,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_s6_track4_link8',
                {
                    'spatial_radius': 6,
                    'min_track_bins': 4,
                    'max_link_distance': 8.0,
                },
            ),
        ):
            candidates.append((
                name,
                0.719,
                (30000, 0.718, 0.719),
                3,
                5,
                0.53,
                35000,
                overrides,
            ))
        return candidates

    if phase == 'p18':
        candidates = [
            ('p18_baseline_f053_max30000', 0.719, None, 3, 5, 0.53, 30000),
        ]
        for candidate_floor in (0.50, 0.51, 0.52, 0.54, 0.55):
            candidates.append((
                'p18_floor_{:.2f}'.format(candidate_floor),
                0.719,
                None,
                3,
                5,
                candidate_floor,
                30000,
            ))
        for max_event_count in (10000, 15000, 20000, 25000, 35000, 40000, 60000):
            candidates.append((
                'p18_max_events_{}'.format(max_event_count),
                0.719,
                None,
                3,
                5,
                0.53,
                max_event_count,
            ))
        candidates.append((
            'p18_max_events_35000_m10low_t0718',
            0.719,
            (30000, 0.718, 0.719),
            3,
            5,
            0.53,
            35000,
        ))
        for name, overrides in (
            ('p18_link_distance_4', {'max_link_distance': 4.0}),
            ('p18_link_distance_5', {'max_link_distance': 5.0}),
            ('p18_link_distance_7', {'max_link_distance': 7.0}),
            ('p18_link_distance_8', {'max_link_distance': 8.0}),
            ('p18_max_gap_bins_2', {'max_gap_bins': 2}),
            ('p18_min_track_bins_3', {'min_track_bins': 3}),
            ('p18_spatial_radius_1', {'spatial_radius': 1}),
            ('p18_spatial_radius_3', {'spatial_radius': 3}),
            (
                'p18_spatial_radius_3_min_track_bins_3',
                {'spatial_radius': 3, 'min_track_bins': 3},
            ),
            (
                'p18_spatial_radius_3_link_distance_8',
                {'spatial_radius': 3, 'max_link_distance': 8.0},
            ),
            (
                'p18_spatial_radius_3_min_track_bins_3_link_distance_8',
                {
                    'spatial_radius': 3,
                    'min_track_bins': 3,
                    'max_link_distance': 8.0,
                },
            ),
            (
                'p18_spatial_radius_3_max_gap_bins_2',
                {'spatial_radius': 3, 'max_gap_bins': 2},
            ),
        ):
            candidates.append((
                name,
                0.719,
                (30000, 0.718, 0.719),
                3,
                5,
                0.53,
                35000,
                overrides,
            ))
        return candidates

    candidates = [
        ('baseline_t0719', 0.719, None, 3, 5, 0.53, 30000),
        ('p0_min_events2_t0719', 0.719, None, 2, 5, 0.53, 30000),
        ('p0_min_events4_t0719', 0.719, None, 4, 5, 0.53, 30000),
        ('p0_min_duration4_t0719', 0.719, None, 3, 4, 0.53, 30000),
        ('p0_min_duration6_t0719', 0.719, None, 3, 6, 0.53, 30000),
    ]
    for cutoff in (20000, 30000, 40000):
        for low_threshold in (0.718, 0.719):
            for high_threshold in (0.719, 0.720):
                candidates.append((
                    'density_c{}_l{:.3f}_h{:.3f}'.format(
                        cutoff,
                        low_threshold,
                        high_threshold,
                    ),
                    0.719,
                    (cutoff, low_threshold, high_threshold),
                    3,
                    5,
                    0.53,
                    30000,
                ))
    # M10 is used exactly in this low-density range. Sweep it independently
    # while holding M15 at its already-selected threshold.
    for low_threshold in (
        0.700,
        0.705,
        0.710,
        0.712,
        0.714,
        0.716,
        0.717,
        0.7172,
        0.7174,
        0.7176,
        0.7178,
        0.718,
        0.7182,
        0.7184,
        0.7186,
        0.7188,
        0.719,
    ):
        candidates.append((
            'm10low_c30000_t{:.3f}'.format(low_threshold),
            0.719,
            (30000, low_threshold, 0.719),
            3,
            5,
            0.53,
            30000,
        ))
    return candidates


def blend_candidates():
    """Probe calibrated M15/M16 mixtures with the frozen production policy."""
    candidates = []
    for primary_weight in (
        0.76, 0.78, 0.79, 0.80, 0.81, 0.82, 0.84, 0.86, 0.88,
    ):
        for high_threshold in (
            0.7175, 0.7180, 0.7185, 0.7188, 0.7190, 0.7192, 0.7194,
            0.7196, 0.7198, 0.7200,
        ):
            candidates.append((
                'm15w{:.2f}_high_t{:.3f}'.format(
                    primary_weight, high_threshold
                ),
                primary_weight,
                0.719,
                (30000, 0.718, high_threshold),
                3,
                5,
                0.53,
                35000,
                {
                    'spatial_radius': 5,
                    'max_link_distance': 8.0,
                    'min_track_bins': 4,
                },
            ))
    return candidates


def run_blend_phase(primary_records, secondary_records):
    """Evaluate all score mixtures while retaining shared low-density routing."""
    results = []
    for candidate in blend_candidates():
        (
            name,
            primary_weight,
            static_threshold,
            density_rule,
            min_events,
            min_duration,
            p18_candidate_floor,
            p18_max_event_count,
            p18_overrides,
        ) = candidate
        metrics = evaluate_candidate(
            blend_records(primary_records, secondary_records, primary_weight),
            name,
            static_threshold=static_threshold,
            density_rule=density_rule,
            min_cluster_events=min_events,
            min_duration_bins=min_duration,
            p18_candidate_floor=p18_candidate_floor,
            p18_max_event_count=p18_max_event_count,
            p18_overrides=p18_overrides,
        )
        results.append((metrics.score, name, metrics))

    score, name, metrics = max(results, key=lambda item: item[0])
    print(
        'BEST {} Score={:.10f} Pd={:.10f} IoU={:.10f} Acc={:.10f} Fa={:.10e}'.format(
            name,
            score,
            metrics.pd,
            metrics.iou,
            metrics.acc,
            metrics.fa,
        )
    )


def main():
    args = parse_args()
    if not cfg.eval or not cfg.roc:
        raise ValueError('Set TEST.eval=true and TEST.roc=true.')
    records = build_or_load_cache(args)
    routed_videos = sum(record['routed_to_secondary'] for record in records)
    print(
        'cached videos: {}, events: {}, secondary-routed videos: {}'.format(
            len(records),
            sum(record['event_count'] for record in records),
            routed_videos,
        )
    )
    if args.phase == 'blend':
        run_blend_phase(records, load_blend_records(args.blend_cache))
        return

    results = []
    for candidate in focused_candidates(args.phase):
        (
            name,
            static_threshold,
            density_rule,
            min_events,
            min_duration,
            p18_candidate_floor,
            p18_max_event_count,
        ) = candidate[:7]
        p18_overrides = candidate[7] if len(candidate) > 7 else None
        metrics = evaluate_candidate(
            records,
            name,
            static_threshold=static_threshold,
            density_rule=density_rule,
            min_cluster_events=min_events,
            min_duration_bins=min_duration,
            p18_candidate_floor=p18_candidate_floor,
            p18_max_event_count=p18_max_event_count,
            p18_overrides=p18_overrides,
        )
        results.append((metrics.score, name, metrics))

    score, name, metrics = max(results, key=lambda item: item[0])
    print(
        'BEST {} Score={:.10f} Pd={:.10f} IoU={:.10f} Acc={:.10f} Fa={:.10e}'.format(
            name,
            score,
            metrics.pd,
            metrics.iou,
            metrics.acc,
            metrics.fa,
        )
    )


if __name__ == '__main__':
    main()
