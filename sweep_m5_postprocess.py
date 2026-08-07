"""Cache M5 validation scores and compare global post-processing candidates.

The script deliberately never branches on a video name, index, or label.  It
performs the expensive temporal-memory forward pass once, then replays the
official evaluator on cached scores for each global candidate.  This makes
P18/P0b comparisons practical while keeping the selected rule reproducible.

Environment variables:
  M5_CKPT        checkpoint to evaluate (defaults to the released checkpoint)
  M5_SWEEP_CACHE cache path for raw validation scores
  M5_SWEEP_PHASE all, baseline, p18, p0b, focus, or cap64_filter
  (defaults to all)
"""

import hashlib
import os
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
    P0bTrackFilter,
    P0bTrackFilterConfig,
    P18ScoreTrackRecovery,
    P18ScoreTrackRecoveryConfig,
)
from utils.temporal_frame_inference import temporal_frame_video_from_sample
from utils.temporal_memory_inference import (
    load_temporal_memory_model,
    predict_temporal_memory_scores,
)


DEFAULT_CHECKPOINT = 'checkpoints/m4_dacc_m5_best_loss_seed42.pt'
DEFAULT_CACHE = 'log/analysis/m5_validation_scores_seed42.pt'


def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_matches(cache, checkpoint, checksum):
    metadata = cache.get('metadata', {})
    return (
        metadata.get('checkpoint') == str(Path(checkpoint).resolve())
        and metadata.get('sha256') == checksum
        and metadata.get('dataset_root') == str(Path(cfg.root).resolve())
        and metadata.get('temporal_bin_size') == int(cfg.temporal_memory_bin_size)
    )


def build_or_load_cache(checkpoint, cache_path):
    """Return CPU scores and evaluator fields for every validation video."""
    checkpoint = Path(checkpoint).resolve()
    cache_path = Path(cache_path).resolve()
    checksum = _checkpoint_sha256(checkpoint)
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location='cpu')
        if _cache_matches(cached, checkpoint, checksum):
            print('loaded score cache:', cache_path)
            return cached['records']
        print('cache metadata does not match checkpoint/data; rebuilding:', cache_path)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to build the M5 score cache.')
    device = torch.device('cuda:0')
    model, _ = load_temporal_memory_model(
        checkpoint,
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
    for video_index in tqdm.trange(len(dataset), desc='M5 cache', unit='video'):
        sample = dataset[video_index]
        batch = evaluation_batch_from_sample(sample)
        frame_video = temporal_frame_video_from_sample(
            sample,
            cfg.temporal_memory_bin_size,
            cfg.whole_t,
        )
        scores = predict_temporal_memory_scores(
            model,
            frame_video,
            device,
            cfg.temporal_memory_context_bins,
            cfg.res[0],
            cfg.res[1],
            cfg.temporal_memory_inference_batch_size,
            cfg.temporal_memory_log_count_clip,
        ).reshape(-1).cpu().contiguous()
        if scores.numel() != batch['locs'].shape[0]:
            raise RuntimeError('M5 prediction/event count mismatch for {}'.format(
                dataset.file_list[video_index]
            ))
        records.append({
            'name': dataset.file_list[video_index],
            'event_count': int(scores.numel()),
            'scores': scores,
            'seg_label': batch['seg_label'].cpu().contiguous(),
            'locs': batch['locs'].cpu().contiguous(),
            'idx_label': batch['idx_label'],
        })
        del sample, batch, frame_video, scores
        torch.cuda.empty_cache()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'metadata': {
            'checkpoint': str(checkpoint),
            'sha256': checksum,
            'dataset_root': str(Path(cfg.root).resolve()),
            'temporal_bin_size': int(cfg.temporal_memory_bin_size),
        },
        'records': records,
    }, cache_path)
    print('wrote score cache:', cache_path)
    return records


def _p0_postprocessor(threshold, min_duration_bins, retain_min_score, p18=None):
    p0 = P0ClusterFilter(
        P0ClusterFilterConfig(
            enabled=True,
            spatial_radius=2,
            temporal_bin_size=50,
            temporal_radius_bins=1,
            min_cluster_events=3,
            min_duration_bins=min_duration_bins,
            high_confidence_recovery_enabled=True,
            retain_min_score=retain_min_score,
        ),
        threshold,
    )
    recovery = P18ScoreTrackRecovery(
        p18 if p18 is not None else P18ScoreTrackRecoveryConfig(enabled=False),
        threshold,
    )
    return ChallengePostprocessor(p0, recovery)


def _p0b_postprocessor(threshold, spatial_radius, link_distance, gap_bins, min_events, min_frames):
    p0b = P0bTrackFilter(
        P0bTrackFilterConfig(
            enabled=True,
            spatial_radius=spatial_radius,
            temporal_bin_size=50,
            max_link_distance=link_distance,
            max_gap_bins=gap_bins,
            min_track_events=min_events,
            min_track_frames=min_frames,
        ),
        threshold,
    )
    return ChallengePostprocessor(
        p0b,
        P18ScoreTrackRecovery(P18ScoreTrackRecoveryConfig(enabled=False), threshold),
    )


def _p18_config(
    candidate_floor,
    restore_mode='best',
    cap=0,
    max_event_count=0,
    event_count_cutoff=1,
):
    return P18ScoreTrackRecoveryConfig(
        enabled=True,
        # Test every validation video under the same rule.  This is intentionally
        # not keyed to video identity or validation labels.
        event_count_cutoff=event_count_cutoff,
        max_event_count=max_event_count,
        candidate_floor=candidate_floor,
        spatial_radius=2,
        temporal_bin_size=50,
        max_link_distance=6.0,
        max_gap_bins=1,
        min_track_bins=2,
        restore_mode=restore_mode,
        max_restore_events_per_component=cap,
    )


def candidate_specs(phase):
    specs = []
    if phase in {'all', 'baseline', 'focus'}:
        for threshold in (0.68, 0.70, 0.72):
            specs.append((
                'p0_mdb5_retain092_t{:.2f}'.format(threshold),
                threshold,
                _p0_postprocessor(threshold, 5, 0.92),
            ))
        specs.append(('p0_mdb6_retain090_t070', 0.70, _p0_postprocessor(0.70, 6, 0.90)))

    if phase in {'all', 'p18'}:
        for min_duration_bins, retain_min_score, base_name in (
            (5, 0.92, 'p0_mdb5_retain092'),
            (6, 0.90, 'p0_mdb6_retain090'),
        ):
            for candidate_floor in (0.55, 0.60, 0.65):
                specs.append((
                    '{}_p18_best_f{:.2f}'.format(base_name, candidate_floor),
                    0.70,
                    _p0_postprocessor(
                        0.70,
                        min_duration_bins,
                        retain_min_score,
                        _p18_config(candidate_floor),
                    ),
                ))
        specs.extend((
            (
                'p0_mdb5_retain092_p18_topk2_f060',
                0.70,
                _p0_postprocessor(0.70, 5, 0.92, _p18_config(0.60, 'topk', 2)),
            ),
            (
                'p0_mdb6_retain090_p18_component_cap2_f060',
                0.70,
                _p0_postprocessor(0.70, 6, 0.90, _p18_config(0.60, 'component', 2)),
            ),
        ))

    if phase == 'focus':
        specs.append((
            'p0_mdb5_retain092_p18_best_f0.65',
            0.70,
            _p0_postprocessor(0.70, 5, 0.92, _p18_config(0.65)),
        ))

    if phase == 'cap64_filter':
        for threshold in (0.70, 0.72, 0.74):
            for min_duration_bins in (5, 6, 7, 8):
                for retain_min_score in (0.88, 0.90, 0.92, 0.94):
                    specs.append((
                        'cap64_t{:.2f}_mdb{}_retain{:.2f}'.format(
                            threshold, min_duration_bins, retain_min_score
                        ),
                        threshold,
                        _p0_postprocessor(
                            threshold,
                            min_duration_bins,
                            retain_min_score,
                        ),
                    ))

    if phase in {'all', 'p0b'}:
        for threshold in (0.68, 0.70, 0.72):
            specs.append((
                'p0b_sr2_d6_g1_e3_f2_t{:.2f}'.format(threshold),
                threshold,
                _p0b_postprocessor(threshold, 2, 6.0, 1, 3, 2),
            ))
        specs.extend((
            ('p0b_sr1_d4_g1_e3_f2_t070', 0.70, _p0b_postprocessor(0.70, 1, 4.0, 1, 3, 2)),
            ('p0b_sr2_d4_g1_e3_f2_t070', 0.70, _p0b_postprocessor(0.70, 2, 4.0, 1, 3, 2)),
            ('p0b_sr2_d6_g1_e3_f1_t070', 0.70, _p0b_postprocessor(0.70, 2, 6.0, 1, 3, 1)),
            ('p0b_sr2_d8_g2_e3_f2_t070', 0.70, _p0b_postprocessor(0.70, 2, 8.0, 2, 3, 2)),
        ))
    return specs


def evaluate_candidate(records, name, threshold, postprocessor):
    evaluator = evalute(cfg)
    stats = postprocessor.new_stats()
    sample_number = 0
    started = time.monotonic()
    for record in records:
        predictions, video_stats = postprocessor.apply(
            record['scores'].clone(),
            record['locs'],
        )
        stats.merge(video_stats)
        batch = {
            'seg_label': record['seg_label'],
            'locs': record['locs'],
            'idx_label': record['idx_label'],
        }
        sample_number = add_batch_to_evaluator(
            evaluator,
            batch,
            predictions,
            sample_number,
            threshold,
        )
    metrics = evaluate_challenge_metrics(evaluator, threshold)
    elapsed = time.monotonic() - started
    print(
        '{:<48} Score={:.10f} Pd={:.10f} IoU={:.10f} Acc={:.10f} '
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
    print('  ', stats.summary())
    return metrics.score


def main():
    if not cfg.eval or not cfg.roc:
        raise ValueError('Set TEST.eval=true and TEST.roc=true.')
    checkpoint = os.environ.get('M5_CKPT', DEFAULT_CHECKPOINT)
    cache_path = os.environ.get('M5_SWEEP_CACHE', DEFAULT_CACHE)
    phase = os.environ.get('M5_SWEEP_PHASE', 'all').strip().lower()
    if phase not in {
        'all', 'baseline', 'p18', 'p0b', 'focus', 'cap64_filter'
    }:
        raise ValueError(
            'M5_SWEEP_PHASE must be all, baseline, p18, p0b, focus, '
            'or cap64_filter.'
        )
    if not Path(checkpoint).is_file():
        raise FileNotFoundError('M5 checkpoint does not exist: {}'.format(checkpoint))
    records = build_or_load_cache(checkpoint, cache_path)
    print('cached videos: {}, events: {}'.format(
        len(records), sum(record['event_count'] for record in records)
    ))
    scores = []
    for name, threshold, postprocessor in candidate_specs(phase):
        scores.append((evaluate_candidate(records, name, threshold, postprocessor), name))
    best_score, best_name = max(scores)
    print('BEST {} Score={:.10f}'.format(best_name, best_score))


if __name__ == '__main__':
    main()
