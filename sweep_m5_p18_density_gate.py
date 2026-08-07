"""Sweep a label-free event-count gate for P18 score-track recovery.

The M5 diagnostics show that P18 improves lower-density videos but can add
false positives in dense scenes. This script replays P0/P0c on cached raw
scores, enabling P18 only when a video's observable event count is no larger
than a global cutoff. No video identity or annotation is used by the policy.
"""

import argparse
from pathlib import Path

import torch

from configs.configs import cfg
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.eval import evalute
from sweep_m5_postprocess import _p0_postprocessor, _p18_config


def parse_numbers(raw, option_name, value_type):
    values = []
    for item in raw.split(','):
        try:
            values.append(value_type(item.strip()))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                '{} contains an invalid value: {!r}'.format(option_name, item)
            ) from error
    if not values:
        raise argparse.ArgumentTypeError('{} must not be empty.'.format(option_name))
    return tuple(values)


def evaluate_candidate(records, cutoff, candidate_floor):
    evaluator = evalute(cfg)
    base_stats = None
    recovery_stats = None
    sample_number = 0
    p18_videos = 0

    for record in records:
        use_p18 = cutoff > 0 and record['event_count'] <= cutoff
        postprocessor = _p0_postprocessor(
            threshold=0.70,
            min_duration_bins=5,
            retain_min_score=0.92,
            p18=_p18_config(candidate_floor) if use_p18 else None,
        )
        predictions, video_stats = postprocessor.apply(
            record['scores'].clone(),
            record['locs'],
        )
        if base_stats is None:
            base_stats = video_stats.base_stats
        else:
            base_stats.merge(video_stats.base_stats)
        if video_stats.recovery_stats.enabled:
            if recovery_stats is None:
                recovery_stats = video_stats.recovery_stats
            else:
                recovery_stats.merge(video_stats.recovery_stats)
        sample_number = add_batch_to_evaluator(
            evaluator,
            {
                'seg_label': record['seg_label'],
                'locs': record['locs'],
                'idx_label': record['idx_label'],
            },
            predictions,
            sample_number,
            0.70,
        )
        p18_videos += int(use_p18)

    return (
        evaluate_challenge_metrics(evaluator, 0.70),
        base_stats,
        recovery_stats,
        p18_videos,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a global event-count gate for P18 on cached M5 scores.'
    )
    parser.add_argument(
        '--cache',
        default='log/analysis/m5_validation_scores_seed42.pt',
        help='Cached raw-score file created by sweep_m5_postprocess.py.',
    )
    parser.add_argument(
        '--cutoffs',
        default='0,40000,60000,80000,100000,150000,200000,300000,999999999',
        help='Comma-separated P18 event-count cutoffs; 0 disables P18.',
    )
    parser.add_argument(
        '--candidate-floors',
        default='0.60,0.62,0.65',
        help='Comma-separated P18 candidate-score floors.',
    )
    # ``configs.configs`` consumes the shared --config/--set options at import
    # time.  Keep accepting them here so this standalone sweep can use the
    # same invocation pattern as validation and submission scripts.
    args, _ = parser.parse_known_args()
    cutoffs = parse_numbers(args.cutoffs, '--cutoffs', int)
    candidate_floors = parse_numbers(
        args.candidate_floors,
        '--candidate-floors',
        float,
    )
    if any(cutoff < 0 for cutoff in cutoffs):
        parser.error('--cutoffs values must be non-negative.')
    if any(not 0.0 <= floor <= 1.0 for floor in candidate_floors):
        parser.error('--candidate-floors values must be in [0, 1].')
    if not cfg.eval or not cfg.roc:
        raise ValueError('Set TEST.eval=true and TEST.roc=true.')

    cache_path = Path(args.cache)
    if not cache_path.is_file():
        raise FileNotFoundError('Score cache does not exist: {}'.format(cache_path))
    records = torch.load(cache_path, map_location='cpu')['records']
    print('cached videos: {}, events: {}'.format(
        len(records), sum(record['event_count'] for record in records)
    ))

    results = []
    for cutoff in cutoffs:
        floors = candidate_floors if cutoff else (candidate_floors[0],)
        for candidate_floor in floors:
            metrics, base_stats, recovery_stats, p18_videos = evaluate_candidate(
                records,
                cutoff,
                candidate_floor,
            )
            label = 'p18_off' if cutoff == 0 else (
                'p18_events_le_{}_floor_{:.2f}'.format(cutoff, candidate_floor)
            )
            print(
                '{:<38} Score={:.10f} Pd={:.10f} IoU={:.10f} Acc={:.10f} '
                'Fa={:.10e} p18_videos={}'.format(
                    label,
                    metrics.score,
                    metrics.pd,
                    metrics.iou,
                    metrics.acc,
                    metrics.fa,
                    p18_videos,
                )
            )
            print('  ', base_stats.summary())
            if recovery_stats is not None:
                print('  P18:', recovery_stats.summary())
            results.append((metrics.score, label, metrics))

    score, label, metrics = max(results, key=lambda item: item[0])
    print(
        'BEST {} Score={:.10f} Pd={:.10f} IoU={:.10f} Acc={:.10f} Fa={:.10e}'.format(
            label,
            score,
            metrics.pd,
            metrics.iou,
            metrics.acc,
            metrics.fa,
        )
    )


if __name__ == '__main__':
    main()
