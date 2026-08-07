"""Diagnose where a cached temporal-memory model loses validation score.

This is an analysis-only companion to ``sweep_m5_postprocess.py``.  It reads
the cached raw scores, compares the released P0/P0c configuration with the
best P18 probe, and aggregates by event-count strata.  Video names are printed
for inspection only; no inference rule is built from them.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from configs.configs import cfg
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.eval import evalute
from sweep_m5_postprocess import _p0_postprocessor, _p18_config


def evaluate_records(records, postprocessor, threshold=0.70):
    evaluator = evalute(cfg)
    stats = postprocessor.new_stats()
    sample_number = 0
    per_video = []
    for record in records:
        predictions, video_stats = postprocessor.apply(
            record['scores'].clone(),
            record['locs'],
        )
        stats.merge(video_stats)
        single_evaluator = evalute(cfg)
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
        add_batch_to_evaluator(single_evaluator, batch, predictions, 0, threshold)
        per_video.append({
            'name': record['name'],
            'events': record['event_count'],
            'metrics': evaluate_challenge_metrics(single_evaluator, threshold),
            'target_count': int(single_evaluator.obj_num),
            'correct_count': int(single_evaluator.correct_num),
            'false_count': int(single_evaluator.false_num),
            'positive_events': int((predictions >= threshold).sum().item()),
            'restored_events': int(video_stats.recovery_stats.restored_events),
        })
    return evaluate_challenge_metrics(evaluator, threshold), stats, per_video


def print_metrics(label, metrics):
    print(
        '{} Score={:.10f} Pd={:.10f} IoU={:.10f} Acc={:.10f} Fa={:.10e}'.format(
            label,
            metrics.score,
            metrics.pd,
            metrics.iou,
            metrics.acc,
            metrics.fa,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description='Compare P0/P0c with a globally gated P18 recovery rule.'
    )
    parser.add_argument(
        '--cache',
        default=os.environ.get(
            'M5_SWEEP_CACHE', 'log/analysis/m5_validation_scores_seed42.pt'
        ),
        help='Raw-score cache created by sweep_m5_postprocess.py.',
    )
    parser.add_argument('--p18-floor', type=float, default=0.65)
    parser.add_argument('--p18-event-count-cutoff', type=int, default=1)
    parser.add_argument('--p18-max-event-count', type=int, default=0)
    args, _ = parser.parse_known_args()
    if not 0.0 <= args.p18_floor <= 1.0:
        parser.error('--p18-floor must be in [0, 1]')
    if args.p18_event_count_cutoff <= 0:
        parser.error('--p18-event-count-cutoff must be positive')
    if args.p18_max_event_count < 0:
        parser.error('--p18-max-event-count must be non-negative')

    cache_path = Path(args.cache)
    if not cache_path.is_file():
        raise FileNotFoundError('Score cache does not exist: {}'.format(cache_path))
    cache = torch.load(cache_path, map_location='cpu')
    records = cache['records']
    baseline = _p0_postprocessor(0.70, 5, 0.92)
    p18_config = _p18_config(
        args.p18_floor,
        max_event_count=args.p18_max_event_count,
        event_count_cutoff=args.p18_event_count_cutoff,
    )
    p18 = _p0_postprocessor(0.70, 5, 0.92, p18_config)
    base_metrics, _, base_per_video = evaluate_records(records, baseline)
    p18_metrics, p18_stats, p18_per_video = evaluate_records(records, p18)
    print_metrics('P0 baseline', base_metrics)
    print_metrics(
        'P18 events>{} and <= {} f={:.2f}'.format(
            args.p18_event_count_cutoff,
            args.p18_max_event_count or 'unbounded',
            args.p18_floor,
        ),
        p18_metrics,
    )
    print('P18 aggregate:', p18_stats.summary())

    print('\nPer-video changes (ascending baseline Pd)')
    joined = []
    for base, recovered in zip(base_per_video, p18_per_video):
        joined.append((base, recovered))
    for base, recovered in sorted(joined, key=lambda pair: pair[0]['metrics'].pd):
        print(
            '{:<11} events={:<8} targets={:<4} Pd {:.3f}->{:.3f} '
            'IoU {:.3f}->{:.3f} Acc {:.3f}->{:.3f} restored={:<3} '
            'score_delta={:+.6f}'.format(
                base['name'],
                base['events'],
                base['target_count'],
                base['metrics'].pd,
                recovered['metrics'].pd,
                base['metrics'].iou,
                recovered['metrics'].iou,
                base['metrics'].acc,
                recovered['metrics'].acc,
                recovered['restored_events'],
                recovered['metrics'].score - base['metrics'].score,
            )
        )

    event_counts = np.asarray([record['event_count'] for record in records])
    median = int(np.median(event_counts))
    print('\nEvent-count strata (median split at {})'.format(median))
    for label, mask in (
        ('at_or_below_median', event_counts <= median),
        ('above_median', event_counts > median),
    ):
        selected_records = [record for record, selected in zip(records, mask) if selected]
        group_base, _, _ = evaluate_records(selected_records, baseline)
        group_p18, _, _ = evaluate_records(selected_records, p18)
        print_metrics(label + ' baseline', group_base)
        print_metrics(label + ' P18', group_p18)
        print('{} score_delta={:+.10f}'.format(
            label, group_p18.score - group_base.score
        ))


if __name__ == '__main__':
    main()
