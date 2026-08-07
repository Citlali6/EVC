"""Evaluate global event-count-adaptive thresholds on cached M5 scores.

The released M5 validation diagnostic shows a large quality gap between sparse
and high-event-count videos.  This script tests only a global, label-free
policy: a video uses ``high_threshold`` when its event count exceeds one
configured cutoff, otherwise ``low_threshold``.  P0/P0c remains fixed at the
released validation setting (mdb=5, retain=0.92).
"""

import os
import time
from pathlib import Path

import torch

from configs.configs import cfg
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.eval import evalute
from sweep_m5_postprocess import _p0_postprocessor


def evaluate_policy(records, cutoff, low_threshold, high_threshold):
    evaluator = evalute(cfg)
    stats = None
    sample_number = 0
    usage = {low_threshold: 0, high_threshold: 0}
    started = time.monotonic()

    for record in records:
        threshold = (
            high_threshold
            if record['event_count'] > cutoff
            else low_threshold
        )
        postprocessor = _p0_postprocessor(threshold, 5, 0.92)
        predictions, video_stats = postprocessor.apply(
            record['scores'].clone(),
            record['locs'],
        )
        if stats is None:
            stats = postprocessor.new_stats()
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
        usage[threshold] = usage.get(threshold, 0) + 1

    # Every retained prediction is at least the branch threshold, and every
    # branch threshold is >= 0.70.  Therefore 0.70 reproduces each branch's
    # binary decisions for the final semantic metrics.  Pd/Fa were accumulated
    # above using each video's actual threshold.
    metrics = evaluate_challenge_metrics(evaluator, 0.70)
    elapsed = time.monotonic() - started
    return metrics, usage, stats, elapsed


def main():
    cache_path = Path(os.environ.get(
        'M5_SWEEP_CACHE', 'log/analysis/m5_validation_scores_seed42.pt'
    ))
    if not cache_path.is_file():
        raise FileNotFoundError('Score cache does not exist: {}'.format(cache_path))
    if not cfg.eval or not cfg.roc:
        raise ValueError('Set TEST.eval=true and TEST.roc=true.')
    cache = torch.load(cache_path, map_location='cpu')
    records = cache['records']

    candidates = [('static_t{:.2f}'.format(value), None, value, value)
                  for value in (0.66, 0.68, 0.70, 0.72, 0.74)]
    for cutoff in (50000, 70000, 100000, 150000, 200000, 250000, 300000):
        for high_threshold in (0.70, 0.72, 0.74, 0.76, 0.78, 0.80):
            candidates.append((
                'density_cut{}_low070_high{:.2f}'.format(cutoff, high_threshold),
                cutoff,
                0.70,
                high_threshold,
            ))

    results = []
    for name, cutoff, low_threshold, high_threshold in candidates:
        # A static policy is expressed with cutoff=0 so every non-empty video
        # enters the high branch, whose threshold equals the low branch.
        effective_cutoff = 0 if cutoff is None else cutoff
        metrics, usage, stats, elapsed = evaluate_policy(
            records,
            effective_cutoff,
            low_threshold,
            high_threshold,
        )
        usage_text = ','.join(
            '{:.2f}:{}'.format(threshold, count)
            for threshold, count in sorted(usage.items())
        )
        print(
            '{:<45} Score={:.10f} Pd={:.10f} IoU={:.10f} Acc={:.10f} '
            'Fa={:.10e} usage={} time={:.1f}s'.format(
                name,
                metrics.score,
                metrics.pd,
                metrics.iou,
                metrics.acc,
                metrics.fa,
                usage_text,
                elapsed,
            )
        )
        results.append((metrics.score, name, metrics))

    score, name, metrics = max(results)
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
