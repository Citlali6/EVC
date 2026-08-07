"""Report positive-weight clipping in the M4/M5 temporal-memory training set."""

import os
from pathlib import Path

import numpy as np

from configs.configs import cfg


def summarize_video(path, bin_size, target_mass):
    with np.load(path) as data:
        locations = data['ev_loc']
        labels = data['ev']['label'].astype(np.int8, copy=False)
    bins = np.floor_divide(locations[:, 2], bin_size)
    total_events = int(labels.size)
    total_positive = int((labels > 0).sum())
    required_weights = []
    for temporal_bin in np.unique(bins):
        frame_labels = labels[bins == temporal_bin]
        positive_count = int((frame_labels > 0).sum())
        negative_count = int((frame_labels == 0).sum())
        if positive_count and negative_count:
            required_weights.append(
                negative_count / positive_count * target_mass / (1.0 - target_mass)
            )
    required = np.asarray(required_weights, dtype=np.float64)
    return {
        'name': path.name,
        'events': total_events,
        'positive_fraction': total_positive / max(total_events, 1),
        'positive_bins': int(required.size),
        'clip16': int((required > 16.0).sum()),
        'clip32': int((required > 32.0).sum()),
        'clip64': int((required > 64.0).sum()),
        'median_required': float(np.median(required)) if required.size else 1.0,
        'max_required': float(required.max()) if required.size else 1.0,
        'required': required,
    }


def main():
    root = Path(cfg.root) / 'train'
    files = sorted(root.glob('*.npz'))
    if not files:
        raise RuntimeError('No training npz files found in {}'.format(root))
    target_mass = float(cfg.temporal_memory_target_positive_loss_mass)
    if not 0.0 < target_mass < 1.0:
        raise ValueError('temporal_memory_target_positive_loss_mass must be in (0, 1).')
    rows = [
        summarize_video(path, int(cfg.temporal_memory_bin_size), target_mass)
        for path in files
    ]
    all_required = np.concatenate([row['required'] for row in rows if row['required'].size])
    print('training videos:', len(rows))
    print('positive temporal bins:', int(all_required.size))
    for limit in (16.0, 32.0, 64.0):
        clipped = int((all_required > limit).sum())
        print('required_weight > {:>2.0f}: {:>6} / {:>6} ({:.2%})'.format(
            limit, clipped, int(all_required.size), clipped / max(int(all_required.size), 1)
        ))
    print('required weight: median={:.2f}, p90={:.2f}, p99={:.2f}, max={:.2f}'.format(
        float(np.median(all_required)),
        float(np.quantile(all_required, 0.90)),
        float(np.quantile(all_required, 0.99)),
        float(all_required.max()),
    ))
    print('\nMost clipped videos')
    for row in sorted(rows, key=lambda item: (item['clip16'], item['events']), reverse=True)[:20]:
        print(
            '{:<13} events={:<8} pos={:.4%} bins={:<4} '
            'clip16={:<4} clip32={:<4} clip64={:<4} median={:>6.2f} max={:>7.2f}'.format(
                row['name'], row['events'], row['positive_fraction'],
                row['positive_bins'], row['clip16'], row['clip32'], row['clip64'],
                row['median_required'], row['max_required'],
            )
        )


if __name__ == '__main__':
    main()
