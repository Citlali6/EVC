"""Pool the fixed H1/T160 + H2/T32 train-only routing diagnostic.

The input reports must have been produced by
``diagnose_temporal_memory_windowing.py`` from canonical ``train_*.npz``
sources.  This script performs CPU-only sufficient-count aggregation.  It
does not run inference and rejects reports that permit validation inputs.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from utils.challenge_eval import challenge_score


INPUT_SCHEMA = 'temporal-memory-window-train-diagnostic-v1'
OUTPUT_SCHEMA = 'temporal-memory-input-routed-train-summary-v1'
M20_SHA256 = '4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849'
TRAIN_NAME_PATTERN = re.compile(r'^train_[0-9]+\.npz$')
WIDTH = 346
HEIGHT = 260
POLARITY_MINORITY_CUTOFF = 0.20

H1_NAMES = tuple('train_{:03d}.npz'.format(index) for index in range(44, 48))
H2_NAMES = tuple('train_{:03d}.npz'.format(index) for index in range(88, 99))
FOLD_PLAN = (
    ('h1_044_045', H1_NAMES[:2]),
    ('h1_046_047', H1_NAMES[2:]),
    ('h2_088_091', H2_NAMES[:4]),
    ('h2_092_094', H2_NAMES[4:7]),
    ('h2_095_098', H2_NAMES[7:]),
)
COUNT_KEYS = (
    'true_positive_events',
    'false_positive_events',
    'false_negative_events',
    'true_negative_events',
    'correct_target_groups',
    'target_groups',
    'false_components',
    'frame_count',
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    path = Path(path).resolve()
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle), path


def validate_input_report(payload, path, expected_names):
    if payload.get('schema_version') != INPUT_SCHEMA:
        raise ValueError('{} has an unexpected schema.'.format(path))
    guardrails = payload.get('guardrails', {})
    if guardrails.get('validation_inputs_permitted') is not False:
        raise ValueError('{} does not fail closed against validation.'.format(path))
    if (
        guardrails.get('threshold_search') is not False
        or guardrails.get('checkpoint_search') is not False
        or guardrails.get('postprocess_search') is not False
    ):
        raise ValueError('{} is not a fixed diagnostic.'.format(path))
    if float(guardrails.get('prediction_threshold', -1)) != 0.719:
        raise ValueError('{} does not use the fixed 0.719 threshold.'.format(path))
    if guardrails.get('postprocess_profile') != 'released_M20_C00_fixed':
        raise ValueError('{} does not use released M20/C00.'.format(path))
    if payload.get('checkpoint', {}).get('sha256') != M20_SHA256:
        raise ValueError('{} does not use the released M20 checkpoint.'.format(path))
    identity = payload.get('identity_checks', {})
    if identity.get('all_bitwise_equal') is not True:
        raise ValueError('{} failed full-window identity.'.format(path))
    if int(identity.get('completed_video_count', 0)) < 1:
        raise ValueError('{} contains no real-checkpoint identity check.'.format(path))
    selected = tuple(payload.get('data', {}).get('selected_files', []))
    if selected != tuple(expected_names):
        raise ValueError(
            '{} selected files {} instead of {}.'.format(path, selected, expected_names)
        )
    if any(not TRAIN_NAME_PATTERN.fullmatch(name) for name in selected):
        raise ValueError('{} contains a non-train source name.'.format(path))
    records = payload.get('per_video', [])
    if tuple(record.get('name') for record in records) != tuple(expected_names):
        raise ValueError('{} per-video order does not match its manifest.'.format(path))
    for record in records:
        modes = record.get('modes', {})
        if not {'full_stream', 'window_t16', 'window_t32'}.issubset(modes):
            raise ValueError('{} is missing a required inference mode.'.format(path))
        for mode in ('full_stream', 'window_t16', 'window_t32'):
            counts = modes[mode].get('c00_evaluation', {}).get('counts', {})
            if set(COUNT_KEYS).difference(counts):
                raise ValueError(
                    '{} {} {} is missing sufficient counts.'.format(
                        path,
                        record['name'],
                        mode,
                    )
                )


def normalized_counts(value):
    counts = {key: int(value[key]) for key in COUNT_KEYS}
    if any(count < 0 for count in counts.values()):
        raise ValueError('Sufficient counts must be non-negative.')
    if (
        counts['true_positive_events'] + counts['false_negative_events'] <= 0
        or counts['target_groups'] <= 0
        or counts['frame_count'] <= 0
    ):
        raise ValueError('Sufficient counts have a zero metric denominator.')
    return counts


def sum_counts(values):
    total = {key: 0 for key in COUNT_KEYS}
    for value in values:
        counts = normalized_counts(value)
        for key in COUNT_KEYS:
            total[key] += counts[key]
    return total


def metrics_from_counts(counts):
    counts = normalized_counts(counts)
    true_positive = counts['true_positive_events']
    false_positive = counts['false_positive_events']
    false_negative = counts['false_negative_events']
    union = true_positive + false_positive + false_negative
    positive = true_positive + false_negative
    if union <= 0:
        raise ValueError('IoU denominator must be positive.')

    # Match utils.eval: semantic divisions happen in torch.float32, whereas
    # Pd and Fa use Python double division over pooled sufficient counts.
    iou = float(
        (
            torch.tensor(true_positive, dtype=torch.float32)
            / torch.tensor(union, dtype=torch.float32)
        ).item()
    )
    acc = float(
        (
            torch.tensor(true_positive, dtype=torch.float32)
            / torch.tensor(positive, dtype=torch.float32)
        ).item()
    )
    pd = counts['correct_target_groups'] / counts['target_groups']
    fa = counts['false_components'] / (
        counts['frame_count'] * WIDTH * HEIGHT
    )
    score_fa, score = challenge_score(iou, acc, pd, fa)
    if not all(math.isfinite(value) for value in (iou, acc, pd, fa, score_fa, score)):
        raise RuntimeError('Non-finite pooled metric.')
    return {
        'iou': iou,
        'acc': acc,
        'pd': pd,
        'fa': fa,
        'score_fa': score_fa,
        'score': score,
    }


def evaluation_from_counts(counts):
    counts = normalized_counts(counts)
    return {'metrics': metrics_from_counts(counts), 'counts': counts}


def evaluation_delta(baseline, candidate):
    return {
        'metrics': {
            key: candidate['metrics'][key] - baseline['metrics'][key]
            for key in baseline['metrics']
        },
        'counts': {
            key: candidate['counts'][key] - baseline['counts'][key]
            for key in COUNT_KEYS
        },
    }


def polarity_minority_fraction(path):
    path = Path(path)
    with np.load(path) as payload:
        event_input = np.asarray(payload['evs_norm'])
    if event_input.ndim != 2 or event_input.shape[1] < 4 or event_input.shape[0] <= 0:
        raise ValueError('{} has invalid evs_norm input.'.format(path))
    polarity = event_input[:, 3].astype(np.float64, copy=False) > 0.5
    positive_fraction = float(polarity.mean())
    return min(positive_fraction, 1.0 - positive_fraction)


def input_route_for_fraction(minority_fraction):
    if not 0.0 <= float(minority_fraction) <= 0.5:
        raise ValueError('Polarity minority fraction must be in [0, 0.5].')
    if float(minority_fraction) < POLARITY_MINORITY_CUTOFF:
        return {'domain': 'h1', 'mode': 'full_stream', 'window_length': None}
    return {'domain': 'h2', 'mode': 'window_t32', 'window_length': 32}


def record_counts(record, mode):
    return normalized_counts(record['modes'][mode]['c00_evaluation']['counts'])


def aggregate_records(records, mode_by_name):
    return evaluation_from_counts(
        sum_counts(
            record_counts(record, mode_by_name[record['name']])
            for record in records
        )
    )


def consistency_summary(items):
    gates = (
        'score_strictly_improved',
        'pd_non_decrease',
        'fa_non_increase',
        'iou_non_decrease',
    )
    return {
        gate: {
            'passed': int(sum(bool(item['consistency'][gate]) for item in items)),
            'total': len(items),
        }
        for gate in gates
    }


def atomic_json_write(path, payload, force=False):
    path = Path(path).resolve()
    if path.exists() and not force:
        raise FileExistsError('Output exists; pass --force to replace: {}'.format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)
    digest = sha256_file(path)
    sidecar = Path(str(path) + '.sha256')
    sidecar_temporary = Path(str(sidecar) + '.tmp')
    sidecar_temporary.write_text(
        '{}  {}\n'.format(digest, path.name),
        encoding='ascii',
    )
    sidecar_temporary.replace(sidecar)
    return path, digest, sidecar


def run(args):
    h1, h1_path = load_json(args.h1_report)
    h2, h2_path = load_json(args.h2_report)
    validate_input_report(h1, h1_path, H1_NAMES)
    validate_input_report(h2, h2_path, H2_NAMES)
    if h1['checkpoint'] != h2['checkpoint']:
        raise ValueError('H1 and H2 checkpoint identities differ.')
    if h1['inference'] != h2['inference']:
        raise ValueError('H1 and H2 inference configurations differ.')
    if h1['guardrails'] != h2['guardrails']:
        raise ValueError('H1 and H2 guardrails differ.')

    records = list(h1['per_video']) + list(h2['per_video'])
    records_by_name = {record['name']: record for record in records}
    if set(records_by_name) != set(H1_NAMES + H2_NAMES):
        raise ValueError('H1/H2 source population is incomplete or duplicated.')

    input_routes = {}
    input_route_records = []
    for record in records:
        source_path = Path(record['path']).resolve()
        if not source_path.is_file() or source_path.name != record['name']:
            raise FileNotFoundError(source_path)
        if sha256_file(source_path) != record['sha256']:
            raise ValueError('Raw train source SHA mismatch: {}'.format(source_path))
        minority_fraction = polarity_minority_fraction(source_path)
        route = input_route_for_fraction(minority_fraction)
        expected_domain = 'h1' if record['name'] in H1_NAMES else 'h2'
        if route['domain'] != expected_domain:
            raise RuntimeError(
                'Input-only route does not reproduce the source block for {}.'.format(
                    record['name']
                )
            )
        input_routes[record['name']] = route['mode']
        input_route_records.append(
            {
                'name': record['name'],
                'raw_source_path': str(source_path),
                'raw_source_sha256': record['sha256'],
                'polarity_minority_fraction': minority_fraction,
                **route,
            }
        )

    baseline_modes = {name: 'full_stream' for name in records_by_name}
    baseline = aggregate_records(records, baseline_modes)
    routed = aggregate_records(records, input_routes)
    routed_delta = evaluation_delta(baseline, routed)
    combined_replication = None
    combined_path = None
    if args.combined_report is not None:
        combined, combined_path = load_json(args.combined_report)
        if combined.get('schema_version') != INPUT_SCHEMA:
            raise ValueError('Combined replication report has an unexpected schema.')
        if combined.get('guardrails') != h1.get('guardrails'):
            raise ValueError('Combined replication guardrails differ.')
        if combined.get('checkpoint') != h1.get('checkpoint'):
            raise ValueError('Combined replication checkpoint differs.')
        if tuple(combined.get('data', {}).get('selected_files', [])) != (
            H1_NAMES + H2_NAMES
        ):
            raise ValueError('Combined replication source population differs.')
        combined_replication = combined.get('aggregate', {}).get(
            'full_stream', {}
        ).get('c00')
        if combined_replication != baseline:
            raise RuntimeError(
                'Separately generated combined baseline does not exactly match pooling.'
            )

    per_video = []
    for route_record in input_route_records:
        name = route_record['name']
        record = records_by_name[name]
        video_baseline = evaluation_from_counts(record_counts(record, 'full_stream'))
        video_routed = evaluation_from_counts(record_counts(record, input_routes[name]))
        delta = evaluation_delta(video_baseline, video_routed)
        per_video.append(
            {
                **route_record,
                'baseline': video_baseline,
                'routed': video_routed,
                'delta': delta,
                'runtime': {
                    'baseline_inference_seconds': record['modes']['full_stream'][
                        'inference_seconds'
                    ],
                    'routed_inference_seconds': record['modes'][input_routes[name]][
                        'inference_seconds'
                    ],
                    'baseline_peak_cuda_memory_bytes': record['modes']['full_stream'][
                        'peak_cuda_memory_bytes'
                    ],
                    'routed_peak_cuda_memory_bytes': record['modes'][input_routes[name]][
                        'peak_cuda_memory_bytes'
                    ],
                },
                'consistency': {
                    'score_strictly_improved': delta['metrics']['score'] > 0.0,
                    'pd_non_decrease': delta['metrics']['pd'] >= 0.0,
                    'fa_non_increase': delta['metrics']['fa'] <= 0.0,
                    'iou_non_decrease': delta['metrics']['iou'] >= 0.0,
                },
            }
        )

    folds = []
    for fold_id, names in FOLD_PLAN:
        fold_records = [records_by_name[name] for name in names]
        fold_baseline = aggregate_records(fold_records, baseline_modes)
        fold_routed = aggregate_records(fold_records, input_routes)
        delta = evaluation_delta(fold_baseline, fold_routed)
        folds.append(
            {
                'fold_id': fold_id,
                'names': list(names),
                'baseline': fold_baseline,
                'routed': fold_routed,
                'delta': delta,
                'consistency': {
                    'score_strictly_improved': delta['metrics']['score'] > 0.0,
                    'pd_non_decrease': delta['metrics']['pd'] >= 0.0,
                    'fa_non_increase': delta['metrics']['fa'] <= 0.0,
                    'iou_non_decrease': delta['metrics']['iou'] >= 0.0,
                },
            }
        )

    project_root = Path(__file__).resolve().parent
    code_paths = {
        'summarize_temporal_window_routing.py': Path(__file__).resolve(),
        'diagnose_temporal_memory_windowing.py': project_root
        / 'diagnose_temporal_memory_windowing.py',
        'utils/temporal_memory_windowed_inference.py': project_root
        / 'utils'
        / 'temporal_memory_windowed_inference.py',
        'utils/temporal_memory_inference.py': project_root
        / 'utils'
        / 'temporal_memory_inference.py',
        'utils/challenge_eval.py': project_root / 'utils' / 'challenge_eval.py',
        'utils/eval.py': project_root / 'utils' / 'eval.py',
        'utils/postprocess.py': project_root / 'utils' / 'postprocess.py',
        'tests/test_temporal_memory_windowed_inference.py': project_root
        / 'tests'
        / 'test_temporal_memory_windowed_inference.py',
        'tests/test_summarize_temporal_window_routing.py': project_root
        / 'tests'
        / 'test_summarize_temporal_window_routing.py',
    }
    report = {
        'schema_version': OUTPUT_SCHEMA,
        'created_utc': utc_now(),
        'evidence_class': (
            'retrospective_train_only_input_route_diagnostic_not_independent_oof'
        ),
        'split_access': {
            'consumed': ['train report sufficient counts', 'train evs_norm[:,3] polarity'],
            'validation_or_test_read': False,
            'labels_used_for_runtime_route': False,
            'train_labels_used_for_diagnostic_metrics': True,
        },
        'route': {
            'observable': 'complete-video polarity minority fraction',
            'definition': 'min(mean(polarity>0.5), 1-mean(polarity>0.5))',
            'cutoff': POLARITY_MINORITY_CUTOFF,
            'operator': '< cutoff -> full_stream; >= cutoff -> window_t32',
            'h1_mode': 'full_stream_T160',
            'h2_mode': 'window_t32_stride16_nearest_center_stitch',
            'checkpoint': 'released M20 T16-trained weights for both modes',
            'route_matches_all_15_train_sources': True,
        },
        'pooled_h1_h2': {
            'pooling': (
                'sum event/target/component/frame sufficient counts first; '
                'then compute float32 IoU/Acc, Pd, Fa, and Score'
            ),
            'baseline_all_t160': baseline,
            'routed_h1_t160_h2_t32': routed,
            'delta': routed_delta,
        },
        'replication_checks': {
            'full_length_window_identity_bitwise': True,
            'separate_combined_t160_report_provided': combined_path is not None,
            'separate_combined_t160_matches_pooled_exactly': (
                combined_replication == baseline
                if combined_path is not None
                else None
            ),
        },
        'runtime': {
            'baseline_inference_seconds_sum': sum(
                item['runtime']['baseline_inference_seconds'] for item in per_video
            ),
            'routed_inference_seconds_sum': sum(
                item['runtime']['routed_inference_seconds'] for item in per_video
            ),
            'baseline_peak_cuda_memory_bytes_max': max(
                item['runtime']['baseline_peak_cuda_memory_bytes'] for item in per_video
            ),
            'routed_peak_cuda_memory_bytes_max': max(
                item['runtime']['routed_peak_cuda_memory_bytes'] for item in per_video
            ),
        },
        'folds': folds,
        'fold_consistency': consistency_summary(folds),
        'per_video': per_video,
        'video_consistency': consistency_summary(per_video),
        'submission_readiness': {
            'directly_usable_now': False,
            'reasons': [
                (
                    'The released test/submission entry points still call only '
                    'full-stream predict_temporal_memory_scores.'
                ),
                (
                    'The polarity router and windowed predictor are diagnostic-only '
                    'and are not wired into the submission path.'
                ),
                (
                    'The H1/T160 + H2/T32 choice is retrospective train-only evidence, '
                    'not an independent OOF or hidden-test result.'
                ),
            ],
            'required_before_submission': [
                'wire the label-free polarity route into complete-video inference',
                'preserve M10 low-density routing before this high-density route',
                'add routed identity/coverage/integration tests and output provenance',
                'freeze the route before any one-time held-data or platform evaluation',
            ],
        },
        'provenance': {
            'input_reports': {
                'h1': {
                    'path': str(h1_path),
                    'sha256': sha256_file(h1_path),
                },
                'h2': {
                    'path': str(h2_path),
                    'sha256': sha256_file(h2_path),
                },
                **(
                    {
                        'combined_replication': {
                            'path': str(combined_path),
                            'sha256': sha256_file(combined_path),
                        }
                    }
                    if combined_path is not None
                    else {}
                ),
            },
            'checkpoint': h1['checkpoint'],
            'code_sha256': {
                name: sha256_file(path) for name, path in code_paths.items()
            },
            'code_paths': {name: str(path.resolve()) for name, path in code_paths.items()},
            'raw_train_sources': {
                item['name']: {
                    'path': item['raw_source_path'],
                    'sha256': item['raw_source_sha256'],
                }
                for item in input_route_records
            },
        },
    }
    output_path, output_sha, sidecar = atomic_json_write(
        args.output,
        report,
        force=args.force,
    )
    print('report:', output_path)
    print('report_sha256:', output_sha)
    print('sha256_sidecar:', sidecar)
    baseline_metrics = baseline['metrics']
    routed_metrics = routed['metrics']
    delta_metrics = routed_delta['metrics']
    print(
        'baseline Score={:.10f} Pd={:.10f} IoU={:.10f} Fa={:.10e}'.format(
            baseline_metrics['score'],
            baseline_metrics['pd'],
            baseline_metrics['iou'],
            baseline_metrics['fa'],
        )
    )
    print(
        'routed   Score={:.10f} Pd={:.10f} IoU={:.10f} Fa={:.10e}'.format(
            routed_metrics['score'],
            routed_metrics['pd'],
            routed_metrics['iou'],
            routed_metrics['fa'],
        )
    )
    print(
        'delta    Score={:+.10f} Pd={:+.10f} IoU={:+.10f} Fa={:+.10e}'.format(
            delta_metrics['score'],
            delta_metrics['pd'],
            delta_metrics['iou'],
            delta_metrics['fa'],
        )
    )
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h1-report', required=True, type=Path)
    parser.add_argument('--h2-report', required=True, type=Path)
    parser.add_argument('--combined-report', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--force', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
