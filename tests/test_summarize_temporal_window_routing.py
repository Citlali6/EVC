import unittest

import torch

from summarize_temporal_window_routing import (
    evaluation_delta,
    evaluation_from_counts,
    input_route_for_fraction,
    metrics_from_counts,
    sum_counts,
)


def _counts(**overrides):
    values = {
        'true_positive_events': 80,
        'false_positive_events': 10,
        'false_negative_events': 20,
        'true_negative_events': 890,
        'correct_target_groups': 8,
        'target_groups': 10,
        'false_components': 5,
        'frame_count': 20,
    }
    values.update(overrides)
    return values


class TemporalRoutingSummaryTests(unittest.TestCase):
    def test_metrics_use_pooled_counts_and_float32_semantic_division(self):
        first = _counts(true_positive_events=3, false_positive_events=1)
        second = _counts(true_positive_events=1234567, false_positive_events=321)
        pooled = sum_counts((first, second))

        metrics = metrics_from_counts(pooled)

        expected_iou = float(
            (
                torch.tensor(
                    pooled['true_positive_events'],
                    dtype=torch.float32,
                )
                / torch.tensor(
                    pooled['true_positive_events']
                    + pooled['false_positive_events']
                    + pooled['false_negative_events'],
                    dtype=torch.float32,
                )
            ).item()
        )
        self.assertEqual(metrics['iou'], expected_iou)
        self.assertNotEqual(
            metrics['iou'],
            (
                metrics_from_counts(first)['iou']
                + metrics_from_counts(second)['iou']
            )
            / 2.0,
        )

    def test_input_route_cutoff_is_explicit_and_label_free(self):
        self.assertEqual(
            input_route_for_fraction(0.199999)['mode'],
            'full_stream',
        )
        self.assertEqual(
            input_route_for_fraction(0.20)['mode'],
            'window_t32',
        )
        with self.assertRaises(ValueError):
            input_route_for_fraction(0.51)

    def test_delta_preserves_event_and_false_component_signs(self):
        baseline = evaluation_from_counts(_counts())
        candidate = evaluation_from_counts(
            _counts(
                true_positive_events=82,
                false_positive_events=7,
                false_negative_events=18,
                true_negative_events=893,
                correct_target_groups=9,
                false_components=3,
            )
        )

        delta = evaluation_delta(baseline, candidate)

        self.assertEqual(delta['counts']['true_positive_events'], 2)
        self.assertEqual(delta['counts']['false_positive_events'], -3)
        self.assertEqual(delta['counts']['false_components'], -2)
        self.assertGreater(delta['metrics']['pd'], 0.0)
        self.assertLess(delta['metrics']['fa'], 0.0)


if __name__ == '__main__':
    unittest.main()
