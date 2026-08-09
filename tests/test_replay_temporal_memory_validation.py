import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import replay_temporal_memory_validation as replay
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.density_threshold import select_density_threshold
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor


def make_cfg():
    return SimpleNamespace(
        roc=True,
        pd_detT=50,
        correct_thresh=0.5,
        p0_enabled=True,
        p0_spatial_radius=1,
        p0_temporal_bin_size=50,
        p0_temporal_radius_bins=1,
        p0_min_cluster_events=2,
        p0_min_duration_bins=2,
        p0c_high_confidence_recovery_enabled=True,
        p0c_retain_min_score=0.95,
        p0b_enabled=False,
        p18_score_track_recovery_enabled=True,
        p18_event_count_cutoff=1,
        p18_max_event_count=100,
        p18_candidate_floor=0.53,
        p18_spatial_radius=1,
        p18_temporal_bin_size=50,
        p18_max_link_distance=3.0,
        p18_max_gap_bins=1,
        p18_min_track_bins=2,
        p18_restore_mode="best",
        p18_max_restore_events_per_component=0,
    )


def make_record(file_name, scores):
    scores = torch.tensor(scores, dtype=torch.float32)
    repeats = (len(scores) + 7) // 8
    x = torch.tensor([10, 10, 11, 11, 30, 30, 31, 31], dtype=torch.int64).repeat(repeats)[: len(scores)]
    y = torch.tensor([10, 10, 10, 10, 30, 30, 30, 30], dtype=torch.int64).repeat(repeats)[: len(scores)]
    t = torch.tensor([10, 60, 61, 110, 10, 60, 61, 110], dtype=torch.int64).repeat(repeats)[: len(scores)]
    locs = torch.column_stack((torch.zeros(len(scores), dtype=torch.int64), x, y, t))
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32).repeat(repeats)[: len(scores)]
    target_ids = np.tile(
        np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64), repeats
    )[: len(scores)]
    digest = replay.source_digest(locs, labels, target_ids)
    return {
        "file_name": file_name,
        "event_count": len(scores),
        "scores": scores,
        "seg_label": labels,
        "locs": locs,
        "idx_label": target_ids,
        "source_sha256": digest,
    }


def make_payload(records, checkpoint_sha="a" * 64):
    return {
        "metadata": {
            "schema": replay.CACHE_SCHEMA,
            "dataset_split": "val",
            "dataset_signature": replay._dataset_signature(records),
            "video_count": len(records),
            "event_count": sum(record["event_count"] for record in records),
            "checkpoint_sha256": checkpoint_sha,
        },
        "records": records,
    }


def direct_test2_metrics(records, density_cutoff, low_threshold, high_threshold, cfg):
    evaluator = evalute(cfg)
    sample_number = 0
    fallback_threshold = high_threshold
    for record in records:
        threshold = select_density_threshold(
            record.event_count,
            density_cutoff,
            low_threshold,
            high_threshold,
        )
        postprocessor = ChallengePostprocessor.from_cfg(cfg, threshold)
        predictions, _ = postprocessor.apply(record.scores.clone(), record.locs)
        # This is the P6 branch in test2.py: persist each video's selected
        # decision as binary before the global semantic evaluation.
        predictions = (predictions >= threshold).to(predictions.dtype)
        sample_number = add_batch_to_evaluator(
            evaluator,
            {
                "seg_label": record.seg_label,
                "locs": record.locs,
                "idx_label": record.idx_label,
            },
            predictions,
            sample_number,
            prediction_threshold=threshold,
        )
    return evaluate_challenge_metrics(evaluator, fallback_threshold)


class ThresholdGridTest(unittest.TestCase):
    def test_decimal_grid_is_inclusive_and_stable(self):
        values = replay.decimal_grid("0.710", "0.730", "0.001")
        self.assertEqual(len(values), 21)
        self.assertEqual(values[0], 0.710)
        self.assertEqual(values[-1], 0.730)
        self.assertEqual(values[8], 0.718)

    def test_rejects_non_divisible_grid(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            replay.decimal_grid("0.71", "0.73", "0.003")


class CacheValidationAndRoutingTest(unittest.TestCase):
    def test_secondary_routing_depends_only_on_event_count(self):
        # Names deliberately imply the opposite routing; only event count may
        # select the cached checkpoint.
        low_primary = make_record("looks_high.h5", [0.1] * 8)
        high_primary = make_record("looks_low.h5", [0.2] * 80)
        low_secondary = make_record("looks_high.h5", [0.8] * 8)
        high_secondary = make_record("looks_low.h5", [0.9] * 80)
        routed = replay.route_cache_records(
            make_payload([low_primary, high_primary]),
            make_payload([low_secondary, high_secondary], checkpoint_sha="b" * 64),
            secondary_max_events=30,
        )
        self.assertEqual(routed[0].score_source, "secondary")
        self.assertEqual(routed[1].score_source, "primary")
        self.assertTrue(torch.equal(routed[0].scores, low_secondary["scores"]))
        self.assertTrue(torch.equal(routed[1].scores, high_primary["scores"]))

    def test_rejects_tampered_source_fields(self):
        record = make_record("val.h5", [0.1] * 8)
        payload = make_payload([record])
        payload["records"][0]["locs"][0, 1] += 1
        with self.assertRaisesRegex(ValueError, "digest"):
            replay.validate_cache_payload(payload)

    def test_rejects_scores_outside_probability_range(self):
        record = make_record("val.h5", [0.1] * 8)
        record["scores"][0] = 1.1
        payload = make_payload([record])
        with self.assertRaisesRegex(ValueError, "non-probability"):
            replay.validate_cache_payload(payload)


class ExactReplayTest(unittest.TestCase):
    def test_replay_matches_test2_with_p6_p0_p0c_and_p18(self):
        cfg = make_cfg()
        first = make_record(
            "alpha.h5",
            [0.99, 0.80, 0.54, 0.72, 0.80, 0.54, 0.20, 0.72],
        )
        second = make_record(
            "beta.h5",
            [0.96, 0.71, 0.55, 0.73, 0.74, 0.54, 0.10, 0.80] * 10,
        )
        routed = replay.route_cache_records(make_payload([first, second]))
        low_threshold = 0.718
        high_threshold = 0.719
        prepared = replay.precompute_video_counts(
            routed,
            density_cutoff=30,
            low_thresholds=(low_threshold,),
            high_thresholds=(high_threshold,),
            cfg=cfg,
        )
        actual = replay.evaluate_threshold_pair(
            prepared,
            density_cutoff=30,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            cfg=cfg,
        )
        expected = direct_test2_metrics(
            routed,
            density_cutoff=30,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            cfg=cfg,
        )
        self.assertEqual(actual.to_dict(), expected.to_dict())

    def test_formatted_reference_verification_is_strict(self):
        metrics = {
            "iou": 0.9422550201,
            "acc": 0.9767196774,
            "pd": 0.9762704746,
            "fa": 4.6929172975e-06,
            "score_fa": 0.9541549752,
            "score": 0.9628776542,
        }
        replay.verify_formatted_metrics(metrics, metrics)
        changed = dict(metrics)
        changed["score"] += 1e-9
        with self.assertRaisesRegex(RuntimeError, "score"):
            replay.verify_formatted_metrics(changed, metrics)


if __name__ == "__main__":
    unittest.main()
