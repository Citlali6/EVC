import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_temporal_memory_input_route_train import OFFICIAL_TRAIN_NAMES, SCHEMA
from run_temporal_memory_input_route_train import (
    C00_DEFINITION,
    ROUTE_DECISION_FIELDS,
    atomic_npz,
    c00_sha256,
    frozen_route_decision,
    load_input_only_video,
    metrics_from_counts,
    validate_audit,
)
from utils.temporal_memory_input_router import (
    route_policy_definition,
    route_policy_sha256,
)


class TrainCacheEvaluationProtocolTests(unittest.TestCase):
    def test_cache_input_loader_needs_no_label_or_target_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_000.npz"
            evs_norm = np.zeros((6, 4), dtype=np.float32)
            evs_norm[:, 3] = [0, 1, 0, 1, 0, 1]
            locations = np.asarray(
                [[0, 0, 1], [1, 0, 11], [2, 0, 21], [0, 1, 31], [1, 1, 41], [2, 1, 51]],
                dtype=np.int64,
            )
            np.savez(path, evs_norm=evs_norm, ev_loc=locations)

            video = load_input_only_video(path)

            np.testing.assert_array_equal(video.polarities, evs_norm[:, 3])
            np.testing.assert_array_equal(video.labels, np.zeros(6, dtype=np.float32))
            np.testing.assert_array_equal(video.target_ids, np.zeros(6, dtype=np.int64))
            self.assertEqual(video.name, "input_video")
            self.assertEqual(len(video.event_indices_by_bin), 160)

    def test_cache_record_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.npz"
            atomic_npz(path, baseline_scores=np.zeros(2), candidate_scores=np.ones(2))

            with self.assertRaises(FileExistsError):
                atomic_npz(path, baseline_scores=np.zeros(2), candidate_scores=np.ones(2))

    def test_c00_definition_has_stable_digest_and_no_reranker(self):
        self.assertEqual(len(c00_sha256()), 64)
        self.assertFalse(C00_DEFINITION["component_reranker_enabled"])
        self.assertEqual(C00_DEFINITION["p18_candidate_floor"], 0.53)

    def test_route_projection_requires_temporal_bin_count(self):
        decision = {
            "event_count": 39169,
            "polarity_minority_fraction": 0.035,
            "domain": "middle",
            "checkpoint_role": "m20",
            "mode": "full_stream",
            "window_length": None,
            "stride": None,
            "prediction_threshold": 0.719,
            "policy_sha256": route_policy_sha256(),
        }
        with self.assertRaisesRegex(ValueError, "temporal_bin_count"):
            frozen_route_decision(decision)
        decision["temporal_bin_count"] = 160

        projected = frozen_route_decision(decision)

        self.assertEqual(tuple(projected), ROUTE_DECISION_FIELDS)

    def test_sufficient_count_metric_is_finite(self):
        metrics = metrics_from_counts(
            {
                "true_positive_events": 90,
                "false_positive_events": 5,
                "false_negative_events": 10,
                "true_negative_events": 1000,
                "correct_target_groups": 9,
                "target_groups": 10,
                "false_components": 2,
                "frame_count": 160,
            }
        )

        self.assertAlmostEqual(metrics["iou"], 90 / 105, places=6)
        self.assertEqual(metrics["pd"], 0.9)
        self.assertGreater(metrics["score"], 0.0)

    def _audit(self):
        records = [
            {"source_name": name, "source_sha256": "a" * 64}
            for name in OFFICIAL_TRAIN_NAMES
        ]
        return {
            "schema": SCHEMA,
            "split_access": {
                "dataset_split": "train",
                "validation_or_test_read": False,
            },
            "route_independence": {
                "labels_used": False,
                "source_name_used": False,
            },
            "policy": route_policy_definition(),
            "policy_sha256": route_policy_sha256(),
            "population": {
                "video_count": 99,
                "event_count_gt_30000": 54,
                "event_count_gt_200000": 15,
                "event_count_200001_to_250000": 0,
                "gt_200000_matches_existing_15_source_evidence": True,
            },
            "density_gate_protection": {
                "unassessed_below_200k_sources_sent_to_t32": 0,
            },
            "records": records,
        }

    def test_audit_contract_rejects_label_dependent_route(self):
        payload = self._audit()
        payload["route_independence"]["labels_used"] = True

        with self.assertRaisesRegex(ValueError, "label/name"):
            validate_audit(payload, Path("audit.json"))

    def test_audit_contract_accepts_frozen_population(self):
        records = validate_audit(self._audit(), Path("audit.json"))

        self.assertEqual(len(records), 99)


if __name__ == "__main__":
    unittest.main()
