import copy
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_h2_safe_fusion_train_oof as fusion


def _counts(tp=100, fp=10, fn=10, correct=10, targets=10, false=1, frames=100):
    return {
        "true_positive_events": tp,
        "false_positive_events": fp,
        "false_negative_events": fn,
        "true_negative_events": 1000,
        "correct_target_groups": correct,
        "target_groups": targets,
        "false_components": false,
        "frame_count": frames,
    }


class ProtocolTests(unittest.TestCase):
    def test_protocol_and_candidate_catalog_are_frozen(self):
        protocol = fusion.validate_protocol()
        self.assertEqual(protocol["population"]["h2_source_count"], 11)
        self.assertEqual(len(fusion.candidate_catalog()), 16)
        self.assertFalse(protocol["split_access"]["gpu_allowed"])
        self.assertIn("validation NPZ", protocol["split_access"]["forbidden"])


class CandidateGenerationTests(unittest.TestCase):
    def test_convex_grid_is_exact_at_alpha_one_and_shape_checked(self):
        full = np.asarray([0.1, 0.8], dtype=np.float32)
        t32 = np.asarray([0.7, 0.2], dtype=np.float32)
        self.assertTrue(np.array_equal(fusion.convex_blend(full, t32, 1.0), t32))
        with self.assertRaisesRegex(ValueError, "aligned"):
            fusion.convex_blend(full, t32[:1], 0.5)

    def test_component_abstain_uses_only_prediction_geometry(self):
        # Pixel 1 anchors the increment at pixel 2.  Pixel 10 is isolated.
        locations = np.asarray(
            [[1, 1, 1], [2, 1, 2], [10, 10, 3], [20, 20, 55]],
            dtype=np.int64,
        )
        full = np.asarray([0.9, 0.1, 0.1, 0.2], dtype=np.float32)
        t32 = np.asarray([0.95, 0.95, 0.95, 0.8], dtype=np.float32)
        bins = (np.asarray([0, 1, 2], dtype=np.int64), np.asarray([3], dtype=np.int64))
        outputs, stats = fusion.component_increment_candidates(
            locations,
            full,
            t32,
            1.0,
            event_indices_by_frame=bins,
            cv2_module=cv2,
        )
        # k=1 accepts the anchored component (pixels 1-2) but rejects isolated 10/20.
        self.assertGreater(outputs[1][1], full[1])
        self.assertEqual(outputs[1][2], full[2])
        self.assertEqual(outputs[1][3], full[3])
        # k=2 rejects the one-anchor component as well.
        self.assertTrue(np.array_equal(outputs[2], full))
        for anchor in fusion.ANCHOR_MINIMUMS:
            self.assertTrue(np.all(outputs[anchor] >= full))
        self.assertEqual(stats[1]["accepted_incremental_components"], 1)
        self.assertGreaterEqual(stats[1]["abstained_incremental_components"], 2)


class SelectionTests(unittest.TestCase):
    def _tables(self):
        baseline = {group: _counts() for group in fusion.GROUPS}
        candidates = {
            item["candidate_id"]: {
                group: copy.deepcopy(baseline[group]) for group in fusion.GROUPS
            }
            for item in fusion.candidate_catalog()
        }
        for group in ("g1", "g2"):
            candidates["convex_a025"][group]["true_positive_events"] += 2
            candidates["convex_a025"][group]["false_negative_events"] -= 2
        return baseline, candidates

    def test_held_counts_cannot_change_development_selection(self):
        baseline, candidates = self._tables()
        first = fusion.select_candidate(baseline, candidates, ("g1", "g2"))
        self.assertEqual(first["selected_candidate_id"], "convex_a025")
        candidates["convex_a025"]["g3"] = _counts(
            tp=1, fp=900, fn=109, correct=0, targets=10, false=100, frames=100
        )
        second = fusion.select_candidate(baseline, candidates, ("g1", "g2"))
        self.assertEqual(second["selected_candidate_id"], first["selected_candidate_id"])

    def test_no_eligible_candidate_abstains_to_full(self):
        baseline = {group: _counts() for group in fusion.GROUPS}
        candidates = {
            item["candidate_id"]: {
                group: copy.deepcopy(baseline[group]) for group in fusion.GROUPS
            }
            for item in fusion.candidate_catalog()
        }
        selection = fusion.select_candidate(baseline, candidates, ("g1", "g2"))
        self.assertEqual(selection["selected_candidate_id"], "full_abstain")
        self.assertTrue(selection["abstained"])


class MetricParityTests(unittest.TestCase):
    def test_local_counts_match_project_float_semantics(self):
        import run_temporal_memory_input_route_train as project

        counts = _counts(tp=123, fp=7, fn=9, correct=17, targets=18, false=3, frames=77)
        local = fusion.evaluation(counts)
        reference = project.evaluation(counts)
        self.assertEqual(local, reference)


if __name__ == "__main__":
    unittest.main()
