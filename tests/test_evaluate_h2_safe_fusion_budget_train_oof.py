import copy
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_h2_safe_fusion_budget_train_oof as fusion


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
    def test_protocol_and_catalog_are_frozen_adaptive_train_only(self):
        protocol = fusion.validate_protocol()
        self.assertEqual(len(fusion.candidate_catalog()), 16)
        self.assertEqual(
            protocol["evidence_class"],
            "adaptive_train_only_confirmation_after_v1_not_independent_oof",
        )
        self.assertFalse(protocol["split_access"]["gpu_allowed"])
        self.assertIn("validation NPZ", protocol["split_access"]["forbidden"])


class BudgetTests(unittest.TestCase):
    def test_at_budget_retains_increment_vector(self):
        full = np.zeros(40, dtype=np.float32)
        raw = full.copy()
        raw[:32] = np.float32(0.8)
        output, stats = fusion.apply_changed_event_budget(full, raw, 32)
        self.assertTrue(np.array_equal(output, raw))
        self.assertTrue(np.shares_memory(output, raw))
        self.assertFalse(stats["video_abstained"])
        self.assertEqual(stats["output_changed_events"], 32)

    def test_over_budget_returns_exact_complete_full_vector(self):
        full = np.linspace(0.0, 0.5, 40, dtype=np.float32)
        raw = full.copy()
        raw[:33] += np.float32(0.1)
        output, stats = fusion.apply_changed_event_budget(full, raw, 32)
        self.assertTrue(np.array_equal(output, full))
        self.assertTrue(np.shares_memory(output, full))
        self.assertTrue(stats["video_abstained"])
        self.assertEqual(stats["changed_event_count"], 33)
        self.assertEqual(stats["output_changed_events"], 0)

    def test_lowering_increment_is_rejected(self):
        full = np.asarray([0.4, 0.5], dtype=np.float32)
        raw = np.asarray([0.4, 0.3], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "lowered"):
            fusion.apply_changed_event_budget(full, raw, 32)


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
            candidates["budget_a025_m032"][group]["true_positive_events"] += 2
            candidates["budget_a025_m032"][group]["false_negative_events"] -= 2
        return baseline, candidates

    def test_held_group_cannot_change_selection(self):
        baseline, candidates = self._tables()
        first = fusion.select_candidate(baseline, candidates, ("g1", "g2"))
        self.assertEqual(first["selected_candidate_id"], "budget_a025_m032")
        candidates["budget_a025_m032"]["g3"] = _counts(
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


if __name__ == "__main__":
    unittest.main()
