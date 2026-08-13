import argparse
import hashlib
import json
from pathlib import Path
import unittest

import torch

import run_middle_multiscale_temporal_summary_expert as runner


class MiddleTemporalSummaryProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = runner.load_protocol()

    def test_protocol_identity_and_blind_scope(self):
        self.assertEqual(self.protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(
            self.protocol["status"],
            "frozen_before_any_middle_expert_gpu_probe_training_or_held_candidate_evaluation",
        )
        self.assertFalse(self.protocol["science_scope"]["validation_read_allowed"])
        self.assertFalse(self.protocol["science_scope"]["test_read_allowed"])
        self.assertFalse(self.protocol["science_scope"]["source_name_path_hash_or_fold_feature_allowed"])

    def test_input_only_middle_route_boundaries(self):
        self.assertFalse(runner.middle_route(30000))
        self.assertTrue(runner.middle_route(30001))
        self.assertTrue(runner.middle_route(200000))
        self.assertFalse(runner.middle_route(200001))
        self.assertEqual(self.protocol["input_only_route"]["observable_inputs"], ["event_count"])

    def test_exact_continuous_families_cover_39_unique_sources(self):
        families = runner.family_sources(self.protocol)
        self.assertEqual(
            tuple(families),
            ("f1_000_014", "f2_028_032", "f3_040_043", "f4_059_065", "f5_067_074"),
        )
        flattened = runner.all_middle_sources(self.protocol)
        self.assertEqual(len(flattened), 39)
        self.assertEqual(len(set(flattened)), 39)
        for names in families.values():
            indices = [int(name[6:9]) for name in names]
            self.assertEqual(indices, list(range(indices[0], indices[-1] + 1)))

    def test_five_lofo_folds_are_disjoint_and_each_source_held_once(self):
        families = runner.family_sources(self.protocol)
        all_sources = set(runner.all_middle_sources(self.protocol))
        held_seen = []
        for fold in self.protocol["fold_order"]:
            held = set(families[fold["held_family"]])
            fit = {source for family in fold["fit_families"] for source in families[family]}
            self.assertFalse(held & fit)
            self.assertEqual(held | fit, all_sources)
            self.assertEqual(len(held), fold["held_source_count"])
            self.assertEqual(len(fit), fold["fit_source_count"])
            self.assertEqual(
                fold["optimizer_steps"],
                len(fit)
                * self.protocol["training"]["epochs"]
                * self.protocol["training"]["views_per_fit_source_per_epoch"],
            )
            held_seen.extend(held)
        self.assertEqual(len(held_seen), 39)
        self.assertEqual(set(held_seen), all_sources)

    def test_first_fold_and_probe_budget_are_fixed(self):
        first = runner.first_fold_spec(self.protocol)
        self.assertEqual(first["fold_id"], "middle_hold_f1_000_014")
        self.assertEqual(len(first["fit_sources"]), 24)
        self.assertEqual(len(first["held_sources"]), 15)
        self.assertEqual(first["optimizer_steps"], 96)
        probe = self.protocol["eight_step_resource_probe"]
        self.assertEqual(probe["optimizer_steps"], 8)
        self.assertEqual(probe["formal_optimizer_steps"], 0)
        self.assertIn(probe["source"], first["fit_sources"])
        self.assertNotIn(probe["source"], first["held_sources"])

    def test_capacity_supports_pooled_point_zero_one_but_is_diagnostic_only(self):
        capacity = self.protocol["parent_error_capacity_diagnostic"]
        self.assertEqual(capacity["middle_source_count"], 39)
        self.assertEqual(capacity["atomic_components"]["pure_FP"], 1475)
        self.assertGreater(capacity["score_headroom_from_pure_FP_deletion"], 0.01)
        self.assertAlmostEqual(
            capacity["perfect_pure_FP_component_deletion_score"]
            - capacity["M20_metrics"]["score"],
            capacity["score_headroom_from_pure_FP_deletion"],
            places=14,
        )
        self.assertTrue(
            self.protocol["claim_limitations"][
                "all_39_labels_used_once_for_parent_error_capacity_diagnostic_before_candidate_training"
            ]
        )

    def test_architecture_and_loss_are_exact_reused_hashes(self):
        model_path = runner.ROOT / "model" / "h2_multiscale_temporal_pyramid_expert.py"
        loss_path = runner.ROOT / "utils" / "h2_multiscale_pyramid_loss.py"
        self.assertEqual(runner.sha256_file(model_path), runner.EXPECTED_MODEL_SHA256)
        self.assertEqual(runner.sha256_file(loss_path), runner.EXPECTED_LOSS_SHA256)
        architecture = self.protocol["architecture"]
        self.assertEqual(architecture["temporal_summary_scales_bins"], [16, 32, 64, 160])
        self.assertEqual(architecture["trainable_parameter_count"], 3381)
        self.assertEqual(architecture["output_projection_initialization"], "exact_zero_weight_and_bias")

    def test_synthetic_cpu_identity_and_dynamic_loss(self):
        self.assertFalse(torch.cuda.is_initialized())
        audit = runner._synthetic_cpu_audit(self.protocol)
        self.assertTrue(audit["passed"], audit)
        self.assertTrue(audit["checks"]["zero_init_bitwise_M20_identity"])
        self.assertTrue(audit["checks"]["hard_negative_finite_positive"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_cpu_audit_payload_is_fail_closed_and_train_only(self):
        payload = runner.build_audit_payload()
        self.assertTrue(payload["passed"], payload)
        self.assertEqual(len(payload["source_evidence"]), 39)
        self.assertEqual(len(payload["folds"]), 5)
        self.assertFalse(payload["validation_or_test_read"])
        self.assertFalse(payload["gpu_or_cuda_initialized"])
        self.assertEqual(payload["formal_training_steps"], 0)
        self.assertEqual(payload["protocol_sha256"], runner.EXPECTED_PROTOCOL_SHA256)
        self.assertTrue(payload["probe_source_cpu_preflight"]["passed"])
        self.assertEqual(
            payload["probe_source_cpu_preflight"]["eligible_joint_view_count"], 97
        )
        self.assertEqual(
            payload["probe_source_cpu_preflight"]["selected_view_starts"],
            [76, 117, 119, 16, 136, 91, 101, 73],
        )

    def test_cli_has_no_formal_or_evaluation_entry_and_probe_needs_authorization(self):
        parser = runner.build_parser()
        action = next(item for item in parser._actions if isinstance(item, argparse._SubParsersAction))
        self.assertEqual(set(action.choices), {"audit", "plan-first-fold", "probe"})
        with self.assertRaises(PermissionError):
            runner.run_probe(False)


if __name__ == "__main__":
    unittest.main()
