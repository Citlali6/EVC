import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np
import torch

import run_h2_activity_selective_recovery_resource_probe_v2 as resource_probe
from run_high_density_dual_expert_grouped_oof import tensor_state_sha256


class ActivitySelectiveRecoveryResourceProbeV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = resource_probe.load_protocol()

    def test_protocol_is_resource_only_and_formal_stays_blocked(self):
        protocol = self.protocol
        self.assertEqual(protocol["role"], "resource_and_mechanical_evidence_only")
        self.assertFalse(protocol["formal_scientific_evidence"])
        self.assertTrue(protocol["base_science_protocol"]["formal_remains_blocked"])
        self.assertFalse(
            protocol["frozen_old_activity_checkpoint"][
                "formal_weight_initialization_allowed"
            ]
        )
        self.assertFalse(
            protocol["frozen_old_activity_checkpoint"][
                "formal_feature_generation_allowed"
            ]
        )
        self.assertFalse(
            protocol["frozen_old_activity_checkpoint"][
                "formal_cutoff_or_score_evidence_allowed"
            ]
        )
        self.assertTrue(
            protocol["mechanical_atomic_action"][
                "official_candidate_metric_must_not_be_reported"
            ]
        )

    def test_fixed_source_and_scope_firewall(self):
        protocol = self.protocol
        self.assertEqual(
            protocol["scope_firewall"]["source_arrays_allowed"],
            ["train_089.npz"],
        )
        self.assertFalse(protocol["scope_firewall"]["held_g2_train_092_094_allowed"])
        self.assertFalse(protocol["scope_firewall"]["other_g1_or_g3_arrays_allowed"])
        self.assertFalse(protocol["scope_firewall"]["validation_or_test_allowed"])
        self.assertEqual(protocol["resource_budget"]["stage1_optimizer_steps"], 0)
        self.assertEqual(protocol["resource_budget"]["stage2_optimizer_steps"], 8)

    def test_prior_failure_chain_hashes_are_exact(self):
        chain = self.protocol["prior_probe_failure_chain"]
        root = (
            resource_probe.WORKSPACE_ROOT
            / "experiments"
            / "20260811_h2_activity_suppress_selective_recovery_g2_v1"
            / "resource_probe"
        )
        expected = {
            "probe_failure.json": chain["failure_receipt_sha256"],
            "probe_postmortem_cpu.json": chain["postmortem_sha256"],
            "stage1_train095_step8.pt": chain["stage1_checkpoint_sha256"],
            "immutable_probe_input.npz": chain["immutable_input_sha256"],
            "immutable_fit_only_probe_labels.npz": chain[
                "immutable_fit_labels_sha256"
            ],
        }
        for name, digest in expected.items():
            self.assertEqual(resource_probe.sha256_file(root / name), digest)

    def test_old_checkpoint_provenance_is_explicitly_nonformal(self):
        payload = torch.load(
            resource_probe.OLD_ACTIVITY_CHECKPOINT, map_location="cpu"
        )
        frozen = self.protocol["frozen_old_activity_checkpoint"]
        self.assertEqual(payload["high_density_expert"]["input_mode"], "activity_control")
        self.assertEqual(payload["high_density_expert"]["domain"], "h2")
        self.assertEqual(payload["high_density_expert"]["insertion_point"], "level1")
        self.assertEqual(payload["provenance"]["fit_names"], frozen["fit_sources"])
        self.assertEqual(payload["provenance"]["held_names"], frozen["old_held_sources"])
        self.assertIn("train_089.npz", payload["provenance"]["held_names"])
        for held_g2 in ("train_092.npz", "train_093.npz", "train_094.npz"):
            self.assertIn(held_g2, payload["provenance"]["fit_names"])
        scope = payload["provenance"]["training_scope"]
        self.assertEqual(scope["trainable_state_tensor_count"], 14)
        self.assertEqual(scope["trainable_parameter_count"], 1712)
        self.assertTrue(scope["inherited_m20_bitwise_frozen"])
        self.assertEqual(len(payload["model_state_dict"]), 103)
        self.assertEqual(
            tensor_state_sha256(payload["model_state_dict"]),
            frozen["source_final_model_state_sha256"],
        )

    def test_checkpoint_and_runtime_file_hashes(self):
        frozen = self.protocol["frozen_old_activity_checkpoint"]
        self.assertEqual(
            resource_probe.sha256_file(resource_probe.OLD_ACTIVITY_CHECKPOINT),
            frozen["sha256"],
        )
        self.assertEqual(
            resource_probe.sha256_file(resource_probe.OLD_ACTIVITY_RUNTIME),
            frozen["runtime_result_sha256"],
        )

    def test_deterministic_resource_batch_has_both_real_classes(self):
        targets = np.asarray([0, 0, 1], dtype=np.uint8)
        first = resource_probe.deterministic_dual_class_indices(targets, 0, 16)
        repeat = resource_probe.deterministic_dual_class_indices(targets, 0, 16)
        later = resource_probe.deterministic_dual_class_indices(targets, 3, 16)
        self.assertEqual(first, repeat)
        for indices in (first, later):
            selected = targets[np.asarray(indices)]
            self.assertEqual(len(indices), 16)
            self.assertTrue(np.any(selected == 0))
            self.assertTrue(np.any(selected == 1))

    def test_resource_batch_fails_without_both_real_classes(self):
        for targets in (
            np.asarray([0, 0], dtype=np.uint8),
            np.asarray([1, 1], dtype=np.uint8),
        ):
            with self.assertRaises(RuntimeError):
                resource_probe.deterministic_dual_class_indices(targets, 0, 16)

    def test_import_and_protocol_audit_do_not_create_probe_output(self):
        # This test does not assume the directory never existed; it verifies that
        # the read-only protocol load itself has no filesystem side effect.
        before = resource_probe.OUTPUT_ROOT.exists()
        resource_probe.load_protocol()
        after = resource_probe.OUTPUT_ROOT.exists()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
