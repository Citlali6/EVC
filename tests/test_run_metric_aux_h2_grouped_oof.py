import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_metric_aux_h2_grouped_oof as runner


class MetricAuxGroupedOofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = runner.load_protocol()

    def test_protocol_hash_status_and_claim_scope_are_frozen(self):
        self.assertEqual(self.protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(
            self.protocol["status"],
            "frozen_before_any_probe_formal_training_or_held_evaluation",
        )
        self.assertTrue(
            self.protocol["audit_amendment"]["shared_parent_pretraining_exposure"]
        )
        self.assertEqual(
            self.protocol["audit_amendment"]["claim_scope"],
            "incremental_finetune_transfer_not_fold_clean_model_generalization",
        )
        self.assertEqual(len(self.protocol["revision_history"]), 2)

    def test_true_defaults_and_single_m23_loss_candidate(self):
        defaults = self.protocol["history_audit"]["base_config_defaults"]
        self.assertEqual(
            (defaults["metric_target_weight"], defaults["metric_component_weight"]),
            (0.01, 0.002),
        )
        self.assertEqual(defaults["metric_activation_threshold"], 0.7)
        candidate = self.protocol["training"]["candidate"]
        self.assertEqual(
            (
                candidate["metric_target_weight"],
                candidate["metric_component_weight"],
                candidate["metric_warmup_epochs"],
                candidate["metric_activation_threshold"],
            ),
            (0.005, 0.001, 1, 0.719),
        )
        self.assertTrue(self.protocol["training"]["no_parameter_grid"])

    def test_sampling_disclosure_does_not_call_eight_views_exact_m23_reuse(self):
        sampling = self.protocol["training"]["sampling_contract"]
        self.assertIn("16 H2 views", sampling["historical_m23"])
        self.assertIn("uniform 8", sampling["current_grouped_oof"])
        self.assertIn("not an exact replication", sampling["reuse_scope"])

    def test_warmup_semantics_default_inactive_and_candidate_active_e2_e3(self):
        epochs = range(3)
        default_active = [epoch for epoch in epochs if epoch >= 5]
        candidate_active = [epoch for epoch in epochs if epoch >= 1]
        self.assertEqual(default_active, [])
        self.assertEqual(candidate_active, [1, 2])
        self.assertEqual(
            candidate_active,
            self.protocol["training"]["candidate"]["active_zero_based_epochs"],
        )

    def test_three_held_groups_are_disjoint_complete_and_each_source_once(self):
        index = runner.source_index(self.protocol)
        held_sets = []
        occurrences = {name: 0 for name in index}
        for fold in self.protocol["dataset"]["folds"]:
            fit = {item["name"] for item in runner.fit_items(self.protocol, fold)}
            held = {item["name"] for item in runner.held_items(self.protocol, fold)}
            self.assertFalse(fit & held)
            self.assertEqual(fit | held, set(index))
            held_sets.append(held)
            for name in held:
                occurrences[name] += 1
        for position, left in enumerate(held_sets):
            for right in held_sets[position + 1 :]:
                self.assertFalse(left & right)
        self.assertEqual(set().union(*held_sets), set(index))
        self.assertEqual(set(occurrences.values()), {1})

    def test_optimizer_step_budget_is_168_192_168_per_variant(self):
        actual = [
            fold["expected_optimizer_steps_per_run"]
            for fold in self.protocol["dataset"]["folds"]
        ]
        self.assertEqual(actual, [168, 192, 168])
        self.assertEqual(sum(actual) * 2, 1056)

    def test_paired_overrides_differ_only_in_output_and_enabled_flag(self):
        for fold in self.protocol["dataset"]["folds"]:
            data_root = runner.OUTPUT_ROOT / "unit" / fold["fold_id"]
            baseline = runner.override_mapping(
                runner.training_overrides(
                    self.protocol,
                    data_root,
                    runner.OUTPUT_ROOT / "unit" / "baseline",
                    "baseline",
                    3,
                )
            )
            candidate = runner.override_mapping(
                runner.training_overrides(
                    self.protocol,
                    data_root,
                    runner.OUTPUT_ROOT / "unit" / "metric_aux",
                    "metric_aux",
                    3,
                )
            )
            differences = {
                key for key in set(baseline) | set(candidate) if baseline.get(key) != candidate.get(key)
            }
            self.assertEqual(
                differences,
                {
                    "TRAIN.model_save_root",
                    "TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled",
                },
            )
            self.assertEqual(baseline["TEMPORAL_MEMORY.temporal_memory_metric_target_weight"], "0.005000")
            self.assertEqual(candidate["TEMPORAL_MEMORY.temporal_memory_metric_component_weight"], "0.001000")
            self.assertEqual(candidate["TEMPORAL_MEMORY.temporal_memory_metric_warmup_epochs"], "1")

    def test_full_scope_parent_checkpoint_name_shape_hash(self):
        checkpoint = runner.workspace_path(
            self.protocol["parent_checkpoint"]["workspace_relative_path"]
        )
        audit = runner.checkpoint_name_shape_audit(self.protocol, checkpoint)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["state_tensor_count"], 89)
        self.assertEqual(audit["state_parameter_count"], 1924716)
        self.assertEqual(
            audit["name_shape_canonical_sha256"],
            "3bbe7100b5be460eeeea5218cccfb27d9ed697e449d5a3abe1babf918023f05e",
        )

    def test_clean_cpu_architecture_scope_hash_and_no_cuda(self):
        audit = runner.model_architecture_scope_audit_cpu(self.protocol)
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["checks"]["cuda_not_initialized"])

    def test_probe_view_dataset_uses_nested_train_root_and_only_train096(self):
        item = runner.source_index(self.protocol)[self.protocol["resource_probe"]["source"]]
        view = runner.materialize_view(self.protocol, "probe", [item])
        train_root = Path(view["root"]) / "train"
        self.assertEqual({path.stem for path in train_root.glob("*.npz")}, {"train_096"})
        from dataset.temporal_memory import TemporalMemoryTrainDataset

        dataset = TemporalMemoryTrainDataset(
            root=train_root,
            whole_t=8000,
            temporal_bin_size=50,
            context_bins=5,
            sequence_length=16,
            width=128,
            height=128,
            views_per_video=8,
            positive_frame_probability=0.75,
            random_seed=49,
            cache_all_videos=False,
            cache_video_count=2,
            dense_sampling_enabled=False,
            density_bucket_boundaries=[],
            density_bucket_views=[],
            min_event_count_exclusive=200000,
            sparse_target_support_sampling_enabled=False,
        )
        self.assertEqual(len(dataset), 8)
        self.assertEqual([path.stem for path in dataset.file_paths], ["train_096"])

    def test_synthetic_metric_autograd_reaches_both_losses_and_fresh_update(self):
        result = runner.synthetic_metric_gradient_probe(self.protocol, device_name="cpu")
        self.assertTrue(result["passed"])
        self.assertGreater(result["target_gradient_norm"], 0.0)
        self.assertGreater(result["component_gradient_norm"], 0.0)
        self.assertGreater(result["fresh_optimizer_update_norm"], 0.0)

    def _gate_fixture(self):
        comparator_counts = {
            "true_positive_events": 100,
            "false_positive_events": 20,
            "false_negative_events": 10,
            "correct_objects": 10,
            "object_count": 10,
            "false_components": 10,
            "frame_count": 100,
            "event_count": 130,
        }
        candidate_counts = {
            **comparator_counts,
            "false_positive_events": 10,
            "false_components": 8,
        }
        comparator_metrics = {
            "score": 0.7000,
            "pd": 1.0,
            "fa": 0.001,
            "iou": 0.75,
            "acc": 0.90,
            "score_fa": 0.4,
        }
        candidate_metrics = {
            **comparator_metrics,
            "score": 0.7010,
            "fa": 0.0008,
            "iou": 0.76,
        }
        folds = []
        for fold in self.protocol["dataset"]["folds"]:
            folds.append(
                {
                    "fold_id": fold["fold_id"],
                    "metric_aux": {"counts": copy.deepcopy(candidate_counts), "metrics": copy.deepcopy(candidate_metrics)},
                    "baseline": {"counts": copy.deepcopy(comparator_counts), "metrics": copy.deepcopy(comparator_metrics)},
                    "released_m20": {"counts": copy.deepcopy(comparator_counts), "metrics": copy.deepcopy(comparator_metrics)},
                }
            )
        pooled = {}
        for variant, counts, metrics in (
            ("metric_aux", candidate_counts, candidate_metrics),
            ("baseline", comparator_counts, comparator_metrics),
            ("released_m20", comparator_counts, comparator_metrics),
        ):
            pooled[variant] = {
                "counts": {key: value * 3 for key, value in counts.items()},
                "metrics": copy.deepcopy(metrics),
            }
        return folds, pooled

    def test_promotion_requires_both_comparators_and_passes_perfect_fixture(self):
        folds, pooled = self._gate_fixture()
        gates = runner.evaluate_report_gates(self.protocol, folds, pooled)
        self.assertTrue(gates["passed"])
        self.assertTrue(gates["checks"]["every_fold_against_both_comparators"])
        self.assertTrue(gates["checks"]["pooled_against_both_comparators"])

    def test_promotion_fails_per_fold_score_drop_even_if_other_metrics_pass(self):
        folds, pooled = self._gate_fixture()
        folds[0]["metric_aux"]["metrics"]["score"] = 0.6999
        gates = runner.evaluate_report_gates(self.protocol, folds, pooled)
        self.assertFalse(gates["passed"])
        self.assertFalse(
            gates["fold_checks"][folds[0]["fold_id"]]["baseline"]["checks"][
                "score_not_lower"
            ]
        )

    def test_promotion_fails_if_pooled_fp_not_strictly_reduced(self):
        folds, pooled = self._gate_fixture()
        pooled["metric_aux"]["counts"]["false_positive_events"] = pooled["baseline"]["counts"]["false_positive_events"]
        pooled["metric_aux"]["counts"]["event_count"] = pooled["baseline"]["counts"]["event_count"]
        gates = runner.evaluate_report_gates(self.protocol, folds, pooled)
        self.assertFalse(gates["passed"])
        self.assertFalse(
            gates["pooled_checks"]["baseline"]["checks"][
                "false_positive_events_strictly_lower"
            ]
        )

    def test_data_path_guard_rejects_nontrain_and_validation_names(self):
        root = runner.official_train_root(self.protocol)
        with self.assertRaises(RuntimeError):
            runner.require_official_train_source(root / "val_001.npz", self.protocol)
        with self.assertRaises(RuntimeError):
            runner.require_official_train_source(root.parent / "test" / "train_096.npz", self.protocol)

    def test_gpu_authorization_is_fail_closed(self):
        with self.assertRaises(PermissionError):
            runner.require_gpu_authorization(False)

    def test_wddm_graphics_are_ignored_but_python_gpu_process_fails(self):
        graphics = mock.Mock(returncode=0, stdout="111, explorer.exe, N/A\n222, chrome.exe, N/A\n", stderr="")
        tasklist = mock.Mock(returncode=0, stdout='"explorer.exe","111","Console","1","1 K"\n"chrome.exe","222","Console","1","1 K"\n', stderr="")
        with mock.patch.object(runner.subprocess, "run", side_effect=[graphics, tasklist]):
            result = runner.require_idle_gpu()
        self.assertTrue(result["idle_for_python_training"])
        python_gpu = mock.Mock(returncode=0, stdout="333, [N/A], N/A\n", stderr="")
        python_tasklist = mock.Mock(returncode=0, stdout='"python.exe","333","Console","1","1 K"\n', stderr="")
        with mock.patch.object(runner.subprocess, "run", side_effect=[python_gpu, python_tasklist]):
            with self.assertRaises(RuntimeError):
                runner.require_idle_gpu()


if __name__ == "__main__":
    unittest.main()
