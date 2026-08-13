import unittest

import numpy as np

import run_h2_activity_selective_recovery_fresh_g2_formal as formal_runner
from utils.activity_selective_recovery_formal import (
    assess_inner_replay,
    deterministic_epoch_batches,
    exact_confidence_cutoffs,
    select_qualifying_inner_replay,
    source_class_balanced_weights,
)


GROUPS = ("g1_088_091", "g3_095_098")


def metric_payload(score, pd, tp, fp, co, fc):
    return {
        "counts": {
            "true_positive_events": tp,
            "false_positive_events": fp,
            "false_negative_events": 100,
            "correct_objects": co,
            "object_count": 200,
            "false_components": fc,
            "frame_count": 100,
            "event_count": 10000,
        },
        "metrics": {
            "score": score,
            "pd": pd,
            "iou": 0.5,
            "acc": 0.7,
            "fa": 0.001,
            "score_fa": 0.8,
        },
    }


def qualifying_replay(cutoff=0.5, group_gain=0.011, pooled_gain=0.012):
    groups = {}
    for group in GROUPS:
        groups[group] = {
            "m20": metric_payload(0.75, 0.85, 1000, 400, 100, 150),
            "activity": metric_payload(0.765, 0.847, 970, 260, 96, 100),
            "candidate": metric_payload(
                0.75 + group_gain, 0.848, 980, 280, 98, 110
            ),
            "atomic_integrity": True,
        }
    return {
        "cutoff": cutoff,
        "recovered_component_count": 2,
        "groups": groups,
        "pooled": {
            "m20": metric_payload(0.75, 0.85, 2000, 800, 200, 300),
            "activity": metric_payload(0.765, 0.847, 1940, 520, 192, 200),
            "candidate": metric_payload(
                0.75 + pooled_gain, 0.848, 1960, 560, 196, 220
            ),
        },
    }


class ActivitySelectiveRecoveryFormalTests(unittest.TestCase):
    def test_source_class_balancing_equalizes_each_observed_cell(self):
        source_ids = np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
        targets = np.asarray([0, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
        weights = source_class_balanced_weights(source_ids, targets)
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        masses = []
        for source in (0, 1):
            for target in (0, 1):
                masses.append(float(weights[(source_ids == source) & (targets == target)].sum()))
        self.assertTrue(np.allclose(masses, masses[0]))

    def test_deterministic_epoch_batches_visit_every_item_once_per_epoch(self):
        batches = deterministic_epoch_batches(7, 3, 4, 73)
        self.assertEqual(len(batches), 12)
        for epoch in range(4):
            visited = np.concatenate(
                [item["indices"] for item in batches if item["epoch"] == epoch]
            )
            self.assertEqual(sorted(visited.tolist()), list(range(7)))
        second = deterministic_epoch_batches(7, 3, 4, 73)
        self.assertTrue(
            all(np.array_equal(a["indices"], b["indices"]) for a, b in zip(batches, second))
        )

    def test_exact_confidence_cutoffs_are_identity_plus_unique_breakpoints(self):
        cutoffs = exact_confidence_cutoffs(
            [np.asarray([0.2, 0.5]), np.asarray([0.5, 0.8])]
        )
        self.assertGreater(cutoffs[0], 0.8)
        self.assertEqual(cutoffs[1:].tolist(), [0.8, 0.5, 0.2])

    def test_parent_specified_inner_gate_passes_exact_contract(self):
        gate = assess_inner_replay(qualifying_replay())
        self.assertTrue(gate["passed"])
        self.assertGreaterEqual(gate["pooled_score_gain"], 0.01)

    def test_each_group_must_recover_tp_or_co_relative_to_stage1(self):
        replay = qualifying_replay()
        replay["groups"][GROUPS[0]]["candidate"]["counts"][
            "true_positive_events"
        ] = 960
        replay["groups"][GROUPS[0]]["candidate"]["counts"][
            "correct_objects"
        ] = 95
        gate = assess_inner_replay(replay)
        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["group_checks"][GROUPS[0]][
                "recovers_tp_or_co_relative_to_stage1"
            ]
        )

    def test_absolute_pd_delta_is_two_sided(self):
        for pd in (0.844, 0.856):
            replay = qualifying_replay()
            replay["groups"][GROUPS[1]]["candidate"]["metrics"]["pd"] = pd
            self.assertFalse(assess_inner_replay(replay)["passed"])

    def test_selection_uses_maximin_before_pooled_gain(self):
        safer = qualifying_replay(cutoff=0.7, group_gain=0.012, pooled_gain=0.012)
        pooled_larger = qualifying_replay(
            cutoff=0.6, group_gain=0.011, pooled_gain=0.02
        )
        selected, audit = select_qualifying_inner_replay([pooled_larger, safer])
        self.assertEqual(len(audit), 2)
        self.assertEqual(selected["replay"]["cutoff"], 0.7)

    def test_frozen_protocol_has_disjoint_fit_and_sealed_g2(self):
        protocol = formal_runner.load_protocol()
        fit = set(protocol["scope"]["inner_fit_sources"])
        held = set(protocol["scope"]["sealed_outer_held_sources"])
        self.assertEqual(
            held, {"train_092.npz", "train_093.npz", "train_094.npz"}
        )
        self.assertFalse(fit & held)
        self.assertFalse(protocol["scope"]["validation_or_test_allowed"])

    def test_stage1_and_stage2_are_strict_true_oof(self):
        protocol = formal_runner.load_protocol()
        self.assertTrue(protocol["stage1_true_oof"]["fit_g1_predict_only_g3"])
        self.assertTrue(protocol["stage1_true_oof"]["fit_g3_predict_only_g1"])
        self.assertFalse(
            protocol["stage1_true_oof"][
                "own_group_activity_feature_generation_allowed"
            ]
        )
        self.assertTrue(
            protocol["stage2_nested_oof"][
                "recovery_fit_g1_oof_predict_g3_oof"
            ]
        )
        self.assertTrue(
            protocol["final_fit_after_inner_pass"]["stage2"].endswith(
                "true-OOF disagreement features only"
            )
        )

    def test_old_and_resource_checkpoints_are_permanently_forbidden(self):
        protocol = formal_runner.load_protocol()
        forbidden = formal_runner.forbidden_hashes(protocol)
        self.assertEqual(
            forbidden,
            {
                "a73a037f09aadb478540aa40e50f9bbfbec73afb315b0fa54fca0b897b2a5b82",
                "ce5e223655abad5038073e9de165f726767793907c271d372e2faa550a80d9df",
                "874212511c09f034d20e3c14d7e28ecc2db39a1490b3b3ec861a1f607008deb1",
            },
        )

    def test_two_process_held_firewall_is_frozen(self):
        protocol = formal_runner.load_protocol()
        self.assertEqual(protocol["execution"]["train_subcommand"], "train-and-freeze")
        self.assertEqual(
            protocol["execution"]["held_subcommand"],
            "evaluate-held-g2-once",
        )
        self.assertTrue(
            protocol["outer_held_once"]["create_held_open_receipt_before_first_array"]
        )
        self.assertFalse(
            protocol["outer_held_once"]["retry_or_reopen_after_any_result_or_failure"]
        )
        self.assertFalse(protocol["execution"]["gpu_authorized"])
        self.assertFalse(protocol["execution"]["priority_authorized"])

    def test_cpu_protocol_import_has_no_output_side_effect(self):
        before = (formal_runner.FIT_OUTPUT.exists(), formal_runner.HELD_OUTPUT.exists())
        formal_runner.load_protocol()
        after = (formal_runner.FIT_OUTPUT.exists(), formal_runner.HELD_OUTPUT.exists())
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
