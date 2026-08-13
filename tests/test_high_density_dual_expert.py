import inspect
import json
import tempfile
import unittest
from pathlib import Path

import torch

import run_high_density_dual_expert_grouped_oof as runner
from model.high_density_polarity_expert import (
    FineTemporalPolarityMultiScaleExpert,
    HighDensityPolarityExpertMemoryNet,
    configure_expert_only_training,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet


class HighDensityDualExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = runner.load_protocol()

    def test_protocol_hash_and_budget_are_frozen(self):
        self.assertEqual(cls_sha := self.protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(cls_sha, runner.sha256_file(runner.PROTOCOL_PATH))
        self.assertEqual(self.protocol["training"]["expected_optimizer_steps_total"], 416)
        self.assertEqual(len(self.protocol["dataset"]["folds"]), 5)
        probe = self.protocol["resource_probe"]
        self.assertEqual(probe["domains"]["h1"]["source"], "train_045.npz")
        self.assertEqual(probe["domains"]["h1"]["steps_per_arm"], 4)
        self.assertEqual(probe["domains"]["h2"]["source"], "train_096.npz")
        self.assertEqual(probe["domains"]["h2"]["steps_per_arm"], 2)
        self.assertEqual(probe["total_optimizer_steps"], 12)

    def test_fold_membership_is_disjoint_and_domain_partitioned(self):
        held = {"h1": [], "h2": []}
        for fold in self.protocol["dataset"]["folds"]:
            fit_names = {item["name"] for item in runner.fit_items(self.protocol, fold)}
            held_names = {item["name"] for item in runner.held_items(self.protocol, fold)}
            self.assertFalse(fit_names & held_names)
            held[fold["domain"]].extend(held_names)
        index = runner.source_index(self.protocol)
        for domain in ("h1", "h2"):
            expected = {
                name
                for name, item in index.items()
                if item["group"].startswith(domain)
            }
            self.assertEqual(set(held[domain]), expected)
            self.assertEqual(len(held[domain]), len(expected))

    def test_input_only_route_has_no_name_argument(self):
        self.assertEqual(
            list(inspect.signature(runner.observable_route).parameters),
            ["event_count", "polarity_minority_fraction"],
        )
        self.assertEqual(runner.observable_route(200000, 0.49), "released_m20")
        self.assertEqual(runner.observable_route(200001, 0.199), "h1")
        self.assertEqual(runner.observable_route(200001, 0.20), "h2")

    def test_paired_modes_and_h2_material_gate(self):
        self.assertEqual(runner.mode_for_arm(self.protocol, "h1", "baseline"), "activity_control")
        self.assertEqual(runner.mode_for_arm(self.protocol, "h1", "candidate"), "h1_saturation")
        self.assertEqual(runner.mode_for_arm(self.protocol, "h2", "candidate"), "h2_polarity")
        self.assertEqual(
            self.protocol["promotion_gates"][
                "h2_held_pooled_score_delta_minimum_against_each_comparator"
            ],
            0.02,
        )

    def test_feature_banks_are_paired_and_clip8_is_required(self):
        frames = torch.zeros(1, 10, 4, 4)
        frames[:, 0::2] = 0.2
        frames[:, 1::2] = 0.8
        control = FineTemporalPolarityMultiScaleExpert(input_mode="activity_control")
        h2 = FineTemporalPolarityMultiScaleExpert(input_mode="h2_polarity")
        h1 = FineTemporalPolarityMultiScaleExpert(input_mode="h1_saturation")
        control_features = control.paired_input_features(frames)
        h2_features = h2.paired_input_features(frames)
        self.assertEqual(tuple(control_features.shape), (1, 15, 4, 4))
        self.assertTrue(torch.equal(control_features[:, :5], control_features[:, 5:10]))
        self.assertTrue(torch.allclose(h2_features[:, 5:10], torch.full_like(h2_features[:, 5:10], 0.6)))
        with self.assertRaises(ValueError):
            h1.paired_input_features(frames)
        clip8 = frames * 0.5
        h1_features = h1.paired_input_features(frames, clip8)
        self.assertTrue(torch.allclose(h1_features[:, 5:10], torch.full_like(h1_features[:, 5:10], 0.4)))
        self.assertTrue(torch.allclose(h1_features[:, 10:15], torch.full_like(h1_features[:, 10:15], 0.16)))

    def test_zero_projection_and_scope_are_exact(self):
        torch.manual_seed(49)
        model = HighDensityPolarityExpertMemoryNet(expert_input_mode="h2_polarity")
        residual = model.high_density_expert(torch.rand(2, 10, 32, 32))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        names = configure_expert_only_training(model)
        self.assertEqual(len(names), 14)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 1712)
        self.assertTrue(all(name.startswith("high_density_expert.") for name in names))

    def test_initial_expert_model_is_parent_identity(self):
        torch.manual_seed(17)
        parent = BidirectionalTemporalMemoryNet(
            input_channels=10,
            width=16,
            temporal_attention_enabled=True,
            density_calibration_enabled=True,
        ).eval()
        torch.manual_seed(23)
        candidate = HighDensityPolarityExpertMemoryNet(
            input_channels=10,
            width=16,
            temporal_attention_enabled=True,
            density_calibration_enabled=True,
            expert_input_mode="h1_saturation",
        ).eval()
        incompatible = candidate.load_state_dict(parent.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(name.startswith("high_density_expert.") for name in incompatible.missing_keys))
        frames = torch.rand(1, 2, 10, 32, 32)
        clip8 = frames * 0.5
        with torch.no_grad():
            expected = parent(frames)
            actual = candidate(frames, expert_frames=clip8)
        self.assertTrue(torch.equal(expected, actual))

    def test_parallel_collate_preserves_extra_stack(self):
        import numpy as np

        sample = {
            "frames": np.zeros((2, 10, 4, 4), dtype=np.float32),
            "expert_frames": np.ones((2, 10, 4, 4), dtype=np.float32),
            "parallel_clip_audit": {
                "clip4_reconstruction_bitwise_equal": True,
                "shape_aligned": True,
                "clip4_nonzero_cells": 1,
                "clip8_nonzero_cells": 1,
                "clip4_saturated_cells": 1,
                "clip8_recovered_dynamic_cells": 1,
                "clip4_max": 1.0,
                "clip8_max": 0.5,
                "start_bin": 0,
                "sequence_length": 2,
            },
            "event_time_indices": np.asarray([0], dtype=np.int64),
            "event_timestamps": np.asarray([1], dtype=np.int64),
            "event_x": np.asarray([1], dtype=np.int64),
            "event_y": np.asarray([1], dtype=np.int64),
            "labels": np.asarray([0], dtype=np.float32),
            "target_ids": np.asarray([0], dtype=np.int64),
        }
        batch = runner.parallel_clip_collate([sample])
        self.assertEqual(tuple(batch["expert_frames"].shape), (2, 10, 4, 4))
        self.assertTrue(torch.equal(batch["expert_frames"], torch.ones_like(batch["expert_frames"])))
        self.assertTrue(batch["parallel_clip_audit"]["shape_aligned"])

    def test_cli_has_no_validation_test_or_search_command(self):
        choices = runner.build_parser()._subparsers._group_actions[0].choices
        self.assertEqual(
            set(choices),
            {"audit", "probe", "train", "train-all", "pair-audit", "evaluate", "evaluate-all", "report"},
        )
        self.assertTrue(
            {"validation", "test", "search", "sweep"}.isdisjoint(choices)
        )

    def test_write_new_json_is_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            runner.write_new_json(path, {"passed": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"passed": True})
            with self.assertRaises(FileExistsError):
                runner.write_new_json(path, {"passed": False})


if __name__ == "__main__":
    unittest.main()
