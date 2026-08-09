"""CPU-only tests for the opt-in density-aware P0c retain threshold."""

import ast
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.postprocess import (  # noqa: E402
    ChallengePostprocessor,
    P0ClusterFilterConfig,
)


CONFIG_PATHS = tuple(sorted((PROJECT_ROOT / "configs").glob("evisseg_evuav*.yaml")))


def make_cfg(**overrides):
    options = {
        "p0_enabled": True,
        "p0_spatial_radius": 1,
        "p0_temporal_bin_size": 50,
        "p0_temporal_radius_bins": 1,
        "p0_min_cluster_events": 2,
        "p0_min_duration_bins": 1,
        "p0c_high_confidence_recovery_enabled": True,
        "p0c_retain_min_score": 0.95,
        "p0c_density_retain_enabled": True,
        "p0c_density_event_count_cutoff": 100000,
        "p0c_density_retain_min_score": 0.97,
        "p0b_enabled": False,
        "p18_score_track_recovery_enabled": False,
    }
    options.update(overrides)
    return SimpleNamespace(**options)


def apply_singleton(cfg, event_count):
    postprocessor = ChallengePostprocessor.from_cfg(
        cfg,
        prediction_threshold=0.719,
        event_count=event_count,
    )
    predictions = torch.tensor([0.96], dtype=torch.float32)
    locations = torch.tensor([[0, 10, 10, 10]], dtype=torch.int64)
    output, _ = postprocessor.apply(predictions, locations)
    return output


def challenge_from_cfg_event_arguments(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    arguments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (
            node.func.attr != "from_cfg"
            or not isinstance(owner, ast.Name)
            or owner.id != "ChallengePostprocessor"
        ):
            continue
        keyword = next(
            (item for item in node.keywords if item.arg == "event_count"),
            None,
        )
        arguments.append(None if keyword is None else keyword.value)
    return arguments


class P0cDensityRetainTest(unittest.TestCase):
    def test_boundary_is_strictly_greater_than_cutoff(self):
        cfg = make_cfg()
        at_cutoff = P0ClusterFilterConfig.from_cfg(cfg, event_count=100000)
        above_cutoff = P0ClusterFilterConfig.from_cfg(cfg, event_count=100001)

        self.assertEqual(at_cutoff.retain_min_score, 0.95)
        self.assertEqual(above_cutoff.retain_min_score, 0.97)
        self.assertEqual(apply_singleton(cfg, 100000).item(), torch.tensor(0.96).item())
        self.assertEqual(apply_singleton(cfg, 100001).item(), 0.0)

    def test_default_off_is_numerically_legacy_equivalent(self):
        disabled = make_cfg(p0c_density_retain_enabled=False)
        legacy = make_cfg()
        del legacy.p0c_density_retain_enabled
        del legacy.p0c_density_event_count_cutoff
        del legacy.p0c_density_retain_min_score

        disabled_output = apply_singleton(disabled, 100001)
        legacy_output = apply_singleton(legacy, 100001)
        self.assertTrue(torch.equal(disabled_output, legacy_output))
        self.assertEqual(disabled_output.item(), torch.tensor(0.96).item())

    def test_rejects_invalid_density_options_and_event_count(self):
        invalid_options = (
            ("p0c_density_event_count_cutoff", -1, "cutoff"),
            ("p0c_density_retain_min_score", -0.01, "retain_min_score"),
            ("p0c_density_retain_min_score", 1.01, "retain_min_score"),
            ("p0c_density_retain_min_score", math.nan, "retain_min_score"),
        )
        for key, value, message in invalid_options:
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    P0ClusterFilterConfig.from_cfg(
                        make_cfg(**{key: value}),
                        event_count=100001,
                    )

        with self.assertRaisesRegex(ValueError, "event_count"):
            P0ClusterFilterConfig.from_cfg(make_cfg(), event_count=-1)
        with self.assertRaisesRegex(ValueError, "complete-video event_count"):
            P0ClusterFilterConfig.from_cfg(make_cfg())
        with self.assertRaisesRegex(ValueError, "high_confidence_recovery_enabled"):
            ChallengePostprocessor.from_cfg(
                make_cfg(p0c_high_confidence_recovery_enabled=False),
                event_count=100001,
            )

    def test_all_five_yaml_configs_ship_safe_defaults(self):
        self.assertEqual(len(CONFIG_PATHS), 5)
        for path in CONFIG_PATHS:
            with self.subTest(path=path.name):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))["POSTPROCESS"]
                self.assertIs(config["p0c_density_retain_enabled"], False)
                self.assertEqual(config["p0c_density_event_count_cutoff"], 100000)
                self.assertEqual(config["p0c_density_retain_min_score"], 0.97)

    def test_test2_submit_and_replay_pass_complete_video_event_count(self):
        for script_name in ("test2.py", "submit_challenge2.py"):
            path = PROJECT_ROOT / script_name
            source = path.read_text(encoding="utf-8")
            arguments = challenge_from_cfg_event_arguments(path)
            with self.subTest(script=script_name):
                self.assertEqual(len(arguments), 3)
                self.assertEqual(
                    sum(
                        isinstance(argument, ast.Name)
                        and argument.id == "event_count"
                        for argument in arguments
                    ),
                    2,
                )
                self.assertEqual(
                    sum(
                        isinstance(argument, ast.Constant)
                        and argument.value == 0
                        for argument in arguments
                    ),
                    1,
                )
                self.assertIn('event_count = len(sample["ev_loc"])', source)
                self.assertIn('event_count = int(batch["locs"].shape[0])', source)
                self.assertIn("P0c density retain requires batch_size=1.", source)

        replay_path = PROJECT_ROOT / "replay_temporal_memory_validation.py"
        replay_arguments = challenge_from_cfg_event_arguments(replay_path)
        self.assertEqual(len(replay_arguments), 1)
        self.assertIsInstance(replay_arguments[0], ast.Attribute)
        self.assertIsInstance(replay_arguments[0].value, ast.Name)
        self.assertEqual(replay_arguments[0].value.id, "record")
        self.assertEqual(replay_arguments[0].attr, "event_count")


if __name__ == "__main__":
    unittest.main()
