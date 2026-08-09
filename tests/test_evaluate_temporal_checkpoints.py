import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_temporal_checkpoints as evaluator


class ParseMetricsTest(unittest.TestCase):
    def test_parses_all_six_metrics_and_uses_final_occurrence(self):
        output = """
IoU:      0.1000000000
noise from progress output
Challenge 2 validation metrics
IoU:      0.9422550201
Acc:      0.9767196774
Pd:       0.9762704746
Fa:       4.6929172975e-06
Score_Fa: 0.9541549752
Score:    0.9628776542
"""
        self.assertEqual(
            evaluator.parse_metrics(output),
            {
                "iou": 0.9422550201,
                "acc": 0.9767196774,
                "pd": 0.9762704746,
                "fa": 4.6929172975e-06,
                "score_fa": 0.9541549752,
                "score": 0.9628776542,
            },
        )

    def test_missing_metric_fails_loudly(self):
        incomplete = "\n".join(
            (
                "IoU: 0.9",
                "Acc: 0.9",
                "Pd: 0.9",
                "Fa: 1e-6",
                "Score_Fa: 0.9",
            )
        )
        with self.assertRaisesRegex(evaluator.MetricParseError, "score"):
            evaluator.parse_metrics(incomplete)


class CommandConstructionTest(unittest.TestCase):
    def test_uses_list_args_current_python_and_frozen_routing(self):
        root = Path("F:/小目标检测")
        checkpoint = root / "实验 输出" / "epoch_004_seed49.pt"
        m10 = root / "EVC-work" / "checkpoints" / "m10.pt"
        data = root / "datasets" / "EV-UAV-Challenge2"
        config = root / "EVC-work" / "configs" / "evisseg_evuav.yaml"
        test_script = root / "EVC-work" / "test2.py"
        command = evaluator.build_test2_command(
            checkpoint,
            m10,
            data,
            config,
            python_executable=sys.executable,
            test_script=test_script,
        )

        self.assertIsInstance(command, list)
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(test_script.resolve()))
        self.assertEqual(command[2:4], ["--config", str(config.resolve())])
        self.assertEqual(command[4], "--set")
        overrides = command[5:]
        self.assertIn("TEST.eval=true", overrides)
        self.assertIn("TEST.roc=true", overrides)
        self.assertIn("TEST.prediction_threshold=0.719", overrides)
        self.assertIn(
            "TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000",
            overrides,
        )
        self.assertIn("TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0", overrides)
        model_override = next(
            value
            for value in overrides
            if value.startswith("TEMPORAL_MEMORY.temporal_memory_model_path=")
        )
        self.assertIn("epoch_004_seed49.pt", model_override)
        self.assertIn("小目标检测", model_override)


class ResumeManifestTest(unittest.TestCase):
    def test_changed_m10_config_or_data_context_is_not_skipped(self):
        complete_metrics = {
            "iou": 0.94,
            "acc": 0.97,
            "pd": 0.98,
            "fa": 4e-6,
            "score_fa": 0.96,
            "score": 0.963,
        }
        tool_identity = {
            "evaluator_version": "test-v2",
            "evaluator_sha256": "1" * 64,
            "test2_sha256": "2" * 64,
            "frozen_settings_version": "frozen-test-v1",
            "frozen_settings_sha256": "3" * 64,
        }

        def identity(m10="B", config="C", data="D"):
            return evaluator.build_evaluation_identity(
                "A" * 64,
                m10 * 64,
                config * 64,
                data * 64,
                tool_identity,
                "E" * 40,
                "F" * 64,
            )

        original = identity()
        manifest = evaluator.new_manifest()
        manifest["runs"] = [
            {
                "evaluation_identity_sha256": original["sha256"],
                "checkpoint_sha256": "A" * 64,
                "status": "success",
                "exit_code": 0,
                "metrics": complete_metrics,
            },
            {
                "evaluation_identity_sha256": "8" * 64,
                "checkpoint_sha256": "B" * 64,
                "status": "failed",
                "exit_code": 1,
                "metrics": None,
            },
            {
                "evaluation_identity_sha256": "9" * 64,
                "checkpoint_sha256": "C" * 64,
                "status": "success",
                "exit_code": 0,
                "metrics": {"score": 0.99},
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / evaluator.MANIFEST_NAME
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = evaluator.load_manifest(path)

        successful = evaluator.successful_evaluation_identities(loaded)
        self.assertEqual(successful, {str(original["sha256"]).lower()})
        self.assertIn(str(original["sha256"]).lower(), successful)
        self.assertNotIn(str(identity(m10="4")["sha256"]).lower(), successful)
        self.assertNotIn(str(identity(config="5")["sha256"]).lower(), successful)
        self.assertNotIn(str(identity(data="6")["sha256"]).lower(), successful)


class ValidationDataIdentityTest(unittest.TestCase):
    def test_exact_24_expected_files_are_required(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            val_dir = data_root / "val"
            val_dir.mkdir()
            for index, filename in enumerate(evaluator.EXPECTED_VAL_FILENAMES):
                (val_dir / filename).write_bytes(bytes([index + 1]))

            identity = evaluator.validation_data_identity(data_root)
            self.assertEqual(identity["file_count"], 24)
            self.assertEqual(len(identity["files"]), 24)

            (val_dir / "val_023.npz").unlink()
            with self.assertRaisesRegex(ValueError, "exactly the 24 expected"):
                evaluator.validation_data_identity(data_root)

    def test_unexpected_npz_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            val_dir = data_root / "val"
            val_dir.mkdir()
            for filename in evaluator.EXPECTED_VAL_FILENAMES:
                (val_dir / filename).write_bytes(b"x")
            (val_dir / "val_024.npz").write_bytes(b"x")

            with self.assertRaisesRegex(ValueError, "unexpected=val_024.npz"):
                evaluator.validation_data_identity(data_root)


if __name__ == "__main__":
    unittest.main()
