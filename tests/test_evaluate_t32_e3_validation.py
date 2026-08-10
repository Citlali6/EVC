import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_t32_e3_validation as frozen


FAKE_GIT = {
    "head": "a" * 40,
    "clean": True,
    "status_sha256": "0" * 64,
    "required_replay_ancestor_present": True,
}
FAKE_CODE = {path: "c" * 64 for path in frozen.CODE_PATHS}
FAKE_VALIDATION_EVIDENCE = {
    "manifest_path": "synthetic",
    "manifest_sha256": frozen.OFFICIAL_MANIFEST_SHA256,
    "semantic_sha256_scheme": "synthetic",
    "semantic_sha256": frozen.OFFICIAL_VAL_SEMANTIC_SHA256,
    "video_count": 24,
    "files": [],
}
FAKE_MANIFEST_ENTRIES = [
    {"path": "val/{}.npz".format(stem), "size": index + 1, "sha256": "d" * 64}
    for index, stem in enumerate(frozen.OFFICIAL_STEMS)
]


class FrozenT32ValidationTests(unittest.TestCase):
    @contextlib.contextmanager
    def _environment(self, directory):
        root = Path(directory).resolve()
        validation = root / "validation"
        names = tuple(frozen._expected_input_sha256())
        paths = {name: root / "inputs" / (name + ".bin") for name in names}
        paths.update(
            {
                "protocol": validation / "preregistered_execution_protocol.json",
                "claim": validation / "validation_attempt_claim.json",
                "m10_cache": validation / "raw_m10_t16_val24.pt",
                "e3_cache": validation / "raw_e3_t32_val24.pt",
                "report": validation / "frozen_validation_report.json",
            }
        )
        paths["dataset_manifest"] = root / "inputs" / "dataset" / "manifest.json"
        paths["dataset_manifest"].parent.mkdir(parents=True)
        for name in names:
            path = paths[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic-" + name).encode("ascii"))
        expected = {name: frozen.sha256_file(paths[name]) for name in names}
        with mock.patch.object(frozen, "_canonical_paths", return_value=paths), mock.patch.object(
            frozen, "_expected_input_sha256", return_value=expected
        ), mock.patch.object(frozen, "_git_state", return_value=dict(FAKE_GIT)), mock.patch.object(
            frozen, "_code_sha256", return_value=dict(FAKE_CODE)
        ), mock.patch.object(frozen, "_validate_formal_lineage"), mock.patch.object(
            frozen, "_validate_validation_files", return_value=dict(FAKE_VALIDATION_EVIDENCE)
        ), mock.patch.object(
            frozen, "_manifest_val_entries", return_value=list(FAKE_MANIFEST_ENTRIES)
        ):
            yield paths, expected

    def _protocol(self, paths):
        protocol = frozen.build_execution_protocol()
        paths["protocol"].parent.mkdir(parents=True, exist_ok=True)
        paths["protocol"].write_text(
            json.dumps(protocol, sort_keys=True), encoding="utf-8"
        )
        return protocol, frozen.sha256_file(paths["protocol"])

    @staticmethod
    def _fake_outcome(passed=True):
        return {
            "validation_dataset_stages": {
                "before_m10": dict(FAKE_VALIDATION_EVIDENCE),
                "after_m10": dict(FAKE_VALIDATION_EVIDENCE),
                "after_e3": dict(FAKE_VALIDATION_EVIDENCE),
            },
            "cache_sha256": {"m10_t16": "1" * 64, "e3_t32": "2" * 64},
            "route_binding": {"enabled": True},
            "profiles": {
                "C00": {"counts": {}, "metrics": dict(frozen.GOLDEN_C00), "per_video": []},
                "C09": {"counts": {}, "metrics": dict(frozen.C09_ACTUAL), "per_video": []},
            },
            "comparisons": {"C00_vs_released_m20": {}, "C09_vs_frozen_C09": {}},
            "gates": {"synthetic": passed},
            "passed": passed,
        }

    def _materialized_outcome(self, paths, passed=True):
        paths["m10_cache"].write_bytes(b"synthetic-m10-cache")
        paths["e3_cache"].write_bytes(b"synthetic-e3-cache")
        outcome = self._fake_outcome(passed)
        outcome["cache_sha256"] = {
            "m10_t16": frozen.sha256_file(paths["m10_cache"]),
            "e3_t32": frozen.sha256_file(paths["e3_cache"]),
        }
        return outcome

    def test_preflight_creates_only_protocol_without_runtime_or_claim(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
            with mock.patch.object(
                frozen, "_run_claimed", side_effect=AssertionError("runtime loaded")
            ), mock.patch.object(
                frozen, "_validate_validation_files", side_effect=AssertionError("NPZ read")
            ):
                result = frozen.preflight_execution()
            self.assertTrue(paths["protocol"].is_file())
            self.assertFalse(paths["claim"].exists())
            self.assertFalse(paths["m10_cache"].exists())
            self.assertFalse(paths["e3_cache"].exists())
            self.assertFalse(result["validation_or_cache_loaded"])

    def test_run_claim_exists_before_any_cache_runtime(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
            _, digest = self._protocol(paths)

            def claimed(protocol, runtime_paths):
                self.assertTrue(runtime_paths["claim"].is_file())
                self.assertFalse(runtime_paths["m10_cache"].exists())
                self.assertFalse(runtime_paths["e3_cache"].exists())
                return self._materialized_outcome(runtime_paths, True)

            with mock.patch.object(frozen, "_run_claimed", side_effect=claimed):
                report = frozen.run_execution(paths["protocol"], digest)
            self.assertTrue(report["passed"])
            self.assertTrue(paths["report"].is_file())

    def test_claim_is_permanent_and_second_attempt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
            _, digest = self._protocol(paths)
            with mock.patch.object(
                frozen, "_run_claimed",
                side_effect=lambda protocol, runtime_paths: self._materialized_outcome(
                    runtime_paths, False
                ),
            ):
                report = frozen.run_execution(paths["protocol"], digest)
            self.assertFalse(report["passed"])
            with self.assertRaisesRegex(FileExistsError, "claim already exists"):
                frozen.run_execution(paths["protocol"], digest)

    def test_post_claim_failure_writes_failure_report_and_keeps_claim(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
            _, digest = self._protocol(paths)
            with mock.patch.object(frozen, "_run_claimed", side_effect=RuntimeError("gpu failed")):
                report = frozen.run_execution(paths["protocol"], digest)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure"]["message"], "gpu failed")
            self.assertTrue(paths["claim"].is_file())
            self.assertTrue(paths["report"].is_file())

    def test_e3_binding_cannot_be_replaced_by_e1_or_e2(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
            protocol, _ = self._protocol(paths)
            for forbidden_name in ("e1_checkpoint", "e2_checkpoint"):
                tampered = json.loads(json.dumps(protocol))
                tampered["inputs"]["e3_checkpoint"] = dict(tampered["inputs"][forbidden_name])
                with self.assertRaises(ValueError):
                    frozen.validate_execution_protocol(tampered)

    def test_protocol_runtime_threshold_or_output_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
            protocol, _ = self._protocol(paths)
            tampered = json.loads(json.dumps(protocol))
            tampered["runtime"]["thresholds"]["high"] = 0.72
            with self.assertRaisesRegex(ValueError, "runtime"):
                frozen.validate_execution_protocol(tampered)
            tampered = json.loads(json.dumps(protocol))
            tampered["outputs"]["report"] = str(Path(directory) / "other.json")
            with self.assertRaises(ValueError):
                frozen.validate_execution_protocol(tampered)

    def test_wrong_head_or_dirty_tree_rejected_before_claim(self):
        for state in ({**FAKE_GIT, "clean": False}, {**FAKE_GIT, "head": "b" * 40}):
            with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
                _, digest = self._protocol(paths)
                with mock.patch.object(frozen, "_git_state", return_value=state):
                    with self.assertRaises(RuntimeError):
                        frozen.run_execution(paths["protocol"], digest)
                self.assertFalse(paths["claim"].exists())

    def test_changed_input_is_rejected_before_claim(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
            _, digest = self._protocol(paths)
            paths["e3_checkpoint"].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "e3_checkpoint changed"):
                frozen.run_execution(paths["protocol"], digest)
            self.assertFalse(paths["claim"].exists())

    def test_cache_cli_has_exact_fixed_inputs_and_no_force(self):
        calls = []

        class Replay:
            @staticmethod
            def main(arguments):
                calls.append(arguments)
                return 0

        with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
            protocol = frozen.build_execution_protocol()
            frozen._cache_cli(
                Replay,
                protocol,
                paths,
                "e3_checkpoint",
                "e3_cache",
                "e3_cache_overrides",
            )
        self.assertEqual(len(calls), 1)
        self.assertNotIn("--force", calls[0])
        self.assertEqual(calls[0].count("--checkpoint"), 1)
        self.assertIn("TEMPORAL_MEMORY.temporal_memory_sequence_length=32", calls[0])

    def test_warm_route_uses_exact_30000_boundary_and_flag(self):
        primary = {
            "metadata": {
                "dataset_split": "val",
                "video_count": 24,
                "event_count": frozen.OFFICIAL_EVENT_COUNT,
                "dataset_signature": frozen.OFFICIAL_CACHE_DATASET_SIGNATURE,
                "checkpoint_path": "e3",
                "checkpoint_sha256": frozen.E3_SHA256,
                "code_sha256": {"cache.py": "c" * 64},
            },
            "records": [{"file_name": stem + ".npz"} for stem in frozen.OFFICIAL_STEMS],
        }
        secondary = {
            "metadata": {
                "dataset_split": "val",
                "video_count": 24,
                "event_count": frozen.OFFICIAL_EVENT_COUNT,
                "dataset_signature": frozen.OFFICIAL_CACHE_DATASET_SIGNATURE,
                "checkpoint_path": "m10",
                "checkpoint_sha256": frozen.M10_SHA256,
                "code_sha256": {"cache.py": "c" * 64},
            }
        }
        captured = {}

        class Replay:
            CACHE_CODE_PROVENANCE_PATHS = ("cache.py",)

            @staticmethod
            def _inference_settings(cfg):
                return {"sequence": 32}

            @staticmethod
            def _validate_cache_compatibility(*args, **kwargs):
                captured["compat"] = kwargs
                return {"enabled": True}

            @staticmethod
            def route_cache_records(*args, **kwargs):
                captured["route"] = (args, kwargs)
                return [
                    SimpleNamespace(event_count=30000, score_source="secondary"),
                    SimpleNamespace(event_count=30001, score_source="primary"),
                ]

        paths = {"e3_checkpoint": Path("e3").resolve(), "m10_checkpoint": Path("m10").resolve()}
        protocol = {
            "repository": {"code_sha256": {"cache.py": "c" * 64}},
            "inputs": {
                "e3_checkpoint": {"sha256": frozen.E3_SHA256},
                "m10_checkpoint": {"sha256": frozen.M10_SHA256},
            }
        }
        binding, records = frozen._validate_and_route(
            Replay, protocol, paths, primary, secondary, object()
        )
        self.assertTrue(binding["enabled"])
        self.assertEqual(captured["compat"]["secondary_max_events"], 30000)
        self.assertTrue(captured["compat"]["allow_warm_primary_t32_secondary_m10_t16"])
        self.assertEqual(captured["route"][0][2], 30000)
        self.assertEqual([r.score_source for r in records], ["secondary", "primary"])

    def test_cache_dataset_signature_and_code_tamper_are_rejected(self):
        primary = {
            "metadata": {
                "dataset_split": "val", "video_count": 24,
                "event_count": frozen.OFFICIAL_EVENT_COUNT,
                "dataset_signature": "bad", "checkpoint_path": "e3",
                "checkpoint_sha256": frozen.E3_SHA256,
                "code_sha256": {"cache.py": "c" * 64},
            },
            "records": [{"file_name": stem + ".npz"} for stem in frozen.OFFICIAL_STEMS],
        }
        secondary = {
            "metadata": {
                "dataset_split": "val", "video_count": 24,
                "event_count": frozen.OFFICIAL_EVENT_COUNT,
                "dataset_signature": frozen.OFFICIAL_CACHE_DATASET_SIGNATURE,
                "checkpoint_path": "m10", "checkpoint_sha256": frozen.M10_SHA256,
                "code_sha256": {"cache.py": "c" * 64},
            }
        }
        replay = SimpleNamespace(CACHE_CODE_PROVENANCE_PATHS=("cache.py",))
        protocol = {
            "repository": {"code_sha256": {"cache.py": "c" * 64}},
            "inputs": {
                "e3_checkpoint": {"sha256": frozen.E3_SHA256},
                "m10_checkpoint": {"sha256": frozen.M10_SHA256},
            },
        }
        paths = {"e3_checkpoint": Path("e3").resolve(), "m10_checkpoint": Path("m10").resolve()}
        with self.assertRaisesRegex(ValueError, "dataset signature"):
            frozen._validate_and_route(replay, protocol, paths, primary, secondary, object())
        primary["metadata"]["dataset_signature"] = frozen.OFFICIAL_CACHE_DATASET_SIGNATURE
        primary["metadata"]["code_sha256"]["cache.py"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "inference code"):
            frozen._validate_and_route(replay, protocol, paths, primary, secondary, object())

    def test_validation_files_are_manifest_locked_and_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            val = root / "val"
            val.mkdir()
            entries = []
            semantic = __import__("hashlib").sha256()
            for index, stem in enumerate(frozen.OFFICIAL_STEMS):
                name = stem + ".npz"
                payload = ("payload-{}".format(index)).encode("ascii")
                path = val / name
                path.write_bytes(payload)
                digest = frozen.sha256_file(path)
                entries.append({"path": "val/" + name, "size": len(payload), "sha256": digest})
                semantic.update(name.encode("utf-8"))
                semantic.update(b"\0")
                semantic.update(bytes.fromhex(digest))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "files": entries}), encoding="utf-8")
            paths = {"dataset_manifest": manifest}
            with mock.patch.object(frozen, "OFFICIAL_MANIFEST_SHA256", frozen.sha256_file(manifest)), mock.patch.object(
                frozen, "OFFICIAL_VAL_SEMANTIC_SHA256", semantic.hexdigest()
            ):
                evidence = frozen._validate_validation_files(paths)
                self.assertEqual(evidence["video_count"], 24)
                (val / "val_023.npz").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "size differs|SHA-256 differs"):
                    frozen._validate_validation_files(paths)

    def test_cache_replacement_after_snapshot_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
            _, digest = self._protocol(paths)

            def stale_outcome(protocol, runtime_paths):
                outcome = self._materialized_outcome(runtime_paths, True)
                outcome["cache_sha256"]["e3_t32"] = "f" * 64
                return outcome

            with mock.patch.object(frozen, "_run_claimed", side_effect=stale_outcome):
                report = frozen.run_execution(paths["protocol"], digest)
            self.assertFalse(report["passed"])
            self.assertIn("raw cache changed", report["failure"]["message"])

    def test_claimed_orchestration_is_exactly_m10_then_e3_then_same_records_c00_c09(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
            protocol = frozen.build_execution_protocol()
            paths["protocol"].parent.mkdir(parents=True, exist_ok=True)
            events = []
            records = object()

            def validate_files(runtime_paths):
                events.append("validate_npz")
                return dict(FAKE_VALIDATION_EVIDENCE)

            def cache_cli(replay, bound_protocol, runtime_paths, checkpoint, cache, overrides):
                events.append("cache:" + checkpoint)
                runtime_paths[cache].write_bytes(cache.encode("ascii"))

            def load_cache(path):
                events.append("load:" + path.name)
                return {"metadata": {}, "records": []}, frozen.sha256_file(path)

            configs = []

            def load_config(path, overrides):
                name = "C09" if "POSTPROCESS.p0c_density_retain_enabled=true" in overrides else "C00"
                cfg = SimpleNamespace(name=name)
                configs.append(cfg)
                return cfg

            fake_replay = SimpleNamespace(
                load_cache_snapshot=load_cache,
                load_flat_config=load_config,
            )

            def route(replay, bound_protocol, runtime_paths, primary, secondary, cfg):
                events.append("route")
                return {"enabled": True}, records

            def evaluate(replay, supplied_records, cfg):
                self.assertIs(supplied_records, records)
                events.append("evaluate:" + cfg.name)
                metrics = dict(frozen.C09_ACTUAL if cfg.name == "C09" else frozen.GOLDEN_C00)
                return {"counts": {}, "metrics": metrics, "per_video": []}

            with mock.patch.dict(sys.modules, {"replay_temporal_memory_validation": fake_replay}), mock.patch.object(
                frozen, "_validate_validation_files", side_effect=validate_files
            ), mock.patch.object(frozen, "_cache_cli", side_effect=cache_cli), mock.patch.object(
                frozen, "_validate_and_route", side_effect=route
            ), mock.patch.object(frozen, "_evaluate_profile", side_effect=evaluate):
                outcome = frozen._run_claimed(protocol, paths)
            self.assertEqual(
                events,
                [
                    "validate_npz",
                    "cache:m10_checkpoint",
                    "validate_npz",
                    "cache:e3_checkpoint",
                    "validate_npz",
                    "load:raw_m10_t16_val24.pt",
                    "load:raw_e3_t32_val24.pt",
                    "route",
                    "evaluate:C00",
                    "evaluate:C09",
                ],
            )
            self.assertEqual(len(configs), 2)
            self.assertIn("C09_vs_frozen_C09", outcome["comparisons"])

    def test_keyboard_interrupt_after_claim_gets_best_effort_failure_report(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory) as (paths, _):
            _, digest = self._protocol(paths)
            with mock.patch.object(frozen, "_run_claimed", side_effect=KeyboardInterrupt("stop")):
                report = frozen.run_execution(paths["protocol"], digest)
            self.assertEqual(report["failure"]["type"], "KeyboardInterrupt")
            integrity = report["post_failure_integrity_observation"]
            self.assertTrue(integrity["protocol"]["equals_claimed"])
            self.assertTrue(integrity["claim"]["equals_created"])
            self.assertTrue(paths["claim"].is_file())

    def test_formal_lineage_rejects_e1_selection(self):
        formal = {
            "schema": frozen.FORMAL_PROTOCOL_SCHEMA,
            "created_before_formal_training": True,
            "git_commit": frozen.FORMAL_TRAINING_COMMIT,
            "parent_checkpoint": {"sha256": frozen.M20_SHA256, "sequence_length": 16},
            "dataset": {
                "official_drive_manifest_sha256": frozen.OFFICIAL_MANIFEST_SHA256,
                "source_video_count": 99,
                "selected_video_count": 54,
            },
            "training": {
                "selection_checkpoint": "epoch_001_seed49.pt",
                "e1_e2_must_not_be_used_for_model_selection": True,
                "source_sequence_length": 16,
                "target_sequence_length": 32,
            },
            "frozen_validation_plan": {},
        }
        with mock.patch.object(frozen, "_load_json", side_effect=[formal, {}, {}]):
            with self.assertRaisesRegex(ValueError, "E3 exclusively"):
                frozen._validate_formal_lineage(
                    {"formal_protocol": Path("x"), "formal_audit": Path("y"), "run_summary": Path("z")},
                    {name: {"sha256": digest} for name, digest in frozen._expected_input_sha256().items()},
                )

    def test_offline_report_member_extraction_and_tamper_rejection(self):
        payload = (
            '{"counts": ' + json.dumps(frozen.GOLDEN_COUNTS) +
            ', "broken_path": "F:\\bad\\q", "metrics": ' +
            json.dumps(frozen.GOLDEN_C00) + "}"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(payload, encoding="utf-8")
            digest = frozen.sha256_file(path)
            frozen._validate_offline_report(
                path, digest, frozen.GOLDEN_C00, frozen.GOLDEN_COUNTS, "synthetic"
            )
            with self.assertRaisesRegex(ValueError, "metrics differ"):
                frozen._validate_offline_report(
                    path, digest, {**frozen.GOLDEN_C00, "score": 0.0},
                    frozen.GOLDEN_COUNTS, "synthetic",
                )

    def test_cli_exposes_no_checkpoint_threshold_cache_output_or_force(self):
        forbidden = ("--checkpoint", "--threshold", "--cache", "--output", "--force")
        for option in forbidden:
            with self.assertRaises(SystemExit):
                frozen.parse_args(
                    [
                        "run",
                        "--execution-protocol",
                        "x",
                        "--expected-execution-protocol-sha256",
                        "f" * 64,
                        option,
                        "x",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
