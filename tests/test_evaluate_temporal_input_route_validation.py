import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_temporal_input_route_validation as frozen


def _science():
    with frozen.SCIENCE_PROTOCOL_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _protocol():
    science = _science()
    code = {relative: "a" * 64 for relative in frozen.CODE_PATHS}
    git = {"head": "b" * 40, "clean": True, "status_sha256": "0" * 64}
    return frozen.build_execution_protocol(
        science,
        frozen.canonical_sha256(science),
        git,
        code,
    )


class ScienceProtocolTests(unittest.TestCase):
    def test_frozen_science_protocol_is_valid_and_complete(self):
        science = frozen.validate_science_protocol(_science())
        self.assertEqual(science["attempt_budget"], 1)
        self.assertEqual(len(science["validation_dataset"]["manifest_files"]), 24)
        self.assertEqual(
            science["route_policy"]["sha256"],
            frozen.canonical_sha256(science["route_policy"]["definition"]),
        )
        self.assertFalse(science["persistence"]["enabled"])
        self.assertEqual(science["postprocess"]["profile"], "C00")

    def test_route_or_gate_tamper_fails_closed(self):
        science = _science()
        science["route_policy"]["definition"]["h2"]["stride"] = 8
        science["route_policy"]["sha256"] = frozen.canonical_sha256(
            science["route_policy"]["definition"]
        )
        with self.assertRaisesRegex(ValueError, "route differs"):
            frozen.validate_science_protocol(science)

        science = _science()
        science["promotion_gates"][
            "candidate_score_strictly_greater_than_golden_plus"
        ] = 0.0
        with self.assertRaisesRegex(ValueError, "promotion gates"):
            frozen.validate_science_protocol(science)

    def test_manifest_member_tamper_fails_semantic_digest(self):
        science = _science()
        science["validation_dataset"]["manifest_files"][0]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "member list"):
            frozen.validate_science_protocol(science)

    def test_execution_protocol_binds_all_code_and_inputs(self):
        protocol = _protocol()
        frozen.validate_execution_protocol(protocol)
        self.assertEqual(set(protocol["repository"]["code_sha256"]), set(frozen.CODE_PATHS))
        self.assertEqual(set(protocol["inputs"]), set(frozen.INPUT_NAMES))
        self.assertEqual(protocol["attempt_budget"], 1)
        self.assertFalse(protocol["persistence"]["enabled"])


class NoEarlyValidationAccessTests(unittest.TestCase):
    def test_freeze_never_hashes_manifest_or_score_caches(self):
        science = _science()
        science_sha = frozen.canonical_sha256(science)
        fake_git = {"head": "b" * 40, "clean": True, "status_sha256": "0" * 64}
        fake_code = {relative: "a" * 64 for relative in frozen.CODE_PATHS}
        hashed = []

        def fake_hash(path):
            name = next(
                (name for name, value in frozen._canonical_inputs().items() if value == Path(path)),
                None,
            )
            hashed.append(name or Path(path).name)
            if name is not None:
                return frozen._expected_input_sha256()[name]
            return "e" * 64

        with mock.patch.object(
            frozen, "_load_json_snapshot", return_value=(science, science_sha)
        ), mock.patch.object(frozen, "_git_state", return_value=fake_git), mock.patch.object(
            frozen, "_code_sha256", return_value=fake_code
        ), mock.patch.object(
            frozen, "sha256_file", side_effect=fake_hash
        ), mock.patch.object(
            frozen, "_atomic_json_no_clobber", return_value="d" * 64
        ), mock.patch.object(
            frozen,
            "_validate_train_prerequisite_files",
            return_value={"passed": True},
        ), mock.patch.object(
            frozen.Path, "exists", return_value=False
        ):
            result = frozen.freeze_execution_protocol()

        self.assertFalse(result["validation_npz_read"])
        self.assertFalse(result["validation_cache_read"])
        self.assertNotIn("m10_golden_cache", hashed)
        self.assertNotIn("m20_golden_cache", hashed)
        self.assertNotIn("official_manifest", hashed)

    def test_preclaim_defers_every_validation_bound_input(self):
        protocol = _protocol()
        paths = frozen._canonical_paths()
        paths["execution_protocol"] = paths["execution_protocol"].parent / "synthetic.json"
        fake_git = {"head": "b" * 40, "clean": True, "status_sha256": "0" * 64}
        protocol["repository"]["expected_clean_git_head"] = fake_git["head"]
        code = protocol["repository"]["code_sha256"]
        science = protocol["science_protocol"]["payload"]
        science_sha = protocol["science_protocol"]["sha256"]
        hashed = []

        def fake_hash(path):
            path = Path(path)
            for name, binding in protocol["inputs"].items():
                if path == Path(binding["path"]):
                    hashed.append(name)
                    return binding["sha256"]
            return science_sha

        with mock.patch.object(
            frozen,
            "_load_execution_protocol",
            return_value=(protocol, paths, "c" * 64),
        ), mock.patch.object(frozen, "_git_state", return_value=fake_git), mock.patch.object(
            frozen, "_code_sha256", return_value=code
        ), mock.patch.object(
            frozen,
            "_load_json_snapshot",
            return_value=(science, science_sha),
        ), mock.patch.object(frozen, "sha256_file", side_effect=fake_hash), mock.patch.object(
            frozen.Path, "exists", return_value=False
        ), mock.patch.object(
            frozen.Path, "is_dir", return_value=True
        ), mock.patch.object(
            frozen,
            "_validate_train_prerequisite_files",
            return_value={"passed": True},
        ):
            state = frozen._preclaim_validate("c" * 64)

        self.assertEqual(set(state[6]), set(frozen.INPUT_NAMES) - set(frozen.DEFERRED_VAL_INPUTS))
        self.assertTrue(set(hashed).isdisjoint(frozen.DEFERRED_VAL_INPUTS))

    def test_runtime_smoke_uses_only_valid_zero_one_polarities(self):
        import numpy as np

        locations, polarities, route_probe = frozen._synthetic_runtime_arrays(
            np, _protocol()
        )
        self.assertEqual(locations.shape, (160, 3))
        self.assertEqual(set(np.unique(polarities).tolist()), {0.0, 1.0})
        self.assertEqual(set(np.unique(route_probe).tolist()), {0.0, 1.0})
        self.assertEqual(route_probe.shape[0], frozen.HIGH_EVENT_COUNT_MAX + 2)
        self.assertGreaterEqual(
            min(float(np.mean(route_probe == 0)), float(np.mean(route_probe == 1))),
            frozen.POLARITY_MINORITY_CUTOFF,
        )


class H2OnlyRouteTests(unittest.TestCase):
    def test_non_h2_returns_same_object_without_calling_predictor(self):
        scores = object()
        calls = []

        def predictor():
            calls.append(True)
            return object()

        for domain in ("low", "middle", "h1"):
            selected, preserved = frozen.choose_candidate_scores(
                SimpleNamespace(domain=domain), scores, predictor
            )
            self.assertIs(selected, scores)
            self.assertTrue(preserved)
        self.assertEqual(calls, [])

    def test_h2_calls_predictor_exactly_once(self):
        baseline = object()
        candidate = object()
        calls = []

        def predictor():
            calls.append(True)
            return candidate

        selected, preserved = frozen.choose_candidate_scores(
            SimpleNamespace(domain="h2"), baseline, predictor
        )
        self.assertIs(selected, candidate)
        self.assertFalse(preserved)
        self.assertEqual(len(calls), 1)

    def test_score_gate_is_strict_and_metric_guards_are_nondegrading(self):
        at_boundary = dict(frozen.GOLDEN_METRICS)
        at_boundary["score"] = frozen.GOLDEN_METRICS["score"] + 0.0001
        gates = frozen.promotion_gate_results(
            dict(frozen.GOLDEN_COUNTS),
            dict(frozen.GOLDEN_METRICS),
            at_boundary,
            True,
            1,
            1,
        )
        self.assertFalse(
            gates["candidate_score_strictly_greater_than_golden_plus_0p0001"]
        )
        self.assertTrue(gates["candidate_pd_not_lower_than_golden"])
        self.assertTrue(gates["candidate_iou_not_lower_than_golden"])
        self.assertTrue(gates["candidate_fa_not_higher_than_golden"])

        passing = dict(at_boundary)
        passing["score"] += 1e-12
        gates = frozen.promotion_gate_results(
            dict(frozen.GOLDEN_COUNTS),
            dict(frozen.GOLDEN_METRICS),
            passing,
            True,
            1,
            1,
        )
        self.assertTrue(
            gates["candidate_score_strictly_greater_than_golden_plus_0p0001"]
        )

    def test_golden_cache_inference_settings_must_equal_frozen_protocol(self):
        protocol = _protocol()
        inputs = {
            name: Path(binding["path"])
            for name, binding in protocol["inputs"].items()
        }
        expected = {
            "temporal_memory_bin_size": 50,
            "temporal_memory_context_bins": 5,
            "temporal_memory_width": 16,
            "temporal_memory_sequence_length": 16,
            "temporal_memory_inference_batch_size": 8,
            "temporal_memory_log_count_clip": 4.0,
            "whole_t": 8000,
            "resolution": [346, 260],
        }

        def payload(checkpoint):
            return {
                "metadata": {
                    "dataset_split": "val",
                    "video_count": frozen.OFFICIAL_VIDEO_COUNT,
                    "event_count": frozen.OFFICIAL_EVENT_COUNT,
                    "dataset_signature": frozen.OFFICIAL_DATASET_SIGNATURE,
                    "checkpoint_sha256": protocol["inputs"][checkpoint]["sha256"],
                    "checkpoint_path": str(inputs[checkpoint]),
                    "inference_settings": dict(expected),
                }
            }

        primary = payload("m20_checkpoint")
        secondary = payload("m10_checkpoint")
        primary["metadata"]["inference_settings"]["whole_t"] = 4000
        replay = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "inference settings"):
            frozen._validate_loaded_golden_caches(
                replay, protocol, primary, secondary, inputs
            )


class AttemptClaimTests(unittest.TestCase):
    def test_claim_is_exclusive_and_second_attempt_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claim.json"
            report = Path(directory) / "report.json"
            with mock.patch.object(
                frozen,
                "_canonical_paths",
                return_value={"execution_protocol": Path(directory) / "protocol.json"},
            ):
                payload, digest = frozen._atomic_claim(
                    path, "a" * 64, report, "b" * 64
                )
                self.assertEqual(payload["attempt_budget"], 1)
                self.assertEqual(
                    payload["runtime_preflight_receipt_sha256"], "b" * 64
                )
                self.assertEqual(len(digest), 64)
                with self.assertRaises(FileExistsError):
                    frozen._atomic_claim(path, "a" * 64, report, "b" * 64)

    def test_run_claims_before_claimed_work(self):
        protocol = _protocol()
        paths = frozen._canonical_paths()
        git = {"head": "b" * 40, "clean": True, "status_sha256": "0" * 64}
        code = protocol["repository"]["code_sha256"]
        inputs = {name: Path(binding["path"]) for name, binding in protocol["inputs"].items()}
        events = []
        outcome = {
            "validation_integrity": {},
            "golden_cache_sha256": {},
            "golden_cache_binding": {},
            "h2_cache": {},
            "route_summary": {},
            "per_video": [],
            "aggregate": {},
            "gates": {"synthetic": True},
            "passed": True,
        }

        def claim(*_):
            events.append("claim")
            return {"attempt": 1}, "d" * 64

        def claimed(*_):
            events.append("claimed_work")
            return outcome

        verified = {
            name: protocol["inputs"][name]["sha256"]
            for name in frozen.INPUT_NAMES
            if name not in frozen.DEFERRED_VAL_INPUTS
        }
        runtime = dict(frozen.EXPECTED_RUNTIME)
        runtime["python_executable"] = "C:\\synthetic\\python.exe"
        smoke = {
            "synthetic_only": True,
            "validation_or_cache_read": False,
            "scores_finite": True,
            "scores_in_probability_range": True,
            "route": {"domain": "h2"},
        }
        runtime_receipt = {"runtime": runtime, "smoke": smoke}
        runtime_bundle = {"model": object()}

        def prepare(*_):
            events.append("runtime")
            return runtime, smoke, runtime_bundle

        def stable_hash(path):
            path = Path(path)
            if path == paths["claim"]:
                return "d" * 64
            if path == paths["execution_protocol"]:
                return "c" * 64
            if path == paths["runtime_receipt"]:
                return "r" * 64
            for name, input_path in inputs.items():
                if path == input_path:
                    return verified.get(name, protocol["inputs"][name]["sha256"])
            return "f" * 64

        with mock.patch.object(
            frozen,
            "_preclaim_validate",
            return_value=(
                protocol,
                paths,
                "c" * 64,
                git,
                code,
                inputs,
                verified,
                {"passed": True},
            ),
        ), mock.patch.object(
            frozen,
            "_load_runtime_receipt",
            return_value=(runtime_receipt, "r" * 64),
        ), mock.patch.object(
            frozen, "_prepare_runtime_before_claim", side_effect=prepare
        ), mock.patch.object(frozen, "_atomic_claim", side_effect=claim), mock.patch.object(
            frozen, "_run_claimed", side_effect=claimed
        ), mock.patch.object(frozen, "_git_state", return_value=git), mock.patch.object(
            frozen, "_code_sha256", return_value=code
        ), mock.patch.object(frozen, "sha256_file", side_effect=stable_hash), mock.patch.object(
            frozen, "_atomic_json_no_clobber", return_value="e" * 64
        ):
            report = frozen.run_execution("c" * 64)

        self.assertEqual(events, ["runtime", "claim", "claimed_work"])
        self.assertTrue(report["passed"])

    def test_runtime_failure_does_not_consume_attempt(self):
        protocol = _protocol()
        paths = frozen._canonical_paths()
        git = {"head": "b" * 40, "clean": True, "status_sha256": "0" * 64}
        code = protocol["repository"]["code_sha256"]
        inputs = {
            name: Path(binding["path"])
            for name, binding in protocol["inputs"].items()
        }
        receipt = {"runtime": {}, "smoke": {}}
        claim = mock.Mock()
        with mock.patch.object(
            frozen,
            "_preclaim_validate",
            return_value=(
                protocol,
                paths,
                "c" * 64,
                git,
                code,
                inputs,
                {},
                {"passed": True},
            ),
        ), mock.patch.object(
            frozen,
            "_load_runtime_receipt",
            return_value=(receipt, "r" * 64),
        ), mock.patch.object(
            frozen,
            "_prepare_runtime_before_claim",
            side_effect=RuntimeError("synthetic CUDA failure"),
        ), mock.patch.object(frozen, "_atomic_claim", claim):
            with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
                frozen.run_execution("c" * 64)
        claim.assert_not_called()

    def test_failure_after_claim_writes_failure_report_and_reraises(self):
        protocol = _protocol()
        paths = frozen._canonical_paths()
        git = {"head": "b" * 40, "clean": True, "status_sha256": "0" * 64}
        code = protocol["repository"]["code_sha256"]
        inputs = {name: Path(binding["path"]) for name, binding in protocol["inputs"].items()}
        written = []
        runtime = dict(frozen.EXPECTED_RUNTIME)
        runtime["python_executable"] = "C:\\synthetic\\python.exe"
        smoke = {
            "synthetic_only": True,
            "validation_or_cache_read": False,
            "scores_finite": True,
            "scores_in_probability_range": True,
            "route": {"domain": "h2"},
        }
        receipt = {"runtime": runtime, "smoke": smoke}
        with mock.patch.object(
            frozen,
            "_preclaim_validate",
            return_value=(
                protocol,
                paths,
                "c" * 64,
                git,
                code,
                inputs,
                {},
                {"passed": True},
            ),
        ), mock.patch.object(
            frozen,
            "_load_runtime_receipt",
            return_value=(receipt, "r" * 64),
        ), mock.patch.object(
            frozen,
            "_prepare_runtime_before_claim",
            return_value=(runtime, smoke, {"model": object()}),
        ), mock.patch.object(
            frozen, "sha256_file", return_value="r" * 64
        ), mock.patch.object(
            frozen, "_atomic_claim", return_value=({"attempt": 1}, "d" * 64)
        ), mock.patch.object(
            frozen, "_run_claimed", side_effect=RuntimeError("synthetic failure")
        ), mock.patch.object(
            frozen, "_artifact_observation", return_value={}
        ), mock.patch.object(
            frozen,
            "_atomic_json_no_clobber",
            side_effect=lambda path, payload: written.append(payload) or "e" * 64,
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                frozen.run_execution("c" * 64)
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["status"], "failed")
        self.assertFalse(written[0]["passed"])


if __name__ == "__main__":
    unittest.main()
