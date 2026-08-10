import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_component_reranker_validation as frozen
import replay_temporal_memory_validation as replay
from utils.component_reranker import (
    ARTIFACT_SCHEMA,
    FEATURE_NAMES,
    FEATURE_SEMANTICS_VERSION,
    TRAIN_CACHE_SCHEMA,
    ComponentTopology,
    input_postprocess_mapping,
    sha256_file,
    sha256_json,
)
from utils.postprocess import P0ClusterFilterConfig


SYNTHETIC_EVENT_COUNT = 100116
FAKE_GIT = {"head": "a" * 40, "clean": True, "status_sha256": "0" * 64}
INFERENCE_SETTINGS = {
    "temporal_memory_bin_size": 50,
    "temporal_memory_context_bins": 5,
    "temporal_memory_width": 16,
    "temporal_memory_sequence_length": 16,
    "temporal_memory_inference_batch_size": 8,
    "temporal_memory_log_count_clip": 4.0,
    "whole_t": 8000,
    "resolution": [346, 260],
}


def _config_payload():
    return {
        "DATA": {"res": [346, 260], "whole_t": 8000},
        "TEST": {"prediction_threshold": 0.9, "roc": True, "pd_detT": 50, "correct_thresh": 0.0001},
        "POSTPROCESS": {
            "p0_enabled": False,
            "p0_spatial_radius": 1,
            "p0_temporal_bin_size": 50,
            "p0_temporal_radius_bins": 1,
            "p0_min_cluster_events": 2,
            "p0_min_duration_bins": 1,
            "p0c_high_confidence_recovery_enabled": False,
            "p0c_retain_min_score": 0.98,
            "p0c_density_retain_enabled": False,
            "p0c_density_event_count_cutoff": 100000,
            "p0c_density_retain_min_score": 0.97,
            "component_reranker_enabled": False,
            "component_reranker_event_count_cutoff": 100000,
            "component_reranker_model_path": "",
            "component_reranker_expected_sha256": "",
            "p0b_enabled": False,
            "p18_score_track_recovery_enabled": False,
            "p18_event_count_cutoff": 100000,
            "p18_max_event_count": 0,
            "p18_candidate_floor": 0.8,
            "p18_spatial_radius": 2,
            "p18_temporal_bin_size": 50,
            "p18_max_link_distance": 6.0,
            "p18_max_gap_bins": 1,
            "p18_min_track_bins": 2,
            "p18_restore_mode": "best",
            "p18_max_restore_events_per_component": 0,
            "p6_density_threshold_enabled": False,
            "p6_event_count_cutoff": 0,
            "p6_low_density_threshold": 0.9,
            "p6_high_density_threshold": 0.9,
        },
        "TEMPORAL_FRAME": {"temporal_frame_enabled": False},
        "TEMPORAL_MEMORY": {
            "temporal_memory_enabled": False,
            "temporal_memory_sparse_weight": 0.5,
            "temporal_memory_temporal_attention_enabled": False,
            "temporal_memory_model_path": "",
            "temporal_memory_secondary_model_path": "",
            "temporal_memory_secondary_max_event_count": 0,
            "temporal_memory_blend_model_path": "",
            "temporal_memory_bin_size": 50,
            "temporal_memory_context_bins": 5,
            "temporal_memory_width": 16,
            "temporal_memory_sequence_length": 16,
            "temporal_memory_inference_batch_size": 8,
            "temporal_memory_log_count_clip": 4.0,
        },
        "FUSION": {"dense_expert_enabled": False, "ensemble_enabled": False},
    }


def _record(stem, high=False):
    count = 100001 if high else 5
    scores = torch.full((count,), 0.1, dtype=torch.float32)
    labels = torch.zeros(count, dtype=torch.float32)
    locs = torch.zeros((count, 4), dtype=torch.int64)
    target_ids = np.zeros(count, dtype=np.int64)
    times = torch.tensor([10, 60, 110, 160, 210], dtype=torch.int64)
    scores[:5], labels[:5], target_ids[:5] = 0.9, 1, 1
    locs[:5, 1], locs[:5, 2], locs[:5, 3] = 10, 10, times
    if high:
        scores[5:10] = 0.72
        locs[5:10, 1], locs[5:10, 2], locs[5:10, 3] = 50, 50, times
        locs[10:, 1], locs[10:, 2], locs[10:, 3] = 200, 200, 400
    return {
        "file_name": stem + ".npz",
        "event_count": count,
        "scores": scores,
        "seg_label": labels,
        "locs": locs,
        "idx_label": target_ids,
        "source_sha256": replay.source_digest(locs, labels, target_ids),
    }


def _payload(records, checkpoint):
    return {
        "metadata": {
            "schema": replay.CACHE_SCHEMA,
            "dataset_split": "val",
            "dataset_signature": replay._dataset_signature(records),
            "video_count": 24,
            "event_count": sum(item["event_count"] for item in records),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "inference_settings": dict(INFERENCE_SETTINGS),
            "code_sha256": {path: "c" * 64 for path in replay.CACHE_CODE_PROVENANCE_PATHS},
        },
        "records": records,
    }


class ProtocolValidationTests(unittest.TestCase):
    @contextlib.contextmanager
    def _environment(self, directory):
        with mock.patch.object(frozen, "FROZEN_EXPERIMENT_DIRECTORY", Path(directory)), mock.patch.object(
            frozen, "OFFICIAL_VALIDATION_EVENT_COUNT", SYNTHETIC_EVENT_COUNT
        ), mock.patch.object(frozen, "_git_state", return_value=dict(FAKE_GIT)):
            yield

    def _fixture(self, directory):
        root = Path(directory).resolve()
        paths = frozen._canonical_experiment_paths()
        config = root / "config.yaml"
        config.write_text(yaml.safe_dump(_config_payload(), sort_keys=False), encoding="utf-8")
        primary_checkpoint, secondary_checkpoint = root / "m20.pt", root / "m10.pt"
        primary_checkpoint.write_bytes(b"synthetic-m20")
        secondary_checkpoint.write_bytes(b"synthetic-m10")
        records = [_record(stem, high=stem == "val_023") for stem in replay.OFFICIAL_VALIDATION_STEMS]
        primary_cache, secondary_cache = root / "primary.pt", root / "secondary.pt"
        torch.save(_payload(records, primary_checkpoint), primary_cache)
        torch.save(_payload(records, secondary_checkpoint), secondary_cache)

        cfg = replay.load_flat_config(config, frozen.FROZEN_CONFIG_OVERRIDES)
        cfg.temporal_memory_model_path = str(primary_checkpoint)
        cfg.temporal_memory_secondary_model_path = str(secondary_checkpoint)
        cfg.temporal_memory_secondary_max_event_count = 30000
        cfg.component_reranker_enabled = False
        routed = replay.route_cache_records(replay.load_cache(primary_cache), replay.load_cache(secondary_cache), 30000)
        totals = []
        for record in routed:
            threshold = 0.718 if record.event_count <= 30000 else 0.719
            totals.append(frozen._evaluate_one(record, threshold, cfg)[1])
        golden = replay.metrics_from_counts_exact(replay._sum_counts(totals), cfg).to_dict()
        policy = {
            "schema": frozen.POLICY_SCHEMA,
            "status": "frozen_before_v2_artifact_and_before_validation_replay",
            "evaluation_budget": {"full_validation_replays": 1, "threshold_or_hyperparameter_search_after_replay": False},
            "frozen_inference_contract": {
                "low_model_route": "M10 when event_count <= 30000",
                "primary_model_route": "released M20 when event_count > 30000",
                "component_reranker_route": "enabled only when event_count > 100000",
                "low_threshold": 0.718,
                "primary_threshold": 0.719,
                "p0c_retain_min_score": 0.95,
                "p0c_density_retain_enabled": False,
                "postprocess_order": "P0/P0c -> component reranker -> P18",
            },
            "golden_baseline": golden,
            "existing_exploratory_c09": {"score": golden["score"]},
            "promotion_gates": {
                "minimum_score": golden["score"] + 0.0001,
                "minimum_score_delta_over_golden": 0.0001,
                "must_exceed_existing_c09": True,
                "minimum_pd": golden["pd"],
                "minimum_iou": golden["iou"],
                "maximum_fa_exclusive": golden["fa"],
                "noneligible_videos_bitwise_unchanged": True,
                "eligible_video_rule": "event_count > 100000",
                "each_eligible_video_score_delta_nonnegative": True,
                "at_least_one_eligible_video_score_delta_strictly_positive": True,
                "all_gates_required": True,
            },
        }
        paths["policy"].write_text(json.dumps(policy, sort_keys=True), encoding="utf-8")
        policy_sha = sha256_file(paths["policy"])
        p0 = P0ClusterFilterConfig.from_cfg(cfg, event_count=100001)
        provenance = {
            "dataset_split": "train",
            "train_cache_schema": TRAIN_CACHE_SCHEMA,
            "train_cache_manifest_sha256": "1" * 64,
            "base_checkpoint_sha256": sha256_file(primary_checkpoint),
            "deployment_event_count_cutoff": 100000,
            "input_postprocess": input_postprocess_mapping(p0),
            "inference_settings": dict(INFERENCE_SETTINGS),
            "config_sha256": sha256_file(config),
            "config_overrides": list(frozen.FROZEN_CONFIG_OVERRIDES),
            "crossfit_candidate_profile": "posthoc_pw4_kp040_v2",
            "crossfit_hypothesis": {"validation_acceptance_policy": {"schema": frozen.POLICY_SCHEMA, "sha256": policy_sha}},
            "crossfit_code_sha256": {path: sha256_file(PROJECT_ROOT / path) for path in (
                "replay_temporal_memory_validation.py", "utils/challenge_eval.py", "utils/component_reranker.py", "utils/eval.py", "utils/postprocess.py"
            )},
        }
        provenance["input_postprocess_sha256"] = sha256_json(provenance["input_postprocess"])
        provenance["inference_settings_sha256"] = sha256_json(provenance["inference_settings"])
        coefficients = [0.0] * len(FEATURE_NAMES)
        coefficients[FEATURE_NAMES.index("score_mean")] = 20.0
        artifact = {
            "schema": ARTIFACT_SCHEMA,
            "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": [0.0] * len(FEATURE_NAMES),
            "feature_scale": [1.0] * len(FEATURE_NAMES),
            "coefficients": coefficients,
            "intercept": -16.0,
            "keep_probability": 0.4,
            "prediction_threshold": 0.719,
            "topology": ComponentTopology().to_dict(),
            "fit": {"positive_weight": 4.0, "l2": 0.1},
            "provenance": provenance,
        }
        paths["artifact"].write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
        inputs = {
            "policy": paths["policy"],
            "artifact": paths["artifact"],
            "primary_cache": primary_cache,
            "secondary_cache": secondary_cache,
            "config": config,
            "primary_checkpoint": primary_checkpoint,
            "secondary_checkpoint": secondary_checkpoint,
        }
        protocol = frozen.build_execution_protocol(inputs=inputs)
        paths["execution_protocol"].write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
        return paths, sha256_file(paths["execution_protocol"])

    @staticmethod
    def _rewrite_protocol(paths, mutate):
        protocol = json.loads(paths["execution_protocol"].read_text(encoding="utf-8"))
        mutate(protocol)
        paths["execution_protocol"].write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
        return sha256_file(paths["execution_protocol"])

    def test_preflight_checks_metadata_without_claim_or_score(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory):
            paths, digest = self._fixture(directory)
            with mock.patch.object(frozen, "_evaluate_one", side_effect=AssertionError("scored")):
                result = frozen.preflight_execution(paths["execution_protocol"], digest)
            self.assertFalse(result["claim_created"])
            self.assertFalse(result["validation_scored"])
            self.assertFalse(paths["claim"].exists())

    def test_run_passes_then_canonical_claim_blocks_second_attempt(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory):
            paths, digest = self._fixture(directory)
            report = frozen.run_execution(paths["execution_protocol"], digest)
            self.assertTrue(report["passed"])
            self.assertTrue(paths["claim"].is_file())
            with self.assertRaisesRegex(FileExistsError, "claim already exists"):
                frozen.run_execution(paths["execution_protocol"], digest)

    def test_crash_after_claim_consumes_attempt(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory):
            paths, digest = self._fixture(directory)
            with mock.patch.object(replay, "load_cache_snapshot", side_effect=RuntimeError("crash")):
                with self.assertRaisesRegex(RuntimeError, "crash"):
                    frozen.run_execution(paths["execution_protocol"], digest)
            self.assertTrue(paths["claim"].is_file())
            self.assertFalse(paths["report"].exists())
            with self.assertRaisesRegex(FileExistsError, "claim already exists"):
                frozen.run_execution(paths["execution_protocol"], digest)

    def test_noneligible_change_is_failed_report_not_exception(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory):
            paths, digest = self._fixture(directory)
            original = frozen._evaluate_one

            def changed(record, threshold, cfg):
                predictions, counts, metrics = original(record, threshold, cfg)
                if getattr(cfg, "component_reranker_enabled", False) and record.event_count <= 100000:
                    predictions = predictions.clone()
                    predictions[0] = 0
                return predictions, counts, metrics

            with mock.patch.object(frozen, "_evaluate_one", side_effect=changed):
                report = frozen.run_execution(paths["execution_protocol"], digest)
            self.assertFalse(report["passed"])
            self.assertFalse(report["gates"]["noneligible_videos_bitwise_unchanged"])
            self.assertEqual(report["failure_action"], "archive_without_validation_tuning")

    def test_protocol_tamper_is_rejected_before_claim(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory):
            paths, digest = self._fixture(directory)
            self._rewrite_protocol(paths, lambda p: p["runtime"].update({"high_threshold": 0.718}))
            with self.assertRaises(ValueError):
                frozen.run_execution(paths["execution_protocol"], digest)
            self.assertFalse(paths["claim"].exists())

    def test_dirty_or_wrong_head_rejected_before_claim(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory):
            paths, digest = self._fixture(directory)
            for state in (
                {**FAKE_GIT, "clean": False},
                {**FAKE_GIT, "head": "b" * 40},
            ):
                with mock.patch.object(frozen, "_git_state", return_value=state):
                    with self.assertRaises(RuntimeError):
                        frozen.run_execution(paths["execution_protocol"], digest)
                self.assertFalse(paths["claim"].exists())

    def test_wrong_code_or_dataset_rejected_before_claim(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory):
            paths, _ = self._fixture(directory)
            digest = self._rewrite_protocol(
                paths,
                lambda p: p["repository"]["code_sha256"].update(
                    {"utils/eval.py": "f" * 64}
                ),
            )
            with self.assertRaises(RuntimeError):
                frozen.run_execution(paths["execution_protocol"], digest)
            self.assertFalse(paths["claim"].exists())
        with tempfile.TemporaryDirectory() as directory, self._environment(directory):
            paths, _ = self._fixture(directory)
            digest = self._rewrite_protocol(
                paths, lambda p: p["validation_dataset"].update({"event_count": 100117})
            )
            with self.assertRaises(ValueError):
                frozen.run_execution(paths["execution_protocol"], digest)
            self.assertFalse(paths["claim"].exists())

    def test_changed_input_rejected_before_claim(self):
        for input_name in (
            "primary_cache",
            "primary_checkpoint",
            "artifact",
            "config",
        ):
            with tempfile.TemporaryDirectory() as directory, self._environment(directory):
                paths, digest = self._fixture(directory)
                protocol = json.loads(paths["execution_protocol"].read_text(encoding="utf-8"))
                target = Path(protocol["inputs"][input_name]["path"])
                with target.open("ab") as stream:
                    stream.write(b"tamper")
                with self.assertRaises(ValueError):
                    frozen.run_execution(paths["execution_protocol"], digest)
                self.assertFalse(paths["claim"].exists())

    def test_canonical_paths_prevent_alternate_protocol_claim_or_output(self):
        with tempfile.TemporaryDirectory() as directory, self._environment(directory):
            paths, _ = self._fixture(directory)
            digest = self._rewrite_protocol(
                paths,
                lambda p: p["outputs"].update({"report_path": str(Path(directory) / "other.json")}),
            )
            with self.assertRaises(ValueError):
                frozen.run_execution(paths["execution_protocol"], digest)
            alternate = Path(directory) / "alternate_protocol.json"
            alternate.write_bytes(paths["execution_protocol"].read_bytes())
            with self.assertRaises(ValueError):
                frozen.run_execution(alternate, sha256_file(alternate))
            self.assertFalse(paths["claim"].exists())

    def test_cli_run_has_only_protocol_and_hash_inputs(self):
        with self.assertRaises(SystemExit):
            frozen.parse_args(
                [
                    "run",
                    "--execution-protocol",
                    "x",
                    "--expected-execution-protocol-sha256",
                    "f" * 64,
                    "--output",
                    "other.json",
                ]
            )


if __name__ == "__main__":
    unittest.main()
