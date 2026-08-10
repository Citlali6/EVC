"""CPU-only contract tests for the train-only component reranker."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import replay_temporal_memory_validation as replay  # noqa: E402
from train_component_reranker import (  # noqa: E402
    _atomic_npz,
    _load_cache_record,
    fit_weighted_logistic,
    main as reranker_train_main,
)
from utils.component_reranker import (  # noqa: E402
    ARTIFACT_SCHEMA,
    FEATURE_SEMANTICS_VERSION,
    FEATURE_NAMES,
    TRAIN_CACHE_SCHEMA,
    ComponentLinearModel,
    ComponentReranker,
    ComponentRerankerConfig,
    ComponentTopology,
    extract_component_examples,
    input_postprocess_mapping,
    sha256_file,
    sha256_json,
    temporal_memory_inference_mapping,
)
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
        "p0_min_cluster_events": 1,
        "p0_min_duration_bins": 1,
        "p0c_high_confidence_recovery_enabled": False,
        "p0c_retain_min_score": 0.95,
        "p0c_density_retain_enabled": False,
        "p0c_density_event_count_cutoff": 100000,
        "p0c_density_retain_min_score": 0.97,
        "p0b_enabled": False,
        "p18_score_track_recovery_enabled": False,
        "component_reranker_enabled": False,
        "component_reranker_event_count_cutoff": 100000,
        "component_reranker_model_path": "",
        "component_reranker_expected_sha256": "",
        "temporal_memory_enabled": True,
        "temporal_memory_sparse_weight": 0.0,
        "temporal_memory_model_path": "unused.pt",
        "temporal_memory_bin_size": 50,
        "temporal_memory_context_bins": 5,
        "temporal_memory_width": 16,
        "temporal_memory_sequence_length": 16,
        "temporal_memory_inference_batch_size": 8,
        "temporal_memory_log_count_clip": 4.0,
        "whole_t": 8000,
        "res": [346, 260],
        "temporal_memory_secondary_model_path": "",
        "temporal_memory_secondary_max_event_count": 0,
        "temporal_memory_blend_model_path": "",
        "temporal_frame_enabled": False,
        "dense_expert_enabled": False,
        "ensemble_enabled": False,
    }
    options.update(overrides)
    return SimpleNamespace(**options)


def artifact_payload(checkpoint_sha256, p0_config, inference_settings=None, **overrides):
    if inference_settings is None:
        inference_settings = temporal_memory_inference_mapping(make_cfg())
    provenance = {
        "dataset_split": "train",
        "base_checkpoint_sha256": checkpoint_sha256,
        "train_cache_manifest_sha256": "1" * 64,
        "train_cache_schema": TRAIN_CACHE_SCHEMA,
        "deployment_event_count_cutoff": 100000,
        "input_postprocess": input_postprocess_mapping(p0_config),
        "inference_settings": inference_settings,
    }
    provenance["input_postprocess_sha256"] = sha256_json(
        provenance["input_postprocess"]
    )
    provenance["inference_settings_sha256"] = sha256_json(
        provenance["inference_settings"]
    )
    payload = {
        "schema": ARTIFACT_SCHEMA,
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": [0.0] * len(FEATURE_NAMES),
        "feature_scale": [1.0] * len(FEATURE_NAMES),
        "coefficients": [0.0] * len(FEATURE_NAMES),
        "intercept": -2.0,
        "keep_probability": 0.5,
        "prediction_threshold": 0.719,
        "topology": ComponentTopology().to_dict(),
        "provenance": provenance,
    }
    payload.update(overrides)
    return payload


class ComponentRerankerTest(unittest.TestCase):
    def test_default_off_is_bitwise_identity_and_does_not_open_artifact(self):
        cfg = make_cfg(
            component_reranker_model_path="Z:/definitely/missing.json",
            component_reranker_expected_sha256="not-a-sha",
        )
        postprocessor = ChallengePostprocessor.from_cfg(
            cfg, prediction_threshold=0.719, event_count=100001
        )
        predictions = torch.tensor([0.8, 0.2], dtype=torch.float32)
        locations = torch.tensor(
            [[0, 10, 10, 0], [0, 20, 20, 0]], dtype=torch.int64
        )
        output, stats = postprocessor.apply(predictions, locations)
        self.assertIs(output, predictions)
        self.assertTrue(torch.equal(output, predictions))
        self.assertFalse(stats.reranker_stats.enabled)

    def test_dense_gate_is_strict_and_ineligible_path_is_not_loaded(self):
        cfg = make_cfg(
            component_reranker_enabled=True,
            component_reranker_model_path="Z:/missing.json",
            component_reranker_expected_sha256="0" * 64,
            temporal_memory_model_path="Z:/missing.pt",
        )
        reranker = ComponentReranker.from_cfg(
            cfg,
            prediction_threshold=0.719,
            event_count=100000,
            input_postprocess=P0ClusterFilterConfig.from_cfg(cfg, event_count=100000),
        )
        self.assertTrue(reranker.enabled)
        self.assertFalse(reranker.eligible)
        self.assertIsNone(reranker.model)

    def test_rejects_score_routes_not_bound_to_m20_primary(self):
        base = {
            "component_reranker_enabled": True,
            "component_reranker_model_path": "Z:/missing.json",
            "component_reranker_expected_sha256": "0" * 64,
            "temporal_memory_model_path": "Z:/missing.pt",
        }
        invalid = (
            ({"temporal_memory_sparse_weight": 0.1}, "pure temporal-memory"),
            ({"temporal_frame_enabled": True}, "temporal-frame"),
            ({"temporal_memory_blend_model_path": "blend.pt"}, "high-density blend"),
            ({"dense_expert_enabled": True}, "dense_expert"),
            ({"ensemble_enabled": True}, "ensemble"),
            (
                {
                    "temporal_memory_secondary_model_path": "secondary.pt",
                    "temporal_memory_secondary_max_event_count": 100001,
                },
                "secondary temporal-memory",
            ),
        )
        for overrides, message in invalid:
            with self.subTest(overrides=overrides):
                cfg = make_cfg(**base, **overrides)
                with self.assertRaisesRegex(ValueError, message):
                    ComponentReranker.from_cfg(
                        cfg,
                        0.719,
                        event_count=100000,
                        input_postprocess=P0ClusterFilterConfig.from_cfg(
                            cfg, event_count=100000
                        ),
                    )

    def test_radius_one_matches_eight_connectivity(self):
        scores = np.asarray([0.9, 0.9], dtype=np.float64)
        locations = np.asarray(
            [[0, 10, 10, 0], [0, 12, 10, 0]], dtype=np.int64
        )
        examples = extract_component_examples(
            scores,
            locations,
            prediction_threshold=0.719,
            topology=ComponentTopology(spatial_radius=1),
            video_event_count=100001,
        )
        self.assertEqual(len(examples), 2)
        self.assertEqual([item.event_indices.tolist() for item in examples], [[0], [1]])

    def test_eligible_apply_requires_complete_single_video_locations(self):
        model = SimpleNamespace(
            topology=ComponentTopology(),
            keep_probability=0.5,
            predict_keep_probability=lambda features: np.ones(features.shape[0]),
        )
        mismatch = ComponentReranker(
            ComponentRerankerConfig(
                enabled=True,
                event_count_cutoff=1,
                model_path="unused.json",
                expected_sha256="0" * 64,
                base_checkpoint_path="unused.pt",
                event_count=3,
            ),
            prediction_threshold=0.719,
            model=model,
        )
        with self.assertRaisesRegex(ValueError, "event_count 3 does not match"):
            mismatch.apply(
                torch.tensor([0.9, 0.8]),
                torch.tensor([[0, 1, 1, 0], [0, 2, 2, 0]], dtype=torch.int64),
            )

        multi_batch = ComponentReranker(
            ComponentRerankerConfig(
                enabled=True,
                event_count_cutoff=1,
                model_path="unused.json",
                expected_sha256="0" * 64,
                base_checkpoint_path="unused.pt",
                event_count=2,
            ),
            prediction_threshold=0.719,
            model=model,
        )
        with self.assertRaisesRegex(ValueError, "exactly one complete-video batch id"):
            multi_batch.apply(
                torch.tensor([0.9, 0.8]),
                torch.tensor([[0, 1, 1, 0], [1, 2, 2, 0]], dtype=torch.int64),
            )

    def test_labels_never_change_inference_features(self):
        scores = np.asarray([0.91, 0.89, 0.92], dtype=np.float64)
        locations = np.asarray(
            [[0, 10, 10, 0], [0, 11, 10, 0], [0, 10, 10, 50]],
            dtype=np.int64,
        )
        negative = extract_component_examples(
            scores,
            locations,
            0.719,
            ComponentTopology(),
            120000,
            labels=np.zeros(3, dtype=np.uint8),
        )
        positive = extract_component_examples(
            scores,
            locations,
            0.719,
            ComponentTopology(),
            120000,
            labels=np.ones(3, dtype=np.uint8),
        )
        self.assertEqual(len(negative), len(positive))
        for left, right in zip(negative, positive):
            np.testing.assert_array_equal(left.event_indices, right.event_indices)
            np.testing.assert_array_equal(left.features, right.features)
            self.assertEqual(left.label, 0)
            self.assertEqual(right.label, 1)
        forbidden = {"file_name", "target_id", "label", "source_name"}
        self.assertTrue(forbidden.isdisjoint(FEATURE_NAMES))

    def test_strict_json_sha_checkpoint_and_p0_binding(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_dir = Path(temporary_dir)
            checkpoint = temporary_dir / "m20.pt"
            checkpoint.write_bytes(b"trusted-test-checkpoint")
            p0_config = P0ClusterFilterConfig.from_cfg(
                make_cfg(), event_count=100001
            )
            artifact = temporary_dir / "reranker.json"
            artifact.write_text(
                json.dumps(
                    artifact_payload(sha256_file(checkpoint), p0_config),
                    sort_keys=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            digest = sha256_file(artifact)
            loaded = ComponentLinearModel.load(
                artifact,
                digest,
                checkpoint,
                0.719,
                100000,
                p0_config,
                temporal_memory_inference_mapping(make_cfg()),
            )
            self.assertEqual(loaded.artifact_sha256, digest)
            with self.assertRaisesRegex(ValueError, "P0/P0c input contract"):
                ComponentLinearModel.load(
                    artifact,
                    digest,
                    checkpoint,
                    0.719,
                    100000,
                    P0ClusterFilterConfig.from_cfg(
                        make_cfg(p0_min_duration_bins=2), event_count=100001
                    ),
                    temporal_memory_inference_mapping(make_cfg()),
                )
            with self.assertRaisesRegex(ValueError, "inference settings differ"):
                ComponentLinearModel.load(
                    artifact,
                    digest,
                    checkpoint,
                    0.719,
                    100000,
                    p0_config,
                    temporal_memory_inference_mapping(
                        make_cfg(temporal_memory_log_count_clip=3.5)
                    ),
                )
            with self.assertRaisesRegex(ValueError, "P0/P0c input contract"):
                ComponentLinearModel.load(
                    artifact,
                    digest,
                    checkpoint,
                    0.719,
                    100000,
                    P0ClusterFilterConfig.from_cfg(
                        make_cfg(
                            p0c_high_confidence_recovery_enabled=True,
                            p0c_density_retain_enabled=True,
                        ),
                        event_count=100001,
                    ),
                    temporal_memory_inference_mapping(make_cfg()),
                )

    def test_runtime_order_is_p0_then_reranker_then_p18(self):
        calls = []

        class Stage:
            enabled = True

            def __init__(self, name):
                self.name = name

            def apply(self, predictions, locations):
                calls.append(self.name)
                return predictions, SimpleNamespace(enabled=True)

            def new_stats(self):
                return SimpleNamespace(enabled=True)

            def describe(self):
                return self.name

        pipeline = ChallengePostprocessor(
            Stage("p0"),
            Stage("p18"),
            component_reranker=Stage("reranker"),
        )
        predictions = torch.tensor([0.8])
        locations = torch.tensor([[0, 1, 1, 0]], dtype=torch.int64)
        pipeline.apply(predictions, locations)
        self.assertEqual(calls, ["p0", "reranker", "p18"])

    def test_rejects_val_missing_or_invalid_train_cache_lineage(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_dir = Path(temporary_dir)
            checkpoint = temporary_dir / "m20.pt"
            checkpoint.write_bytes(b"lineage-checkpoint")
            p0_config = P0ClusterFilterConfig.from_cfg(
                make_cfg(), event_count=100001
            )
            cases = (
                ("val", lambda value: value.__setitem__("dataset_split", "val"), "dataset_split=train"),
                ("missing_schema", lambda value: value.pop("train_cache_schema"), "train_cache_schema"),
                (
                    "invalid_manifest",
                    lambda value: value.__setitem__(
                        "train_cache_manifest_sha256", "not-a-sha"
                    ),
                    "train_cache_manifest_sha256",
                ),
            )
            for name, mutate, message in cases:
                with self.subTest(name=name):
                    payload = artifact_payload(sha256_file(checkpoint), p0_config)
                    mutate(payload["provenance"])
                    artifact = temporary_dir / (name + ".json")
                    artifact.write_text(
                        json.dumps(payload, sort_keys=True, allow_nan=False),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        ComponentLinearModel.load(
                            artifact,
                            sha256_file(artifact),
                            checkpoint,
                            0.719,
                            100000,
                            p0_config,
                            temporal_memory_inference_mapping(make_cfg()),
                        )

    def test_rejects_changed_feature_semantics_version(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_dir = Path(temporary_dir)
            checkpoint = temporary_dir / "m20.pt"
            checkpoint.write_bytes(b"semantic-checkpoint")
            p0_config = P0ClusterFilterConfig.from_cfg(
                make_cfg(), event_count=100001
            )
            payload = artifact_payload(sha256_file(checkpoint), p0_config)
            payload["feature_semantics_version"] = "incompatible-v0"
            artifact = temporary_dir / "reranker.json"
            artifact.write_text(
                json.dumps(payload, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "feature semantics version"):
                ComponentLinearModel.load(
                    artifact,
                    sha256_file(artifact),
                    checkpoint,
                    0.719,
                    100000,
                    p0_config,
                    temporal_memory_inference_mapping(make_cfg()),
                )

    def test_weighted_logistic_fit_is_real_and_deterministic(self):
        rng = np.random.default_rng(17)
        features = rng.normal(size=(80, len(FEATURE_NAMES)))
        labels = (features[:, 1] + 0.5 * features[:, 2] > 0).astype(np.uint8)
        first = fit_weighted_logistic(features, labels, 3.0, 0.1, 50)
        second = fit_weighted_logistic(features, labels, 3.0, 0.1, 50)
        np.testing.assert_allclose(first["coefficients"], second["coefficients"])
        self.assertGreater(np.linalg.norm(first["coefficients"]), 0.1)
        self.assertLess(first["weighted_loss"], 0.69)

    def test_atomic_npz_round_trip_and_digest_guard(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache_dir = Path(temporary_dir).resolve()
            record_path = cache_dir / "records" / "000.npz"
            arrays = {
                "scores": np.asarray([0.2, 0.9], dtype=np.float32),
                "locs": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int16),
                "labels": np.asarray([0, 1], dtype=np.uint8),
                "target_ids": np.asarray([0, 7], dtype=np.int16),
            }
            _atomic_npz(record_path, **arrays)
            metadata = {
                "record": "records/000.npz",
                "record_sha256": sha256_file(record_path),
                "event_count": 2,
            }
            loaded = _load_cache_record(cache_dir, metadata)
            for name, expected in arrays.items():
                np.testing.assert_array_equal(loaded[name], expected)
            metadata["record_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                _load_cache_record(cache_dir, metadata)

    def test_synthetic_cache_fit_json_runtime_apply_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir).resolve()
            cache_dir = root / "cache"
            record_path = cache_dir / "records" / "000.npz"
            scores = np.asarray([0.92, 0.88, 0.91, 0.87], dtype=np.float32)
            locations = np.asarray(
                [[10, 10, 0], [11, 10, 0], [30, 30, 0], [31, 30, 0]],
                dtype=np.int16,
            )
            labels = np.asarray([1, 1, 0, 0], dtype=np.uint8)
            target_ids = np.asarray([1, 1, 0, 0], dtype=np.int16)
            _atomic_npz(
                record_path,
                scores=scores,
                locs=locations,
                labels=labels,
                target_ids=target_ids,
            )
            checkpoint = root / "m20.pt"
            checkpoint.write_bytes(b"synthetic-m20-checkpoint")
            config_path = PROJECT_ROOT / "configs" / "evisseg_evuav.yaml"
            overrides = [
                "POSTPROCESS.p0_enabled=true",
                "POSTPROCESS.p0_spatial_radius=1",
                "POSTPROCESS.p0_temporal_bin_size=50",
                "POSTPROCESS.p0_temporal_radius_bins=1",
                "POSTPROCESS.p0_min_cluster_events=1",
                "POSTPROCESS.p0_min_duration_bins=1",
                "POSTPROCESS.p0c_high_confidence_recovery_enabled=false",
                "POSTPROCESS.p0c_density_retain_enabled=false",
                "POSTPROCESS.p0b_enabled=false",
                "POSTPROCESS.p18_score_track_recovery_enabled=false",
            ]
            fit_cfg = replay.load_flat_config(config_path, overrides)
            manifest = {
                "schema": TRAIN_CACHE_SCHEMA,
                "dataset_split": "train",
                "selection": {
                    "observable": "complete_video_event_count",
                    "operator": ">",
                    "min_event_count_exclusive": 0,
                },
                "selected_video_count": 1,
                "base_checkpoint_sha256": sha256_file(checkpoint),
                "inference_settings": temporal_memory_inference_mapping(fit_cfg),
                "records": [
                    {
                        "record": "records/000.npz",
                        "record_sha256": sha256_file(record_path),
                        "source_name": "train_000.npz",
                        "event_count": 4,
                    }
                ],
            }
            (cache_dir / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            artifact_path = root / "reranker.json"
            arguments = [
                "fit",
                "--config",
                str(config_path),
                "--cache-dir",
                str(cache_dir),
                "--output-model",
                str(artifact_path),
                "--prediction-threshold",
                "0.719",
                "--positive-weight",
                "2.0",
                "--keep-probability",
                "0.5",
                "--minimum-train-component-recall",
                "0.0",
            ]
            for override in overrides:
                arguments.extend(("--override", override))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(reranker_train_main(arguments), 0)
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["feature_semantics_version"], FEATURE_SEMANTICS_VERSION
            )
            self.assertEqual(artifact["provenance"]["dataset_split"], "train")

            runtime_cfg = replay.load_flat_config(config_path, overrides)
            runtime_cfg.component_reranker_enabled = True
            runtime_cfg.component_reranker_event_count_cutoff = 100000
            runtime_cfg.component_reranker_model_path = str(artifact_path)
            runtime_cfg.component_reranker_expected_sha256 = sha256_file(
                artifact_path
            )
            runtime_cfg.temporal_memory_enabled = True
            runtime_cfg.temporal_memory_sparse_weight = 0.0
            runtime_cfg.temporal_memory_model_path = str(checkpoint)
            runtime_cfg.temporal_memory_secondary_model_path = ""
            runtime_cfg.temporal_memory_secondary_max_event_count = 0
            runtime_cfg.temporal_memory_blend_model_path = ""
            runtime_cfg.temporal_frame_enabled = False
            runtime_cfg.dense_expert_enabled = False
            runtime_cfg.ensemble_enabled = False
            postprocessor = ChallengePostprocessor.from_cfg(
                runtime_cfg,
                prediction_threshold=0.719,
                event_count=100001,
            )
            runtime_scores = torch.full((100001,), 0.1, dtype=torch.float32)
            runtime_scores[:4] = torch.from_numpy(scores)
            runtime_locations = torch.zeros((100001, 4), dtype=torch.int64)
            runtime_locations[:4, 1:] = torch.from_numpy(
                locations.astype(np.int64)
            )
            output, stats = postprocessor.apply(runtime_scores, runtime_locations)
            self.assertEqual(output.shape, runtime_scores.shape)
            self.assertEqual(stats.reranker_stats.eligible_videos, 1)
            self.assertEqual(stats.reranker_stats.candidate_components, 2)

    def test_all_yaml_configs_ship_safe_defaults(self):
        self.assertEqual(len(CONFIG_PATHS), 5)
        for path in CONFIG_PATHS:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))["POSTPROCESS"]
            with self.subTest(path=path.name):
                self.assertIs(config["component_reranker_enabled"], False)
                self.assertEqual(config["component_reranker_event_count_cutoff"], 100000)
                self.assertEqual(config["component_reranker_model_path"], "")
                self.assertEqual(config["component_reranker_expected_sha256"], "")


if __name__ == "__main__":
    unittest.main()
