import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import replay_temporal_memory_validation as replay
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.component_reranker import (
    ARTIFACT_SCHEMA,
    FEATURE_SEMANTICS_VERSION,
    FEATURE_NAMES,
    TRAIN_CACHE_SCHEMA,
    ComponentTopology,
    input_postprocess_mapping,
    sha256_file,
    sha256_json,
)
from utils.density_threshold import select_density_threshold
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor, P0ClusterFilterConfig


def make_cfg(**overrides):
    options = dict(
        roc=True,
        pd_detT=50,
        correct_thresh=0.5,
        p0_enabled=True,
        p0_spatial_radius=1,
        p0_temporal_bin_size=50,
        p0_temporal_radius_bins=1,
        p0_min_cluster_events=2,
        p0_min_duration_bins=2,
        p0c_high_confidence_recovery_enabled=True,
        p0c_retain_min_score=0.95,
        p0c_density_retain_enabled=False,
        p0c_density_event_count_cutoff=100000,
        p0c_density_retain_min_score=0.97,
        p0b_enabled=False,
        p18_score_track_recovery_enabled=True,
        p18_event_count_cutoff=1,
        p18_max_event_count=100,
        p18_candidate_floor=0.53,
        p18_spatial_radius=1,
        p18_temporal_bin_size=50,
        p18_max_link_distance=3.0,
        p18_max_gap_bins=1,
        p18_min_track_bins=2,
        p18_restore_mode="best",
        p18_max_restore_events_per_component=0,
        component_reranker_enabled=False,
        component_reranker_event_count_cutoff=100000,
        component_reranker_model_path="",
        component_reranker_expected_sha256="",
        temporal_memory_enabled=True,
        temporal_memory_model_path="",
        temporal_memory_sparse_weight=0.0,
        temporal_memory_secondary_model_path="",
        temporal_memory_secondary_max_event_count=30000,
        temporal_memory_blend_model_path="",
        temporal_memory_bin_size=50,
        temporal_memory_context_bins=5,
        temporal_memory_width=16,
        temporal_memory_sequence_length=16,
        temporal_memory_inference_batch_size=8,
        temporal_memory_log_count_clip=4.0,
        temporal_frame_enabled=False,
        dense_expert_enabled=False,
        ensemble_enabled=False,
        whole_t=5000,
        res=[346, 260],
    )
    options.update(overrides)
    return SimpleNamespace(**options)


def make_record(file_name, scores):
    scores = torch.tensor(scores, dtype=torch.float32)
    repeats = (len(scores) + 7) // 8
    x = torch.tensor([10, 10, 11, 11, 30, 30, 31, 31], dtype=torch.int64).repeat(repeats)[: len(scores)]
    y = torch.tensor([10, 10, 10, 10, 30, 30, 30, 30], dtype=torch.int64).repeat(repeats)[: len(scores)]
    t = torch.tensor([10, 60, 61, 110, 10, 60, 61, 110], dtype=torch.int64).repeat(repeats)[: len(scores)]
    locs = torch.column_stack((torch.zeros(len(scores), dtype=torch.int64), x, y, t))
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32).repeat(repeats)[: len(scores)]
    target_ids = np.tile(
        np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64), repeats
    )[: len(scores)]
    digest = replay.source_digest(locs, labels, target_ids)
    return {
        "file_name": file_name,
        "event_count": len(scores),
        "scores": scores,
        "seg_label": labels,
        "locs": locs,
        "idx_label": target_ids,
        "source_sha256": digest,
    }


DEFAULT_INFERENCE_SETTINGS = {
    "temporal_memory_bin_size": 50,
    "temporal_memory_context_bins": 5,
    "temporal_memory_width": 16,
    "temporal_memory_sequence_length": 16,
    "temporal_memory_inference_batch_size": 8,
    "temporal_memory_log_count_clip": 4.0,
    "whole_t": 5000,
    "resolution": [346, 260],
}


def complete_official_records(records):
    records = list(records)
    used_stems = {Path(record["file_name"]).stem for record in records}
    for stem in replay.OFFICIAL_VALIDATION_STEMS:
        if stem not in used_stems:
            records.append(
                make_record(
                    stem + ".npz",
                    [0.99, 0.80, 0.54, 0.72, 0.80, 0.54, 0.20, 0.72],
                )
            )
    return sorted(records, key=lambda record: Path(record["file_name"]).stem)


def make_payload(
    records,
    checkpoint_sha="a" * 64,
    checkpoint_path=None,
    complete=True,
    inference_settings=None,
    code_sha256=None,
):
    records = complete_official_records(records) if complete else list(records)
    if inference_settings is None:
        inference_settings = dict(DEFAULT_INFERENCE_SETTINGS)
    if code_sha256 is None:
        code_sha256 = {
            relative_path: "c" * 64
            for relative_path in replay.CACHE_CODE_PROVENANCE_PATHS
        }
    metadata = {
            "schema": replay.CACHE_SCHEMA,
            "dataset_split": "val",
            "dataset_signature": replay._dataset_signature(records),
            "video_count": len(records),
            "event_count": sum(record["event_count"] for record in records),
            "checkpoint_sha256": checkpoint_sha,
            "inference_settings": inference_settings,
            "code_sha256": code_sha256,
        }
    if checkpoint_path is not None:
        metadata["checkpoint_path"] = str(Path(checkpoint_path).resolve())
    return {
        "metadata": metadata,
        "records": records,
    }


def make_warm_t32_checkpoint(directory, mutate=None):
    directory = Path(directory)
    parent_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "m20_attn_dense_views8_epoch_003_seed48.pt"
    ).resolve()
    parent_checkpoint = replay._torch_load_cpu(parent_path)
    migration = {
        "name": replay.WARM_ROUTE_MIGRATION_NAME,
        "source_sequence_length": 16,
        "target_sequence_length": 32,
        "metadata_difference_allowlist": ["sequence_length"],
        "state_dict_strict": True,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": replay.RELEASED_M20_SHA256,
        "source_model_state_sha256": replay.RELEASED_M20_STATE_SHA256,
        "loaded_model_state_sha256": replay.RELEASED_M20_STATE_SHA256,
    }
    model_state = copy.deepcopy(parent_checkpoint["model_state_dict"])
    model_state["temporal_attn.output_projection.bias"] = (
        model_state["temporal_attn.output_projection.bias"] + 0.125
    )
    checkpoint = {
        "checkpoint_format_version": 2,
        "model_state_dict": model_state,
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "rng_state": {},
        "temporal_memory": {
            "temporal_bin_size": 50,
            "context_bins": 5,
            "width": 16,
            "sequence_length": 32,
            "log_count_clip": 4.0,
            "density_calibration_enabled": True,
            "density_calibration_version": 1,
            "trajectory_extrapolation_enabled": False,
            "confidence_head_enabled": False,
            "confidence_only_enabled": False,
            "freeze_base_enabled": False,
            "head_only_enabled": False,
            "dacc_v2_only_enabled": False,
            "attention_projection_only_enabled": True,
            "init_sequence_length_warm_start_enabled": True,
            "temporal_attention_enabled": True,
        },
        "provenance": {
            "initialized_from": str(parent_path),
            "initialized_from_sha256": replay.RELEASED_M20_SHA256,
            "initialization_migrations": [migration],
            "resolved_config": {
                "TEMPORAL_MEMORY": {
                    "temporal_memory_sequence_length": 32,
                    "temporal_memory_init_sequence_length_warm_start_enabled": True,
                    "temporal_memory_attention_projection_only_enabled": True,
                }
            },
            "training_scope": {
                "name": "temporal_attention_projection_only",
                "trainable_parameter_count": 9312,
                "mutable_state_keys": [
                    "temporal_attn.output_projection.bias",
                    "temporal_attn.output_projection.weight",
                ],
                "frozen_state_reference_sha256": (
                    replay.RELEASED_M20_FROZEN_STATE_SHA256
                ),
            },
        },
    }
    if mutate is not None:
        mutate(checkpoint)
    checkpoint_path = directory / "warm-t32.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, replay.sha256_file(checkpoint_path)


def make_warm_route_payloads(directory, primary_settings=None):
    directory = Path(directory)
    primary_checkpoint, primary_sha256 = make_warm_t32_checkpoint(directory)
    secondary_checkpoint = (
        PROJECT_ROOT / "checkpoints" / "m10_dense_views2_epoch_002_seed42.pt"
    ).resolve()
    if primary_settings is None:
        primary_settings = dict(DEFAULT_INFERENCE_SETTINGS)
        primary_settings["temporal_memory_sequence_length"] = 32
    primary = make_payload(
        [],
        checkpoint_sha=primary_sha256,
        checkpoint_path=primary_checkpoint,
        inference_settings=primary_settings,
    )
    secondary = make_payload(
        [],
        checkpoint_sha=replay.RELEASED_M10_SHA256,
        checkpoint_path=secondary_checkpoint,
        inference_settings=DEFAULT_INFERENCE_SETTINGS,
    )
    return primary, secondary, primary_settings


def make_reranker_fixture(directory):
    directory = Path(directory)
    checkpoint = directory / "m20.pt"
    checkpoint.write_bytes(b"replay-bound-m20")
    checkpoint_sha256 = sha256_file(checkpoint)
    cfg = make_cfg(
        component_reranker_enabled=True,
        temporal_memory_model_path=str(checkpoint),
    )
    p0_config = P0ClusterFilterConfig.from_cfg(cfg, event_count=100001)
    inference_settings = replay._inference_settings(cfg)
    provenance = {
        "dataset_split": "train",
        "train_cache_schema": TRAIN_CACHE_SCHEMA,
        "train_cache_manifest_sha256": "1" * 64,
        "base_checkpoint_sha256": checkpoint_sha256,
        "deployment_event_count_cutoff": 100000,
        "input_postprocess": input_postprocess_mapping(p0_config),
        "inference_settings": inference_settings,
    }
    provenance["input_postprocess_sha256"] = sha256_json(
        provenance["input_postprocess"]
    )
    provenance["inference_settings_sha256"] = sha256_json(inference_settings)
    artifact = directory / "reranker.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": ARTIFACT_SCHEMA,
                "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "feature_mean": [0.0] * len(FEATURE_NAMES),
                "feature_scale": [1.0] * len(FEATURE_NAMES),
                "coefficients": [0.0] * len(FEATURE_NAMES),
                "intercept": 0.0,
                "keep_probability": 0.5,
                "prediction_threshold": 0.719,
                "topology": ComponentTopology().to_dict(),
                "provenance": provenance,
            },
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    cfg.component_reranker_model_path = str(artifact)
    cfg.component_reranker_expected_sha256 = sha256_file(artifact)
    return cfg, checkpoint_sha256


def direct_test2_metrics(records, density_cutoff, low_threshold, high_threshold, cfg):
    evaluator = evalute(cfg)
    sample_number = 0
    fallback_threshold = high_threshold
    for record in records:
        threshold = select_density_threshold(
            record.event_count,
            density_cutoff,
            low_threshold,
            high_threshold,
        )
        postprocessor = ChallengePostprocessor.from_cfg(
            cfg,
            threshold,
            event_count=record.event_count,
        )
        predictions, _ = postprocessor.apply(record.scores.clone(), record.locs)
        # This is the P6 branch in test2.py: persist each video's selected
        # decision as binary before the global semantic evaluation.
        predictions = (predictions >= threshold).to(predictions.dtype)
        sample_number = add_batch_to_evaluator(
            evaluator,
            {
                "seg_label": record.seg_label,
                "locs": record.locs,
                "idx_label": record.idx_label,
            },
            predictions,
            sample_number,
            prediction_threshold=threshold,
        )
    return evaluate_challenge_metrics(evaluator, fallback_threshold)


class ThresholdGridTest(unittest.TestCase):
    def test_decimal_grid_is_inclusive_and_stable(self):
        values = replay.decimal_grid("0.710", "0.730", "0.001")
        self.assertEqual(len(values), 21)
        self.assertEqual(values[0], 0.710)
        self.assertEqual(values[-1], 0.730)
        self.assertEqual(values[8], 0.718)

    def test_rejects_non_divisible_grid(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            replay.decimal_grid("0.71", "0.73", "0.003")


class CacheValidationAndRoutingTest(unittest.TestCase):
    def test_warm_t32_m10_route_is_explicit_and_uses_fixed_boundary(self):
        parsed_default = replay.parse_args(
            [
                "replay",
                "--primary-cache",
                "primary.pt",
                "--output-json",
                "result.json",
                "--output-csv",
                "result.csv",
            ]
        )
        self.assertFalse(
            parsed_default.allow_warm_primary_t32_secondary_m10_t16
        )
        parsed_enabled = replay.parse_args(
            [
                "replay",
                "--primary-cache",
                "primary.pt",
                "--output-json",
                "result.json",
                "--output-csv",
                "result.csv",
                "--allow-warm-primary-t32-secondary-m10-t16",
            ]
        )
        self.assertTrue(
            parsed_enabled.allow_warm_primary_t32_secondary_m10_t16
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            primary_template, secondary_template, runtime_settings = (
                make_warm_route_payloads(temporary_directory)
            )
            low_primary = make_record("val_000.npz", [0.1] * 30000)
            high_primary = make_record("val_001.npz", [0.2] * 30001)
            low_secondary = make_record("val_000.npz", [0.8] * 30000)
            high_secondary = make_record("val_001.npz", [0.9] * 30001)
            primary = make_payload(
                [low_primary, high_primary],
                checkpoint_sha=primary_template["metadata"]["checkpoint_sha256"],
                checkpoint_path=primary_template["metadata"]["checkpoint_path"],
                inference_settings=runtime_settings,
            )
            secondary = make_payload(
                [low_secondary, high_secondary],
                checkpoint_sha=replay.RELEASED_M10_SHA256,
                checkpoint_path=secondary_template["metadata"]["checkpoint_path"],
                inference_settings=DEFAULT_INFERENCE_SETTINGS,
            )
            with self.assertRaisesRegex(ValueError, "inference settings"):
                replay.route_cache_records(
                    primary,
                    secondary,
                    secondary_max_events=30000,
                )
            routed = replay.route_cache_records(
                primary,
                secondary,
                secondary_max_events=30000,
                allow_warm_primary_t32_secondary_m10_t16=True,
                runtime_inference_settings=runtime_settings,
            )
            by_name = {record.file_name: record for record in routed}
            self.assertEqual(by_name["val_000.npz"].score_source, "secondary")
            self.assertEqual(by_name["val_001.npz"].score_source, "primary")
            self.assertTrue(
                torch.equal(by_name["val_000.npz"].scores, low_secondary["scores"])
            )
            self.assertTrue(
                torch.equal(by_name["val_001.npz"].scores, high_primary["scores"])
            )

    def test_warm_t32_m10_route_rejects_every_non_sequence_difference(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            primary, secondary, runtime_settings = make_warm_route_payloads(
                temporary_directory
            )
            changed_primary = copy.deepcopy(primary)
            changed_runtime = dict(runtime_settings)
            changed_primary["metadata"]["inference_settings"][
                "temporal_memory_inference_batch_size"
            ] = 4
            changed_runtime["temporal_memory_inference_batch_size"] = 4
            with self.assertRaisesRegex(ValueError, "may differ only"):
                replay.route_cache_records(
                    changed_primary,
                    secondary,
                    secondary_max_events=30000,
                    allow_warm_primary_t32_secondary_m10_t16=True,
                    runtime_inference_settings=changed_runtime,
                )

            wrong_runtime = dict(runtime_settings)
            wrong_runtime["whole_t"] = 8000
            with self.assertRaisesRegex(ValueError, "differ from runtime"):
                replay.route_cache_records(
                    primary,
                    secondary,
                    secondary_max_events=30000,
                    allow_warm_primary_t32_secondary_m10_t16=True,
                    runtime_inference_settings=wrong_runtime,
                )

            for invalid_cutoff in (29999, 30001):
                with self.subTest(invalid_cutoff=invalid_cutoff):
                    with self.assertRaisesRegex(ValueError, "exactly 30000"):
                        replay.route_cache_records(
                            primary,
                            secondary,
                            secondary_max_events=invalid_cutoff,
                            allow_warm_primary_t32_secondary_m10_t16=True,
                            runtime_inference_settings=runtime_settings,
                        )

            changed_code = copy.deepcopy(secondary)
            changed_code["metadata"]["code_sha256"][
                "model/temporal_memory_net.py"
            ] = "d" * 64
            with self.assertRaisesRegex(ValueError, "inference code"):
                replay.route_cache_records(
                    primary,
                    changed_code,
                    secondary_max_events=30000,
                    allow_warm_primary_t32_secondary_m10_t16=True,
                    runtime_inference_settings=runtime_settings,
                )

    def test_warm_t32_m10_route_rejects_checkpoint_lineage_tampering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            primary, secondary, runtime_settings = make_warm_route_payloads(root)
            primary_checkpoint = replay._torch_load_cpu(
                Path(primary["metadata"]["checkpoint_path"])
            )

            def primary_payload_for(checkpoint_payload, name):
                checkpoint_path = root / name
                torch.save(checkpoint_payload, checkpoint_path)
                changed = copy.deepcopy(primary)
                changed["metadata"]["checkpoint_path"] = str(
                    checkpoint_path.resolve()
                )
                changed["metadata"]["checkpoint_sha256"] = replay.sha256_file(
                    checkpoint_path
                )
                return changed

            primary_checkpoint["provenance"]["initialization_migrations"].append(
                {"name": "unauthorized_extra_migration"}
            )
            tampered_primary = primary_payload_for(
                primary_checkpoint,
                "tampered-primary.pt",
            )
            with self.assertRaisesRegex(ValueError, "sole initialization"):
                replay.route_cache_records(
                    tampered_primary,
                    secondary,
                    secondary_max_events=30000,
                    allow_warm_primary_t32_secondary_m10_t16=True,
                    runtime_inference_settings=runtime_settings,
                )

            wrong_state_lineage = replay._torch_load_cpu(
                Path(primary["metadata"]["checkpoint_path"])
            )
            wrong_state_lineage["provenance"]["initialization_migrations"][0][
                "source_model_state_sha256"
            ] = "3" * 64
            wrong_state_lineage["provenance"]["initialization_migrations"][0][
                "loaded_model_state_sha256"
            ] = "3" * 64
            with self.assertRaisesRegex(ValueError, "inherited state provenance"):
                replay.route_cache_records(
                    primary_payload_for(
                        wrong_state_lineage,
                        "wrong-state-lineage.pt",
                    ),
                    secondary,
                    secondary_max_events=30000,
                    allow_warm_primary_t32_secondary_m10_t16=True,
                    runtime_inference_settings=runtime_settings,
                )

            changed_frozen_tensor = replay._torch_load_cpu(
                Path(primary["metadata"]["checkpoint_path"])
            )
            changed_frozen_tensor["model_state_dict"][
                "memory_projection.bias"
            ][0] += 1.0
            with self.assertRaisesRegex(ValueError, "frozen model state"):
                replay.route_cache_records(
                    primary_payload_for(
                        changed_frozen_tensor,
                        "changed-frozen-tensor.pt",
                    ),
                    secondary,
                    secondary_max_events=30000,
                    allow_warm_primary_t32_secondary_m10_t16=True,
                    runtime_inference_settings=runtime_settings,
                )

            fake_parent = root / "fake-m20-parent.pt"
            fake_parent.write_bytes(b"not-released-m20")
            wrong_parent = replay._torch_load_cpu(
                Path(primary["metadata"]["checkpoint_path"])
            )
            wrong_parent["provenance"]["initialized_from"] = str(fake_parent)
            wrong_parent["provenance"]["initialization_migrations"][0][
                "parent_checkpoint"
            ] = str(fake_parent)
            with self.assertRaisesRegex(ValueError, "file is not released M20"):
                replay.route_cache_records(
                    primary_payload_for(wrong_parent, "wrong-parent.pt"),
                    secondary,
                    secondary_max_events=30000,
                    allow_warm_primary_t32_secondary_m10_t16=True,
                    runtime_inference_settings=runtime_settings,
                )

            incomplete_state = replay._torch_load_cpu(
                Path(primary["metadata"]["checkpoint_path"])
            )
            incomplete_state["model_state_dict"].pop(
                next(iter(incomplete_state["model_state_dict"]))
            )
            with self.assertRaisesRegex(ValueError, "complete 89-key"):
                replay.route_cache_records(
                    primary_payload_for(incomplete_state, "incomplete-state.pt"),
                    secondary,
                    secondary_max_events=30000,
                    allow_warm_primary_t32_secondary_m10_t16=True,
                    runtime_inference_settings=runtime_settings,
                )

            fake_m10_path = root / "fake-m10.pt"
            torch.save({"temporal_memory": {}}, fake_m10_path)
            fake_secondary = copy.deepcopy(secondary)
            fake_secondary["metadata"]["checkpoint_path"] = str(
                fake_m10_path.resolve()
            )
            fake_secondary["metadata"]["checkpoint_sha256"] = replay.sha256_file(
                fake_m10_path
            )
            with self.assertRaisesRegex(ValueError, "not released M10"):
                replay.route_cache_records(
                    primary,
                    fake_secondary,
                    secondary_max_events=30000,
                    allow_warm_primary_t32_secondary_m10_t16=True,
                    runtime_inference_settings=runtime_settings,
                )

            missing_path = copy.deepcopy(primary)
            missing_path["metadata"].pop("checkpoint_path")
            with self.assertRaisesRegex(ValueError, "checkpoint path provenance"):
                replay.route_cache_records(
                    missing_path,
                    secondary,
                    secondary_max_events=30000,
                    allow_warm_primary_t32_secondary_m10_t16=True,
                    runtime_inference_settings=runtime_settings,
                )

    def test_secondary_routing_depends_only_on_event_count(self):
        # Suffixes deliberately imply the opposite routing; only event count
        # may select the cached checkpoint.  Canonical stems remain mandatory.
        low_primary = make_record("val_000.looks_high", [0.1] * 8)
        high_primary = make_record("val_001.looks_low", [0.2] * 80)
        low_secondary = make_record("val_000.looks_high", [0.8] * 8)
        high_secondary = make_record("val_001.looks_low", [0.9] * 80)
        routed = replay.route_cache_records(
            make_payload([low_primary, high_primary]),
            make_payload([low_secondary, high_secondary], checkpoint_sha="b" * 64),
            secondary_max_events=30,
        )
        self.assertEqual(routed[0].score_source, "secondary")
        self.assertEqual(routed[1].score_source, "primary")
        self.assertTrue(torch.equal(routed[0].scores, low_secondary["scores"]))
        self.assertTrue(torch.equal(routed[1].scores, high_primary["scores"]))

    def test_rejects_tampered_source_fields(self):
        record = make_record("val_000.npz", [0.1] * 8)
        payload = make_payload([record])
        payload["records"][0]["locs"][0, 1] += 1
        with self.assertRaisesRegex(ValueError, "digest"):
            replay.validate_cache_payload(payload)

    def test_rejects_scores_outside_probability_range(self):
        record = make_record("val_000.npz", [0.1] * 8)
        record["scores"][0] = 1.1
        payload = make_payload([record])
        with self.assertRaisesRegex(ValueError, "non-probability"):
            replay.validate_cache_payload(payload)

    def test_rejects_incomplete_or_noncanonical_validation_split(self):
        incomplete = make_payload(
            [make_record("val_000.npz", [0.1] * 8)],
            complete=False,
        )
        with self.assertRaisesRegex(ValueError, "exactly 24"):
            replay.validate_cache_payload(incomplete)

        records = complete_official_records([])
        records[-1]["file_name"] = "validation_023.npz"
        noncanonical = make_payload(records, complete=False)
        with self.assertRaisesRegex(ValueError, "canonical"):
            replay.validate_cache_payload(noncanonical)

        with self.assertRaisesRegex(ValueError, "mandatory"):
            replay._validate_expected_video_count(0)

    def test_rejects_non_float32_scores_and_nonbinary_labels(self):
        float64_record = make_record("val_000.npz", [0.1] * 8)
        float64_record["scores"] = float64_record["scores"].double()
        with self.assertRaisesRegex(ValueError, "float32"):
            replay.validate_cache_payload(make_payload([float64_record]))

        nonbinary_record = make_record("val_000.npz", [0.1] * 8)
        nonbinary_record["seg_label"][0] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            replay.validate_cache_payload(make_payload([nonbinary_record]))

    def test_rejects_secondary_inference_or_code_mismatch(self):
        primary = make_payload([])
        changed_settings = dict(DEFAULT_INFERENCE_SETTINGS)
        changed_settings["temporal_memory_bin_size"] = 25
        secondary = make_payload(
            [],
            checkpoint_sha="b" * 64,
            inference_settings=changed_settings,
        )
        with self.assertRaisesRegex(ValueError, "inference settings"):
            replay.route_cache_records(primary, secondary, secondary_max_events=30)

        changed_code = {
            relative_path: "c" * 64
            for relative_path in replay.CACHE_CODE_PROVENANCE_PATHS
        }
        changed_code["model/temporal_frame_net.py"] = "d" * 64
        secondary = make_payload(
            [],
            checkpoint_sha="b" * 64,
            code_sha256=changed_code,
        )
        with self.assertRaisesRegex(ValueError, "inference code"):
            replay.route_cache_records(primary, secondary, secondary_max_events=30)

    def test_reranker_replay_binds_actual_primary_checkpoint_and_settings(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cfg, checkpoint_sha256 = make_reranker_fixture(temporary_dir)
            primary = make_payload([], checkpoint_sha=checkpoint_sha256)
            binding = replay.validate_component_reranker_cache_binding(
                cfg, primary, secondary_max_events=30000
            )
            self.assertEqual(
                binding["primary_cache_checkpoint_sha256"], checkpoint_sha256
            )
            self.assertEqual(
                binding["inference_settings_sha256"],
                sha256_json(DEFAULT_INFERENCE_SETTINGS),
            )

            wrong_checkpoint = make_payload([], checkpoint_sha="b" * 64)
            with self.assertRaisesRegex(ValueError, "Primary replay cache checkpoint"):
                replay.validate_component_reranker_cache_binding(
                    cfg, wrong_checkpoint, secondary_max_events=30000
                )

            changed_settings = dict(DEFAULT_INFERENCE_SETTINGS)
            changed_settings["temporal_memory_log_count_clip"] = 3.5
            wrong_settings = make_payload(
                [],
                checkpoint_sha=checkpoint_sha256,
                inference_settings=changed_settings,
            )
            with self.assertRaisesRegex(ValueError, "Primary replay cache inference"):
                replay.validate_component_reranker_cache_binding(
                    cfg, wrong_settings, secondary_max_events=30000
                )

    def test_reranker_replay_rejects_secondary_above_cutoff_or_dense_route(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cfg, checkpoint_sha256 = make_reranker_fixture(temporary_dir)
            primary = make_payload([], checkpoint_sha=checkpoint_sha256)
            with self.assertRaisesRegex(ValueError, "secondary routing"):
                replay.validate_component_reranker_cache_binding(
                    cfg, primary, secondary_max_events=100001
                )

            dense_secondary = replay.RoutedRecord(
                file_name="val_023.npz",
                event_count=100001,
                scores=torch.tensor([0.9], dtype=torch.float32),
                seg_label=torch.tensor([0.0], dtype=torch.float32),
                locs=torch.tensor([[0, 1, 1, 1]], dtype=torch.int64),
                idx_label=np.asarray([0], dtype=np.int64),
                source_sha256="0" * 64,
                score_source="secondary",
            )
            with self.assertRaisesRegex(ValueError, "must come from primary"):
                replay.validate_component_reranker_dense_routes(
                    cfg, [dense_secondary]
                )

    def test_disabled_reranker_does_not_open_artifact_during_replay(self):
        cfg = make_cfg(
            component_reranker_enabled=False,
            component_reranker_model_path="Z:/missing.json",
            component_reranker_expected_sha256="invalid",
        )
        self.assertIsNone(
            replay.validate_component_reranker_cache_binding(
                cfg,
                make_payload([]),
                secondary_max_events=100001,
            )
        )


class ExactReplayTest(unittest.TestCase):
    def test_replay_matches_test2_with_p6_p0_p0c_and_p18(self):
        cfg = make_cfg()
        first = make_record(
            "val_000.npz",
            [0.99, 0.80, 0.54, 0.72, 0.80, 0.54, 0.20, 0.72],
        )
        second = make_record(
            "val_001.npz",
            [0.96, 0.71, 0.55, 0.73, 0.74, 0.54, 0.10, 0.80] * 10,
        )
        routed = replay.route_cache_records(make_payload([first, second]))
        low_threshold = 0.718
        high_threshold = 0.719
        prepared = replay.precompute_video_counts(
            routed,
            density_cutoff=30,
            low_thresholds=(low_threshold,),
            high_thresholds=(high_threshold,),
            cfg=cfg,
        )
        actual = replay.evaluate_threshold_pair(
            prepared,
            density_cutoff=30,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            cfg=cfg,
        )
        expected = direct_test2_metrics(
            routed,
            density_cutoff=30,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            cfg=cfg,
        )
        self.assertEqual(actual.to_dict(), expected.to_dict())

    def test_formatted_reference_verification_is_strict(self):
        metrics = {
            "iou": 0.9422550201,
            "acc": 0.9767196774,
            "pd": 0.9762704746,
            "fa": 4.6929172975e-06,
            "score_fa": 0.9541549752,
            "score": 0.9628776542,
        }
        replay.verify_formatted_metrics(metrics, metrics)
        changed = dict(metrics)
        changed["score"] += 1e-9
        with self.assertRaisesRegex(RuntimeError, "score"):
            replay.verify_formatted_metrics(changed, metrics)


class FilesystemSafetyTest(unittest.TestCase):
    def test_atomic_torch_cache_is_no_clobber_unless_overwrite_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "cache.pt"
            output.write_bytes(b"concurrent-sentinel")
            with self.assertRaises(FileExistsError):
                replay._atomic_torch_save({"new": True}, output)
            self.assertEqual(output.read_bytes(), b"concurrent-sentinel")
            self.assertEqual(list(output.parent.glob(output.name + ".*.tmp")), [])
            replay._atomic_torch_save({"new": True}, output, overwrite=True)
            self.assertEqual(replay._torch_load_cpu(output), {"new": True})

    def test_atomic_json_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.json"
            output.write_text("previous-result\n", encoding="utf-8")
            with mock.patch.object(
                replay.json,
                "dump",
                side_effect=RuntimeError("serialization failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "serialization failed"):
                    replay._write_json(output, {"score": 1.0})
            self.assertEqual(output.read_text(encoding="utf-8"), "previous-result\n")
            self.assertEqual(list(output.parent.glob(output.name + ".*.tmp")), [])

    def test_rejects_input_output_and_output_output_path_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = Path(temporary_directory) / "cache.pt"
            cache.write_bytes(b"cache")
            with self.assertRaisesRegex(ValueError, "Path conflict"):
                replay._require_distinct_paths(
                    (("primary-cache", cache), ("output-json", cache))
                )
            with self.assertRaisesRegex(ValueError, "Path conflict"):
                replay._require_distinct_paths(
                    (("output-json", cache), ("output-csv", cache))
                )

    def test_existing_outputs_require_force(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.json"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--force"):
                replay._require_outputs_available((("output-json", output),), False)
            replay._require_outputs_available((("output-json", output),), True)

    def test_failed_reference_check_does_not_replace_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache = root / "primary.pt"
            output_json = root / "result.json"
            output_csv = root / "result.csv"
            output_json.write_text("previous-json\n", encoding="utf-8")
            output_csv.write_text("previous-csv\n", encoding="utf-8")
            replay._atomic_torch_save(make_payload([]), cache)

            arguments = [
                "replay",
                "--config",
                str(replay.DEFAULT_CONFIG),
                "--primary-cache",
                str(cache),
                "--density-cutoff",
                "30",
                "--low-min",
                "0.700",
                "--low-max",
                "0.700",
                "--high-min",
                "0.700",
                "--high-max",
                "0.700",
                "--threshold-step",
                "0.001",
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--force",
                "--reference-low",
                "0.700",
                "--reference-high",
                "0.700",
                "--expect-metric",
                "score=-1",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "score"):
                    replay.main(arguments)
            self.assertEqual(output_json.read_text(encoding="utf-8"), "previous-json\n")
            self.assertEqual(output_csv.read_text(encoding="utf-8"), "previous-csv\n")

    def test_atomic_json_success_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.json"
            replay._write_json(output, {"score": 0.9})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"score": 0.9})

    def test_successful_main_records_cache_and_code_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache = root / "primary.pt"
            output_json = root / "result.json"
            output_csv = root / "result.csv"
            replay._atomic_torch_save(make_payload([]), cache)
            arguments = [
                "replay",
                "--config",
                str(replay.DEFAULT_CONFIG),
                "--primary-cache",
                str(cache),
                "--density-cutoff",
                "30",
                "--low-min",
                "0.700",
                "--low-max",
                "0.700",
                "--high-min",
                "0.700",
                "--high-max",
                "0.700",
                "--threshold-step",
                "0.001",
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(replay.main(arguments), 0)
            result = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(result["tool_schema"], "evc-temporal-memory-replay-results-v2")
            self.assertEqual(result["video_count"], 24)
            self.assertEqual(result["primary_cache_sha256"], replay.sha256_file(cache))
            self.assertEqual(
                set(result["replay_code_sha256"]),
                set(replay.REPLAY_CODE_PROVENANCE_PATHS),
            )
            self.assertTrue(output_csv.read_text(encoding="utf-8").startswith("rank,"))


if __name__ == "__main__":
    unittest.main()
