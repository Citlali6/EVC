import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import crossfit_component_reranker as crossfit
import replay_temporal_memory_validation as replay
import train_component_reranker as trainer
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.component_reranker import (
    ARTIFACT_SCHEMA,
    ComponentLinearModel,
    FEATURE_NAMES,
    TRAIN_CACHE_SCHEMA,
    sha256_file,
    temporal_memory_inference_mapping,
)
from utils.postprocess import P0ClusterFilter
from utils.eval import evalute


def c00_overrides():
    return [
        "TEST.prediction_threshold=0.719",
        "TEMPORAL_FRAME.temporal_frame_enabled=false",
        "TEMPORAL_MEMORY.temporal_memory_enabled=true",
        "TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0",
        "TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true",
        "POSTPROCESS.p0_enabled=true",
        "POSTPROCESS.p0_spatial_radius=2",
        "POSTPROCESS.p0_temporal_bin_size=50",
        "POSTPROCESS.p0_temporal_radius_bins=1",
        "POSTPROCESS.p0_min_cluster_events=3",
        "POSTPROCESS.p0_min_duration_bins=5",
        "POSTPROCESS.p0c_high_confidence_recovery_enabled=true",
        "POSTPROCESS.p0c_retain_min_score=0.95",
        "POSTPROCESS.p0c_density_retain_enabled=false",
        "POSTPROCESS.p0c_density_event_count_cutoff=100000",
        "POSTPROCESS.p0c_density_retain_min_score=0.97",
        "POSTPROCESS.p0b_enabled=false",
        "POSTPROCESS.p18_score_track_recovery_enabled=true",
        "POSTPROCESS.p18_event_count_cutoff=1",
        "POSTPROCESS.p18_max_event_count=35000",
        "POSTPROCESS.p18_candidate_floor=0.53",
        "POSTPROCESS.p18_spatial_radius=5",
        "POSTPROCESS.p18_temporal_bin_size=50",
        "POSTPROCESS.p18_max_link_distance=8.0",
        "POSTPROCESS.p18_max_gap_bins=1",
        "POSTPROCESS.p18_min_track_bins=4",
        "POSTPROCESS.p18_restore_mode=best",
        "POSTPROCESS.p18_max_restore_events_per_component=0",
        "POSTPROCESS.p6_density_threshold_enabled=true",
        "POSTPROCESS.p6_event_count_cutoff=30000",
        "POSTPROCESS.p6_low_density_threshold=0.718",
        "POSTPROCESS.p6_high_density_threshold=0.719",
    ]


def synthetic_cache_manifest(cfg):
    official_sources = []
    for index in range(99):
        name = "train_{:03d}.npz".format(index)
        official_sources.append(
            {
                "source_name": name,
                "source_sha256": hashlib.sha256(
                    ("source:" + name).encode("utf-8")
                ).hexdigest(),
            }
        )
    official_sha256 = trainer.source_manifest_sha256(official_sources)
    official_by_name = {
        entry["source_name"]: entry["source_sha256"]
        for entry in official_sources
    }
    high_names = list(crossfit.H1_NAMES + crossfit.H2_NAMES)
    middle_names = [
        name
        for name in crossfit.EXPECTED_SELECTED_NAMES
        if name not in set(high_names)
    ]
    counts = {}
    for index, name in enumerate(high_names):
        counts[name] = 450000 if index < 14 else 519439
    for index, name in enumerate(middle_names):
        counts[name] = 44000 if index < 38 else 64323
    records = []
    for index, name in enumerate(crossfit.EXPECTED_SELECTED_NAMES):
        records.append(
            {
                "record": "records/{:03d}.npz".format(index),
                "record_sha256": hashlib.sha256(
                    ("record:" + name).encode("utf-8")
                ).hexdigest(),
                "source_name": name,
                "source_sha256": official_by_name[name],
                "event_count": counts[name],
            }
        )
    manifest = {
        "schema": TRAIN_CACHE_SCHEMA,
        "dataset_split": "train",
        "selection": {
            "observable": "complete_video_event_count",
            "operator": ">",
            "min_event_count_exclusive": 30000,
        },
        "total_train_video_count": 99,
        "selected_video_count": 54,
        "selected_event_count": sum(counts.values()),
        "official_train_source_manifest_scheme": trainer.TRAIN_SOURCE_MANIFEST_SCHEME,
        "official_train_source_manifest_sha256": official_sha256,
        "official_train_sources": official_sources,
        "base_checkpoint_sha256": crossfit.RELEASED_M20_CHECKPOINT_SHA256,
        "inference_settings": temporal_memory_inference_mapping(cfg),
        "records": records,
    }
    assert manifest["selected_event_count"] == crossfit.EXPECTED_SELECTED_EVENT_COUNT
    return manifest, official_sha256


def make_video(name, block, component_count, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(component_count, len(FEATURE_NAMES)))
    labels = np.asarray(
        [(index + seed) % 2 for index in range(component_count)], dtype=np.uint8
    )
    return crossfit.PreparedVideo(
        source_name=name,
        block=block,
        event_count=100001 if block != "middle" else 40000,
        features=features,
        component_labels=labels,
        event_indices=tuple(np.asarray([index]) for index in range(component_count)),
        p0_scores=None,
        locations=None,
        event_labels=None,
        target_ids=None,
        baseline_counts=crossfit.SufficientCounts(
            true_positive_events=10,
            false_positive_events=10,
            false_negative_events=0,
            correct_objects=1,
            object_count=1,
            false_components=5,
            frame_count=2,
            event_count=20,
        ),
    )


def make_runtime_high_video(name, block, seed=0):
    event_count = 100001
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(4, len(FEATURE_NAMES)))
    component_labels = np.asarray([1, 0, 1, 0], dtype=np.uint8)
    scores = np.full(event_count, 0.1, dtype=np.float32)
    scores[:4] = 0.8
    locations = np.zeros((event_count, 4), dtype=np.int64)
    locations[:4] = np.asarray(
        [[0, 1, 1, 10], [0, 10, 10, 20], [0, 2, 2, 60], [0, 20, 20, 110]],
        dtype=np.int64,
    )
    labels = np.zeros(event_count, dtype=np.uint8)
    labels[[0, 2]] = 1
    target_ids = np.zeros(event_count, dtype=np.int16)
    target_ids[[0, 2]] = 1
    return crossfit.PreparedVideo(
        source_name=name,
        block=block,
        event_count=event_count,
        features=features,
        component_labels=component_labels,
        event_indices=tuple(np.asarray([index]) for index in range(4)),
        p0_scores=scores,
        locations=locations,
        event_labels=labels,
        target_ids=target_ids,
        baseline_counts=crossfit.sufficient_counts_for_video(
            scores, labels, target_ids, locations
        ),
    )


def make_fold_result(fold_id, candidate_id, baseline_counts, selected_counts):
    baseline_metrics = crossfit.metrics_from_counts(baseline_counts)
    selected_metrics = crossfit.metrics_from_counts(selected_counts)
    return {
        "fold_id": fold_id,
        "baseline": {
            "counts": baseline_counts.to_dict(),
            "metrics": baseline_metrics,
        },
        "candidate_results": [
            {
                "candidate_id": candidate_id,
                "counts": selected_counts.to_dict(),
                "metrics": selected_metrics,
                "delta": {
                    name: selected_metrics[name] - baseline_metrics[name]
                    for name in selected_metrics
                },
                "false_component_delta": (
                    selected_counts.false_components
                    - baseline_counts.false_components
                ),
            }
        ],
        "winner_candidate_id": candidate_id,
    }


def make_synthetic_posthoc_sources(root):
    cache_dir = root / "cache"
    cache_dir.mkdir()
    manifest_path = cache_dir / "manifest.json"
    crossfit._atomic_json(manifest_path, {"synthetic": True})
    manifest_sha256 = sha256_file(manifest_path)
    common = {
        "dataset": {"cache_manifest_sha256": manifest_sha256},
        "blocks": {"h1": ["a"], "h2": ["b"], "middle": ["c"]},
        "fold_plan": [dict(item) for item in crossfit.FOLD_PLAN],
        "topology": crossfit.TOPOLOGY.to_dict(),
        "fit": {"algorithm": "synthetic"},
        "scoring": {"prediction_threshold": 0.719},
        "gates": {"pooled_score_delta_minimum": 0.0002},
        "promotion": {"deployment_event_count_cutoff": 100000},
        "config": {"sha256": "1" * 64, "overrides": c00_overrides()},
    }
    current_definition = {
        **copy.deepcopy(common),
        "candidate_profile": crossfit.POSTHOC_V2_PROFILE,
        "candidates": crossfit.candidate_definitions(crossfit.POSTHOC_V2_PROFILE),
        "oof": crossfit._oof_definition(crossfit.POSTHOC_V2_PROFILE),
    }
    source_definition = {
        **copy.deepcopy(common),
        "candidates": crossfit.candidate_definitions(
            crossfit.CONSERVATIVE_V1_PROFILE
        ),
        "oof": crossfit._oof_definition(crossfit.CONSERVATIVE_V1_PROFILE),
    }
    source_protocol = {
        "schema": crossfit.PROTOCOL_SCHEMA,
        "created_utc": "synthetic",
        "definition": source_definition,
        "definition_sha256": crossfit.sha256_json(source_definition),
    }
    protocol_path = root / "source_protocol.json"
    crossfit._atomic_json(protocol_path, source_protocol)
    protocol_sha256 = sha256_file(protocol_path)

    counts = crossfit.SufficientCounts(
        true_positive_events=100,
        false_positive_events=10,
        false_negative_events=5,
        correct_objects=10,
        object_count=10,
        false_components=5,
        frame_count=10,
        event_count=1000,
    )
    metrics = crossfit.metrics_from_counts(counts)
    zero_delta = {name: 0.0 for name in metrics}
    candidates = crossfit.candidate_definitions(crossfit.CONSERVATIVE_V1_PROFILE)
    folds = []
    for fold_id in ("holdout_h1", "holdout_h2"):
        folds.append(
            {
                "fold_id": fold_id,
                "baseline": {"counts": counts.to_dict(), "metrics": metrics},
                "candidate_results": [
                    {
                        **candidate,
                        "counts": counts.to_dict(),
                        "metrics": metrics,
                        "delta": zero_delta,
                        "false_component_delta": 0,
                    }
                    for candidate in candidates
                ],
                "winner_candidate_id": candidates[0]["candidate_id"],
            }
        )
    report = {
        "schema": crossfit.REPORT_SCHEMA,
        "dataset_split": "train",
        "evidence_class": (
            "train_only_cross_source_held_block_consistency_not_unbiased_oof"
        ),
        "cache_manifest_path": str(manifest_path.resolve()),
        "cache_manifest_sha256": manifest_sha256,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_file_sha256": protocol_sha256,
        "protocol_definition_sha256": source_protocol["definition_sha256"],
        "fold_results": folds,
        "promotion_gates": {
            "passed": False,
            "baseline_pooled": {"counts": counts.to_dict()},
            "selected_pooled_oof": {
                "counts": counts.to_dict(),
                "delta": zero_delta,
            },
        },
        "artifact": {"emitted": False, "path": None, "sha256": None},
    }
    report_path = root / "source_report.json"
    crossfit._atomic_json(report_path, report)
    report_sha256 = sha256_file(report_path)

    singleton = dict(crossfit.candidate_definitions(crossfit.POSTHOC_V2_PROFILE)[0])
    singleton["policy"] = crossfit.POSTHOC_SINGLETON_POLICY
    diagnostic = {
        "schema": crossfit.POSTHOC_DIAGNOSTIC_SCHEMA,
        "evidence_class": crossfit.POSTHOC_DIAGNOSTIC_EVIDENCE_CLASS,
        "source_protocol_sha256": protocol_sha256,
        "source_crossfit_report_sha256": report_sha256,
        "train_cache_manifest_sha256": manifest_sha256,
        "original_grid": {
            "positive_weights": [4.0, 8.0],
            "keep_probabilities": [0.02, 0.05, 0.10, 0.20],
            "outcome": "all candidates were exact no-ops",
        },
        "frozen_followup_hypothesis": singleton,
    }
    diagnostic_path = root / "posthoc_threshold_diagnostic.json"
    crossfit._atomic_json(diagnostic_path, diagnostic)
    diagnostic_sha256 = sha256_file(diagnostic_path)

    policy = {
        "schema": crossfit.POSTHOC_VALIDATION_POLICY_SCHEMA,
        "status": "frozen_before_v2_artifact_and_before_validation_replay",
        "evidence_class": (
            "retrospective_train_hypothesis_with_one_frozen_validation_check"
        ),
        "hypothesis_source": {
            "path": str(diagnostic_path.resolve()),
            "sha256": diagnostic_sha256,
            "positive_weight": 4.0,
            "keep_probability": 0.4,
            "l2": 0.1,
        },
        "evaluation_budget": {
            "full_validation_replays": 1,
            "threshold_or_hyperparameter_search_after_replay": False,
            "allowed_follow_up_on_failure": (
                "archive this suppression branch without tuning it on validation"
            ),
        },
    }
    policy_path = root / "validation_acceptance_policy.json"
    crossfit._atomic_json(policy_path, policy)
    policy_sha256 = sha256_file(policy_path)
    return {
        "current_definition": current_definition,
        "manifest_path": manifest_path,
        "report_path": report_path,
        "report_sha256": report_sha256,
        "diagnostic_path": diagnostic_path,
        "diagnostic_sha256": diagnostic_sha256,
        "policy_path": policy_path,
        "policy_sha256": policy_sha256,
    }


class ComponentRerankerCrossfitTests(unittest.TestCase):
    def test_candidate_grid_is_exact_and_label_free(self):
        candidates = crossfit.candidate_definitions()
        self.assertEqual(len(candidates), 8)
        self.assertEqual(
            [item["candidate_id"] for item in candidates],
            [
                "pw04_kp020",
                "pw04_kp050",
                "pw04_kp100",
                "pw04_kp200",
                "pw08_kp020",
                "pw08_kp050",
                "pw08_kp100",
                "pw08_kp200",
            ],
        )
        self.assertEqual({item["l2"] for item in candidates}, {0.1})
        self.assertEqual(
            candidates,
            crossfit.candidate_definitions(crossfit.CONSERVATIVE_V1_PROFILE),
        )
        self.assertEqual(
            crossfit.candidate_definitions(crossfit.POSTHOC_V2_PROFILE),
            [
                {
                    "candidate_id": "pw04_kp400",
                    "positive_weight": 4.0,
                    "keep_probability": 0.4,
                    "l2": 0.1,
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            crossfit.candidate_definitions("unknown")
        self.assertEqual(crossfit.TOPOLOGY.spatial_radius, 1)
        self.assertEqual(crossfit.TOPOLOGY.max_component_events, 3)
        self.assertTrue(
            {"file_name", "source_name", "target_id", "label"}.isdisjoint(
                FEATURE_NAMES
            )
        )

    def test_time_boundary_semantics_are_explicitly_frozen(self):
        locations = np.asarray(
            [[0, 5, 5, 50], [0, 6, 5, 60], [0, 0, 0, 110]], dtype=np.int64
        )
        examples = crossfit.extract_component_examples(
            np.asarray([0.8, 0.8, 0.1], dtype=np.float32),
            locations,
            0.719,
            crossfit.TOPOLOGY,
            3,
        )
        self.assertEqual(len(examples), 1)
        np.testing.assert_array_equal(examples[0].event_indices, [0, 1])

        # Official Pd/Fa open intervals exclude the t=50 prediction, while
        # semantic counts still include it as one false-positive event.
        counts = crossfit.sufficient_counts_for_video(
            np.asarray([0.8, 0.1, 0.1], dtype=np.float32),
            np.zeros(3, dtype=np.uint8),
            np.zeros(3, dtype=np.int16),
            locations,
        )
        self.assertEqual(counts.false_positive_events, 1)
        self.assertEqual(counts.false_components, 0)

    def test_hierarchical_weights_balance_domains_videos_and_components(self):
        videos = [
            make_video("train_044.npz", "h1", 2, 1),
            make_video("train_045.npz", "h1", 4, 2),
            make_video("train_000.npz", "middle", 1, 3),
            make_video("train_001.npz", "middle", 5, 4),
        ]
        _, _, weights, sources = crossfit.balanced_component_dataset(videos, {"h1"})
        mass = {
            source: float(weights[np.asarray(sources) == source].sum())
            for source in set(sources)
        }
        self.assertAlmostEqual(mass["train_044.npz"], 0.25, places=14)
        self.assertAlmostEqual(mass["train_045.npz"], 0.25, places=14)
        self.assertAlmostEqual(mass["train_000.npz"], 0.25, places=14)
        self.assertAlmostEqual(mass["train_001.npz"], 0.25, places=14)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=14)

    def test_fold_partitions_are_disjoint_and_middle_is_fit_only(self):
        videos = [
            make_video("train_044.npz", "h1", 2, 1),
            make_video("train_088.npz", "h2", 2, 2),
            make_video("train_000.npz", "middle", 2, 3),
        ]
        fit_videos, held = crossfit.partition_fold_videos(
            crossfit.FOLD_PLAN[0], videos
        )
        self.assertEqual({video.block for video in fit_videos}, {"h2", "middle"})
        self.assertEqual({video.block for video in held}, {"h1"})
        self.assertTrue(
            {video.source_name for video in fit_videos}.isdisjoint(
                video.source_name for video in held
            )
        )

    def test_balanced_logistic_is_real_deterministic_and_fit_side_only(self):
        rng = np.random.default_rng(28)
        features = rng.normal(size=(60, len(FEATURE_NAMES)))
        labels = (features[:, 2] + 0.3 * features[:, 3] > 0).astype(np.uint8)
        weights = np.full(60, 1.0 / 60, dtype=np.float64)
        first = crossfit.fit_balanced_logistic(features, labels, weights, 4.0)
        second = crossfit.fit_balanced_logistic(features, labels, weights, 4.0)
        np.testing.assert_allclose(first["coefficients"], second["coefficients"])
        self.assertGreater(np.linalg.norm(first["coefficients"]), 0.1)
        self.assertLess(first["weighted_loss"], 0.7)
        with self.assertRaisesRegex(ValueError, "candidate set"):
            crossfit.fit_balanced_logistic(features, labels, weights, 3.0)

    def test_synthetic_held_block_runs_all_eight_real_candidates(self):
        config_path = PROJECT_ROOT / "configs" / "evisseg_evuav.yaml"
        cfg = replay.load_flat_config(config_path, c00_overrides())
        videos = [
            make_runtime_high_video("train_044.npz", "h1", 1),
            make_runtime_high_video("train_088.npz", "h2", 2),
            make_video("train_000.npz", "middle", 4, 3),
        ]
        result = crossfit._evaluate_fold(crossfit.FOLD_PLAN[0], videos, cfg)
        self.assertEqual(len(result["candidate_results"]), 8)
        self.assertNotIn("train_044.npz", result["fit_video_names"])
        self.assertEqual(result["held_video_names"], ["train_044.npz"])
        self.assertIn(
            result["winner_candidate_id"],
            {item["candidate_id"] for item in crossfit.candidate_definitions()},
        )

    def test_posthoc_profile_runs_exactly_one_candidate_without_selection(self):
        config_path = PROJECT_ROOT / "configs" / "evisseg_evuav.yaml"
        cfg = replay.load_flat_config(config_path, c00_overrides())
        videos = [
            make_runtime_high_video("train_044.npz", "h1", 1),
            make_runtime_high_video("train_088.npz", "h2", 2),
            make_video("train_000.npz", "middle", 4, 3),
        ]
        result = crossfit._evaluate_fold(
            crossfit.FOLD_PLAN[0],
            videos,
            cfg,
            crossfit.POSTHOC_V2_PROFILE,
        )
        self.assertEqual(len(result["candidate_results"]), 1)
        self.assertEqual(result["winner_candidate_id"], "pw04_kp400")
        self.assertEqual(
            result["candidate_results"][0]["keep_probability"], 0.4
        )

    def test_posthoc_source_diagnostic_lineage_and_tamper_guards(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sources = make_synthetic_posthoc_sources(Path(temporary_dir))
            patches = {
                "POSTHOC_V1_REPORT_SHA256": sources["report_sha256"],
                "POSTHOC_V1_REPORT_PATH": sources["report_path"].resolve(),
                "POSTHOC_DIAGNOSTIC_PATH": sources["diagnostic_path"].resolve(),
                "POSTHOC_DIAGNOSTIC_SHA256": sources["diagnostic_sha256"],
                "POSTHOC_VALIDATION_POLICY_PATH": sources["policy_path"].resolve(),
                "POSTHOC_VALIDATION_POLICY_SHA256": sources["policy_sha256"],
            }
            with mock.patch.multiple(crossfit, **patches):
                metadata = crossfit._validate_posthoc_hypothesis_source(
                    sources["report_path"],
                    sources["report_sha256"],
                    sources["diagnostic_path"],
                    sources["diagnostic_sha256"],
                    sources["current_definition"],
                    sources["manifest_path"],
                )
            self.assertEqual(
                metadata["hypothesis_origin"],
                crossfit.POSTHOC_HYPOTHESIS_ORIGIN,
            )
            self.assertFalse(metadata["independent_oof_claim"])
            self.assertEqual(
                metadata["source_train_diagnostic"]["frozen_followup_hypothesis"],
                {
                    **crossfit.candidate_definitions(crossfit.POSTHOC_V2_PROFILE)[0],
                    "policy": crossfit.POSTHOC_SINGLETON_POLICY,
                },
            )
            self.assertEqual(
                metadata["validation_acceptance_policy"]["full_validation_replays"],
                1,
            )

            original_report = sources["report_path"].read_bytes()
            sources["report_path"].write_bytes(original_report + b" ")
            with mock.patch.multiple(crossfit, **patches):
                with self.assertRaisesRegex(ValueError, "report SHA-256"):
                    crossfit._validate_posthoc_hypothesis_source(
                        sources["report_path"],
                        sources["report_sha256"],
                        sources["diagnostic_path"],
                        sources["diagnostic_sha256"],
                        sources["current_definition"],
                        sources["manifest_path"],
                    )
            sources["report_path"].write_bytes(original_report)

            original_policy = sources["policy_path"].read_bytes()
            sources["policy_path"].write_bytes(original_policy + b" ")
            with mock.patch.multiple(crossfit, **patches):
                with self.assertRaisesRegex(ValueError, "policy SHA-256"):
                    crossfit._validate_posthoc_hypothesis_source(
                        sources["report_path"],
                        sources["report_sha256"],
                        sources["diagnostic_path"],
                        sources["diagnostic_sha256"],
                        sources["current_definition"],
                        sources["manifest_path"],
                    )
            sources["policy_path"].write_bytes(original_policy)

            diagnostic = json.loads(
                sources["diagnostic_path"].read_text(encoding="utf-8")
            )
            diagnostic["frozen_followup_hypothesis"]["keep_probability"] = 0.41
            crossfit._atomic_json(sources["diagnostic_path"], diagnostic)
            tampered_sha256 = sha256_file(sources["diagnostic_path"])
            with mock.patch.multiple(
                crossfit,
                POSTHOC_V1_REPORT_SHA256=sources["report_sha256"],
                POSTHOC_V1_REPORT_PATH=sources["report_path"].resolve(),
                POSTHOC_DIAGNOSTIC_PATH=sources["diagnostic_path"].resolve(),
                POSTHOC_DIAGNOSTIC_SHA256=tampered_sha256,
                POSTHOC_VALIDATION_POLICY_PATH=sources["policy_path"].resolve(),
                POSTHOC_VALIDATION_POLICY_SHA256=sources["policy_sha256"],
            ):
                with self.assertRaisesRegex(ValueError, "exact singleton policy"):
                    crossfit._validate_posthoc_hypothesis_source(
                        sources["report_path"],
                        sources["report_sha256"],
                        sources["diagnostic_path"],
                        tampered_sha256,
                        sources["current_definition"],
                        sources["manifest_path"],
                    )

    def test_profile_arguments_require_both_frozen_posthoc_sources(self):
        crossfit._validate_profile_arguments(crossfit.CONSERVATIVE_V1_PROFILE)
        with self.assertRaisesRegex(ValueError, "does not accept"):
            crossfit._validate_profile_arguments(
                crossfit.CONSERVATIVE_V1_PROFILE,
                "unexpected.json",
                "0" * 64,
            )
        with self.assertRaisesRegex(ValueError, "requires the source v1 report"):
            crossfit._validate_profile_arguments(crossfit.POSTHOC_V2_PROFILE)

    def test_promotion_gates_pass_and_mismatched_winner_fails(self):
        baseline = crossfit.SufficientCounts(
            true_positive_events=100,
            false_positive_events=100,
            false_negative_events=0,
            correct_objects=10,
            object_count=10,
            false_components=100,
            frame_count=100,
            event_count=1000,
        )
        selected = crossfit.SufficientCounts(
            true_positive_events=100,
            false_positive_events=80,
            false_negative_events=0,
            correct_objects=10,
            object_count=10,
            false_components=80,
            frame_count=100,
            event_count=1000,
        )
        middle = crossfit.SufficientCounts(
            true_positive_events=100,
            false_positive_events=50,
            false_negative_events=0,
            correct_objects=10,
            object_count=10,
            false_components=50,
            frame_count=100,
            event_count=1000,
        )
        folds = [
            make_fold_result("holdout_h1", "pw04_kp020", baseline, selected),
            make_fold_result("holdout_h2", "pw04_kp020", baseline, selected),
        ]
        gates = crossfit.evaluate_promotion_gates(folds, middle)
        self.assertTrue(gates["passed"])
        self.assertGreaterEqual(
            gates["held_high_false_components"]["reduction_fraction"], 0.01
        )

        mismatched = copy.deepcopy(folds)
        mismatched[1]["candidate_results"][0]["candidate_id"] = "pw08_kp020"
        mismatched[1]["winner_candidate_id"] = "pw08_kp020"
        failed = crossfit.evaluate_promotion_gates(mismatched, middle)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["fold_winner_candidate_ids_match"])

    def test_artifact_is_emitted_only_after_pass(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            failed_report = root / "failed_report.json"
            failed_model = root / "failed_model.json"
            digest = crossfit.publish_crossfit_outputs(
                failed_report,
                failed_model,
                None,
                lambda artifact_sha: {
                    "schema": crossfit.REPORT_SCHEMA,
                    "artifact": {"emitted": False, "sha256": artifact_sha},
                },
            )
            self.assertIsNone(digest)
            self.assertTrue(failed_report.is_file())
            self.assertFalse(failed_model.exists())

            passed_report = root / "passed_report.json"
            passed_model = root / "passed_model.json"
            digest = crossfit.publish_crossfit_outputs(
                passed_report,
                passed_model,
                {"schema": ARTIFACT_SCHEMA},
                lambda artifact_sha: {
                    "schema": crossfit.REPORT_SCHEMA,
                    "artifact": {"emitted": True, "sha256": artifact_sha},
                },
            )
            self.assertEqual(digest, sha256_file(passed_model))
            self.assertTrue(passed_report.is_file())

    def test_report_and_artifact_paths_cannot_alias(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "same.json"
            with self.assertRaisesRegex(ValueError, "paths must differ"):
                crossfit.run_crossfit(
                    SimpleNamespace(
                        output_report=str(path),
                        output_model=str(path),
                        protocol="unused",
                        expected_protocol_sha256="0" * 64,
                        cache_dir="unused",
                    )
                )

    def test_report_construction_failure_occurs_before_artifact_publish(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            report = root / "report.json"
            artifact = root / "artifact.json"

            def fail_before_publish(_):
                raise RuntimeError("synthetic report construction failure")

            with self.assertRaisesRegex(RuntimeError, "construction failure"):
                crossfit.publish_crossfit_outputs(
                    report,
                    artifact,
                    {"schema": ARTIFACT_SCHEMA},
                    fail_before_publish,
                )
            self.assertFalse(report.exists())
            self.assertFalse(artifact.exists())

    def test_failure_after_artifact_publish_removes_orphan(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            report = root / "report.json"
            artifact = root / "artifact.json"
            real_replace = crossfit.os.replace
            replace_count = 0

            def fail_second_replace(source, destination):
                nonlocal replace_count
                replace_count += 1
                if replace_count == 2:
                    raise OSError("synthetic report publish failure")
                return real_replace(source, destination)

            with mock.patch.object(
                crossfit.os, "replace", side_effect=fail_second_replace
            ):
                with self.assertRaisesRegex(OSError, "report publish failure"):
                    crossfit.publish_crossfit_outputs(
                        report,
                        artifact,
                        {"schema": ARTIFACT_SCHEMA},
                        lambda artifact_sha: {
                            "schema": crossfit.REPORT_SCHEMA,
                            "artifact": {
                                "emitted": True,
                                "sha256": artifact_sha,
                            },
                        },
                    )
            self.assertFalse(report.exists())
            self.assertFalse(artifact.exists())

    def test_full_refit_artifact_is_runtime_compatible(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "m20.pt"
            checkpoint.write_bytes(b"crossfit-runtime-checkpoint")
            config_path = PROJECT_ROOT / "configs" / "evisseg_evuav.yaml"
            overrides = c00_overrides()
            cfg = replay.load_flat_config(config_path, overrides)
            fitted = {
                "feature_mean": np.zeros(len(FEATURE_NAMES), dtype=np.float64),
                "feature_scale": np.ones(len(FEATURE_NAMES), dtype=np.float64),
                "coefficients": np.zeros(len(FEATURE_NAMES), dtype=np.float64),
                "intercept": 0.0,
                "iterations": 1,
                "converged": True,
                "weighted_loss": 0.5,
                "positive_weight": 4.0,
            }
            manifest = {
                "schema": TRAIN_CACHE_SCHEMA,
                "selection": {
                    "observable": "complete_video_event_count",
                    "operator": ">",
                    "min_event_count_exclusive": 30000,
                },
                "base_checkpoint_sha256": sha256_file(checkpoint),
                "inference_settings": temporal_memory_inference_mapping(cfg),
                "official_train_source_manifest_scheme": trainer.TRAIN_SOURCE_MANIFEST_SCHEME,
                "official_train_source_manifest_sha256": "9" * 64,
            }
            protocol = {
                "definition_sha256": "b" * 64,
                "definition": {
                    "config": {"overrides": overrides},
                    "frozen_document": {"sha256": "c" * 64},
                    "code_sha256": {"crossfit_component_reranker.py": "d" * 64},
                    "software_versions": {
                        "numpy": np.__version__,
                        "torch": torch.__version__,
                        "opencv": crossfit.cv2.__version__,
                    },
                },
            }
            video = make_video("train_000.npz", "middle", 2, 8)
            artifact_payload = crossfit._artifact_for_full_refit(
                fitted,
                crossfit.candidate_definitions()[0],
                protocol,
                "e" * 64,
                manifest,
                "f" * 64,
                cfg,
                config_path,
                {"passed": True},
                [{"fold_id": "synthetic"}],
                [video],
            )
            artifact_path = root / "artifact.json"
            crossfit._atomic_json(artifact_path, artifact_payload)
            p0_config = P0ClusterFilter.from_cfg(
                cfg,
                0.719,
                event_count=100001,
            ).config
            loaded = ComponentLinearModel.load(
                artifact_path,
                sha256_file(artifact_path),
                checkpoint,
                0.719,
                100000,
                p0_config,
                temporal_memory_inference_mapping(cfg),
            )
            self.assertEqual(loaded.topology.spatial_radius, 1)
            self.assertEqual(
                loaded.provenance["crossfit_candidate_id"], "pw04_kp020"
            )

            posthoc_protocol = copy.deepcopy(protocol)
            posthoc_protocol["definition"]["candidate_profile"] = (
                crossfit.POSTHOC_V2_PROFILE
            )
            posthoc_protocol["definition"]["hypothesis"] = {
                "hypothesis_origin": crossfit.POSTHOC_HYPOTHESIS_ORIGIN,
                "independent_oof_claim": False,
                "source_train_diagnostic": {"sha256": "7" * 64},
                "validation_acceptance_policy": {"sha256": "8" * 64},
            }
            posthoc_payload = crossfit._artifact_for_full_refit(
                fitted,
                crossfit.candidate_definitions(crossfit.POSTHOC_V2_PROFILE)[0],
                posthoc_protocol,
                "e" * 64,
                manifest,
                "f" * 64,
                cfg,
                config_path,
                {"passed": True},
                [{"fold_id": "synthetic"}],
                [video],
            )
            self.assertEqual(
                posthoc_payload["fit"]["hyperparameter_selection"],
                "retrospective_train_only_singleton_after_v1_noop_no_further_selection",
            )
            self.assertEqual(
                posthoc_payload["provenance"]["crossfit_candidate_profile"],
                crossfit.POSTHOC_V2_PROFILE,
            )
            self.assertEqual(
                posthoc_payload["provenance"]["crossfit_hypothesis"],
                posthoc_protocol["definition"]["hypothesis"],
            )

    def test_train_source_manifest_hash_has_injectable_synthetic_expectation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = []
            entries = []
            for index in range(99):
                path = root / "train_{:03d}.npz".format(index)
                path.write_bytes("synthetic-{}".format(index).encode("ascii"))
                paths.append(path)
                entries.append(
                    {
                        "source_name": path.name,
                        "source_sha256": sha256_file(path),
                    }
                )
            expected = trainer.source_manifest_sha256(entries)
            actual, loaded_entries = trainer.hash_train_sources(paths, expected)
            self.assertEqual(actual, expected)
            self.assertEqual(loaded_entries, entries)
            with self.assertRaisesRegex(ValueError, "does not match required"):
                trainer.hash_train_sources(paths, "0" * 64)

    def test_cache_builder_official_hash_gate_precedes_output_and_gpu_work(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            train_dir = root / "dataset" / "train"
            train_dir.mkdir(parents=True)
            for index in range(99):
                (train_dir / "train_{:03d}.npz".format(index)).write_bytes(b"x")
            checkpoint = root / "m20.pt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "cache"
            args = SimpleNamespace(
                output_cache_dir=str(output),
                data_root=str(root / "dataset"),
                checkpoint=str(checkpoint),
                expected_total_videos=99,
                require_canonical_names=True,
            )
            with mock.patch.object(
                trainer,
                "hash_train_sources",
                side_effect=ValueError("synthetic official hash rejection"),
            ):
                with self.assertRaisesRegex(ValueError, "hash rejection"):
                    trainer.build_train_cache(args)
            self.assertFalse(output.exists())

    def test_population_rejects_cross_fold_record_aliases_and_bad_totals(self):
        config_path = PROJECT_ROOT / "configs" / "evisseg_evuav.yaml"
        cfg = replay.load_flat_config(config_path, c00_overrides())
        manifest, official_sha256 = synthetic_cache_manifest(cfg)
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache_dir = Path(temporary_dir)
            aliased = copy.deepcopy(manifest)
            by_name = {
                item["source_name"]: item for item in aliased["records"]
            }
            by_name["train_088.npz"]["record"] = by_name["train_044.npz"][
                "record"
            ]
            with self.assertRaisesRegex(ValueError, "paths alias"):
                crossfit._validate_cache_population(
                    aliased,
                    cache_dir,
                    expected_official_train_sha256=official_sha256,
                )

            duplicate_digest = copy.deepcopy(manifest)
            by_name = {
                item["source_name"]: item for item in duplicate_digest["records"]
            }
            by_name["train_088.npz"]["record_sha256"] = by_name[
                "train_044.npz"
            ]["record_sha256"]
            with self.assertRaisesRegex(ValueError, "record SHA-256 values"):
                crossfit._validate_cache_population(
                    duplicate_digest,
                    cache_dir,
                    expected_official_train_sha256=official_sha256,
                )

            duplicate_source = copy.deepcopy(manifest)
            by_name = {
                item["source_name"]: item for item in duplicate_source["records"]
            }
            by_name["train_088.npz"]["source_sha256"] = by_name[
                "train_044.npz"
            ]["source_sha256"]
            with self.assertRaisesRegex(ValueError, "source SHA-256 values"):
                crossfit._validate_cache_population(
                    duplicate_source,
                    cache_dir,
                    expected_official_train_sha256=official_sha256,
                )

            noncanonical_path = copy.deepcopy(manifest)
            noncanonical_path["records"][0]["record"] = "records/x/../000.npz"
            with self.assertRaisesRegex(ValueError, "not canonical"):
                crossfit._validate_cache_population(
                    noncanonical_path,
                    cache_dir,
                    expected_official_train_sha256=official_sha256,
                )

            wrong_total = copy.deepcopy(manifest)
            wrong_total["selected_event_count"] -= 1
            with self.assertRaisesRegex(ValueError, "selected_event_count"):
                crossfit._validate_cache_population(
                    wrong_total,
                    cache_dir,
                    expected_official_train_sha256=official_sha256,
                )

            wrong_population = copy.deepcopy(manifest)
            wrong_population["total_train_video_count"] = 98
            with self.assertRaisesRegex(ValueError, "all 99"):
                crossfit._validate_cache_population(
                    wrong_population,
                    cache_dir,
                    expected_official_train_sha256=official_sha256,
                )

            wrong_checkpoint = copy.deepcopy(manifest)
            wrong_checkpoint["base_checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "released M20"):
                crossfit._validate_cache_population(
                    wrong_checkpoint,
                    cache_dir,
                    expected_official_train_sha256=official_sha256,
                )

    def test_c00_requires_cfg_threshold_and_primary_only_cache_route(self):
        config_path = PROJECT_ROOT / "configs" / "evisseg_evuav.yaml"
        bad_threshold = replay.load_flat_config(
            config_path, c00_overrides() + ["TEST.prediction_threshold=0.720"]
        )
        with self.assertRaisesRegex(ValueError, "cfg.prediction_threshold"):
            crossfit.validate_c00_config(bad_threshold, 0.719)

        secondary = replay.load_flat_config(
            config_path,
            c00_overrides()
            + [
                "TEMPORAL_MEMORY.temporal_memory_secondary_model_path=secondary.pt",
                "TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000",
            ],
        )
        with self.assertRaisesRegex(ValueError, "only the primary M20"):
            crossfit.validate_c00_config(secondary, 0.719)

    def test_pooled_sufficient_counts_exactly_match_original_evaluator(self):
        videos = [
            {
                "scores": np.asarray([0.8, 0.8, 0.1, 0.8, 0.1, 0.1], np.float32),
                "labels": np.asarray([1, 0, 1, 0, 0, 0], np.uint8),
                "ids": np.asarray([1, 0, 1, 0, 0, 0], np.int16),
                "locations": np.asarray(
                    [
                        [0, 1, 1, 10],
                        [0, 10, 10, 20],
                        [0, 2, 2, 60],
                        [0, 20, 20, 70],
                        [0, 0, 0, 110],
                        [0, 0, 0, 160],
                    ],
                    np.int64,
                ),
            },
            {
                "scores": np.asarray([0.8, 0.1, 0.8, 0.1], np.float32),
                "labels": np.asarray([1, 1, 0, 0], np.uint8),
                "ids": np.asarray([2, 2, 0, 0], np.int16),
                "locations": np.asarray(
                    [
                        [0, 3, 3, 15],
                        [0, 4, 4, 65],
                        [0, 30, 30, 115],
                        [0, 0, 0, 165],
                    ],
                    np.int64,
                ),
            },
        ]
        evaluator = evalute(
            SimpleNamespace(roc=True, pd_detT=50, correct_thresh=0.0001)
        )
        pooled = crossfit.SufficientCounts()
        sample_number = 0
        for video in videos:
            batch = {
                "seg_label": torch.from_numpy(video["labels"].astype(np.float32)),
                "locs": torch.from_numpy(video["locations"]),
                "idx_label": video["ids"],
            }
            sample_number = add_batch_to_evaluator(
                evaluator,
                batch,
                torch.from_numpy(video["scores"]),
                sample_number,
                0.719,
            )
            pooled = pooled + crossfit.sufficient_counts_for_video(
                video["scores"],
                video["labels"],
                video["ids"],
                video["locations"],
            )
        official = evaluate_challenge_metrics(evaluator, 0.719).to_dict()
        reconstructed = crossfit.metrics_from_counts(pooled)
        self.assertEqual(pooled.correct_objects, evaluator.correct_num)
        self.assertEqual(pooled.object_count, evaluator.obj_num)
        self.assertEqual(pooled.false_components, evaluator.false_num)
        self.assertEqual(pooled.frame_count, evaluator.frame_num)
        for name in official:
            self.assertAlmostEqual(reconstructed[name], official[name], places=14)

    def test_protocol_preregister_hash_and_tamper_guards(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            config_path = PROJECT_ROOT / "configs" / "evisseg_evuav.yaml"
            overrides = c00_overrides()
            cfg = replay.load_flat_config(config_path, overrides)
            manifest, synthetic_official_sha256 = synthetic_cache_manifest(cfg)
            (cache_dir / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            protocol_path = root / "protocol.json"
            arguments = [
                "preregister",
                "--cache-dir",
                str(cache_dir),
                "--config",
                str(config_path),
                "--output-protocol",
                str(protocol_path),
                "--expected-selected-videos",
                "54",
            ]
            for override in overrides:
                arguments.extend(("--override", override))
            with mock.patch.object(
                crossfit,
                "OFFICIAL_TRAIN_SOURCE_MANIFEST_SHA256",
                synthetic_official_sha256,
            ):
                self.assertEqual(crossfit.main(arguments), 0)
            original_sha = sha256_file(protocol_path)
            payload, actual_sha = crossfit.load_frozen_protocol(
                protocol_path, original_sha
            )
            self.assertEqual(actual_sha, original_sha)
            self.assertEqual(len(payload["definition"]["blocks"]["middle"]), 39)
            self.assertEqual(
                payload["definition"]["frozen_document"]["sha256"],
                sha256_file(PROJECT_ROOT / crossfit.PROTOCOL_DOCUMENT),
            )

            payload["definition"]["candidates"][0]["positive_weight"] = 999
            protocol_path.write_text(
                json.dumps(payload, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match expected"):
                crossfit.load_frozen_protocol(protocol_path, original_sha)
            with self.assertRaisesRegex(ValueError, "canonical SHA-256 mismatch"):
                crossfit.load_frozen_protocol(protocol_path, sha256_file(protocol_path))


if __name__ == "__main__":
    unittest.main()
