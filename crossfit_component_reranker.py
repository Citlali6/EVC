"""Preregister and run the train-only dense component-reranker cross-fit.

This module consumes only the immutable official-train cache produced by
``train_component_reranker.py cache``.  It never accepts a validation path.
The protocol is deliberately narrow: two independent high-density source
blocks are held out in turn, the remaining middle-density videos are used for
fitting but receive an identity reranker in pooled OOF scoring, and a runtime
artifact is emitted only if every preregistered promotion gate passes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import torch

import replay_temporal_memory_validation as replay
from train_component_reranker import (
    OFFICIAL_TRAIN_SOURCE_MANIFEST_SHA256,
    TRAIN_SOURCE_MANIFEST_SCHEME,
    _atomic_json,
    _load_cache_record,
    _require_new_output,
    load_train_cache,
    source_manifest_sha256,
)
from utils.challenge_eval import challenge_score
from utils.component_reranker import (
    ARTIFACT_SCHEMA,
    FEATURE_NAMES,
    FEATURE_SEMANTICS_VERSION,
    TRAIN_CACHE_SCHEMA,
    ComponentTopology,
    extract_component_examples,
    input_postprocess_mapping,
    sha256_file,
    sha256_json,
    temporal_memory_inference_mapping,
)
from utils.eval import evalute
from utils.postprocess import (
    ChallengePostprocessor,
    P0ClusterFilter,
    P18ScoreTrackRecovery,
)


PROTOCOL_SCHEMA = "ev-uav-component-reranker-crossfit-protocol-v1"
REPORT_SCHEMA = "ev-uav-component-reranker-crossfit-report-v1"
PROTOCOL_DOCUMENT = "docs/COMPONENT_RERANKER_CROSSFIT_PROTOCOL.md"
CODE_PATHS = (
    "crossfit_component_reranker.py",
    "train_component_reranker.py",
    "utils/component_reranker.py",
    "utils/postprocess.py",
    "utils/challenge_eval.py",
    "utils/eval.py",
    "replay_temporal_memory_validation.py",
)

CACHE_MIN_EVENT_COUNT_EXCLUSIVE = 30000
EXPECTED_SELECTED_VIDEO_COUNT = 54
EXPECTED_SELECTED_EVENT_COUNT = 8555762
DEPLOYMENT_EVENT_COUNT_CUTOFF = 100000
RELEASED_M20_CHECKPOINT_SHA256 = (
    "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
)
PREDICTION_THRESHOLD = 0.719
PD_DETECTION_INTERVAL = 50
CORRECT_THRESHOLD = 0.0001
RESOLUTION = (346, 260)

H1_NAMES = tuple("train_{:03d}.npz".format(index) for index in range(44, 48))
H2_NAMES = tuple("train_{:03d}.npz".format(index) for index in range(88, 99))
EXPECTED_SELECTED_INDICES = (
    tuple(range(0, 15))
    + tuple(range(28, 33))
    + tuple(range(40, 48))
    + tuple(range(59, 66))
    + tuple(range(67, 75))
    + tuple(range(88, 99))
)
EXPECTED_SELECTED_NAMES = tuple(
    "train_{:03d}.npz".format(index) for index in EXPECTED_SELECTED_INDICES
)
FOLD_PLAN = (
    {
        "fold_id": "holdout_h1",
        "fit_high_block": "h2",
        "held_block": "h1",
        "fit_middle": True,
        "middle_oof_reranker": "identity",
    },
    {
        "fold_id": "holdout_h2",
        "fit_high_block": "h1",
        "held_block": "h2",
        "fit_middle": True,
        "middle_oof_reranker": "identity",
    },
)
POSITIVE_WEIGHTS = (4.0, 8.0)
KEEP_PROBABILITIES = (0.02, 0.05, 0.10, 0.20)
L2_PENALTY = 0.1
MAX_ITERATIONS = 50
CONSERVATIVE_V1_PROFILE = "conservative_v1"
POSTHOC_V2_PROFILE = "posthoc_pw4_kp040_v2"
CANDIDATE_PROFILES = (CONSERVATIVE_V1_PROFILE, POSTHOC_V2_PROFILE)
POSTHOC_HYPOTHESIS_ORIGIN = "retrospective_train_only_after_v1_noop"
POSTHOC_V1_REPORT_SHA256 = (
    "e06182a03667e169b16fee7b02e7e44dd636457bc2b4d6f62265dcb6300577d0"
)
POSTHOC_V1_REPORT_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "20260810_component_reranker_crosssource_v1"
    / "crossfit_report.json"
).resolve()
POSTHOC_DIAGNOSTIC_SCHEMA = "ev-uav-component-reranker-posthoc-train-diagnostic-v1"
POSTHOC_DIAGNOSTIC_EVIDENCE_CLASS = "retrospective_train_only_not_independent_oof"
POSTHOC_DIAGNOSTIC_SHA256 = (
    "5ba11e24c8f820773b9baf2c1b778131cdf30876dffbc64b4f2d4c06b17ef8e3"
)
POSTHOC_DIAGNOSTIC_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "20260810_component_reranker_crosssource_v1"
    / "posthoc_threshold_diagnostic.json"
).resolve()
POSTHOC_SINGLETON_POLICY = (
    "Run as a singleton with no further candidate selection, then at most one "
    "24-val replay. Do not tune 0.39, 0.41, or any other threshold from validation."
)
POSTHOC_VALIDATION_POLICY_SCHEMA = (
    "ev-uav-component-reranker-singleton-validation-policy-v1"
)
POSTHOC_VALIDATION_POLICY_SHA256 = (
    "8f26a10b8585e31a2bef26177a2bc6390292549f66ce4e0b39f6c51eecf9d987"
)
POSTHOC_VALIDATION_POLICY_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "20260810_component_reranker_posthoc_singleton_v2"
    / "validation_acceptance_policy.json"
).resolve()
POOLED_SCORE_DELTA_GATE = 0.0002
FALSE_COMPONENT_REDUCTION_GATE = 0.01
TOPOLOGY = ComponentTopology(
    spatial_radius=1,
    temporal_bin_size=50,
    max_link_distance=6.0,
    max_gap_bins=1,
    max_component_events=3,
)
CANONICAL_NAME = re.compile(r"^train_(\d{3})\.npz$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_sha256(value, name):
    value = str(value).strip().lower()
    if HEX64.fullmatch(value) is None:
        raise ValueError("{} must be a 64-character lowercase SHA-256.".format(name))
    return value


def _code_sha256(project_root):
    result = {}
    for relative in CODE_PATHS:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError("Cross-fit code source is missing: {}".format(path))
        result[relative] = sha256_file(path)
    return result


def candidate_definitions(profile=CONSERVATIVE_V1_PROFILE):
    """Return the exact candidates frozen by the named experiment profile."""
    if profile not in CANDIDATE_PROFILES:
        raise ValueError("Unsupported component-reranker candidate profile: {!r}.".format(profile))
    if profile == POSTHOC_V2_PROFILE:
        return [
            {
                "candidate_id": "pw04_kp400",
                "positive_weight": 4.0,
                "keep_probability": 0.40,
                "l2": L2_PENALTY,
            }
        ]
    candidates = []
    for positive_weight in POSITIVE_WEIGHTS:
        for keep_probability in KEEP_PROBABILITIES:
            candidates.append(
                {
                    "candidate_id": "pw{:02d}_kp{:03d}".format(
                        int(positive_weight), int(round(keep_probability * 1000))
                    ),
                    "positive_weight": positive_weight,
                    "keep_probability": keep_probability,
                    "l2": L2_PENALTY,
                }
            )
    return candidates


def _postprocess_contract(cfg):
    names = (
        "p0_enabled",
        "p0_spatial_radius",
        "p0_temporal_bin_size",
        "p0_temporal_radius_bins",
        "p0_min_cluster_events",
        "p0_min_duration_bins",
        "p0c_high_confidence_recovery_enabled",
        "p0c_retain_min_score",
        "p0c_density_retain_enabled",
        "p0c_density_event_count_cutoff",
        "p0c_density_retain_min_score",
        "component_reranker_enabled",
        "component_reranker_event_count_cutoff",
        "p0b_enabled",
        "p18_score_track_recovery_enabled",
        "p18_event_count_cutoff",
        "p18_max_event_count",
        "p18_candidate_floor",
        "p18_spatial_radius",
        "p18_temporal_bin_size",
        "p18_max_link_distance",
        "p18_max_gap_bins",
        "p18_min_track_bins",
        "p18_restore_mode",
        "p18_max_restore_events_per_component",
        "p6_density_threshold_enabled",
        "p6_event_count_cutoff",
        "p6_low_density_threshold",
        "p6_high_density_threshold",
    )
    missing = [name for name in names if not hasattr(cfg, name)]
    if missing:
        raise ValueError("C00 config is missing: {}.".format(", ".join(missing)))
    return {name: getattr(cfg, name) for name in names}


def validate_c00_config(cfg, prediction_threshold=PREDICTION_THRESHOLD):
    """Fail closed unless the exact released C00 route is configured."""
    if not math.isclose(
        float(prediction_threshold), PREDICTION_THRESHOLD, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Cross-fit prediction threshold is frozen at 0.719.")
    if not math.isclose(
        float(getattr(cfg, "prediction_threshold", float("nan"))),
        PREDICTION_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("C00 cfg.prediction_threshold must equal 0.719.")
    expected = {
        "p0_enabled": True,
        "p0_spatial_radius": 2,
        "p0_temporal_bin_size": 50,
        "p0_temporal_radius_bins": 1,
        "p0_min_cluster_events": 3,
        "p0_min_duration_bins": 5,
        "p0c_high_confidence_recovery_enabled": True,
        "p0c_retain_min_score": 0.95,
        "p0c_density_retain_enabled": False,
        "p0c_density_event_count_cutoff": 100000,
        "p0c_density_retain_min_score": 0.97,
        "component_reranker_enabled": False,
        "component_reranker_event_count_cutoff": 100000,
        "p0b_enabled": False,
        "p18_score_track_recovery_enabled": True,
        "p18_event_count_cutoff": 1,
        "p18_max_event_count": 35000,
        "p18_candidate_floor": 0.53,
        "p18_spatial_radius": 5,
        "p18_temporal_bin_size": 50,
        "p18_max_link_distance": 8.0,
        "p18_max_gap_bins": 1,
        "p18_min_track_bins": 4,
        "p18_restore_mode": "best",
        "p18_max_restore_events_per_component": 0,
        "p6_density_threshold_enabled": True,
        "p6_event_count_cutoff": 30000,
        "p6_low_density_threshold": 0.718,
        "p6_high_density_threshold": 0.719,
    }
    actual = _postprocess_contract(cfg)
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if isinstance(expected_value, float):
            matches = math.isclose(
                float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual_value == expected_value
        if not matches:
            raise ValueError(
                "C00 contract mismatch for {}: expected {!r}, got {!r}.".format(
                    name, expected_value, actual_value
                )
            )
    if not bool(getattr(cfg, "temporal_memory_enabled", False)):
        raise ValueError("C00 cross-fit requires temporal_memory_enabled=true.")
    if float(getattr(cfg, "temporal_memory_sparse_weight", -1.0)) != 0.0:
        raise ValueError("C00 cross-fit requires temporal_memory_sparse_weight=0.")
    if not bool(getattr(cfg, "temporal_memory_temporal_attention_enabled", False)):
        raise ValueError(
            "C00 cross-fit requires temporal_memory_temporal_attention_enabled=true."
        )
    if bool(getattr(cfg, "temporal_frame_enabled", False)):
        raise ValueError("C00 cross-fit rejects temporal-frame blending.")
    if bool(getattr(cfg, "dense_expert_enabled", False)):
        raise ValueError("C00 cross-fit rejects dense-expert routing.")
    if bool(getattr(cfg, "ensemble_enabled", False)):
        raise ValueError("C00 cross-fit rejects sparse-model ensembles.")
    if str(getattr(cfg, "temporal_memory_blend_model_path", "")).strip():
        raise ValueError("C00 cross-fit rejects a dense temporal blend model.")
    if str(getattr(cfg, "temporal_memory_secondary_model_path", "")).strip():
        raise ValueError("Cross-fit cache input must use only the primary M20 model.")
    if int(getattr(cfg, "temporal_memory_secondary_max_event_count", -1)) != 0:
        raise ValueError("Cross-fit secondary_max_event_count must equal zero.")
    if not math.isclose(
        float(getattr(cfg, "temporal_memory_primary_weight", float("nan"))),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Cross-fit temporal_memory_primary_weight must equal one.")
    resolution = tuple(int(value) for value in getattr(cfg, "res"))
    if resolution != RESOLUTION:
        raise ValueError("C00 evaluation resolution must be [346, 260].")
    if int(getattr(cfg, "pd_detT")) != PD_DETECTION_INTERVAL:
        raise ValueError("C00 pd_detT must be 50.")
    if not math.isclose(
        float(getattr(cfg, "correct_thresh")),
        CORRECT_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("C00 correct_thresh must be 0.0001.")
    return actual


def _validate_cache_population(
    manifest,
    cache_dir,
    expected_official_train_sha256=None,
):
    if manifest.get("schema") != TRAIN_CACHE_SCHEMA:
        raise ValueError("Cross-fit requires the component train-cache schema.")
    if manifest.get("dataset_split") != "train":
        raise ValueError("Cross-fit accepts only dataset_split=train.")
    if int(manifest.get("total_train_video_count", -1)) != 99:
        raise ValueError("Cross-fit cache must originate from all 99 train videos.")
    if manifest.get("official_train_source_manifest_scheme") != TRAIN_SOURCE_MANIFEST_SCHEME:
        raise ValueError("Cross-fit cache has an invalid official source-manifest scheme.")
    official_sources = manifest.get("official_train_sources")
    if not isinstance(official_sources, list):
        raise ValueError("Cross-fit cache is missing all-99 official source hashes.")
    official_manifest_sha256 = source_manifest_sha256(official_sources)
    recorded_official_sha256 = _require_sha256(
        manifest.get("official_train_source_manifest_sha256", ""),
        "official train source manifest SHA-256",
    )
    if official_manifest_sha256 != recorded_official_sha256:
        raise ValueError("Official train source manifest canonical SHA-256 mismatch.")
    required_official_sha256 = (
        OFFICIAL_TRAIN_SOURCE_MANIFEST_SHA256
        if expected_official_train_sha256 is None
        else _require_sha256(
            expected_official_train_sha256,
            "expected official train source manifest SHA-256",
        )
    )
    if recorded_official_sha256 != required_official_sha256:
        raise ValueError(
            "Official train source manifest SHA-256 {} does not match required {}."
            .format(recorded_official_sha256, required_official_sha256)
        )
    official_sha_by_name = {
        entry["source_name"]: entry["source_sha256"] for entry in official_sources
    }
    if len(set(official_sha_by_name.values())) != 99:
        raise ValueError("Official train raw source SHA-256 values must be unique.")
    selection = manifest.get("selection")
    expected_selection = {
        "observable": "complete_video_event_count",
        "operator": ">",
        "min_event_count_exclusive": CACHE_MIN_EVENT_COUNT_EXCLUSIVE,
    }
    if selection != expected_selection:
        raise ValueError("Cross-fit cache selection must be strict event_count > 30000.")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_SELECTED_VIDEO_COUNT:
        raise ValueError("Cross-fit cache must contain exactly 54 selected videos.")
    if int(manifest.get("selected_video_count", -1)) != len(records):
        raise ValueError("Cross-fit selected_video_count differs from records.")
    names = []
    metadata_by_name = {}
    cache_dir = Path(cache_dir).resolve()
    normalized_record_paths = set()
    record_sha256_values = set()
    selected_source_sha256_values = set()
    selected_event_count = 0
    for metadata in records:
        if not isinstance(metadata, dict):
            raise ValueError("Cross-fit record metadata must be JSON objects.")
        name = str(metadata.get("source_name", ""))
        match = CANONICAL_NAME.fullmatch(name)
        if match is None or not 0 <= int(match.group(1)) < 99:
            raise ValueError("Non-canonical train source name: {!r}.".format(name))
        if name in metadata_by_name:
            raise ValueError("Duplicate train source name: {}.".format(name))
        event_count = int(metadata.get("event_count", -1))
        if event_count <= CACHE_MIN_EVENT_COUNT_EXCLUSIVE:
            raise ValueError("Cache record does not satisfy strict >30000: {}.".format(name))
        selected_event_count += event_count
        raw_record_path = str(metadata.get("record", ""))
        relative_record_path = Path(raw_record_path)
        if (
            not raw_record_path
            or relative_record_path.is_absolute()
            or ".." in relative_record_path.parts
            or raw_record_path != relative_record_path.as_posix()
        ):
            raise ValueError("Cross-fit record path is not canonical relative POSIX: {!r}.".format(raw_record_path))
        resolved_record_path = (cache_dir / relative_record_path).resolve()
        try:
            resolved_record_path.relative_to(cache_dir)
        except ValueError as exc:
            raise ValueError("Cross-fit record path escaped the cache directory.") from exc
        normalized_key = os.path.normcase(str(resolved_record_path))
        if normalized_key in normalized_record_paths:
            raise ValueError("Cross-fit cache record paths alias across train sources.")
        normalized_record_paths.add(normalized_key)
        record_sha256 = _require_sha256(
            metadata.get("record_sha256", ""), "cache record SHA-256"
        )
        if record_sha256 in record_sha256_values:
            raise ValueError("Cross-fit cache record SHA-256 values must be unique.")
        record_sha256_values.add(record_sha256)
        source_sha256 = _require_sha256(
            metadata.get("source_sha256", ""), "selected source SHA-256"
        )
        if source_sha256 in selected_source_sha256_values:
            raise ValueError("Selected train source SHA-256 values must be unique.")
        selected_source_sha256_values.add(source_sha256)
        if official_sha_by_name.get(name) != source_sha256:
            raise ValueError(
                "Selected source SHA-256 differs from the official all-99 manifest: {}."
                .format(name)
            )
        names.append(name)
        metadata_by_name[name] = metadata
    if names != sorted(names):
        raise ValueError("Cross-fit cache records must use canonical sorted order.")
    if tuple(names) != EXPECTED_SELECTED_NAMES:
        raise ValueError("Cross-fit selected train names differ from the official >30000 population.")
    if int(manifest.get("selected_event_count", -1)) != selected_event_count:
        raise ValueError("Cross-fit selected_event_count differs from record totals.")
    if selected_event_count != EXPECTED_SELECTED_EVENT_COUNT:
        raise ValueError(
            "Cross-fit selected event total differs from official 8,555,762."
        )
    missing_high = sorted((set(H1_NAMES) | set(H2_NAMES)).difference(names))
    if missing_high:
        raise ValueError("Cross-fit cache is missing high block videos: {}.".format(missing_high))
    for name in H1_NAMES + H2_NAMES:
        if int(metadata_by_name[name]["event_count"]) <= DEPLOYMENT_EVENT_COUNT_CUTOFF:
            raise ValueError("High block is not deployment-eligible: {}.".format(name))
    middle_names = tuple(
        name for name in names if name not in set(H1_NAMES) and name not in set(H2_NAMES)
    )
    if len(middle_names) != 39:
        raise ValueError("Cross-fit requires exactly 39 middle videos.")
    for name in middle_names:
        if int(metadata_by_name[name]["event_count"]) > DEPLOYMENT_EVENT_COUNT_CUTOFF:
            raise ValueError("A deployment-eligible video escaped H1/H2: {}.".format(name))
    base_checkpoint_sha256 = _require_sha256(
        manifest.get("base_checkpoint_sha256", ""), "base checkpoint SHA-256"
    )
    if base_checkpoint_sha256 != RELEASED_M20_CHECKPOINT_SHA256:
        raise ValueError(
            "Cross-fit cache base checkpoint is not the released M20 checkpoint."
        )
    return tuple(names), middle_names


def _oof_definition(profile):
    winner_rule = (
        "maximum_held_score_then_candidate_id"
        if profile == CONSERVATIVE_V1_PROFILE
        else "singleton_candidate_no_hyperparameter_selection"
    )
    return {
        "middle_reranker": "identity_after_full_c00_including_p18",
        "winner_rule": winner_rule,
        "pooling": "h1_held_plus_h2_held_plus_middle_identity_sufficient_counts_once",
    }


def _validate_profile_arguments(
    profile,
    hypothesis_source_report=None,
    expected_hypothesis_source_report_sha256=None,
    hypothesis_source_diagnostic=None,
    expected_hypothesis_source_diagnostic_sha256=None,
):
    if profile not in CANDIDATE_PROFILES:
        raise ValueError("Unsupported component-reranker candidate profile: {!r}.".format(profile))
    has_report = hypothesis_source_report is not None
    has_report_sha = expected_hypothesis_source_report_sha256 is not None
    has_diagnostic = hypothesis_source_diagnostic is not None
    has_diagnostic_sha = expected_hypothesis_source_diagnostic_sha256 is not None
    if profile == CONSERVATIVE_V1_PROFILE:
        if has_report or has_report_sha or has_diagnostic or has_diagnostic_sha:
            raise ValueError(
                "conservative_v1 does not accept retrospective hypothesis-source inputs."
            )
        return
    if not (has_report and has_report_sha and has_diagnostic and has_diagnostic_sha):
        raise ValueError(
            "posthoc_pw4_kp040_v2 requires the source v1 report and post-hoc "
            "diagnostic, each with an expected SHA-256."
        )
    expected = _require_sha256(
        expected_hypothesis_source_report_sha256,
        "expected hypothesis source report SHA-256",
    )
    if expected != POSTHOC_V1_REPORT_SHA256:
        raise ValueError(
            "posthoc_pw4_kp040_v2 is bound to the first v1 no-op report SHA-256."
        )
    report_path = Path(hypothesis_source_report).resolve()
    if os.path.normcase(str(report_path)) != os.path.normcase(
        str(POSTHOC_V1_REPORT_PATH)
    ):
        raise ValueError(
            "posthoc_pw4_kp040_v2 requires the frozen first v1 report path."
        )
    expected_diagnostic = _require_sha256(
        expected_hypothesis_source_diagnostic_sha256,
        "expected hypothesis source diagnostic SHA-256",
    )
    if expected_diagnostic != POSTHOC_DIAGNOSTIC_SHA256:
        raise ValueError(
            "posthoc_pw4_kp040_v2 is bound to the frozen train-only diagnostic SHA-256."
        )
    diagnostic_path = Path(hypothesis_source_diagnostic).resolve()
    if os.path.normcase(str(diagnostic_path)) != os.path.normcase(
        str(POSTHOC_DIAGNOSTIC_PATH)
    ):
        raise ValueError(
            "posthoc_pw4_kp040_v2 requires the frozen train-only diagnostic path."
        )


def _no_op_v1_report_check(report):
    expected_candidates = candidate_definitions(CONSERVATIVE_V1_PROFILE)
    expected_ids = [candidate["candidate_id"] for candidate in expected_candidates]
    folds = report.get("fold_results")
    if not isinstance(folds, list) or [fold.get("fold_id") for fold in folds] != [
        "holdout_h1",
        "holdout_h2",
    ]:
        raise ValueError("Hypothesis source report does not contain the frozen two folds.")
    for fold in folds:
        baseline = fold.get("baseline")
        if not isinstance(baseline, dict):
            raise ValueError("Hypothesis source report fold is missing its baseline.")
        results = fold.get("candidate_results")
        if not isinstance(results, list) or [
            result.get("candidate_id") for result in results
        ] != expected_ids:
            raise ValueError(
                "Hypothesis source report is not the exact conservative_v1 grid."
            )
        for expected, result in zip(expected_candidates, results):
            observed = {name: result.get(name) for name in expected}
            if observed != expected:
                raise ValueError(
                    "Hypothesis source report candidate definition was modified."
                )
            if result.get("counts") != baseline.get("counts"):
                raise ValueError(
                    "Hypothesis source report is not a strict all-candidate no-op."
                )
            if result.get("metrics") != baseline.get("metrics"):
                raise ValueError(
                    "Hypothesis source report no-op metrics differ from baseline."
                )
            delta = result.get("delta")
            if not isinstance(delta, dict) or any(float(value) != 0.0 for value in delta.values()):
                raise ValueError(
                    "Hypothesis source report contains a nonzero candidate delta."
                )
            if int(result.get("false_component_delta", 1)) != 0:
                raise ValueError(
                    "Hypothesis source report contains a false-component change."
                )
        if fold.get("winner_candidate_id") != expected_ids[0]:
            raise ValueError("Hypothesis source report v1 tie-break winner was modified.")

    gates = report.get("promotion_gates")
    if not isinstance(gates, dict) or gates.get("passed") is not False:
        raise ValueError("Hypothesis source v1 report must have failed its frozen gates.")
    baseline_pooled = gates.get("baseline_pooled", {})
    selected_pooled = gates.get("selected_pooled_oof", {})
    if selected_pooled.get("counts") != baseline_pooled.get("counts"):
        raise ValueError("Hypothesis source v1 pooled result is not a strict no-op.")
    pooled_delta = selected_pooled.get("delta")
    if not isinstance(pooled_delta, dict) or any(
        float(value) != 0.0 for value in pooled_delta.values()
    ):
        raise ValueError("Hypothesis source v1 pooled delta is not zero.")
    artifact = report.get("artifact")
    if artifact != {"emitted": False, "path": None, "sha256": None}:
        raise ValueError("Hypothesis source v1 report must not have emitted an artifact.")


def _validation_acceptance_policy_metadata(source_diagnostic_path):
    policy_path = Path(POSTHOC_VALIDATION_POLICY_PATH).resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(
            "Frozen singleton validation acceptance policy does not exist: {}".format(
                policy_path
            )
        )
    actual_sha256 = sha256_file(policy_path)
    if actual_sha256 != POSTHOC_VALIDATION_POLICY_SHA256:
        raise ValueError("Frozen singleton validation acceptance policy SHA-256 differs.")
    with policy_path.open("r", encoding="utf-8") as stream:
        policy = json.load(stream)
    if not isinstance(policy, dict) or policy.get("schema") != (
        POSTHOC_VALIDATION_POLICY_SCHEMA
    ):
        raise ValueError("Frozen singleton validation acceptance policy schema is invalid.")
    if policy.get("status") != "frozen_before_v2_artifact_and_before_validation_replay":
        raise ValueError("Singleton validation policy was not frozen before replay.")
    if policy.get("evidence_class") != (
        "retrospective_train_hypothesis_with_one_frozen_validation_check"
    ):
        raise ValueError("Singleton validation policy evidence class is invalid.")
    hypothesis_source = policy.get("hypothesis_source")
    if not isinstance(hypothesis_source, dict):
        raise ValueError("Singleton validation policy is missing its hypothesis source.")
    policy_diagnostic_path = Path(str(hypothesis_source.get("path", ""))).resolve()
    if os.path.normcase(str(policy_diagnostic_path)) != os.path.normcase(
        str(Path(source_diagnostic_path).resolve())
    ):
        raise ValueError("Singleton validation policy points to a different diagnostic.")
    expected_source = {
        "sha256": POSTHOC_DIAGNOSTIC_SHA256,
        "positive_weight": 4.0,
        "keep_probability": 0.4,
        "l2": 0.1,
    }
    if {name: hypothesis_source.get(name) for name in expected_source} != expected_source:
        raise ValueError("Singleton validation policy hypothesis binding differs.")
    budget = policy.get("evaluation_budget")
    if not isinstance(budget, dict) or {
        "full_validation_replays": budget.get("full_validation_replays"),
        "threshold_or_hyperparameter_search_after_replay": budget.get(
            "threshold_or_hyperparameter_search_after_replay"
        ),
        "allowed_follow_up_on_failure": budget.get("allowed_follow_up_on_failure"),
    } != {
        "full_validation_replays": 1,
        "threshold_or_hyperparameter_search_after_replay": False,
        "allowed_follow_up_on_failure": (
            "archive this suppression branch without tuning it on validation"
        ),
    }:
        raise ValueError("Singleton validation policy budget differs from the frozen rule.")
    return {
        "path": str(policy_path),
        "sha256": actual_sha256,
        "schema": POSTHOC_VALIDATION_POLICY_SCHEMA,
        "status": policy["status"],
        "full_validation_replays": 1,
        "threshold_or_hyperparameter_search_after_replay": False,
        "failure_action": "archive_without_validation_tuning",
    }


def _validate_posthoc_hypothesis_source(
    source_report_path,
    expected_source_report_sha256,
    source_diagnostic_path,
    expected_source_diagnostic_sha256,
    current_definition,
    current_manifest_path,
):
    """Validate and summarize the immutable v1 report that generated v2."""
    _validate_profile_arguments(
        POSTHOC_V2_PROFILE,
        source_report_path,
        expected_source_report_sha256,
        source_diagnostic_path,
        expected_source_diagnostic_sha256,
    )
    source_diagnostic_path = Path(source_diagnostic_path).resolve()
    expected_source_diagnostic_sha256 = _require_sha256(
        expected_source_diagnostic_sha256,
        "expected hypothesis source diagnostic SHA-256",
    )
    actual_diagnostic_sha256 = sha256_file(source_diagnostic_path)
    if actual_diagnostic_sha256 != expected_source_diagnostic_sha256:
        raise ValueError(
            "Hypothesis source diagnostic SHA-256 {} does not match expected {}."
            .format(actual_diagnostic_sha256, expected_source_diagnostic_sha256)
        )
    with source_diagnostic_path.open("r", encoding="utf-8") as stream:
        diagnostic = json.load(stream)
    if not isinstance(diagnostic, dict) or diagnostic.get("schema") != (
        POSTHOC_DIAGNOSTIC_SCHEMA
    ):
        raise ValueError("Hypothesis source train-only diagnostic schema is invalid.")
    if diagnostic.get("evidence_class") != POSTHOC_DIAGNOSTIC_EVIDENCE_CLASS:
        raise ValueError("Hypothesis source train-only diagnostic evidence class is invalid.")
    if diagnostic.get("original_grid") != {
        "positive_weights": [4.0, 8.0],
        "keep_probabilities": [0.02, 0.05, 0.10, 0.20],
        "outcome": "all candidates were exact no-ops",
    }:
        raise ValueError("Hypothesis source diagnostic v1 outcome is invalid.")
    expected_singleton = dict(candidate_definitions(POSTHOC_V2_PROFILE)[0])
    expected_singleton["policy"] = POSTHOC_SINGLETON_POLICY
    if diagnostic.get("frozen_followup_hypothesis") != expected_singleton:
        raise ValueError(
            "Hypothesis source diagnostic does not freeze the exact singleton policy."
        )

    source_report_path = Path(source_report_path).resolve()
    if not source_report_path.is_file():
        raise FileNotFoundError(
            "Hypothesis source v1 report does not exist: {}".format(source_report_path)
        )
    expected_source_report_sha256 = _require_sha256(
        expected_source_report_sha256,
        "expected hypothesis source report SHA-256",
    )
    actual_report_sha256 = sha256_file(source_report_path)
    if actual_report_sha256 != expected_source_report_sha256:
        raise ValueError(
            "Hypothesis source v1 report SHA-256 {} does not match expected {}."
            .format(actual_report_sha256, expected_source_report_sha256)
        )
    with source_report_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise ValueError("Hypothesis source v1 report schema is invalid.")
    if report.get("dataset_split") != "train":
        raise ValueError("Hypothesis source report must use only the train split.")
    if report.get("evidence_class") != (
        "train_only_cross_source_held_block_consistency_not_unbiased_oof"
    ):
        raise ValueError("Hypothesis source report evidence class is invalid.")

    current_manifest_path = Path(current_manifest_path).resolve()
    current_manifest_sha256 = current_definition["dataset"]["cache_manifest_sha256"]
    if sha256_file(current_manifest_path) != current_manifest_sha256:
        raise ValueError("Current train-cache manifest changed during source validation.")
    report_manifest_path = Path(str(report.get("cache_manifest_path", ""))).resolve()
    if os.path.normcase(str(report_manifest_path)) != os.path.normcase(
        str(current_manifest_path)
    ):
        raise ValueError("Hypothesis source report points to a different train cache.")
    if report.get("cache_manifest_sha256") != current_manifest_sha256:
        raise ValueError("Hypothesis source report train-cache SHA-256 lineage differs.")

    source_protocol_path = Path(str(report.get("protocol_path", ""))).resolve()
    source_protocol_sha256 = _require_sha256(
        report.get("protocol_file_sha256", ""),
        "hypothesis source protocol file SHA-256",
    )
    source_protocol, actual_protocol_sha256 = load_frozen_protocol(
        source_protocol_path,
        source_protocol_sha256,
    )
    if actual_protocol_sha256 != source_protocol_sha256:
        raise RuntimeError("Hypothesis source protocol SHA-256 validation was inconsistent.")
    if report.get("protocol_definition_sha256") != source_protocol.get(
        "definition_sha256"
    ):
        raise ValueError("Hypothesis source report/protocol definition lineage differs.")
    if diagnostic.get("source_protocol_sha256") != source_protocol_sha256:
        raise ValueError("Hypothesis diagnostic/source protocol SHA-256 lineage differs.")
    if diagnostic.get("source_crossfit_report_sha256") != actual_report_sha256:
        raise ValueError("Hypothesis diagnostic/source report SHA-256 lineage differs.")
    if diagnostic.get("train_cache_manifest_sha256") != current_manifest_sha256:
        raise ValueError("Hypothesis diagnostic/train-cache SHA-256 lineage differs.")
    source_definition = source_protocol["definition"]
    if source_definition.get("candidate_profile", CONSERVATIVE_V1_PROFILE) != (
        CONSERVATIVE_V1_PROFILE
    ):
        raise ValueError("Hypothesis source protocol is not conservative_v1.")
    if source_definition.get("candidates") != candidate_definitions(
        CONSERVATIVE_V1_PROFILE
    ):
        raise ValueError("Hypothesis source protocol candidate grid was modified.")
    if source_definition.get("oof") != _oof_definition(CONSERVATIVE_V1_PROFILE):
        raise ValueError("Hypothesis source protocol v1 winner rule was modified.")
    common_lineage_keys = (
        "dataset",
        "blocks",
        "fold_plan",
        "topology",
        "fit",
        "scoring",
        "gates",
        "promotion",
        "config",
    )
    for name in common_lineage_keys:
        if source_definition.get(name) != current_definition.get(name):
            raise ValueError(
                "Hypothesis source protocol differs in frozen {} lineage.".format(name)
            )
    _no_op_v1_report_check(report)
    return {
        "hypothesis_origin": POSTHOC_HYPOTHESIS_ORIGIN,
        "retrospective": True,
        "independent_oof_claim": False,
        "candidate_selection_after_source": "singleton_only_no_further_search",
        "source_v1_report": {
            "path": str(source_report_path),
            "expected_sha256": expected_source_report_sha256,
            "sha256": actual_report_sha256,
            "protocol_path": str(source_protocol_path),
            "protocol_file_sha256": source_protocol_sha256,
            "protocol_definition_sha256": source_protocol["definition_sha256"],
            "cache_manifest_path": str(current_manifest_path),
            "cache_manifest_sha256": current_manifest_sha256,
            "promotion_gates_passed": False,
            "artifact_emitted": False,
            "candidate_outcome": "all_eight_conservative_v1_candidates_strict_noop",
        },
        "source_train_diagnostic": {
            "path": str(source_diagnostic_path),
            "expected_sha256": expected_source_diagnostic_sha256,
            "sha256": actual_diagnostic_sha256,
            "schema": POSTHOC_DIAGNOSTIC_SCHEMA,
            "evidence_class": POSTHOC_DIAGNOSTIC_EVIDENCE_CLASS,
            "source_protocol_sha256": source_protocol_sha256,
            "source_crossfit_report_sha256": actual_report_sha256,
            "train_cache_manifest_sha256": current_manifest_sha256,
            "frozen_followup_hypothesis": expected_singleton,
        },
        "validation_acceptance_policy": _validation_acceptance_policy_metadata(
            source_diagnostic_path
        ),
    }


def _build_protocol_definition(
    cache_dir,
    config_path,
    overrides,
    prediction_threshold=PREDICTION_THRESHOLD,
    candidate_profile=CONSERVATIVE_V1_PROFILE,
    hypothesis_source_report=None,
    expected_hypothesis_source_report_sha256=None,
    hypothesis_source_diagnostic=None,
    expected_hypothesis_source_diagnostic_sha256=None,
):
    _validate_profile_arguments(
        candidate_profile,
        hypothesis_source_report,
        expected_hypothesis_source_report_sha256,
        hypothesis_source_diagnostic,
        expected_hypothesis_source_diagnostic_sha256,
    )
    project_root = Path(__file__).resolve().parent
    cache_dir, manifest_path, manifest_sha256, manifest = load_train_cache(cache_dir)
    selected_names, middle_names = _validate_cache_population(manifest, cache_dir)
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError("Cross-fit config does not exist: {}".format(config_path))
    cfg = replay.load_flat_config(config_path, list(overrides))
    postprocess_contract = validate_c00_config(cfg, prediction_threshold)
    inference_settings = temporal_memory_inference_mapping(cfg)
    if manifest.get("inference_settings") != inference_settings:
        raise ValueError("C00 inference settings differ from the train-cache manifest.")
    document_path = project_root / PROTOCOL_DOCUMENT
    if not document_path.is_file():
        raise FileNotFoundError("Frozen cross-fit protocol document is missing.")
    definition = {
        "dataset": {
            "dataset_split": "train",
            "cache_schema": TRAIN_CACHE_SCHEMA,
            "cache_manifest_sha256": manifest_sha256,
            "selection": manifest["selection"],
            "expected_selected_video_count": EXPECTED_SELECTED_VIDEO_COUNT,
            "expected_selected_event_count": EXPECTED_SELECTED_EVENT_COUNT,
            "selected_video_names": list(selected_names),
            "official_train_source_manifest_scheme": manifest[
                "official_train_source_manifest_scheme"
            ],
            "official_train_source_manifest_sha256": manifest[
                "official_train_source_manifest_sha256"
            ],
            "selected_source_identities": [
                {
                    "source_name": metadata["source_name"],
                    "source_sha256": metadata["source_sha256"],
                    "event_count": int(metadata["event_count"]),
                    "record": metadata["record"],
                    "record_sha256": metadata["record_sha256"],
                }
                for metadata in manifest["records"]
            ],
            "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
        },
        "blocks": {
            "h1": list(H1_NAMES),
            "h2": list(H2_NAMES),
            "middle": list(middle_names),
        },
        "fold_plan": [dict(item) for item in FOLD_PLAN],
        "candidate_profile": candidate_profile,
        "candidates": candidate_definitions(candidate_profile),
        "topology": TOPOLOGY.to_dict(),
        "fit": {
            "algorithm": "deterministic_domain_video_component_weighted_logistic_newton",
            "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "l2": L2_PENALTY,
            "max_iterations": MAX_ITERATIONS,
            "scaler_weighting": "base_domain_video_component_weights",
            "domain_base_weight": {"high": 0.5, "middle": 0.5},
            "within_domain": "equal_weight_per_video",
            "within_video": "equal_weight_per_component",
            "class_weighting": "multiply_positive_components_after_base_weight",
            "source_identity_is_feature": False,
            "block_identity_is_feature": False,
        },
        "oof": _oof_definition(candidate_profile),
        "scoring": {
            "prediction_threshold": float(prediction_threshold),
            "pd_detection_interval": PD_DETECTION_INTERVAL,
            "correct_threshold": CORRECT_THRESHOLD,
            "resolution": list(RESOLUTION),
            "false_component_connectivity": 8,
            "component_feature_time_binning": "floor_divide_integer_t_by_50",
            "pd_fa_time_binning": "unchanged_official_open_intervals",
            "boundary_events_in_semantic_iou_acc": True,
            "score_formula": "0.4*Pd + 0.3*exp(-10000*Fa) + 0.2*IoU + 0.1*Acc",
        },
        "gates": {
            "h1_score_delta_strictly_greater_than": 0.0,
            "h2_score_delta_strictly_greater_than": 0.0,
            "pooled_pd_delta_minimum": 0.0,
            "pooled_iou_delta_minimum": 0.0,
            "held_false_components_each_delta_maximum": 0,
            "held_false_components_combined_reduction_minimum": FALSE_COMPONENT_REDUCTION_GATE,
            "pooled_score_delta_minimum": POOLED_SCORE_DELTA_GATE,
            "fold_winner_candidate_ids_must_match": True,
        },
        "promotion": {
            "deployment_event_count_cutoff": DEPLOYMENT_EVENT_COUNT_CUTOFF,
            "full_refit_population": "all_54_selected_train_videos",
            "emit_artifact_only_if_all_gates_pass": True,
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "overrides": list(overrides),
            "postprocess_contract": postprocess_contract,
            "postprocess_contract_sha256": sha256_json(postprocess_contract),
            "inference_settings": inference_settings,
            "inference_settings_sha256": sha256_json(inference_settings),
        },
        "frozen_document": {
            "path": PROTOCOL_DOCUMENT,
            "sha256": sha256_file(document_path),
        },
        "code_sha256": _code_sha256(project_root),
        "software_versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
        },
    }
    if candidate_profile == POSTHOC_V2_PROFILE:
        definition["hypothesis"] = _validate_posthoc_hypothesis_source(
            hypothesis_source_report,
            expected_hypothesis_source_report_sha256,
            hypothesis_source_diagnostic,
            expected_hypothesis_source_diagnostic_sha256,
            definition,
            manifest_path,
        )
    return definition


def preregister_protocol(args):
    output_path = _require_new_output(args.output_protocol, "Cross-fit protocol")
    if int(args.expected_selected_videos) != EXPECTED_SELECTED_VIDEO_COUNT:
        raise ValueError("Frozen cross-fit expected-selected-videos must equal 54.")
    definition = _build_protocol_definition(
        args.cache_dir,
        args.config,
        args.override,
        prediction_threshold=float(args.prediction_threshold),
        candidate_profile=getattr(
            args, "candidate_profile", CONSERVATIVE_V1_PROFILE
        ),
        hypothesis_source_report=getattr(args, "hypothesis_source_report", None),
        expected_hypothesis_source_report_sha256=(
            getattr(args, "expected_hypothesis_source_report_sha256", None)
        ),
        hypothesis_source_diagnostic=getattr(
            args, "hypothesis_source_diagnostic", None
        ),
        expected_hypothesis_source_diagnostic_sha256=(
            getattr(args, "expected_hypothesis_source_diagnostic_sha256", None)
        ),
    )
    payload = {
        "schema": PROTOCOL_SCHEMA,
        "created_utc": _utc_now(),
        "definition": definition,
        "definition_sha256": sha256_json(definition),
    }
    _atomic_json(output_path, payload)
    print("wrote frozen cross-fit protocol:", output_path)
    print("protocol file sha256:", sha256_file(output_path))
    print("protocol definition sha256:", payload["definition_sha256"])
    return 0


def load_frozen_protocol(path, expected_sha256):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Cross-fit protocol does not exist: {}".format(path))
    expected_sha256 = _require_sha256(expected_sha256, "expected protocol SHA-256")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Cross-fit protocol SHA-256 {} does not match expected {}.".format(
                actual_sha256, expected_sha256
            )
        )
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "created_utc",
        "definition",
        "definition_sha256",
    }:
        raise ValueError("Cross-fit protocol top-level schema is invalid.")
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported cross-fit protocol schema.")
    definition = payload.get("definition")
    if not isinstance(definition, dict):
        raise ValueError("Cross-fit protocol definition must be a JSON object.")
    expected_definition_sha = _require_sha256(
        payload.get("definition_sha256", ""), "protocol definition SHA-256"
    )
    if sha256_json(definition) != expected_definition_sha:
        raise ValueError("Cross-fit protocol definition canonical SHA-256 mismatch.")
    return payload, actual_sha256


@dataclass(frozen=True)
class SufficientCounts:
    true_positive_events: int = 0
    false_positive_events: int = 0
    false_negative_events: int = 0
    correct_objects: int = 0
    object_count: int = 0
    false_components: int = 0
    frame_count: int = 0
    event_count: int = 0

    def __add__(self, other):
        if not isinstance(other, SufficientCounts):
            return NotImplemented
        return SufficientCounts(
            **{
                field: int(getattr(self, field) + getattr(other, field))
                for field in self.__dataclass_fields__
            }
        )

    def to_dict(self):
        return asdict(self)


def metrics_from_counts(counts):
    if not isinstance(counts, SufficientCounts):
        raise TypeError("counts must be SufficientCounts.")
    union = (
        counts.true_positive_events
        + counts.false_positive_events
        + counts.false_negative_events
    )
    positives = counts.true_positive_events + counts.false_negative_events
    denominator_fa = counts.frame_count * RESOLUTION[0] * RESOLUTION[1]
    if union <= 0 or positives <= 0 or counts.object_count <= 0 or denominator_fa <= 0:
        raise ValueError("Sufficient counts cannot produce finite Challenge metrics.")
    # The unchanged evaluator performs IoU and Acc division in float32.
    iou = float(
        torch.tensor(counts.true_positive_events, dtype=torch.float32)
        / torch.tensor(union, dtype=torch.float32)
    )
    acc = float(
        torch.tensor(counts.true_positive_events, dtype=torch.float32)
        / torch.tensor(positives, dtype=torch.float32)
    )
    pd = counts.correct_objects / counts.object_count
    fa = counts.false_components / denominator_fa
    score_fa, score = challenge_score(iou, acc, pd, fa)
    return {
        "iou": iou,
        "acc": acc,
        "pd": pd,
        "fa": fa,
        "score_fa": score_fa,
        "score": score,
    }


def sufficient_counts_for_video(
    scores,
    labels,
    target_ids,
    locations,
    prediction_threshold=PREDICTION_THRESHOLD,
):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    target_ids = np.asarray(target_ids).reshape(-1)
    locations = np.asarray(locations)
    if locations.ndim != 2 or locations.shape[1] != 4:
        raise ValueError("Evaluation locations must have shape [N, 4].")
    if not (
        scores.size == labels.size == target_ids.size == locations.shape[0]
    ):
        raise ValueError("Evaluation record lengths differ.")
    if np.unique(locations[:, 0]).size != 1:
        raise ValueError("Each sufficient-count record must contain one batch id.")
    evaluator = evalute(
        SimpleNamespace(
            roc=True,
            pd_detT=PD_DETECTION_INTERVAL,
            correct_thresh=CORRECT_THRESHOLD,
        )
    )
    score_tensor = torch.from_numpy(scores.copy())
    label_tensor = torch.from_numpy(labels.astype(np.float32, copy=False))
    location_tensor = torch.from_numpy(locations.astype(np.int64, copy=False))
    evaluator.roc_update(
        location_tensor[:, 3],
        score_tensor.clone(),
        target_ids,
        label_tensor,
        location_tensor,
        thresh=float(prediction_threshold),
    )
    predicted = scores >= float(prediction_threshold)
    positive = labels > 0
    return SufficientCounts(
        true_positive_events=int(np.sum(predicted & positive)),
        false_positive_events=int(np.sum(predicted & ~positive)),
        false_negative_events=int(np.sum(~predicted & positive)),
        correct_objects=int(evaluator.correct_num),
        object_count=int(evaluator.obj_num),
        false_components=int(evaluator.false_num),
        frame_count=int(evaluator.frame_num),
        event_count=int(scores.size),
    )


@dataclass
class PreparedVideo:
    source_name: str
    block: str
    event_count: int
    features: np.ndarray
    component_labels: np.ndarray
    event_indices: tuple
    p0_scores: np.ndarray | None
    locations: np.ndarray | None
    event_labels: np.ndarray | None
    target_ids: np.ndarray | None
    baseline_counts: SufficientCounts


def _block_for_name(name, middle_names):
    if name in H1_NAMES:
        return "h1"
    if name in H2_NAMES:
        return "h2"
    if name in middle_names:
        return "middle"
    raise ValueError("Train source is outside frozen cross-fit blocks: {}.".format(name))


def _prepare_videos(cache_dir, manifest, cfg, middle_names):
    videos = []
    for index, metadata in enumerate(manifest["records"], start=1):
        record = _load_cache_record(cache_dir, metadata)
        event_count = int(metadata["event_count"])
        raw_scores = torch.from_numpy(
            record["scores"].reshape(-1).astype(np.float32, copy=False)
        )
        locations = np.column_stack(
            (
                np.zeros(event_count, dtype=np.int64),
                record["locs"].astype(np.int64, copy=False),
            )
        )
        location_tensor = torch.from_numpy(locations).to(torch.int64).contiguous()

        # Baseline always traverses the complete frozen C00 chain.  This is
        # material for the 30k--35k middle videos where P18 is eligible.
        baseline_processor = ChallengePostprocessor.from_cfg(
            cfg, PREDICTION_THRESHOLD, event_count=event_count
        )
        baseline_scores, _ = baseline_processor.apply(raw_scores, location_tensor)

        p0_filter = P0ClusterFilter.from_cfg(
            cfg, PREDICTION_THRESHOLD, event_count=event_count
        )
        p0_scores, _ = p0_filter.apply(raw_scores, location_tensor)
        p18 = P18ScoreTrackRecovery.from_cfg(cfg, PREDICTION_THRESHOLD)
        chained_scores, _ = p18.apply(p0_scores, location_tensor)
        if not torch.equal(chained_scores, baseline_scores):
            raise RuntimeError("Manual P0->P18 chain differs from full C00 postprocessor.")

        examples = extract_component_examples(
            p0_scores.numpy(),
            locations,
            PREDICTION_THRESHOLD,
            TOPOLOGY,
            event_count,
            labels=record["labels"],
        )
        if not examples:
            raise RuntimeError(
                "Frozen equal-video weighting is undefined for zero candidates: {}."
                .format(metadata["source_name"])
            )
        features = np.stack([example.features for example in examples]).astype(
            np.float64, copy=False
        )
        component_labels = np.asarray(
            [example.label for example in examples], dtype=np.uint8
        )
        if features.shape[1] != len(FEATURE_NAMES):
            raise RuntimeError("Cross-fit feature width differs from frozen schema.")
        baseline_counts = sufficient_counts_for_video(
            baseline_scores.numpy(),
            record["labels"],
            record["target_ids"],
            locations,
        )
        block = _block_for_name(metadata["source_name"], middle_names)
        keep_runtime_arrays = block != "middle"
        videos.append(
            PreparedVideo(
                source_name=metadata["source_name"],
                block=block,
                event_count=event_count,
                features=features,
                component_labels=component_labels,
                event_indices=tuple(
                    np.asarray(example.event_indices, dtype=np.int64)
                    for example in examples
                ),
                p0_scores=(
                    p0_scores.numpy().astype(np.float32, copy=True)
                    if keep_runtime_arrays
                    else None
                ),
                locations=(locations if keep_runtime_arrays else None),
                event_labels=(
                    record["labels"].astype(np.uint8, copy=True)
                    if keep_runtime_arrays
                    else None
                ),
                target_ids=(
                    record["target_ids"].copy() if keep_runtime_arrays else None
                ),
                baseline_counts=baseline_counts,
            )
        )
        print(
            "prepare {}/{}: {} [{}] -> {} candidates".format(
                index,
                len(manifest["records"]),
                metadata["source_name"],
                block,
                len(examples),
            ),
            flush=True,
        )
    return videos


def balanced_component_dataset(videos, high_blocks):
    """Build features and exact domain/video/component base weights."""
    high_blocks = set(high_blocks)
    high = [video for video in videos if video.block in high_blocks]
    middle = [video for video in videos if video.block == "middle"]
    if not high or not middle:
        raise ValueError("Balanced fitting requires non-empty high and middle domains.")
    feature_batches = []
    label_batches = []
    weight_batches = []
    source_batches = []
    for domain_videos in (high, middle):
        video_mass = 0.5 / len(domain_videos)
        for video in domain_videos:
            component_count = int(video.features.shape[0])
            if component_count <= 0:
                raise ValueError("Every weighted fit video needs a component.")
            feature_batches.append(video.features)
            label_batches.append(video.component_labels)
            weight_batches.append(
                np.full(component_count, video_mass / component_count, dtype=np.float64)
            )
            source_batches.extend([video.source_name] * component_count)
    features = np.concatenate(feature_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    base_weights = np.concatenate(weight_batches, axis=0)
    if not math.isclose(float(base_weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Balanced component base weights do not sum to one.")
    return features, labels, base_weights, tuple(source_batches)


def _weighted_logistic_loss(design, labels, weights, parameters, l2):
    logits = design @ parameters
    losses = np.logaddexp(0.0, logits) - labels * logits
    return float(
        np.dot(weights, losses) / weights.sum()
        + 0.5 * l2 * np.dot(parameters[:-1], parameters[:-1])
    )


def fit_balanced_logistic(
    features,
    labels,
    base_weights,
    positive_weight,
    l2=L2_PENALTY,
    max_iterations=MAX_ITERATIONS,
):
    """Fit a deterministic logistic model using frozen hierarchical weights."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    base_weights = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape != (labels.size, len(FEATURE_NAMES)):
        raise ValueError("Balanced fit features/labels have incompatible shapes.")
    if base_weights.size != labels.size:
        raise ValueError("Balanced fit base_weights length differs.")
    if not np.isfinite(features).all() or not np.isfinite(base_weights).all():
        raise ValueError("Balanced fit inputs must be finite.")
    if (base_weights <= 0).any():
        raise ValueError("Balanced fit base weights must be positive.")
    if not np.isin(labels, (0.0, 1.0)).all() or np.unique(labels).size != 2:
        raise ValueError("Balanced fit labels must contain both binary classes.")
    if positive_weight not in POSITIVE_WEIGHTS:
        raise ValueError("positive_weight is outside the frozen candidate set.")
    if not math.isclose(float(l2), L2_PENALTY, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Cross-fit L2 is frozen at 0.1.")
    if int(max_iterations) != MAX_ITERATIONS:
        raise ValueError("Cross-fit max_iterations is frozen at 50.")

    normalized_base = base_weights / base_weights.sum()
    feature_mean = np.sum(features * normalized_base[:, None], axis=0)
    centered = features - feature_mean
    feature_scale = np.sqrt(np.sum(centered * centered * normalized_base[:, None], axis=0))
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = centered / feature_scale
    design = np.column_stack((standardized, np.ones(labels.size, dtype=np.float64)))
    sample_weights = base_weights * np.where(labels > 0.5, positive_weight, 1.0)
    weighted_positives = float(np.dot(sample_weights, labels))
    weighted_negatives = float(np.dot(sample_weights, 1.0 - labels))
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    parameters[-1] = math.log(
        max(weighted_positives, 1e-12) / max(weighted_negatives, 1e-12)
    )
    converged = False
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        logits = design @ parameters
        probabilities = np.empty_like(logits)
        nonnegative = logits >= 0
        probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
        exp_negative = np.exp(logits[~nonnegative])
        probabilities[~nonnegative] = exp_negative / (1.0 + exp_negative)
        normalization = sample_weights.sum()
        gradient = design.T @ (sample_weights * (probabilities - labels)) / normalization
        gradient[:-1] += l2 * parameters[:-1]
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design / normalization
        hessian[:-1, :-1] += np.eye(features.shape[1]) * l2
        hessian[-1, -1] += 1e-12
        try:
            newton_step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            newton_step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if np.max(np.abs(newton_step)) < 1e-9:
            converged = True
            break
        current_loss = _weighted_logistic_loss(
            design, labels, sample_weights, parameters, l2
        )
        step_scale = 1.0
        accepted = False
        while step_scale >= 2.0 ** -20:
            candidate_parameters = parameters - step_scale * newton_step
            candidate_loss = _weighted_logistic_loss(
                design, labels, sample_weights, candidate_parameters, l2
            )
            if candidate_loss <= current_loss:
                parameters = candidate_parameters
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        if abs(current_loss - candidate_loss) < 1e-12:
            converged = True
            break
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "coefficients": parameters[:-1],
        "intercept": float(parameters[-1]),
        "iterations": iterations,
        "converged": converged,
        "weighted_loss": _weighted_logistic_loss(
            design, labels, sample_weights, parameters, l2
        ),
        "positive_weight": float(positive_weight),
    }


def _predict_probabilities(features, fitted):
    standardized = (
        np.asarray(features, dtype=np.float64) - fitted["feature_mean"]
    ) / fitted["feature_scale"]
    logits = standardized @ fitted["coefficients"] + fitted["intercept"]
    probabilities = np.empty_like(logits)
    nonnegative = logits >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    negative_exp = np.exp(logits[~nonnegative])
    probabilities[~nonnegative] = negative_exp / (1.0 + negative_exp)
    return probabilities


def _json_model(fitted):
    return {
        "feature_mean": fitted["feature_mean"].tolist(),
        "feature_scale": fitted["feature_scale"].tolist(),
        "coefficients": fitted["coefficients"].tolist(),
        "intercept": fitted["intercept"],
        "iterations": fitted["iterations"],
        "converged": fitted["converged"],
        "weighted_loss": fitted["weighted_loss"],
        "positive_weight": fitted["positive_weight"],
    }


def _candidate_counts(video, fitted, keep_probability, cfg):
    if video.p0_scores is None or video.locations is None:
        raise ValueError("Held-video runtime arrays are unavailable.")
    probabilities = _predict_probabilities(video.features, fitted)
    keep = probabilities >= float(keep_probability)
    scores = video.p0_scores.copy()
    for event_indices, keep_component in zip(video.event_indices, keep):
        if not keep_component:
            scores[event_indices] = 0.0
    # Preserve deployment order even though >100k makes P18 ineligible.
    recovery = P18ScoreTrackRecovery.from_cfg(cfg, PREDICTION_THRESHOLD)
    score_tensor, _ = recovery.apply(
        torch.from_numpy(scores),
        torch.from_numpy(video.locations.astype(np.int64, copy=False)),
    )
    return sufficient_counts_for_video(
        score_tensor.numpy(),
        video.event_labels,
        video.target_ids,
        video.locations,
    )


def _sum_counts(videos, attribute="baseline_counts"):
    total = SufficientCounts()
    for video in videos:
        total = total + getattr(video, attribute)
    return total


def partition_fold_videos(fold, videos):
    """Return disjoint fit/held partitions for one frozen block holdout."""
    if fold not in FOLD_PLAN:
        raise ValueError("Fold definition is outside the frozen fold plan.")
    held = [video for video in videos if video.block == fold["held_block"]]
    fit_videos = [
        video
        for video in videos
        if video.block in {fold["fit_high_block"], "middle"}
    ]
    held_names = {video.source_name for video in held}
    fit_names = {video.source_name for video in fit_videos}
    if held_names & fit_names:
        raise RuntimeError("Cross-fit source leakage between fit and held partitions.")
    if not held or not fit_videos:
        raise RuntimeError("Frozen cross-fit produced an empty partition.")
    return fit_videos, held


def select_fold_winner(candidate_results):
    if not candidate_results:
        raise ValueError("Cannot select a winner from no candidates.")
    return sorted(
        candidate_results,
        key=lambda result: (-float(result["metrics"]["score"]), result["candidate_id"]),
    )[0]


def _evaluate_fold(
    fold,
    videos,
    cfg,
    candidate_profile=CONSERVATIVE_V1_PROFILE,
):
    fit_videos, held = partition_fold_videos(fold, videos)
    features, labels, base_weights, fit_sources = balanced_component_dataset(
        fit_videos, {fold["fit_high_block"]}
    )
    held_baseline_counts = _sum_counts(held)
    held_baseline_metrics = metrics_from_counts(held_baseline_counts)
    candidates = candidate_definitions(candidate_profile)
    fitted_by_weight = {
        weight: fit_balanced_logistic(features, labels, base_weights, weight)
        for weight in sorted({candidate["positive_weight"] for candidate in candidates})
    }
    candidate_results = []
    for candidate in candidates:
        fitted = fitted_by_weight[candidate["positive_weight"]]
        counts = SufficientCounts()
        for video in held:
            counts = counts + _candidate_counts(
                video, fitted, candidate["keep_probability"], cfg
            )
        metrics = metrics_from_counts(counts)
        model_json = _json_model(fitted)
        candidate_results.append(
            {
                **candidate,
                "fit_model_sha256": sha256_json(model_json),
                "fit_model": model_json,
                "counts": counts.to_dict(),
                "metrics": metrics,
                "delta": {
                    name: metrics[name] - held_baseline_metrics[name]
                    for name in metrics
                },
                "false_component_delta": (
                    counts.false_components - held_baseline_counts.false_components
                ),
            }
        )
    winner = select_fold_winner(candidate_results)
    return {
        "fold_id": fold["fold_id"],
        "fit_high_block": fold["fit_high_block"],
        "held_block": fold["held_block"],
        "fit_video_names": sorted(set(fit_sources)),
        "held_video_names": [video.source_name for video in held],
        "fit_component_count": int(features.shape[0]),
        "baseline": {
            "counts": held_baseline_counts.to_dict(),
            "metrics": held_baseline_metrics,
        },
        "candidate_results": candidate_results,
        "winner_candidate_id": winner["candidate_id"],
    }


def _result_by_id(fold_result, candidate_id):
    for result in fold_result["candidate_results"]:
        if result["candidate_id"] == candidate_id:
            return result
    raise KeyError(candidate_id)


def evaluate_promotion_gates(fold_results, middle_baseline_counts):
    if len(fold_results) != 2:
        raise ValueError("Promotion gates require exactly two fold results.")
    by_id = {fold["fold_id"]: fold for fold in fold_results}
    if set(by_id) != {"holdout_h1", "holdout_h2"}:
        raise ValueError("Promotion gates require holdout_h1 and holdout_h2.")
    winners = {
        fold_id: _result_by_id(fold, fold["winner_candidate_id"])
        for fold_id, fold in by_id.items()
    }
    baseline_high_counts = SufficientCounts()
    selected_high_counts = SufficientCounts()
    for fold in fold_results:
        baseline_high_counts = baseline_high_counts + SufficientCounts(
            **fold["baseline"]["counts"]
        )
        selected = winners[fold["fold_id"]]
        selected_high_counts = selected_high_counts + SufficientCounts(
            **selected["counts"]
        )
    baseline_pooled_counts = baseline_high_counts + middle_baseline_counts
    selected_pooled_counts = selected_high_counts + middle_baseline_counts
    baseline_pooled_metrics = metrics_from_counts(baseline_pooled_counts)
    selected_pooled_metrics = metrics_from_counts(selected_pooled_counts)
    baseline_false = baseline_high_counts.false_components
    selected_false = selected_high_counts.false_components
    combined_false_reduction = (
        (baseline_false - selected_false) / baseline_false
        if baseline_false > 0
        else 0.0
    )
    checks = {
        "h1_score_delta_positive": (
            winners["holdout_h1"]["delta"]["score"] > 0.0
        ),
        "h2_score_delta_positive": (
            winners["holdout_h2"]["delta"]["score"] > 0.0
        ),
        "pooled_pd_nondecrease": (
            selected_pooled_metrics["pd"] >= baseline_pooled_metrics["pd"]
        ),
        "pooled_iou_nondecrease": (
            selected_pooled_metrics["iou"] >= baseline_pooled_metrics["iou"]
        ),
        "h1_false_components_nonincrease": (
            winners["holdout_h1"]["false_component_delta"] <= 0
        ),
        "h2_false_components_nonincrease": (
            winners["holdout_h2"]["false_component_delta"] <= 0
        ),
        "combined_false_component_reduction_at_least_1pct": (
            combined_false_reduction >= FALSE_COMPONENT_REDUCTION_GATE
        ),
        "pooled_score_delta_at_least_0p0002": (
            selected_pooled_metrics["score"] - baseline_pooled_metrics["score"]
            >= POOLED_SCORE_DELTA_GATE
        ),
        "fold_winner_candidate_ids_match": (
            by_id["holdout_h1"]["winner_candidate_id"]
            == by_id["holdout_h2"]["winner_candidate_id"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fold_winner_candidate_ids": {
            fold_id: fold["winner_candidate_id"] for fold_id, fold in by_id.items()
        },
        "baseline_pooled": {
            "counts": baseline_pooled_counts.to_dict(),
            "metrics": baseline_pooled_metrics,
        },
        "selected_pooled_oof": {
            "counts": selected_pooled_counts.to_dict(),
            "metrics": selected_pooled_metrics,
            "delta": {
                name: selected_pooled_metrics[name] - baseline_pooled_metrics[name]
                for name in selected_pooled_metrics
            },
        },
        "held_high_false_components": {
            "baseline": baseline_false,
            "selected": selected_false,
            "reduction_fraction": combined_false_reduction,
        },
    }


def _artifact_for_full_refit(
    fitted,
    candidate,
    protocol_payload,
    protocol_file_sha256,
    manifest,
    manifest_sha256,
    cfg,
    config_path,
    gates,
    fold_results,
    videos,
):
    protocol_definition = protocol_payload["definition"]
    candidate_profile = protocol_definition.get(
        "candidate_profile", CONSERVATIVE_V1_PROFILE
    )
    hypothesis = protocol_definition.get("hypothesis")
    if candidate_profile == POSTHOC_V2_PROFILE:
        hyperparameter_selection = (
            "retrospective_train_only_singleton_after_v1_noop_no_further_selection"
        )
    else:
        hyperparameter_selection = "frozen_train_only_two_block_crossfit_common_winner"
    p0_config = P0ClusterFilter.from_cfg(
        cfg,
        PREDICTION_THRESHOLD,
        event_count=DEPLOYMENT_EVENT_COUNT_CUTOFF + 1,
    ).config
    p0_mapping = input_postprocess_mapping(p0_config)
    inference_settings = manifest["inference_settings"]
    oof_evidence = {
        "fold_results": fold_results,
        "gates": gates,
        "protocol_definition_sha256": protocol_payload["definition_sha256"],
    }
    project_root = Path(__file__).resolve().parent
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "created_utc": _utc_now(),
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": fitted["feature_mean"].tolist(),
        "feature_scale": fitted["feature_scale"].tolist(),
        "coefficients": fitted["coefficients"].tolist(),
        "intercept": fitted["intercept"],
        "keep_probability": candidate["keep_probability"],
        "prediction_threshold": PREDICTION_THRESHOLD,
        "topology": TOPOLOGY.to_dict(),
        "fit": {
            "algorithm": "deterministic_domain_video_component_weighted_logistic_newton",
            "positive_weight": candidate["positive_weight"],
            "l2": L2_PENALTY,
            "max_iterations": MAX_ITERATIONS,
            "iterations": fitted["iterations"],
            "converged": fitted["converged"],
            "weighted_loss": fitted["weighted_loss"],
            "hyperparameter_selection": hyperparameter_selection,
        },
        "provenance": {
            "dataset_split": "train",
            "training_selection": manifest["selection"],
            "deployment_event_count_cutoff": DEPLOYMENT_EVENT_COUNT_CUTOFF,
            "input_postprocess": p0_mapping,
            "input_postprocess_sha256": sha256_json(p0_mapping),
            "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
            "inference_settings": inference_settings,
            "inference_settings_sha256": sha256_json(inference_settings),
            "train_cache_manifest_sha256": manifest_sha256,
            "train_cache_schema": manifest["schema"],
            "official_train_source_manifest_scheme": manifest[
                "official_train_source_manifest_scheme"
            ],
            "official_train_source_manifest_sha256": manifest[
                "official_train_source_manifest_sha256"
            ],
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "config_overrides": protocol_definition["config"]["overrides"],
            "fit_script_sha256": sha256_file(Path(__file__).resolve()),
            "component_module_sha256": sha256_file(
                project_root / "utils" / "component_reranker.py"
            ),
            "crossfit_protocol_schema": PROTOCOL_SCHEMA,
            "crossfit_protocol_file_sha256": protocol_file_sha256,
            "crossfit_protocol_definition_sha256": protocol_payload[
                "definition_sha256"
            ],
            "crossfit_protocol_document_sha256": protocol_definition[
                "frozen_document"
            ]["sha256"],
            "crossfit_code_sha256": protocol_definition["code_sha256"],
            "crossfit_software_versions": protocol_definition["software_versions"],
            "crossfit_oof_evidence_sha256": sha256_json(oof_evidence),
            "crossfit_candidate_id": candidate["candidate_id"],
            "crossfit_candidate_profile": candidate_profile,
            "crossfit_evidence_class": (
                "retrospective_train_only_not_independent_oof"
                if candidate_profile == POSTHOC_V2_PROFILE
                else "train_only_cross_source_held_block_consistency_not_unbiased_oof"
            ),
            "crossfit_hypothesis": hypothesis,
        },
        "train_diagnostics_in_sample_only": {
            "video_count": len(videos),
            "component_count": int(sum(video.features.shape[0] for video in videos)),
            "positive_components": int(
                sum(video.component_labels.sum() for video in videos)
            ),
            "negative_components": int(
                sum((video.component_labels == 0).sum() for video in videos)
            ),
            "note": "Full-54 refit diagnostics are not validation or leaderboard evidence.",
        },
    }
    return artifact


def _json_file_bytes(payload):
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    return text.encode("utf-8")


def _stage_bytes(destination, payload):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".pending",
        dir=str(destination.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return temporary_path


def publish_crossfit_outputs(
    output_report,
    output_model,
    artifact,
    report_factory,
):
    """Stage both outputs before publishing; never leave an orphan artifact."""
    output_report = Path(output_report).resolve()
    output_model = Path(output_model).resolve()
    if output_report.exists() or output_model.exists():
        raise FileExistsError("Cross-fit output path already exists.")
    if os.path.normcase(str(output_report)) == os.path.normcase(str(output_model)):
        raise ValueError("Cross-fit report and model artifact paths must differ.")

    artifact_bytes = None
    artifact_sha256 = None
    if artifact is not None:
        if not isinstance(artifact, dict) or artifact.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError("Promoted cross-fit artifact has an invalid schema.")
        artifact_bytes = _json_file_bytes(artifact)
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()

    # The complete report, including any potentially failing metric/JSON work,
    # is constructed before either destination becomes visible.
    report = report_factory(artifact_sha256)
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise ValueError("Cross-fit report has an invalid schema.")
    expected_artifact = report.get("artifact")
    if not isinstance(expected_artifact, dict):
        raise ValueError("Cross-fit report is missing artifact audit metadata.")
    if expected_artifact.get("sha256") != artifact_sha256:
        raise ValueError("Cross-fit report artifact SHA-256 differs from staged bytes.")
    if bool(expected_artifact.get("emitted")) != (artifact is not None):
        raise ValueError("Cross-fit report artifact-emitted flag is inconsistent.")
    report_bytes = _json_file_bytes(report)

    staged_artifact = None
    staged_report = None
    artifact_published = False
    try:
        if artifact_bytes is not None:
            staged_artifact = _stage_bytes(output_model, artifact_bytes)
        staged_report = _stage_bytes(output_report, report_bytes)
        if staged_artifact is not None:
            os.replace(str(staged_artifact), str(output_model))
            artifact_published = True
            staged_artifact = None
        os.replace(str(staged_report), str(output_report))
        staged_report = None
    except BaseException:
        if artifact_published and output_model.exists():
            output_model.unlink()
        raise
    finally:
        for staged in (staged_artifact, staged_report):
            if staged is not None and staged.exists():
                staged.unlink()
    return artifact_sha256


def run_crossfit(args):
    output_report = _require_new_output(args.output_report, "Cross-fit report")
    output_model = _require_new_output(args.output_model, "Cross-fit model artifact")
    if os.path.normcase(str(output_report)) == os.path.normcase(str(output_model)):
        raise ValueError("Cross-fit report and model artifact paths must differ.")
    protocol_payload, protocol_file_sha256 = load_frozen_protocol(
        args.protocol, args.expected_protocol_sha256
    )
    definition = protocol_payload["definition"]
    candidate_profile = definition.get(
        "candidate_profile", CONSERVATIVE_V1_PROFILE
    )
    expected_candidates = candidate_definitions(candidate_profile)
    if definition.get("candidates") != expected_candidates:
        raise ValueError("Frozen candidate profile and candidate definitions differ.")
    hypothesis_source_report = None
    expected_hypothesis_source_report_sha256 = None
    hypothesis_source_diagnostic = None
    expected_hypothesis_source_diagnostic_sha256 = None
    if candidate_profile == POSTHOC_V2_PROFILE:
        hypothesis = definition.get("hypothesis")
        if not isinstance(hypothesis, dict) or hypothesis.get(
            "hypothesis_origin"
        ) != POSTHOC_HYPOTHESIS_ORIGIN:
            raise ValueError("Frozen post-hoc hypothesis provenance is invalid.")
        source_v1_report = hypothesis.get("source_v1_report")
        if not isinstance(source_v1_report, dict):
            raise ValueError("Frozen post-hoc hypothesis is missing its v1 source report.")
        hypothesis_source_report = source_v1_report.get("path")
        expected_hypothesis_source_report_sha256 = source_v1_report.get(
            "expected_sha256"
        )
        source_diagnostic = hypothesis.get("source_train_diagnostic")
        if not isinstance(source_diagnostic, dict):
            raise ValueError(
                "Frozen post-hoc hypothesis is missing its train-only diagnostic."
            )
        hypothesis_source_diagnostic = source_diagnostic.get("path")
        expected_hypothesis_source_diagnostic_sha256 = source_diagnostic.get(
            "expected_sha256"
        )
    config_definition = definition.get("config", {})
    config_path = Path(str(config_definition.get("path", ""))).resolve()
    overrides = config_definition.get("overrides")
    if not isinstance(overrides, list) or not all(isinstance(item, str) for item in overrides):
        raise ValueError("Frozen config overrides must be a list of strings.")
    rebuilt_definition = _build_protocol_definition(
        args.cache_dir,
        config_path,
        overrides,
        prediction_threshold=PREDICTION_THRESHOLD,
        candidate_profile=candidate_profile,
        hypothesis_source_report=hypothesis_source_report,
        expected_hypothesis_source_report_sha256=(
            expected_hypothesis_source_report_sha256
        ),
        hypothesis_source_diagnostic=hypothesis_source_diagnostic,
        expected_hypothesis_source_diagnostic_sha256=(
            expected_hypothesis_source_diagnostic_sha256
        ),
    )
    if rebuilt_definition != definition:
        raise ValueError(
            "Current cache/config/code/document state differs from the frozen protocol."
        )

    cache_dir, manifest_path, manifest_sha256, manifest = load_train_cache(
        args.cache_dir
    )
    _, middle_names = _validate_cache_population(manifest, cache_dir)
    cfg = replay.load_flat_config(config_path, overrides)
    validate_c00_config(cfg)
    videos = _prepare_videos(cache_dir, manifest, cfg, set(middle_names))
    fold_results = [
        _evaluate_fold(fold, videos, cfg, candidate_profile)
        for fold in FOLD_PLAN
    ]
    middle_baseline_counts = _sum_counts(
        [video for video in videos if video.block == "middle"]
    )
    gates = evaluate_promotion_gates(fold_results, middle_baseline_counts)
    if (
        _build_protocol_definition(
            args.cache_dir,
            config_path,
            overrides,
            prediction_threshold=PREDICTION_THRESHOLD,
            candidate_profile=candidate_profile,
            hypothesis_source_report=hypothesis_source_report,
            expected_hypothesis_source_report_sha256=(
                expected_hypothesis_source_report_sha256
            ),
            hypothesis_source_diagnostic=hypothesis_source_diagnostic,
            expected_hypothesis_source_diagnostic_sha256=(
                expected_hypothesis_source_diagnostic_sha256
            ),
        )
        != definition
    ):
        raise RuntimeError(
            "Cache/config/code/document state changed during the cross-fit run."
        )
    artifact = None
    if gates["passed"]:
        common_candidate_id = fold_results[0]["winner_candidate_id"]
        candidate = next(
            item
            for item in candidate_definitions(candidate_profile)
            if item["candidate_id"] == common_candidate_id
        )
        features, labels, base_weights, _ = balanced_component_dataset(
            videos, {"h1", "h2"}
        )
        fitted = fit_balanced_logistic(
            features,
            labels,
            base_weights,
            candidate["positive_weight"],
        )
        artifact = _artifact_for_full_refit(
            fitted,
            candidate,
            protocol_payload,
            protocol_file_sha256,
            manifest,
            manifest_sha256,
            cfg,
            config_path,
            gates,
            fold_results,
            videos,
        )

    def build_report(artifact_sha256):
        if candidate_profile == POSTHOC_V2_PROFILE:
            evidence_class = (
                "retrospective_train_only_cross_source_consistency_not_independent_oof"
            )
            note = (
                "The singleton pw4/kp0.40 hypothesis was chosen retrospectively after "
                "the bound conservative_v1 train-only no-op report. The same held train "
                "blocks are reused here only as a consistency gate, so this is not an "
                "independent or unbiased OOF estimate. No validation or leaderboard data "
                "was used."
            )
        else:
            evidence_class = (
                "train_only_cross_source_held_block_consistency_not_unbiased_oof"
            )
            note = (
                "Train-only cross-source held-block consistency evidence; the held "
                "blocks also select candidates, so this is not claimed as an unbiased "
                "independent OOF estimate. No validation or leaderboard data was used."
            )
        return {
            "schema": REPORT_SCHEMA,
            "created_utc": _utc_now(),
            "dataset_split": "train",
            "evidence_class": evidence_class,
            "candidate_profile": candidate_profile,
            "hypothesis": definition.get("hypothesis"),
            "protocol_path": str(Path(args.protocol).resolve()),
            "protocol_file_sha256": protocol_file_sha256,
            "protocol_definition_sha256": protocol_payload["definition_sha256"],
            "cache_manifest_path": str(manifest_path),
            "cache_manifest_sha256": manifest_sha256,
            "fold_results": fold_results,
            "middle_identity": {
                "video_names": list(middle_names),
                "counts": middle_baseline_counts.to_dict(),
                "metrics": metrics_from_counts(middle_baseline_counts),
                "postprocessor": "full_frozen_c00_including_p18",
            },
            "promotion_gates": gates,
            "artifact": {
                "emitted": artifact is not None,
                "path": str(output_model) if artifact is not None else None,
                "sha256": artifact_sha256,
            },
            "note": note,
        }

    artifact_sha256 = publish_crossfit_outputs(
        output_report,
        output_model,
        artifact,
        build_report,
    )
    print("wrote train-only cross-fit report:", output_report)
    print("promotion gates passed:", gates["passed"])
    if gates["passed"]:
        print("wrote promoted component reranker:", output_model)
        print("artifact sha256:", artifact_sha256)
    else:
        print("no artifact emitted because at least one frozen gate failed")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preregister/run the train-only component reranker cross-fit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preregister = subparsers.add_parser(
        "preregister", help="Freeze cache, config, candidates, folds, and gates."
    )
    preregister.add_argument("--cache-dir", required=True)
    preregister.add_argument("--config", required=True)
    preregister.add_argument("--override", action="append", default=[])
    preregister.add_argument("--output-protocol", required=True)
    preregister.add_argument("--prediction-threshold", type=float, default=0.719)
    preregister.add_argument("--expected-selected-videos", type=int, required=True)
    preregister.add_argument(
        "--candidate-profile",
        choices=CANDIDATE_PROFILES,
        default=CONSERVATIVE_V1_PROFILE,
    )
    preregister.add_argument("--hypothesis-source-report")
    preregister.add_argument("--expected-hypothesis-source-report-sha256")
    preregister.add_argument("--hypothesis-source-diagnostic")
    preregister.add_argument("--expected-hypothesis-source-diagnostic-sha256")
    preregister.set_defaults(handler=preregister_protocol)

    run = subparsers.add_parser(
        "run", help="Run the frozen train-only cross-fit and promotion gates."
    )
    run.add_argument("--protocol", required=True)
    run.add_argument("--expected-protocol-sha256", required=True)
    run.add_argument("--cache-dir", required=True)
    run.add_argument("--output-report", required=True)
    run.add_argument("--output-model", required=True)
    run.set_defaults(handler=run_crossfit)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
