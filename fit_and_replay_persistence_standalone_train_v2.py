"""Fit/replay the corrected H2-only persistence_pw08_kp050 artifact.

The final estimator uses exactly train_088..098 with equal video mass 1/11,
matching the fit population used by the H2 OOF folds.  train_044..047 are used
only to prove runtime identity and never enter feature extraction or fitting.
Only immutable train inputs are accepted; validation, test, and T32 are absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

import crossfit_component_reranker as component_crossfit
import crossfit_persistent_pixel_prior as persistence_crossfit
import fit_and_replay_persistence_standalone_train as common
import replay_temporal_memory_validation as replay
from train_component_reranker import load_train_cache
from utils.component_reranker import (
    sha256_file,
    sha256_json,
    temporal_memory_inference_mapping,
)
from utils.persistence_component_suppressor import (
    ARTIFACT_SCHEMA,
    DEFAULT_TOPOLOGY,
    FEATURE_NAMES,
    FEATURE_SEMANTICS_VERSION,
    PREDICTION_THRESHOLD,
    PersistenceArtifact,
    PersistenceComponentSuppressor,
    derive_pixel_prior_from_arrays,
    extract_persistence_components,
)
from utils.postprocess import P18ScoreTrackRecovery


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
PROTOCOL_PATH = (
    PROJECT_ROOT / "protocols" / "persistence_standalone_train_fit_v2.json"
).resolve()
EXPECTED_PROTOCOL_SHA256 = "d7e064a5b941453e0a940d3401c6734b3fc25fe2461e8758a392f78810b69878"
EXPECTED_OOF_REPORT_SHA256 = common.EXPECTED_OOF_REPORT_SHA256
EXPECTED_CACHE_MANIFEST_SHA256 = common.EXPECTED_CACHE_MANIFEST_SHA256
EXPECTED_REFERENCE_PROTOCOL_SHA256 = common.EXPECTED_REFERENCE_PROTOCOL_SHA256
EXPECTED_CONFIG_SHA256 = common.EXPECTED_CONFIG_SHA256
EXPECTED_M20_SHA256 = common.EXPECTED_M20_SHA256
H1_NAMES = persistence_crossfit.H1_NAMES
H2_NAMES = persistence_crossfit.H2_NAMES
HIGH_NAMES = H1_NAMES + H2_NAMES
REPORT_SCHEMA = "ev-uav-persistence-standalone-h2only-train-replay-v1"
EXPECTED_EFFECTIVE_C00_SHA256 = (
    "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413"
)


def validate_protocol_and_winner():
    protocol = common._load_json_snapshot(
        PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256, "H2-only train-fit protocol"
    )
    if (
        protocol.get("schema")
        != "ev-uav-persistence-standalone-train-fit-protocol-v2"
        or protocol.get("status") != "frozen_before_h2_only_final_fit"
        or protocol.get("candidate_id") != "persistence_pw08_kp050"
        or protocol.get("split_access", {}).get("gpu_allowed") is not False
        or protocol.get("standalone_runtime", {}).get("t32_allowed") is not False
        or protocol.get("estimator_correction", {}).get(
            "superseded_outputs_must_not_enter_validation"
        )
        is not True
    ):
        raise ValueError("Frozen H2-only train-fit protocol identity differs.")
    population = protocol["population"]
    if (
        tuple(population["fit_h2_sources"]) != H2_NAMES
        or tuple(population["identity_only_h1_sources"]) != H1_NAMES
        or population["fit_source_count"] != 11
        or population["identity_audit_source_count"] != 4
        or float(population["h1_fit_mass"]) != 0.0
    ):
        raise ValueError("Frozen H2-fit/H1-identity population differs.")
    final_fit = protocol["final_fit"]
    if (
        final_fit["family"] != "persistence_only"
        or final_fit["feature_dtype"] != "float64"
        or final_fit["feature_names"] != list(FEATURE_NAMES)
        or final_fit["equal_video_weighting"]
        != "each of 11 H2 fit videos has mass 1/11; each candidate component within a fit video receives mass (1/11)/component_count before class weighting"
        or float(final_fit["positive_weight"]) != 8.0
        or float(final_fit["keep_probability"]) != 0.5
        or float(final_fit["l2_penalty"]) != 0.1
        or int(final_fit["max_newton_iterations"]) != 50
        or final_fit["component_topology"] != DEFAULT_TOPOLOGY.to_dict()
        or float(final_fit["prediction_threshold"]) != PREDICTION_THRESHOLD
    ):
        raise ValueError("Frozen H2-only fit definition differs.")
    feature_semantics = final_fit["feature_semantics"]
    if (
        feature_semantics["complete_temporal_bins"] != 160
        or "unclipped numpy.log1p" not in feature_semantics["log_count"]
        or "divided by 160" not in feature_semantics["active_fraction"]
        or "divided by 160" not in feature_semantics["longest_run_fraction"]
        or "BORDER_REPLICATE" not in feature_semantics["neighbor_active_fraction"]
        or "repeated events" not in feature_semantics["component_aggregation"]
    ):
        raise ValueError("Frozen persistence feature semantics differ.")
    runtime_contract = protocol["standalone_runtime"]
    if (
        runtime_contract["effective_c00_canonical_sha256"]
        != EXPECTED_EFFECTIVE_C00_SHA256
        or runtime_contract["stage_order"]
        != [
            "released raw M20 full-stream T160 probabilities",
            "frozen P0/P0c postprocess",
            "extract candidates with topology spatial_radius=1, temporal_bin_size=50, max_link_distance=6, max_gap_bins=1, max_component_events=3",
            "derive the frozen 14 persistence features in their listed order and compute the logistic keep probability",
            "keep probability >=0.5; zero only rejected candidate component events and never raise or otherwise alter a score",
            "frozen P18 recovery",
        ]
    ):
        raise ValueError("Frozen standalone stage order differs.")
    disclosure = protocol["selection_disclosure"]
    if (
        disclosure["candidate_grid_count"] != 7
        or disclosure["candidate_grid_shared_the_same_five_oof_folds"] is not True
        or disclosure["pooled_oof_delta_is_selection_affected_not_independent"]
        is not True
        or disclosure["independent_held_claim_allowed"] is not False
    ):
        raise ValueError("Frozen selection-bias disclosure differs.")

    binding = protocol["winner_binding"]
    report_path = common._workspace_input(
        binding["report_workspace_relative_path"], "train-only OOF report"
    )
    report = common._load_json_snapshot(
        report_path, EXPECTED_OOF_REPORT_SHA256, "train-only OOF report"
    )
    if (
        binding["report_sha256"] != EXPECTED_OOF_REPORT_SHA256
        or report.get("schema") != binding["report_schema"]
        or report["pooled_oof"]["conservative_winner_candidate_id"]
        != binding["expected_conservative_winner"]
    ):
        raise ValueError("Hash-bound conservative winner differs.")
    winner = next(
        item
        for item in report["pooled_oof"]["candidates"]
        if item["candidate_id"] == "persistence_pw08_kp050"
    )
    required = binding["required_oof_checks"]
    if (
        winner["family"] != "persistence"
        or float(winner["positive_weight"]) != 8.0
        or float(winner["keep_probability"]) != 0.5
        or winner["conservative_gate_passed"] is not True
        or winner["nonnegative_score_fold_count"]
        != required["nonnegative_score_fold_count"]
        or winner["count_delta"]["true_positive_events"]
        != required["true_positive_event_delta"]
        or winner["count_delta"]["correct_objects"]
        != required["correct_object_delta"]
    ):
        raise ValueError("Hash-bound winner fails the frozen conservative checks.")
    return protocol, report_path, winner


def _model_payload(fitted):
    return {
        "feature_mean": fitted["feature_mean"].tolist(),
        "feature_scale": fitted["feature_scale"].tolist(),
        "coefficients": fitted["coefficients"].tolist(),
        "intercept": float(fitted["intercept"]),
        "positive_weight": float(fitted["positive_weight"]),
        "keep_probability": 0.5,
        "l2_penalty": 0.1,
        "max_newton_iterations": 50,
        "iterations": int(fitted["iterations"]),
        "converged": bool(fitted["converged"]),
        "weighted_loss": float(fitted["weighted_loss"]),
    }


def run(output_directory):
    started = time.monotonic()
    protocol, oof_report_path, winner = validate_protocol_and_winner()
    inputs = protocol["inputs"]
    manifest_path = common._workspace_input(
        inputs["full_m20_train_cache_manifest"]["workspace_relative_path"],
        "full M20 train cache manifest",
    )
    reference_protocol_path = common._workspace_input(
        inputs["reference_protocol"]["workspace_relative_path"],
        "reference train protocol",
    )
    config_path = common._workspace_input(
        inputs["config"]["workspace_relative_path"], "config"
    )
    raw_train_root = common._workspace_input(
        inputs["raw_train_root_workspace_relative_path"], "raw train root"
    )
    if raw_train_root.name.lower() != "train":
        raise ValueError("Raw input root is not the official train directory.")
    fixed_hashes = (
        (manifest_path, EXPECTED_CACHE_MANIFEST_SHA256, "full M20 train manifest"),
        (reference_protocol_path, EXPECTED_REFERENCE_PROTOCOL_SHA256, "reference protocol"),
        (config_path, EXPECTED_CONFIG_SHA256, "config"),
    )
    for path, expected, description in fixed_hashes:
        if sha256_file(path) != expected:
            raise ValueError("{} SHA-256 differs.".format(description))
    for relative, expected in inputs["source_code_sha256"].items():
        if sha256_file(PROJECT_ROOT / relative) != expected:
            raise ValueError("Frozen source dependency differs: {}".format(relative))

    _, overrides = persistence_crossfit._read_reference_protocol(
        reference_protocol_path
    )
    cfg = replay.load_flat_config(config_path, overrides)
    component_crossfit.validate_c00_config(cfg)
    effective_c00 = component_crossfit._postprocess_contract(cfg)
    effective_c00_sha = sha256_json(effective_c00)
    if (
        effective_c00_sha != EXPECTED_EFFECTIVE_C00_SHA256
        or inputs["config"]["effective_c00_canonical_sha256"]
        != EXPECTED_EFFECTIVE_C00_SHA256
        or inputs["config"]["config_files_are_byte_different"] is not True
        or inputs["config"]["effective_c00_matches_cache_generation_contract"]
        is not True
    ):
        raise ValueError("Effective C00 canonical contract differs.")
    cache_dir, loaded_manifest_path, manifest_sha, manifest = load_train_cache(
        manifest_path.parent
    )
    if (
        loaded_manifest_path.resolve() != manifest_path
        or manifest_sha != EXPECTED_CACHE_MANIFEST_SHA256
        or manifest.get("base_checkpoint_sha256") != EXPECTED_M20_SHA256
    ):
        raise ValueError("Loaded full M20 train cache identity differs.")
    records_by_name = {item["source_name"]: item for item in manifest["records"]}
    if not set(HIGH_NAMES).issubset(records_by_name):
        raise ValueError("Full M20 train cache lacks the 15 replay sources.")
    selected_manifest = dict(manifest)
    selected_manifest["records"] = [records_by_name[name] for name in HIGH_NAMES]

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized; this fit/replay is CPU-only.")
    prepared = component_crossfit._prepare_videos(
        cache_dir, selected_manifest, cfg, middle_names=set()
    )
    if tuple(item.source_name for item in prepared) != HIGH_NAMES:
        raise RuntimeError("Prepared replay source order differs.")

    h2_fit_videos = []
    source_inputs = {}
    chain_audit = []
    for index, video in enumerate(prepared, start=1):
        raw_path = raw_train_root / video.source_name
        expected_source_sha = protocol["population"]["raw_source_sha256"][
            video.source_name
        ]
        if sha256_file(raw_path) != expected_source_sha:
            raise ValueError("Raw train source SHA-256 differs: {}".format(video.source_name))
        locations, polarities = common._raw_xytp(raw_path)
        if not np.array_equal(locations, video.locations[:, 1:4]):
            raise RuntimeError("Raw/cache locations differ: {}".format(video.source_name))
        source_inputs[video.source_name] = (locations, polarities)
        if video.block == "h1":
            chain_audit.append(
                {
                    "source_name": video.source_name,
                    "domain": "h1",
                    "fit_mass": 0.0,
                    "offline_component_chain_called": False,
                    "role": "runtime_identity_only",
                }
            )
            print("identity source {}/15: {} [h1]".format(index, video.source_name), flush=True)
            continue

        reference_prior = persistence_crossfit.derive_pixel_prior(
            raw_path, video.locations[:, 1:4]
        )
        runtime_prior = derive_pixel_prior_from_arrays(locations, polarities)
        if not common._prior_fields_equal(reference_prior, runtime_prior):
            raise RuntimeError("Standalone prior differs: {}".format(video.source_name))
        reference_features = persistence_crossfit.component_persistence_features(
            reference_prior, video.event_indices
        )
        runtime_indices, runtime_features, runtime_summary = extract_persistence_components(
            video.p0_scores,
            locations,
            polarities,
            topology=DEFAULT_TOPOLOGY,
            prediction_threshold=PREDICTION_THRESHOLD,
        )
        indices_exact = len(runtime_indices) == len(video.event_indices) and all(
            np.array_equal(left, right)
            for left, right in zip(runtime_indices, video.event_indices)
        )
        features_exact = np.array_equal(runtime_features, reference_features)
        if not indices_exact or not features_exact or runtime_summary != reference_prior.summary:
            raise RuntimeError("H2 component/feature chain differs: {}".format(video.source_name))
        h2_fit_videos.append(
            persistence_crossfit.PriorVideo(video, runtime_features, runtime_summary)
        )
        chain_audit.append(
            {
                "source_name": video.source_name,
                "domain": "h2",
                "fit_mass": 1.0 / len(H2_NAMES),
                "offline_component_chain_called": True,
                "role": "h2_fit_and_runtime_replay",
                "candidate_component_count": len(runtime_indices),
                "positive_candidate_component_count": int(
                    np.count_nonzero(video.component_labels)
                ),
                "component_indices_sha256": common._component_indices_sha256(
                    runtime_indices
                ),
                "features_sha256": common._array_sha256(runtime_features),
                "prior_summary_sha256": sha256_json(runtime_summary),
                "component_indices_exact": True,
                "features_exact": True,
                "prior_exact": True,
            }
        )
        print(
            "fit chain {}/15: {} [h2] {} components".format(
                index, video.source_name, len(runtime_indices)
            ),
            flush=True,
        )

    if tuple(item.source_name for item in h2_fit_videos) != H2_NAMES:
        raise RuntimeError("Final fit population is not exactly the 11 H2 sources.")
    features, labels, base_weights = persistence_crossfit._balanced_dataset(
        h2_fit_videos, "persistence"
    )
    expected_video_mass = 1.0 / len(H2_NAMES)
    offset = 0
    for video in h2_fit_videos:
        count = video.persistence_features.shape[0]
        expected_component_mass = expected_video_mass / count
        if not np.all(base_weights[offset : offset + count] == expected_component_mass):
            raise RuntimeError("Equal-H2-video base weights differ.")
        offset += count
    fitted = persistence_crossfit._fit_balanced_logistic(
        features,
        labels,
        base_weights,
        positive_weight=8.0,
        l2=0.1,
        max_iterations=50,
    )

    code_paths = (
        "fit_and_replay_persistence_standalone_train_v2.py",
        "fit_and_replay_persistence_standalone_train.py",
        "utils/persistence_component_suppressor.py",
        "protocols/persistence_standalone_train_fit_v2.json",
        "crossfit_persistent_pixel_prior.py",
        "crossfit_component_reranker.py",
        "utils/component_reranker.py",
        "utils/postprocess.py",
        "utils/eval.py",
        "utils/challenge_eval.py",
        "utils/temporal_memory_input_router.py",
    )
    artifact_payload = {
        "schema": ARTIFACT_SCHEMA,
        "created_utc": common.utc_now(),
        "candidate_id": "persistence_pw08_kp050",
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "component_topology": DEFAULT_TOPOLOGY.to_dict(),
        "model": _model_payload(fitted),
        "runtime_contract": {
            "stage_order": protocol["standalone_runtime"]["stage_order"],
            "prediction_threshold": PREDICTION_THRESHOLD,
            "event_count_cutoff_exclusive": 200000,
            "polarity_minority_cutoff": 0.2,
            "eligible_route": "h2",
            "non_h2_identity": True,
            "component_calls_only_h2": True,
            "t32_allowed": False,
            "effective_c00_canonical_sha256": effective_c00_sha,
        },
        "training_provenance": {
            "evidence_class": "train_only_h2_final_fit_after_selection_affected_five_fold_oof",
            "selection_disclosure": protocol["selection_disclosure"],
            "estimator_correction": protocol["estimator_correction"],
            "train_fit_protocol": {
                "path": str(PROTOCOL_PATH),
                "sha256": EXPECTED_PROTOCOL_SHA256,
            },
            "winner_report": {
                "path": str(oof_report_path),
                "sha256": EXPECTED_OOF_REPORT_SHA256,
                "winner": winner["candidate_id"],
            },
            "full_m20_train_cache_manifest": {
                "path": str(manifest_path),
                "sha256": EXPECTED_CACHE_MANIFEST_SHA256,
            },
            "released_m20_checkpoint_sha256": EXPECTED_M20_SHA256,
            "fit_source_names": list(H2_NAMES),
            "identity_only_source_names": list(H1_NAMES),
            "fit_source_sha256": {
                name: protocol["population"]["raw_source_sha256"][name]
                for name in H2_NAMES
            },
            "equal_h2_video_weighting": True,
            "fit_video_mass": 1.0 / len(H2_NAMES),
            "h1_fit_mass": 0.0,
            "candidate_component_count": int(features.shape[0]),
            "positive_candidate_component_count": int(np.count_nonzero(labels)),
            "negative_candidate_component_count": int(
                labels.size - np.count_nonzero(labels)
            ),
            "effective_c00_postprocess": effective_c00,
            "effective_c00_canonical_sha256": effective_c00_sha,
            "effective_temporal_memory_inference": temporal_memory_inference_mapping(cfg),
            "code_sha256": {
                relative: sha256_file(PROJECT_ROOT / relative) for relative in code_paths
            },
        },
    }
    artifact_model = PersistenceArtifact.from_payload(artifact_payload)
    if not (
        np.array_equal(artifact_model.feature_mean, fitted["feature_mean"])
        and np.array_equal(artifact_model.feature_scale, fitted["feature_scale"])
        and np.array_equal(artifact_model.coefficients, fitted["coefficients"])
        and artifact_model.intercept == fitted["intercept"]
    ):
        raise RuntimeError("H2-only artifact differs from the direct final fit.")
    probability_audit = {}
    for fit_video in h2_fit_videos:
        reference_probabilities = component_crossfit._predict_probabilities(
            fit_video.persistence_features, fitted
        )
        artifact_probabilities = artifact_model.predict_probabilities(
            fit_video.persistence_features
        )
        if not np.array_equal(reference_probabilities, artifact_probabilities):
            raise RuntimeError(
                "Per-component probability replay differs: {}".format(
                    fit_video.source_name
                )
            )
        probability_audit[fit_video.source_name] = {
            "probability_count": int(reference_probabilities.size),
            "probabilities_exact": True,
            "probabilities_sha256": common._array_sha256(reference_probabilities),
            "keep_decisions_sha256": common._array_sha256(
                reference_probabilities >= 0.5
            ),
        }
    for row in chain_audit:
        if row["domain"] == "h2":
            row.update(probability_audit[row["source_name"]])
    suppressor = PersistenceComponentSuppressor(artifact_model)
    h2_video_by_name = {item.source_name: item for item in h2_fit_videos}
    recovery = P18ScoreTrackRecovery.from_cfg(cfg, PREDICTION_THRESHOLD)
    baseline_all = component_crossfit.SufficientCounts()
    candidate_all = component_crossfit.SufficientCounts()
    baseline_h2 = component_crossfit.SufficientCounts()
    candidate_h2 = component_crossfit.SufficientCounts()
    replay_rows = []
    runtime_call_domains = []
    h1_identity = True
    outputs_exact = True
    for video in prepared:
        locations, polarities = source_inputs[video.source_name]
        p0_tensor = torch.from_numpy(video.p0_scores.copy())
        location_tensor = torch.from_numpy(video.locations.astype(np.int64, copy=False))
        baseline_c00_tensor, _ = recovery.apply(p0_tensor, location_tensor)
        if not torch.equal(baseline_c00_tensor, p0_tensor):
            raise RuntimeError("P18 unexpectedly changed {}.".format(video.source_name))
        suppressed_p0_tensor, stats = suppressor.apply(
            p0_tensor, locations, polarities
        )
        output, _ = recovery.apply(suppressed_p0_tensor, location_tensor)
        if stats.component_chain_called:
            runtime_call_domains.append(video.block)
        if video.block == "h1":
            source_identity = (
                suppressed_p0_tensor is p0_tensor
                and torch.equal(suppressed_p0_tensor, p0_tensor)
                and torch.equal(output, baseline_c00_tensor)
            )
            h1_identity = h1_identity and source_identity
            reference_output = baseline_c00_tensor
        else:
            source_identity = False
            fit_video = h2_video_by_name[video.source_name]
            probabilities = component_crossfit._predict_probabilities(
                fit_video.persistence_features, fitted
            )
            keep = probabilities >= 0.5
            reference_scores = video.p0_scores.copy()
            for indices, keep_component in zip(video.event_indices, keep):
                if not keep_component:
                    reference_scores[indices] = 0.0
            reference_output, _ = recovery.apply(
                torch.from_numpy(reference_scores), location_tensor
            )
        source_exact = torch.equal(output, reference_output)
        outputs_exact = outputs_exact and source_exact
        candidate_counts = component_crossfit.sufficient_counts_for_video(
            output.numpy(),
            video.event_labels,
            video.target_ids,
            video.locations,
        )
        baseline_all = baseline_all + video.baseline_counts
        candidate_all = candidate_all + candidate_counts
        if video.block == "h2":
            baseline_h2 = baseline_h2 + video.baseline_counts
            candidate_h2 = candidate_h2 + candidate_counts
        replay_rows.append(
            {
                "source_name": video.source_name,
                "domain": video.block,
                "fit_mass": 0.0 if video.block == "h1" else 1.0 / len(H2_NAMES),
                "component_chain_called": stats.component_chain_called,
                "input_object_identity": source_identity,
                "reference_output_exact": source_exact,
                "runtime_stats": stats.to_dict(),
                "baseline_counts": video.baseline_counts.to_dict(),
                "candidate_counts": candidate_counts.to_dict(),
                "count_delta": persistence_crossfit._count_delta(
                    candidate_counts, video.baseline_counts
                ),
            }
        )

    only_h2_called = len(runtime_call_domains) == len(H2_NAMES) and set(
        runtime_call_domains
    ) == {"h2"}
    h2_zero_target_loss = bool(
        candidate_h2.true_positive_events == baseline_h2.true_positive_events
        and candidate_h2.correct_objects == baseline_h2.correct_objects
    )
    passed = bool(
        h1_identity
        and outputs_exact
        and only_h2_called
        and len(h2_fit_videos) == len(H2_NAMES)
        and h2_zero_target_loss
        and all(
            row.get("component_indices_exact", True)
            and row.get("features_exact", True)
            and row.get("prior_exact", True)
            for row in chain_audit
        )
        and not torch.cuda.is_initialized()
    )
    if not passed:
        raise RuntimeError("Corrected H2-only train replay failed.")

    output_directory = Path(output_directory).resolve()
    if any(part.lower() in {"val", "validation", "test"} for part in output_directory.parts):
        raise ValueError("Output directory contains a forbidden split token.")
    artifact_path = output_directory / protocol["outputs"]["artifact"]
    artifact_sha = common._atomic_json_no_clobber(artifact_path, artifact_payload)
    reloaded = PersistenceArtifact.load(artifact_path, artifact_sha)
    if not np.array_equal(reloaded.coefficients, fitted["coefficients"]):
        raise RuntimeError("Written H2-only artifact differs from direct fit.")
    common._atomic_json_no_clobber(
        output_directory / protocol["outputs"]["artifact_sha256_sidecar"],
        {"path": str(artifact_path), "sha256": artifact_sha, "schema": ARTIFACT_SCHEMA},
    )

    baseline_all_metrics = component_crossfit.metrics_from_counts(baseline_all)
    candidate_all_metrics = component_crossfit.metrics_from_counts(candidate_all)
    baseline_h2_metrics = component_crossfit.metrics_from_counts(baseline_h2)
    candidate_h2_metrics = component_crossfit.metrics_from_counts(candidate_h2)
    report = {
        "schema": REPORT_SCHEMA,
        "created_utc": common.utc_now(),
        "passed": passed,
        "evidence_class": "train_only_h2_final_fit_and_replay_not_independent_held",
        "selection_disclosure": protocol["selection_disclosure"],
        "estimator_correction": protocol["estimator_correction"],
        "input_integrity": {
            "train_only": True,
            "validation_or_test_read": False,
            "t32_read_or_combined": False,
            "gpu_used": False,
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "h2_fit_source_count": len(h2_fit_videos),
            "h1_identity_only_source_count": len(H1_NAMES),
        },
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": EXPECTED_PROTOCOL_SHA256},
        "winner_report": {
            "path": str(oof_report_path),
            "sha256": EXPECTED_OOF_REPORT_SHA256,
            "candidate_id": winner["candidate_id"],
        },
        "artifact": {"path": str(artifact_path), "sha256": artifact_sha},
        "fit": {
            "source_names": list(H2_NAMES),
            "identity_only_source_names": list(H1_NAMES),
            "equal_h2_video_weighting": True,
            "fit_video_mass": 1.0 / len(H2_NAMES),
            "h1_fit_mass": 0.0,
            "candidate_component_count": int(features.shape[0]),
            "positive_candidate_component_count": int(np.count_nonzero(labels)),
            "negative_candidate_component_count": int(
                labels.size - np.count_nonzero(labels)
            ),
            "model": artifact_payload["model"],
        },
        "chain_audit": chain_audit,
        "runtime_replay": {
            "h1_identity": h1_identity,
            "only_h2_component_calls": only_h2_called,
            "reference_output_exact_all_sources": outputs_exact,
            "h2_zero_true_positive_and_correct_object_loss": h2_zero_target_loss,
            "component_call_domains": runtime_call_domains,
            "per_source": replay_rows,
        },
        "all15_metrics": {
            "baseline": {"counts": baseline_all.to_dict(), "metrics": baseline_all_metrics},
            "candidate": {"counts": candidate_all.to_dict(), "metrics": candidate_all_metrics},
            "count_delta": persistence_crossfit._count_delta(candidate_all, baseline_all),
            "metric_delta": persistence_crossfit._metric_delta(
                candidate_all_metrics, baseline_all_metrics
            ),
        },
        "h2_metrics": {
            "baseline": {"counts": baseline_h2.to_dict(), "metrics": baseline_h2_metrics},
            "candidate": {"counts": candidate_h2.to_dict(), "metrics": candidate_h2_metrics},
            "count_delta": persistence_crossfit._count_delta(candidate_h2, baseline_h2),
            "metric_delta": persistence_crossfit._metric_delta(
                candidate_h2_metrics, baseline_h2_metrics
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "elapsed_seconds": time.monotonic() - started,
        },
        "code_sha256": {
            relative: sha256_file(PROJECT_ROOT / relative) for relative in code_paths
        },
    }
    report_path = output_directory / protocol["outputs"]["train_replay_report"]
    report_sha = common._atomic_json_no_clobber(report_path, report)
    common._atomic_json_no_clobber(
        output_directory / protocol["outputs"]["train_replay_report_sha256_sidecar"],
        {"path": str(report_path), "sha256": report_sha, "schema": REPORT_SCHEMA},
    )
    return artifact_path, artifact_sha, report_path, report_sha, report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        default=str(WORKSPACE_ROOT / "experiments" / "20260810_persistence_standalone_v2"),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    artifact_path, artifact_sha, report_path, report_sha, report = run(
        args.output_directory
    )
    print("artifact:", artifact_path)
    print("artifact_sha256:", artifact_sha)
    print("report:", report_path)
    print("report_sha256:", report_sha)
    print("passed:", report["passed"])
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
