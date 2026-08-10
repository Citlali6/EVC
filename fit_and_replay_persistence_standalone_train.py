"""Fit and CPU-replay the frozen persistence_pw08_kp050 standalone artifact.

This entry point accepts only the frozen 15-source official-train population and
the immutable full-stream M20 train cache.  It never accepts validation, test,
or T32 inputs.  Candidate identity is taken from the hash-bound train-only OOF
v2 report before any final all-source fit is performed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

import crossfit_component_reranker as component_crossfit
import crossfit_persistent_pixel_prior as persistence_crossfit
import replay_temporal_memory_validation as replay
from train_component_reranker import load_train_cache
from utils.component_reranker import sha256_file, sha256_json, temporal_memory_inference_mapping
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
    PROJECT_ROOT / "protocols" / "persistence_standalone_train_fit_v1.json"
).resolve()
EXPECTED_PROTOCOL_SHA256 = "131096c0968994c7098ddf117e410ebda2bcd26ba8e0fe18e771a1f9c237177c"
EXPECTED_OOF_REPORT_SHA256 = "acfdda1910305834ad217117b3c237fb496050aa1c6d3f0669594feaac3e96ae"
EXPECTED_CACHE_MANIFEST_SHA256 = "05a707dcfeb8487fafdb99599abfff81b452c6fac9d1938da47f711097257f82"
EXPECTED_REFERENCE_PROTOCOL_SHA256 = "babd85216e83c3e6324d2ddc06d39c0c38669e4a7da6f65ea5884dca6ecfb9d2"
EXPECTED_CONFIG_SHA256 = "c4157f6e04fb96be1fe9bef6ed87004b1e7da0d72507a43091e9f929345f2ec9"
EXPECTED_M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
ARTIFACT_NAME = "persistence_pw08_kp050.json"
REPORT_SCHEMA = "ev-uav-persistence-standalone-train-replay-v1"
H1_NAMES = persistence_crossfit.H1_NAMES
H2_NAMES = persistence_crossfit.H2_NAMES
HIGH_NAMES = persistence_crossfit.HIGH_NAMES


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _load_json_snapshot(path, expected_sha256, description):
    path = Path(path).resolve()
    before = sha256_file(path)
    if before != expected_sha256:
        raise ValueError("{} SHA-256 differs.".format(description))
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if sha256_file(path) != before:
        raise RuntimeError("{} changed while being read.".format(description))
    return payload


def _atomic_json_no_clobber(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Refusing to overwrite: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _workspace_input(relative_path, description):
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("{} must be workspace-relative.".format(description))
    resolved = (WORKSPACE_ROOT / relative).resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise ValueError("{} escapes the workspace root.".format(description)) from error
    return resolved


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _component_indices_sha256(values):
    digest = hashlib.sha256()
    for indices in values:
        array = np.ascontiguousarray(indices, dtype=np.int64).reshape(-1)
        digest.update(int(array.size).to_bytes(8, "little", signed=False))
        digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _raw_xytp(path):
    with np.load(path, allow_pickle=False) as source:
        if "ev_loc" not in source.files or "ev" not in source.files:
            raise ValueError("Raw train source lacks ev_loc/ev: {}".format(path))
        locations = np.ascontiguousarray(source["ev_loc"], dtype=np.int64)
        events = source["ev"]
        if events.dtype.names is None or "p" not in events.dtype.names:
            raise ValueError("Raw train source lacks polarity field: {}".format(path))
        polarities = np.ascontiguousarray(events["p"] > 0, dtype=np.uint8)
    return locations, polarities


def _prior_fields_equal(reference, candidate):
    fields = (
        "event_pixel_ids",
        "log_events",
        "active_fraction",
        "longest_run_fraction",
        "collision_fraction",
        "log_max_bin_events",
        "polarity_dominance",
        "neighbor_active_fraction",
    )
    return all(
        np.array_equal(getattr(reference, field), getattr(candidate, field))
        for field in fields
    ) and reference.summary == candidate.summary


def validate_protocol_and_winner():
    protocol = _load_json_snapshot(
        PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256, "standalone train-fit protocol"
    )
    if (
        protocol.get("schema")
        != "ev-uav-persistence-standalone-train-fit-protocol-v1"
        or protocol.get("status") != "frozen_before_final_15_source_fit"
        or protocol.get("candidate_id") != "persistence_pw08_kp050"
        or protocol.get("split_access", {}).get("gpu_allowed") is not False
        or protocol.get("standalone_runtime", {}).get("t32_allowed") is not False
    ):
        raise ValueError("Frozen standalone train-fit protocol identity differs.")
    if tuple(protocol["population"]["h1_sources"]) != H1_NAMES or tuple(
        protocol["population"]["h2_sources"]
    ) != H2_NAMES:
        raise ValueError("Frozen 15-source population differs.")
    if protocol["final_fit"] != {
        "family": "persistence_only",
        "feature_names": list(FEATURE_NAMES),
        "equal_video_weighting": "each of 15 videos has mass 1/15; each candidate component within a video receives mass (1/15)/component_count before class weighting",
        "positive_weight": 8.0,
        "keep_probability": 0.5,
        "l2_penalty": 0.1,
        "max_newton_iterations": 50,
        "standardization": "weighted mean and population scale using equal-video base weights",
        "component_topology": DEFAULT_TOPOLOGY.to_dict(),
        "prediction_threshold": PREDICTION_THRESHOLD,
        "temporal_bin_count": 160,
        "video_duration": 8000,
        "resolution": [346, 260],
    }:
        raise ValueError("Frozen final-fit settings differ.")
    binding = protocol["winner_binding"]
    if binding["report_sha256"] != EXPECTED_OOF_REPORT_SHA256:
        raise ValueError("Frozen OOF report binding differs.")
    report_path = _workspace_input(
        binding["report_workspace_relative_path"], "train-only OOF report"
    )
    report = _load_json_snapshot(
        report_path, EXPECTED_OOF_REPORT_SHA256, "train-only OOF report"
    )
    if (
        report.get("schema") != binding["report_schema"]
        or report.get("pooled_oof", {}).get("conservative_winner_candidate_id")
        != binding["expected_value"]
    ):
        raise ValueError("Hash-bound conservative winner differs.")
    winner = next(
        candidate
        for candidate in report["pooled_oof"]["candidates"]
        if candidate["candidate_id"] == binding["expected_value"]
    )
    checks = binding["required_oof_checks"]
    if (
        winner.get("family") != "persistence"
        or float(winner.get("positive_weight")) != 8.0
        or float(winner.get("keep_probability")) != 0.5
        or winner.get("conservative_gate_passed") is not True
        or int(winner.get("nonnegative_score_fold_count"))
        != checks["nonnegative_score_fold_count"]
        or int(winner["count_delta"]["true_positive_events"])
        != checks["true_positive_event_delta"]
        or int(winner["count_delta"]["correct_objects"])
        != checks["correct_object_delta"]
        or float(winner["metric_delta"]["pd"]) != checks["pd_delta"]
        or float(winner["metric_delta"]["iou"]) < checks["iou_delta_minimum"]
        or float(winner["metric_delta"]["fa"]) > checks["fa_delta_maximum"]
    ):
        raise ValueError("Hash-bound winner no longer satisfies frozen gates.")
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


def _count_delta(candidate, baseline):
    return persistence_crossfit._count_delta(candidate, baseline)


def _metric_delta(candidate, baseline):
    return persistence_crossfit._metric_delta(candidate, baseline)


def run(output_directory):
    started = time.monotonic()
    protocol, oof_report_path, winner = validate_protocol_and_winner()
    inputs = protocol["inputs"]
    manifest_path = _workspace_input(
        inputs["full_m20_train_cache_manifest"]["workspace_relative_path"],
        "full M20 train cache manifest",
    )
    reference_protocol_path = _workspace_input(
        inputs["reference_protocol"]["workspace_relative_path"],
        "reference train protocol",
    )
    config_path = _workspace_input(
        inputs["config"]["workspace_relative_path"], "config"
    )
    raw_train_root = _workspace_input(
        inputs["raw_train_root_workspace_relative_path"], "raw train root"
    )
    if raw_train_root.name.lower() != "train":
        raise ValueError("Raw input root is not the official train directory.")
    if sha256_file(manifest_path) != EXPECTED_CACHE_MANIFEST_SHA256:
        raise ValueError("Full M20 train manifest SHA-256 differs.")
    if sha256_file(reference_protocol_path) != EXPECTED_REFERENCE_PROTOCOL_SHA256:
        raise ValueError("Reference train protocol SHA-256 differs.")
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("Current config SHA-256 differs.")
    for relative, expected_hash in inputs["source_code_sha256"].items():
        if sha256_file(PROJECT_ROOT / relative) != expected_hash:
            raise ValueError("Frozen source dependency differs: {}".format(relative))

    reference_payload, overrides = persistence_crossfit._read_reference_protocol(
        reference_protocol_path
    )
    cfg = replay.load_flat_config(config_path, overrides)
    component_crossfit.validate_c00_config(cfg)
    cache_dir, loaded_manifest_path, manifest_sha, manifest = load_train_cache(
        manifest_path.parent
    )
    if loaded_manifest_path.resolve() != manifest_path or manifest_sha != EXPECTED_CACHE_MANIFEST_SHA256:
        raise ValueError("Loaded full M20 train manifest identity differs.")
    if manifest.get("base_checkpoint_sha256") != EXPECTED_M20_SHA256:
        raise ValueError("Full M20 checkpoint binding differs.")
    records_by_name = {record["source_name"]: record for record in manifest["records"]}
    if not set(HIGH_NAMES).issubset(records_by_name):
        raise ValueError("Full M20 train cache lacks the frozen 15 sources.")
    selected_manifest = dict(manifest)
    selected_manifest["records"] = [records_by_name[name] for name in HIGH_NAMES]

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized; this fit/replay is CPU-only.")
    prepared = component_crossfit._prepare_videos(
        cache_dir, selected_manifest, cfg, middle_names=set()
    )
    if tuple(video.source_name for video in prepared) != HIGH_NAMES:
        raise RuntimeError("Prepared 15-source order differs.")

    prior_videos = []
    source_inputs = {}
    chain_audit = []
    for index, video in enumerate(prepared, start=1):
        raw_path = raw_train_root / video.source_name
        expected_source_hash = protocol["population"]["raw_source_sha256"][
            video.source_name
        ]
        if sha256_file(raw_path) != expected_source_hash:
            raise ValueError("Raw train source SHA-256 differs: {}".format(video.source_name))
        locations, polarities = _raw_xytp(raw_path)
        if not np.array_equal(locations, video.locations[:, 1:4]):
            raise RuntimeError("Raw/cache locations differ: {}".format(video.source_name))
        reference_prior = persistence_crossfit.derive_pixel_prior(
            raw_path, video.locations[:, 1:4]
        )
        runtime_prior = derive_pixel_prior_from_arrays(locations, polarities)
        if not _prior_fields_equal(reference_prior, runtime_prior):
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
        indices_equal = len(runtime_indices) == len(video.event_indices) and all(
            np.array_equal(left, right)
            for left, right in zip(runtime_indices, video.event_indices)
        )
        features_equal = np.array_equal(runtime_features, reference_features)
        if not indices_equal or not features_equal or runtime_summary != reference_prior.summary:
            raise RuntimeError(
                "Standalone component/feature chain differs: {}".format(video.source_name)
            )
        prior_videos.append(
            persistence_crossfit.PriorVideo(video, runtime_features, runtime_summary)
        )
        source_inputs[video.source_name] = (locations, polarities)
        chain_audit.append(
            {
                "source_name": video.source_name,
                "domain": video.block,
                "event_count": video.event_count,
                "candidate_component_count": len(runtime_indices),
                "positive_candidate_component_count": int(
                    np.count_nonzero(video.component_labels)
                ),
                "component_indices_sha256": _component_indices_sha256(runtime_indices),
                "features_sha256": _array_sha256(runtime_features),
                "prior_summary_sha256": sha256_json(runtime_summary),
                "component_indices_exact": True,
                "features_exact": True,
                "prior_exact": True,
            }
        )
        print(
            "chain {}/15: {} [{}] {} components".format(
                index, video.source_name, video.block, len(runtime_indices)
            ),
            flush=True,
        )

    features, labels, base_weights = persistence_crossfit._balanced_dataset(
        prior_videos, "persistence"
    )
    fitted = persistence_crossfit._fit_balanced_logistic(
        features,
        labels,
        base_weights,
        positive_weight=8.0,
        l2=0.1,
        max_iterations=50,
    )
    code_paths = (
        "fit_and_replay_persistence_standalone_train.py",
        "utils/persistence_component_suppressor.py",
        "protocols/persistence_standalone_train_fit_v1.json",
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
        "created_utc": utc_now(),
        "candidate_id": "persistence_pw08_kp050",
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "component_topology": DEFAULT_TOPOLOGY.to_dict(),
        "model": _model_payload(fitted),
        "runtime_contract": {
            "upstream": "released_M20_full_stream_T160_threshold_0.719_C00",
            "prediction_threshold": PREDICTION_THRESHOLD,
            "event_count_cutoff_exclusive": 200000,
            "polarity_minority_cutoff": 0.2,
            "eligible_route": "h2",
            "non_h2_identity": True,
            "component_calls_only_h2": True,
            "t32_allowed": False,
        },
        "training_provenance": {
            "evidence_class": "train_only_final_fit_after_frozen_five_fold_selection",
            "selection_disclosure": protocol["selection_disclosure"],
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
            "source_names": list(HIGH_NAMES),
            "source_sha256": protocol["population"]["raw_source_sha256"],
            "equal_video_weighting": True,
            "video_mass": 1.0 / len(HIGH_NAMES),
            "candidate_component_count": int(features.shape[0]),
            "positive_candidate_component_count": int(np.count_nonzero(labels)),
            "negative_candidate_component_count": int(labels.size - np.count_nonzero(labels)),
            "effective_c00_postprocess": component_crossfit._postprocess_contract(cfg),
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
        raise RuntimeError("Serialized artifact model differs from direct final fit.")
    suppressor = PersistenceComponentSuppressor(artifact_model)

    baseline_all = component_crossfit.SufficientCounts()
    candidate_all = component_crossfit.SufficientCounts()
    baseline_h2 = component_crossfit.SufficientCounts()
    candidate_h2 = component_crossfit.SufficientCounts()
    replay_rows = []
    runtime_call_domains = []
    h1_identity = True
    outputs_match_reference = True
    recovery = P18ScoreTrackRecovery.from_cfg(cfg, PREDICTION_THRESHOLD)
    for video in prepared:
        locations, polarities = source_inputs[video.source_name]
        p0_tensor = torch.from_numpy(video.p0_scores.copy())
        location_tensor = torch.from_numpy(video.locations.astype(np.int64, copy=False))
        c00_tensor, _ = recovery.apply(p0_tensor, location_tensor)
        if not torch.equal(c00_tensor, p0_tensor):
            raise RuntimeError("P18 is unexpectedly active for {}.".format(video.source_name))
        output, stats = suppressor.apply(c00_tensor, locations, polarities)
        if stats.component_chain_called:
            runtime_call_domains.append(video.block)
        if video.block == "h1":
            source_identity = output is c00_tensor and torch.equal(output, c00_tensor)
            h1_identity = h1_identity and source_identity
            reference_output = c00_tensor
        else:
            source_identity = False
            probabilities = component_crossfit._predict_probabilities(
                next(item for item in prior_videos if item.source_name == video.source_name).persistence_features,
                fitted,
            )
            keep = probabilities >= 0.5
            reference_scores = video.p0_scores.copy()
            for indices, keep_component in zip(video.event_indices, keep):
                if not keep_component:
                    reference_scores[indices] = 0.0
            reference_output, _ = recovery.apply(
                torch.from_numpy(reference_scores), location_tensor
            )
        source_match = torch.equal(output, reference_output)
        outputs_match_reference = outputs_match_reference and source_match
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
                "event_count": video.event_count,
                "route": stats.route,
                "component_chain_called": stats.component_chain_called,
                "input_object_identity": source_identity,
                "reference_output_exact": source_match,
                "runtime_stats": stats.to_dict(),
                "baseline_counts": video.baseline_counts.to_dict(),
                "candidate_counts": candidate_counts.to_dict(),
                "count_delta": _count_delta(candidate_counts, video.baseline_counts),
                "output_sha256": _array_sha256(output.numpy()),
            }
        )

    baseline_all_metrics = component_crossfit.metrics_from_counts(baseline_all)
    candidate_all_metrics = component_crossfit.metrics_from_counts(candidate_all)
    baseline_h2_metrics = component_crossfit.metrics_from_counts(baseline_h2)
    candidate_h2_metrics = component_crossfit.metrics_from_counts(candidate_h2)
    only_h2_called = bool(runtime_call_domains) and set(runtime_call_domains) == {"h2"}
    replay_passed = bool(
        h1_identity
        and outputs_match_reference
        and only_h2_called
        and all(row["component_indices_exact"] for row in chain_audit)
        and all(row["features_exact"] for row in chain_audit)
        and all(row["prior_exact"] for row in chain_audit)
        and not torch.cuda.is_initialized()
    )
    if not replay_passed:
        raise RuntimeError("Standalone train replay consistency gate failed.")

    output_directory = Path(output_directory).resolve()
    if any(part.lower() in {"val", "validation", "test"} for part in output_directory.parts):
        raise ValueError("Output directory contains a forbidden split token.")
    artifact_path = output_directory / protocol["outputs"]["artifact"]
    artifact_sha = _atomic_json_no_clobber(artifact_path, artifact_payload)
    loaded_artifact = PersistenceArtifact.load(artifact_path, artifact_sha)
    if not np.array_equal(loaded_artifact.coefficients, fitted["coefficients"]):
        raise RuntimeError("Written artifact reload differs from direct final fit.")
    artifact_sidecar_path = output_directory / protocol["outputs"][
        "artifact_sha256_sidecar"
    ]
    _atomic_json_no_clobber(
        artifact_sidecar_path,
        {"path": str(artifact_path), "sha256": artifact_sha, "schema": ARTIFACT_SCHEMA},
    )

    report = {
        "schema": REPORT_SCHEMA,
        "created_utc": utc_now(),
        "passed": replay_passed,
        "evidence_class": "train_only_final_fit_and_replay_not_independent_held",
        "selection_disclosure": protocol["selection_disclosure"],
        "input_integrity": {
            "train_only": True,
            "validation_or_test_read": False,
            "t32_read_or_combined": False,
            "gpu_used": False,
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "source_count": len(HIGH_NAMES),
            "h1_source_count": len(H1_NAMES),
            "h2_source_count": len(H2_NAMES),
        },
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": EXPECTED_PROTOCOL_SHA256},
        "winner_report": {
            "path": str(oof_report_path),
            "sha256": EXPECTED_OOF_REPORT_SHA256,
            "candidate_id": winner["candidate_id"],
        },
        "artifact": {
            "path": str(artifact_path),
            "sha256": artifact_sha,
            "model_content_sha256": sha256_json(artifact_payload["model"]),
        },
        "fit": {
            "source_names": list(HIGH_NAMES),
            "equal_video_weighting": True,
            "candidate_component_count": int(features.shape[0]),
            "positive_candidate_component_count": int(np.count_nonzero(labels)),
            "negative_candidate_component_count": int(labels.size - np.count_nonzero(labels)),
            "model": artifact_payload["model"],
        },
        "chain_audit": chain_audit,
        "runtime_replay": {
            "h1_identity": h1_identity,
            "only_h2_component_calls": only_h2_called,
            "reference_output_exact_all_sources": outputs_match_reference,
            "component_call_domains": runtime_call_domains,
            "per_source": replay_rows,
        },
        "all15_metrics": {
            "baseline": {"counts": baseline_all.to_dict(), "metrics": baseline_all_metrics},
            "candidate": {"counts": candidate_all.to_dict(), "metrics": candidate_all_metrics},
            "count_delta": _count_delta(candidate_all, baseline_all),
            "metric_delta": _metric_delta(candidate_all_metrics, baseline_all_metrics),
        },
        "h2_metrics": {
            "baseline": {"counts": baseline_h2.to_dict(), "metrics": baseline_h2_metrics},
            "candidate": {"counts": candidate_h2.to_dict(), "metrics": candidate_h2_metrics},
            "count_delta": _count_delta(candidate_h2, baseline_h2),
            "metric_delta": _metric_delta(candidate_h2_metrics, baseline_h2_metrics),
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
    report_sha = _atomic_json_no_clobber(report_path, report)
    report_sidecar_path = output_directory / protocol["outputs"][
        "train_replay_report_sha256_sidecar"
    ]
    _atomic_json_no_clobber(
        report_sidecar_path,
        {"path": str(report_path), "sha256": report_sha, "schema": REPORT_SCHEMA},
    )
    return artifact_path, artifact_sha, report_path, report_sha, report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        default=str(WORKSPACE_ROOT / "experiments" / "20260810_persistence_standalone_v1"),
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
